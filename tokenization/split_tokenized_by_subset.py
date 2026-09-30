# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create subset-level Megatron indexes without copying tokenized binary data.

The original DataTrove runs recursively tokenized every source file below one
top-level dataset into rank-level ``.bin/.idx`` pairs. This tool reconstructs
which documents came from each source subdirectory, writes filtered indexes,
and symlinks each filtered prefix to its original binary shard.

The resulting layout is::

    <tokenized-root>/<dataset>/subsets/<subset>/
      data/dataset_<rank>_tokens.idx
      data/dataset_<rank>_tokens.bin -> ../../../data/dataset_<rank>_tokens.bin
      data_prefixes.txt
      manifest.json

Only the indexes consume additional material space. Token bytes remain in the
original binary shards and are shared by all subset views.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import struct
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


LOGGER = logging.getLogger(__name__)

INDEX_HEADER = b"MMIDIDX\x00\x00"
INDEX_VERSION = 1
INDEX_METADATA_BYTES = 34
INDEX_BYTES_PER_SEQUENCE = 20
INDEX_TRAILING_BYTES = 8
SAVE_FILENAME = "dataset"


@dataclass(frozen=True)
class IndexMetadata:
    """Metadata and array offsets for one Megatron index."""

    dtype_code: int
    sequence_count: int
    document_count: int
    lengths_offset: int
    pointers_offset: int
    documents_offset: int


@dataclass(frozen=True)
class FileAssignment:
    """One source file assigned to an executor rank and logical subset."""

    relative_path: str
    subset: str


@dataclass(frozen=True)
class RankWork:
    """All information required to split one original output shard."""

    dataset: str
    rank: int
    source_root: Path
    parent_data_dir: Path
    subset_root: Path
    text_column: str
    assignments: tuple[FileAssignment, ...]


@dataclass(frozen=True)
class ShardResult:
    """Subset statistics produced from one parent rank."""

    subset: str
    rank: int
    documents: int
    tokens: int
    prefix: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split completed Megatron indexes into zero-copy source-subset views."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing the original top-level dataset directories.",
    )
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        required=True,
        help="Root containing completed top-level tokenization outputs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Top-level dataset to inspect; repeat to select several. Defaults to every completed dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Parallel rank workers used for Parquet footer reads and index generation.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Source Parquet column used by the original identity renderer.",
    )
    parser.add_argument(
        "--scan-empty-text",
        action="store_true",
        help="Scan text columns in a mismatched rank to reproduce DataTrove's empty-text filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report affected datasets and estimated index space without creating subset views.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_index_metadata(index_path: Path) -> IndexMetadata:
    with index_path.open("rb") as index_file:
        header = index_file.read(len(INDEX_HEADER))
        if header != INDEX_HEADER:
            raise ValueError(f"Invalid Megatron index header in {index_path}")
        version = struct.unpack("<Q", index_file.read(8))[0]
        if version != INDEX_VERSION:
            raise ValueError(f"Unsupported Megatron index version {version} in {index_path}")
        dtype_code = struct.unpack("<B", index_file.read(1))[0]
        sequence_count = struct.unpack("<Q", index_file.read(8))[0]
        document_count = struct.unpack("<Q", index_file.read(8))[0]

    lengths_offset = INDEX_METADATA_BYTES
    pointers_offset = lengths_offset + sequence_count * 4
    documents_offset = pointers_offset + sequence_count * 8
    expected_size = documents_offset + document_count * 8
    actual_size = index_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Index size mismatch for {index_path}: expected {expected_size}, found {actual_size}")
    return IndexMetadata(
        dtype_code=dtype_code,
        sequence_count=sequence_count,
        document_count=document_count,
        lengths_offset=lengths_offset,
        pointers_offset=pointers_offset,
        documents_offset=documents_offset,
    )


def _common_directory_prefix(paths: list[str]) -> tuple[str, ...]:
    directories = [Path(path).parts[:-1] for path in paths]
    if not directories:
        return ()
    prefix: list[str] = []
    for components in zip(*directories):
        if len(set(components)) != 1:
            break
        prefix.append(components[0])
    return tuple(prefix)


def _discover_subsets(paths: list[str]) -> tuple[tuple[str, ...], dict[str, str]]:
    common_prefix = _common_directory_prefix(paths)
    subset_by_path: dict[str, str] = {}
    for path in paths:
        parts = Path(path).parts
        directory_parts = parts[:-1]
        if len(directory_parts) <= len(common_prefix):
            subset_by_path[path] = ""
        else:
            subset_by_path[path] = directory_parts[len(common_prefix)]
    return common_prefix, subset_by_path


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as parquet

    return int(parquet.ParquetFile(path).metadata.num_rows)


def _parquet_nonempty_text_count(path: Path, text_column: str) -> int:
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    count = 0
    with path.open("rb") as input_file:
        parquet_file = parquet.ParquetFile(input_file)
        for batch in parquet_file.iter_batches(columns=[text_column]):
            values = batch.column(0)
            valid = compute.is_valid(values)
            nonempty = compute.greater(compute.utf8_length(values), 0)
            count += int(compute.sum(compute.and_(valid, nonempty)).as_py() or 0)
    return count


def _copy_array_segments(
    output_file: BinaryIO,
    source_view: memoryview,
    segments: list[tuple[int, int]],
    *,
    item_size: int,
) -> None:
    for start, end in segments:
        output_file.write(source_view[start * item_size : end * item_size])


def _write_document_indices(output_file: BinaryIO, document_count: int) -> None:
    import numpy as np

    chunk_size = 1_000_000
    for start in range(0, document_count, chunk_size):
        end = min(document_count, start + chunk_size)
        output_file.write(np.arange(start, end, dtype="<i8").tobytes(order="C"))


def _write_filtered_index(
    *,
    source_index: Path,
    target_index: Path,
    metadata: IndexMetadata,
    segments: list[tuple[int, int]],
) -> None:
    sequence_count = sum(end - start for start, end in segments)
    expected_size = INDEX_METADATA_BYTES + INDEX_BYTES_PER_SEQUENCE * sequence_count + INDEX_TRAILING_BYTES
    if target_index.is_file():
        existing = _read_index_metadata(target_index)
        if existing.dtype_code == metadata.dtype_code and existing.sequence_count == sequence_count:
            if target_index.stat().st_size == expected_size:
                return
        raise FileExistsError(f"Existing subset index does not match expected output: {target_index}")

    target_index.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with source_index.open("rb") as source_file:
            import mmap

            with mmap.mmap(source_file.fileno(), length=0, access=mmap.ACCESS_READ) as mapped_index:
                source_view = memoryview(mapped_index)
                lengths: memoryview | None = None
                pointers: memoryview | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=target_index.parent,
                        prefix=f".{target_index.name}.",
                        delete=False,
                    ) as output_file:
                        temporary_path = Path(output_file.name)
                        output_file.write(INDEX_HEADER)
                        output_file.write(struct.pack("<Q", INDEX_VERSION))
                        output_file.write(struct.pack("<B", metadata.dtype_code))
                        output_file.write(struct.pack("<Q", sequence_count))
                        output_file.write(struct.pack("<Q", sequence_count + 1))
                        lengths = source_view[
                            metadata.lengths_offset : metadata.lengths_offset + metadata.sequence_count * 4
                        ]
                        pointers = source_view[
                            metadata.pointers_offset : metadata.pointers_offset + metadata.sequence_count * 8
                        ]
                        _copy_array_segments(output_file, lengths, segments, item_size=4)
                        _copy_array_segments(output_file, pointers, segments, item_size=8)
                        _write_document_indices(output_file, sequence_count + 1)
                        output_file.flush()
                        os.fsync(output_file.fileno())
                finally:
                    if lengths is not None:
                        lengths.release()
                    if pointers is not None:
                        pointers.release()
                    source_view.release()
        if temporary_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Generated index size mismatch for {target_index}: "
                f"expected {expected_size}, found {temporary_path.stat().st_size}"
            )
        os.replace(temporary_path, target_index)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _ensure_bin_symlink(*, source_bin: Path, target_bin: Path) -> None:
    relative_target = Path(os.path.relpath(source_bin, start=target_bin.parent))
    if target_bin.is_symlink():
        if Path(os.readlink(target_bin)) != relative_target:
            raise FileExistsError(f"Existing symlink has the wrong target: {target_bin}")
        return
    if target_bin.exists():
        raise FileExistsError(f"Refusing to replace existing non-symlink binary: {target_bin}")
    target_bin.parent.mkdir(parents=True, exist_ok=True)
    temporary_link = target_bin.parent / f".{target_bin.name}.{os.getpid()}.tmp"
    try:
        temporary_link.symlink_to(relative_target)
        os.replace(temporary_link, target_bin)
    finally:
        temporary_link.unlink(missing_ok=True)


def _sum_lengths(source_index: Path, metadata: IndexMetadata, segments: list[tuple[int, int]]) -> int:
    import numpy as np

    lengths = np.memmap(
        source_index,
        dtype="<i4",
        mode="r",
        offset=metadata.lengths_offset,
        shape=(metadata.sequence_count,),
    )
    try:
        return sum(int(lengths[start:end].sum(dtype=np.int64)) for start, end in segments)
    finally:
        del lengths


def _split_rank(work: RankWork, *, scan_empty_text: bool) -> list[ShardResult]:
    prefix_name = f"{SAVE_FILENAME}_{work.rank:05d}_tokens"
    source_index = work.parent_data_dir / f"{prefix_name}.idx"
    source_bin = work.parent_data_dir / f"{prefix_name}.bin"
    if not source_index.is_file() or not source_bin.is_file():
        raise FileNotFoundError(f"Missing parent shard for rank {work.rank}: {source_index} / {source_bin}")

    metadata = _read_index_metadata(source_index)
    if metadata.document_count != metadata.sequence_count + 1:
        raise ValueError(
            f"Rank {work.rank} is not one-sequence-per-document: "
            f"sequences={metadata.sequence_count}, document_index_entries={metadata.document_count}"
        )

    row_counts = [_parquet_row_count(work.source_root / assignment.relative_path) for assignment in work.assignments]
    if sum(row_counts) != metadata.sequence_count:
        if not scan_empty_text:
            raise ValueError(
                f"Rank {work.rank} has {sum(row_counts)} Parquet rows but {metadata.sequence_count} tokenized documents. "
                "Rerun with --scan-empty-text to reproduce DataTrove's empty-text filtering."
            )
        row_counts = [
            _parquet_nonempty_text_count(work.source_root / assignment.relative_path, work.text_column)
            for assignment in work.assignments
        ]
    if sum(row_counts) != metadata.sequence_count:
        raise ValueError(
            f"Cannot reconcile source rows with tokenized documents for rank {work.rank}: "
            f"source={sum(row_counts)}, tokenized={metadata.sequence_count}"
        )

    segments_by_subset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cursor = 0
    for assignment, document_count in zip(work.assignments, row_counts):
        end = cursor + document_count
        if document_count:
            segments_by_subset[assignment.subset].append((cursor, end))
        cursor = end

    results: list[ShardResult] = []
    for subset, segments in segments_by_subset.items():
        subset_dir = work.subset_root / subset
        target_index = subset_dir / "data" / f"{prefix_name}.idx"
        target_bin = target_index.with_suffix(".bin")
        _write_filtered_index(
            source_index=source_index,
            target_index=target_index,
            metadata=metadata,
            segments=segments,
        )
        _ensure_bin_symlink(source_bin=source_bin, target_bin=target_bin)
        document_count = sum(end - start for start, end in segments)
        results.append(
            ShardResult(
                subset=subset,
                rank=work.rank,
                documents=document_count,
                tokens=_sum_lengths(source_index, metadata, segments),
                prefix=f"data/{prefix_name}",
            )
        )
    return results


def _selected_datasets(tokenized_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(set(requested))
    return sorted(
        path.name
        for path in tokenized_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file() and (path / "result.json").is_file()
    )


def _prepare_dataset(
    *,
    dataset: str,
    source_root: Path,
    tokenized_root: Path,
    text_column: str,
) -> tuple[tuple[str, ...], dict[str, str], list[RankWork], int]:
    dataset_root = tokenized_root / dataset
    manifest = _read_json(dataset_root / "manifest.json")
    pipeline = manifest.get("pipeline")
    renderer = manifest.get("renderer")
    source_metadata = manifest.get("dataset")
    output_metadata = manifest.get("output")
    if not isinstance(pipeline, dict) or not isinstance(source_metadata, dict) or not isinstance(output_metadata, dict):
        raise ValueError(f"Malformed tokenization manifest: {dataset_root / 'manifest.json'}")
    if renderer is not None and (not isinstance(renderer, dict) or renderer.get("name") != "identity"):
        return (), {}, [], 0
    if source_metadata.get("input_format") != "parquet":
        return (), {}, [], 0

    tasks = int(pipeline["tasks"])
    paths_file = dataset_root / str(manifest["input_paths_file"])
    paths = [line for line in paths_file.read_text(encoding="utf-8").splitlines() if line]
    common_prefix, subset_by_path = _discover_subsets(paths)
    subsets = {subset for subset in subset_by_path.values() if subset}
    if "" in subset_by_path.values() or len(subsets) <= 1:
        return common_prefix, subset_by_path, [], 0

    parent_data_dir = dataset_root / str(output_metadata["data_dir"])
    subset_root = dataset_root / "subsets"
    source_dataset_root = source_root / dataset
    work_items = []
    for rank in range(tasks):
        rank_paths = paths[rank::tasks]
        assignments = tuple(FileAssignment(path, subset_by_path[path]) for path in rank_paths)
        if assignments:
            work_items.append(
                RankWork(
                    dataset=dataset,
                    rank=rank,
                    source_root=source_dataset_root,
                    parent_data_dir=parent_data_dir,
                    subset_root=subset_root,
                    text_column=text_column,
                    assignments=assignments,
                )
            )
    index_bytes = sum(path.stat().st_size for path in parent_data_dir.glob(f"{SAVE_FILENAME}_*_tokens.idx"))
    return common_prefix, subset_by_path, work_items, index_bytes


def _write_subset_manifests(
    *,
    dataset: str,
    tokenized_root: Path,
    common_prefix: tuple[str, ...],
    results: list[ShardResult],
) -> None:
    subset_root = tokenized_root / dataset / "subsets"
    results_by_subset: dict[str, list[ShardResult]] = defaultdict(list)
    for result in results:
        results_by_subset[result.subset].append(result)

    top_level_subsets: dict[str, object] = {}
    for subset, subset_results in sorted(results_by_subset.items()):
        subset_results.sort(key=lambda item: item.rank)
        subset_dir = subset_root / subset
        prefixes = [result.prefix for result in subset_results]
        documents = sum(result.documents for result in subset_results)
        tokens = sum(result.tokens for result in subset_results)
        _atomic_write_text(subset_dir / "data_prefixes.txt", "\n".join(prefixes) + "\n")
        manifest = {
            "schema_version": 1,
            "storage": "filtered-index-with-shared-bin-symlink",
            "parent_dataset": dataset,
            "source_directory": "/".join((*common_prefix, subset)),
            "subset": subset,
            "documents": documents,
            "tokens": tokens,
            "shards": [
                {
                    "documents": result.documents,
                    "prefix": result.prefix,
                    "rank": result.rank,
                    "tokens": result.tokens,
                }
                for result in subset_results
            ],
        }
        _atomic_write_text(subset_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        top_level_subsets[subset] = {
            "data_prefixes_file": f"{subset}/data_prefixes.txt",
            "documents": documents,
            "manifest": f"{subset}/manifest.json",
            "shards": len(subset_results),
            "source_directory": "/".join((*common_prefix, subset)),
            "tokens": tokens,
        }

    parent_result = _read_json(tokenized_root / dataset / "result.json")
    result_metadata = parent_result.get("result")
    if not isinstance(result_metadata, dict):
        raise ValueError(f"Malformed parent result for {dataset}")
    total_documents = sum(int(value["documents"]) for value in top_level_subsets.values())
    total_tokens = sum(int(value["tokens"]) for value in top_level_subsets.values())
    if total_documents != int(result_metadata["documents"]) or total_tokens != int(result_metadata["tokens"]):
        raise RuntimeError(
            f"Subset totals do not reconcile for {dataset}: documents={total_documents}, tokens={total_tokens}; "
            f"parent documents={result_metadata['documents']}, tokens={result_metadata['tokens']}"
        )
    top_level_manifest = {
        "schema_version": 1,
        "common_source_prefix": "/".join(common_prefix),
        "documents": total_documents,
        "parent_dataset": dataset,
        "storage": "filtered-index-with-shared-bin-symlink",
        "subsets": top_level_subsets,
        "tokens": total_tokens,
    }
    _atomic_write_text(subset_root / "manifest.json", json.dumps(top_level_manifest, indent=2, sort_keys=True) + "\n")


def _run_dataset(
    *,
    dataset: str,
    source_root: Path,
    tokenized_root: Path,
    text_column: str,
    workers: int,
    scan_empty_text: bool,
    dry_run: bool,
) -> tuple[int, int]:
    common_prefix, subset_by_path, work_items, index_bytes = _prepare_dataset(
        dataset=dataset,
        source_root=source_root,
        tokenized_root=tokenized_root,
        text_column=text_column,
    )
    subsets = sorted({subset for subset in subset_by_path.values() if subset})
    if not work_items:
        LOGGER.info("SKIP %s: no multiple Parquet identity-renderer subsets found", dataset)
        return 0, 0
    LOGGER.info(
        "%s: %d subsets, %d source files, %d parent ranks, estimated new index space %.2f GiB",
        dataset,
        len(subsets),
        len(subset_by_path),
        len(work_items),
        index_bytes / 2**30,
    )
    if dry_run:
        LOGGER.info("  subsets: %s", ", ".join(subsets))
        return len(subsets), index_bytes

    all_results: list[ShardResult] = []
    max_workers = min(workers, len(work_items))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_split_rank, work, scan_empty_text=scan_empty_text): work.rank for work in work_items
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rank = futures[future]
            rank_results = future.result()
            all_results.extend(rank_results)
            if completed == 1 or completed % 16 == 0 or completed == len(futures):
                LOGGER.info("%s: completed %d/%d parent ranks (latest rank=%d)", dataset, completed, len(futures), rank)

    _write_subset_manifests(
        dataset=dataset,
        tokenized_root=tokenized_root,
        common_prefix=common_prefix,
        results=all_results,
    )
    LOGGER.info("DONE %s: subset views are under %s", dataset, tokenized_root / dataset / "subsets")
    return len(subsets), index_bytes


def main() -> None:
    """Split all selected completed datasets into subset-level index views."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    source_root = args.source_root.expanduser().resolve()
    tokenized_root = args.tokenized_root.expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not tokenized_root.is_dir():
        raise NotADirectoryError(tokenized_root)

    affected_datasets = 0
    subset_count = 0
    index_bytes = 0
    for dataset in _selected_datasets(tokenized_root, args.datasets):
        dataset_subsets, dataset_index_bytes = _run_dataset(
            dataset=dataset,
            source_root=source_root,
            tokenized_root=tokenized_root,
            text_column=args.text_column,
            workers=args.workers,
            scan_empty_text=args.scan_empty_text,
            dry_run=args.dry_run,
        )
        if dataset_subsets:
            affected_datasets += 1
            subset_count += dataset_subsets
            index_bytes += dataset_index_bytes

    action = "DRY RUN" if args.dry_run else "COMPLETE"
    LOGGER.info(
        "%s: %d affected datasets, %d subsets, estimated/materialized index space %.2f GiB",
        action,
        affected_datasets,
        subset_count,
        index_bytes / 2**30,
    )


if __name__ == "__main__":
    main()
