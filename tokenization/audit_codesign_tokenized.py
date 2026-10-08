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

"""Audit CodeSign Megatron indexed datasets against their source documents."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


LOGGER = logging.getLogger(__name__)
INDEX_HEADER = b"MMIDIDX\x00\x00"
INDEX_FIXED_HEADER_BYTES = len(INDEX_HEADER) + 8 + 1 + 8 + 8
DTYPE_ITEM_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}
TOKEN_STRUCT_FORMATS = {1: "B", 2: "b", 3: "h", 4: "i", 5: "q", 8: "H"}
SHARD_PATTERN = re.compile(r"dataset_(\d+)_tokens\.idx$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/cb/ml-olympus/codesign_pretrain/tokenized/mellum2"),
        help="Root containing tokenized dataset directories.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination.")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also inspect finalized shards in directories without result.json.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(input_file: BinaryIO, size: int) -> bytes:
    value = input_file.read(size)
    if len(value) != size:
        raise ValueError(f"short read: requested {size} bytes, received {len(value)}")
    return value


def _read_index_header(index_path: Path) -> dict[str, int]:
    with index_path.open("rb") as input_file:
        if _read_exact(input_file, len(INDEX_HEADER)) != INDEX_HEADER:
            raise ValueError("invalid index header")
        version = struct.unpack("<Q", _read_exact(input_file, 8))[0]
        dtype_code = struct.unpack("<B", _read_exact(input_file, 1))[0]
        sequence_count = struct.unpack("<Q", _read_exact(input_file, 8))[0]
        document_index_count = struct.unpack("<Q", _read_exact(input_file, 8))[0]
    if version != 1:
        raise ValueError(f"unsupported index version: {version}")
    if dtype_code not in DTYPE_ITEM_SIZES:
        raise ValueError(f"unsupported dtype code: {dtype_code}")
    expected_index_bytes = INDEX_FIXED_HEADER_BYTES + sequence_count * 12 + document_index_count * 8
    if index_path.stat().st_size != expected_index_bytes:
        raise ValueError(f"index size is {index_path.stat().st_size}, expected {expected_index_bytes} bytes")
    return {
        "dtype_code": dtype_code,
        "item_size": DTYPE_ITEM_SIZES[dtype_code],
        "sequence_count": sequence_count,
        "document_index_count": document_index_count,
    }


def _read_sequence(index_path: Path, sequence_index: int) -> tuple[list[int], dict[str, int]]:
    header = _read_index_header(index_path)
    sequence_count = header["sequence_count"]
    if not 0 <= sequence_index < sequence_count:
        raise IndexError(f"sequence {sequence_index} outside [0, {sequence_count})")
    lengths_offset = INDEX_FIXED_HEADER_BYTES
    pointers_offset = lengths_offset + sequence_count * 4
    documents_offset = pointers_offset + sequence_count * 8
    with index_path.open("rb") as input_file:
        input_file.seek(lengths_offset + sequence_index * 4)
        length = struct.unpack("<i", _read_exact(input_file, 4))[0]
        input_file.seek(pointers_offset + sequence_index * 8)
        pointer = struct.unpack("<q", _read_exact(input_file, 8))[0]
        input_file.seek(documents_offset)
        first_document_index = struct.unpack("<q", _read_exact(input_file, 8))[0]
        input_file.seek(documents_offset + (header["document_index_count"] - 1) * 8)
        last_document_index = struct.unpack("<q", _read_exact(input_file, 8))[0]
    if length <= 0 or pointer < 0:
        raise ValueError(f"invalid sequence length/pointer: {length}/{pointer}")
    if first_document_index != 0 or last_document_index != sequence_count:
        raise ValueError(
            f"document index bounds are {first_document_index}/{last_document_index}, expected 0/{sequence_count}"
        )
    binary_path = index_path.with_suffix(".bin")
    end_pointer = pointer + length * header["item_size"]
    if end_pointer > binary_path.stat().st_size:
        raise ValueError(f"sequence ends at byte {end_pointer}, beyond {binary_path.stat().st_size}")
    token_format = TOKEN_STRUCT_FORMATS.get(header["dtype_code"])
    if token_format is None:
        raise ValueError(f"dtype code {header['dtype_code']} is not an integer token dtype")
    with binary_path.open("rb") as input_file:
        input_file.seek(pointer)
        token_bytes = _read_exact(input_file, length * header["item_size"])
    tokens = [value[0] for value in struct.iter_unpack(f"<{token_format}", token_bytes)]
    return tokens, header


def _sample_positions(values: list[int], sample_count: int) -> list[int]:
    if not values or sample_count <= 0:
        return []
    if len(values) <= sample_count:
        return values
    if sample_count == 1:
        return [values[0]]
    return sorted({values[round(index * (len(values) - 1) / (sample_count - 1))] for index in range(sample_count)})


def _first_parquet_text(path: Path, text_column: str) -> tuple[str, int]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    if text_column not in parquet_file.schema_arrow.names:
        raise KeyError(f"missing column {text_column!r}; columns={parquet_file.schema_arrow.names}")
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=64, columns=[text_column]):
        for value in batch.column(0).to_pylist():
            if value:
                if not isinstance(value, str):
                    raise TypeError(f"{text_column!r} value is {type(value).__name__}, expected str")
                return value, row_index
            row_index += 1
    raise ValueError(f"no non-empty {text_column!r} value")


def _first_json_text(path: Path, text_column: str) -> tuple[str, int]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as input_file:
        for row_index, line in enumerate(input_file):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            value = row.get(text_column)
            if value:
                if not isinstance(value, str):
                    raise TypeError(f"{text_column!r} value is {type(value).__name__}, expected str")
                return value, row_index
    raise ValueError(f"no non-empty {text_column!r} value")


def _first_source_text(
    root: Path,
    input_paths: list[str],
    rank: int,
    tasks: int,
    input_format: str,
    text_column: str,
) -> tuple[str, str, int]:
    for relative_path in input_paths[rank::tasks]:
        source_path = root / relative_path
        try:
            if input_format == "parquet":
                text, row_index = _first_parquet_text(source_path, text_column)
            else:
                text, row_index = _first_json_text(source_path, text_column)
        except ValueError:
            continue
        return text, relative_path, row_index
    raise ValueError(f"rank {rank} has no non-empty source document")


def _read_parquet_row(path: Path, row_index: int, columns: list[str]) -> dict[str, object]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    remaining = row_index
    for row_group_index in range(parquet_file.num_row_groups):
        row_count = parquet_file.metadata.row_group(row_group_index).num_rows
        if remaining < row_count:
            table = parquet_file.read_row_group(row_group_index, columns=columns)
            return table.slice(remaining, 1).to_pylist()[0]
        remaining -= row_count
    raise IndexError(f"row {row_index} outside {path}")


def _preview(text: str, limit: int = 320) -> str:
    return text[:limit].replace("\x00", "\\0")


def _identity_sample(
    dataset_dir: Path,
    manifest: dict[str, object],
    tokenizer: object,
    index_path: Path,
    rank: int,
) -> dict[str, object]:
    dataset = manifest["dataset"]
    pipeline = manifest["pipeline"]
    tasks = int(pipeline["tasks"])
    input_paths = (dataset_dir / str(manifest["input_paths_file"])).read_text(encoding="utf-8").splitlines()
    source_text, source_file, source_row_index = _first_source_text(
        Path(str(dataset["root"])),
        input_paths,
        rank,
        tasks,
        str(dataset["input_format"]),
        str(pipeline["text_column"]),
    )
    actual_tokens, header = _read_sequence(index_path, 0)
    expected_tokens = tokenizer.encode(source_text).ids
    decoded = tokenizer.decode(actual_tokens[:-1], skip_special_tokens=False)
    return {
        "rank": rank,
        "index": str(index_path),
        "sequence_count": header["sequence_count"],
        "source_file": source_file,
        "source_row_index": source_row_index,
        "source_chars": len(source_text),
        "token_count": len(actual_tokens),
        "eos_ok": bool(actual_tokens and actual_tokens[-1] == manifest["tokenizer"]["eos_id"]),
        "token_ids_exact_match": actual_tokens == expected_tokens,
        "decoded_text_exact_match": decoded == source_text,
        "source_preview": _preview(source_text),
        "decoded_preview": _preview(decoded),
        "min_token_id": min(actual_tokens),
        "max_token_id": max(actual_tokens),
    }


def _stack_sample(
    dataset_dir: Path,
    manifest: dict[str, object],
    tokenizer: object,
    index_path: Path,
    rank: int,
) -> dict[str, object]:
    import pyarrow.parquet as pq

    from tokenization.stack_v3_repo_to_text_27_variants import repo_to_text_chunks, source_row_seed

    mapping_path = index_path.with_suffix(".mapping.parquet")
    mapping_file = pq.ParquetFile(mapping_path)
    mapping_row = (
        mapping_file.read_row_group(
            0,
            columns=["source_file", "source_row_index", "chunk_index", "target_document_index"],
        )
        .slice(0, 1)
        .to_pylist()[0]
    )
    source_file = str(mapping_row["source_file"])
    source_row_index = int(mapping_row["source_row_index"])
    chunk_index = int(mapping_row["chunk_index"])
    target_document_index = int(mapping_row["target_document_index"])
    source_path = Path(str(manifest["dataset"]["root"])) / source_file
    source_row = _read_parquet_row(source_path, source_row_index, ["repo_path", "files"])
    renderer = manifest["renderer"]
    chunks = repo_to_text_chunks(
        source_row,
        max_chars=int(renderer["max_chars"]),
        seed=source_row_seed(int(renderer["seed"]), source_file, source_row_index),
    )
    rendered = next(chunk for index, chunk in enumerate(chunks) if index == chunk_index)
    actual_tokens, header = _read_sequence(index_path, target_document_index)
    expected_tokens = tokenizer.encode(rendered).ids
    decoded = tokenizer.decode(actual_tokens[:-1], skip_special_tokens=False)
    return {
        "rank": rank,
        "index": str(index_path),
        "sequence_count": header["sequence_count"],
        "source_file": source_file,
        "source_row_index": source_row_index,
        "chunk_index": chunk_index,
        "target_document_index": target_document_index,
        "source_chars": len(rendered),
        "token_count": len(actual_tokens),
        "eos_ok": bool(actual_tokens and actual_tokens[-1] == manifest["tokenizer"]["eos_id"]),
        "token_ids_exact_match": actual_tokens == expected_tokens,
        "decoded_text_exact_match": decoded == rendered,
        "source_preview": _preview(rendered),
        "decoded_preview": _preview(decoded),
        "min_token_id": min(actual_tokens),
        "max_token_id": max(actual_tokens),
    }


def _audit_dataset(dataset_dir: Path, sample_count: int, include_incomplete: bool) -> dict[str, object] | None:
    manifest_path = dataset_dir / "manifest.json"
    result_path = dataset_dir / "result.json"
    complete = result_path.is_file()
    if not complete and not include_incomplete:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_path = dataset_dir / str(manifest["tokenizer"]["json_path"])
    paths_path = dataset_dir / str(manifest["input_paths_file"])
    tasks = int(manifest["pipeline"]["tasks"])
    data_dir = dataset_dir / str(manifest["output"]["data_dir"])
    index_paths = sorted(data_dir.glob("dataset_*_tokens.idx"))
    binary_paths = sorted(data_dir.glob("dataset_*_tokens.bin"))
    index_prefixes = {path.with_suffix("").name for path in index_paths}
    binary_prefixes = {path.with_suffix("").name for path in binary_paths}
    open_binary_shards = max(0, len(binary_paths) - len(index_paths))
    index_ranks = [int(match.group(1)) for path in index_paths if (match := SHARD_PATTERN.search(path.name))]
    issues = []
    if _sha256(tokenizer_path) != manifest["tokenizer"]["json_sha256"]:
        issues.append("tokenizer hash mismatch")
    if _sha256(paths_path) != manifest["input_paths_sha256"]:
        issues.append("input path-list hash mismatch")
    if complete and len(index_paths) != len(binary_paths):
        issues.append(f"idx/bin count mismatch: {len(index_paths)}/{len(binary_paths)}")
    if complete and index_prefixes != binary_prefixes:
        issues.append("completed dataset has unpaired idx/bin shard names")
    elif not complete and len(binary_paths) < len(index_paths):
        issues.append(f"incomplete dataset has fewer bin files than indexes: {len(binary_paths)}/{len(index_paths)}")
    elif not complete and open_binary_shards > int(manifest["pipeline"]["effective_workers"]):
        issues.append(
            f"incomplete dataset has {open_binary_shards} bin files awaiting indexes, "
            f"more than its {manifest['pipeline']['effective_workers']} configured workers"
        )
    if complete and len(index_paths) != tasks:
        issues.append(f"completed dataset has {len(index_paths)} indexes; expected {tasks}")
    if not complete and len(index_paths) > tasks:
        issues.append(f"incomplete dataset has {len(index_paths)} indexes; expected at most {tasks}")
    if complete:
        prefixes_path = dataset_dir / str(manifest["output"]["data_prefixes_file"])
        if not prefixes_path.is_file():
            issues.append("missing data_prefixes.txt")
        else:
            data_prefixes = prefixes_path.read_text(encoding="utf-8").splitlines()
            if len(data_prefixes) != tasks:
                issues.append("data_prefixes.txt line count does not match tasks")
            elif any(
                not (dataset_dir / f"{prefix}.idx").is_file() or not (dataset_dir / f"{prefix}.bin").is_file()
                for prefix in data_prefixes
            ):
                issues.append("data_prefixes.txt references a missing idx/bin pair")

    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = int(manifest["tokenizer"]["eos_id"])
    tokenizer.post_processor = TemplateProcessing(single="$A <EOS>", special_tokens=[("<EOS>", eos_id)])
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    renderer_name = str(manifest.get("renderer", {"name": "identity"})["name"])
    samples = []
    for rank in _sample_positions(index_ranks, sample_count):
        index_path = data_dir / f"dataset_{rank:05d}_tokens.idx"
        try:
            sample = (
                _stack_sample(dataset_dir, manifest, tokenizer, index_path, rank)
                if renderer_name == "stack-v3"
                else _identity_sample(dataset_dir, manifest, tokenizer, index_path, rank)
            )
            sample["token_ids_in_vocab"] = 0 <= sample["min_token_id"] <= sample["max_token_id"] < vocab_size
            samples.append(sample)
        except Exception as error:  # noqa: BLE001 - each sample must be reported independently
            samples.append({"rank": rank, "error": f"{type(error).__name__}: {error}"})
            issues.append(f"rank {rank} sample failed: {type(error).__name__}: {error}")

    if complete:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_summary = result["result"]
        if int(result_summary["shards"]) != tasks:
            issues.append("result shard count does not match manifest tasks")
    else:
        result_summary = None
    return {
        "dataset": str(dataset_dir),
        "path": str(dataset_dir),
        "complete": complete,
        "renderer": renderer_name,
        "text_column": manifest["pipeline"]["text_column"],
        "source_format": manifest["dataset"]["input_format"],
        "source_files": manifest["dataset"]["selected_files"],
        "tasks": tasks,
        "finalized_shards": len(index_paths),
        "open_binary_shards": open_binary_shards,
        "tokenizer_id": manifest["tokenizer"]["id"],
        "tokenizer_revision": manifest["tokenizer"]["resolved_revision"],
        "tokenizer_sha256": manifest["tokenizer"]["json_sha256"],
        "vocab_size": vocab_size,
        "result": result_summary,
        "issues": issues,
        "samples": samples,
    }


def main() -> None:
    """Run the audit and write a machine-readable report."""
    args = _parse_args()
    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive")
    manifest_paths = sorted(args.root.rglob("manifest.json"))
    datasets = []
    for position, manifest_path in enumerate(manifest_paths, start=1):
        LOGGER.info("Auditing %d/%d: %s", position, len(manifest_paths), manifest_path.parent)
        result = _audit_dataset(manifest_path.parent, args.samples_per_dataset, args.include_incomplete)
        if result is not None:
            result["dataset"] = str(manifest_path.parent.relative_to(args.root))
            datasets.append(result)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(args.root),
        "manifest_directories": len(manifest_paths),
        "audited_datasets": len(datasets),
        "completed_datasets": sum(bool(dataset["complete"]) for dataset in datasets),
        "incomplete_datasets": sum(not bool(dataset["complete"]) for dataset in datasets),
        "datasets_with_issues": sum(bool(dataset["issues"]) for dataset in datasets),
        "sample_count": sum(len(dataset["samples"]) for dataset in datasets),
        "exact_token_matches": sum(
            sample.get("token_ids_exact_match") is True for dataset in datasets for sample in dataset["samples"]
        ),
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
