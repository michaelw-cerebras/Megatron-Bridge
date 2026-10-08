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

"""Stream local or Hugging Face data files into Megatron indexed datasets.

Hugging Face source and tokenizer revisions are resolved to immutable commits. Local
Parquet and gzip-compressed JSONL files are read in place. Source documents are not
copied into the output directory.

Usage examples:
python process_datatrove.py  \
    --dataset openbmb/Ultra-FineWeb \
    --path-glob 'data/ultrafineweb_l1_en_hq/*.parquet' \
    --text-column content \
    --output-folder /cb/cold/michaelw/ultrafineweb_l1_en_hq/ \
    --workers 96

python process_datatrove.py \
    --input-folder /path/to/local/dataset \
    --path-glob '*.jsonl.gz' \
    --output-folder /path/to/tokenized/dataset \
    --workers 96

"""

import argparse
import fcntl
import fnmatch
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import socket
import struct
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


logger = logging.getLogger(__name__)
INDEX_HEADER = b"MMIDIDX\x00\x00"
# Keep each task small enough that preempted workers replay fewer source files.
AUTO_TASKS_PER_WORKER = 8
STACK_V3_RENDERER_RELATIVE_PATH = "tokenization/stack_v3_repo_to_text_27_variants.py"
STACK_V3_RENDERER_PATH = Path(__file__).resolve().parents[1] / STACK_V3_RENDERER_RELATIVE_PATH
DTYPE_ITEM_SIZES = {
    1: 1,  # uint8
    2: 1,  # int8
    3: 2,  # int16
    4: 4,  # int32
    5: 8,  # int64
    6: 8,  # float64
    7: 4,  # float32
    8: 2,  # uint16
}
MAPPING_COLUMNS = {
    "dataset",
    "source_file",
    "source_row_index",
    "chunk_index",
    "target_document_index",
}


@dataclass(frozen=True)
class IndexedDatasetTotals:
    """Aggregate metadata from Megatron indexed-dataset shards."""

    documents: int
    sequences: int
    tokens: int
    output_bytes: int
    shards: int
    mapping_rows: int
    mapping_bytes: int


def available_cpu_count() -> int:
    """Return the number of CPUs available to the current process.

    Returns:
        CPU count from the affinity mask when available.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset", help="Hugging Face dataset repository ID.")
    source_group.add_argument(
        "--input-folder",
        type=Path,
        help="Local input root containing Parquet or line-delimited JSON files.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="New output directory, or an existing run with --resume.",
    )
    parser.add_argument("--dataset-revision", default="main", help="Dataset branch, tag, or commit to pin.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Top-level dataset config directory to include; repeat to select multiple. Defaults to all configs.",
    )
    parser.add_argument(
        "--path-glob",
        default="*.parquet",
        help="fnmatch pattern applied to source-root-relative paths after config filtering.",
    )
    parser.add_argument("--text-column", default="text", help="Source field containing document text.")
    parser.add_argument(
        "--text-renderer",
        choices=("identity", "stack-v3"),
        default="identity",
        help="Optional deterministic source-row renderer applied before tokenization.",
    )
    parser.add_argument(
        "--renderer-seed",
        type=int,
        default=1234,
        help="Global augmentation seed used by a non-identity text renderer.",
    )
    parser.add_argument(
        "--renderer-max-chars",
        type=int,
        help="Maximum rendered characters per Stack v3 document chunk.",
    )
    parser.add_argument(
        "--id-column",
        help="Optional document ID column. Defaults to stable IDs derived from source path and row number.",
    )
    parser.add_argument(
        "--tokenizer",
        default="openai/gpt-oss-20b",
        help="Hugging Face tokenizer repository ID, local directory, or local tokenizer.json.",
    )
    parser.add_argument("--tokenizer-revision", default="main", help="Tokenizer branch, tag, or commit to pin.")
    parser.add_argument("--eos-token", default="<|return|>", help="End-of-document token appended to every document.")
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpu_count(),
        help="Concurrent local worker processes.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        help="Stable output task/shard count. Defaults to min(input files, 8 * workers).",
    )
    parser.add_argument(
        "--reader-batch-size",
        type=int,
        default=1000,
        help="Parquet reader batch size; accepted but unused for JSONL.",
    )
    parser.add_argument("--tokenizer-batch-size", type=int, default=1000, help="Tokenizer encode batch size.")
    parser.add_argument("--save-filename", default="dataset", help="Prefix used for output .bin/.idx shard names.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete tasks from the existing logging directory.",
    )
    return parser.parse_args()


def select_parquet_paths(paths: list[str], *, configs: list[str], path_glob: str) -> list[str]:
    """Select deterministic Parquet paths from a dataset repository listing.

    Args:
        paths: Repository-relative file paths.
        configs: Optional top-level directory prefixes.
        path_glob: fnmatch expression applied to the complete relative path.

    Returns:
        Sorted, unique matching Parquet paths.
    """
    normalized_configs = [config.strip("/") for config in configs]
    selected = []
    for path in paths:
        if not path.endswith(".parquet") or not fnmatch.fnmatch(path, path_glob):
            continue
        if normalized_configs and not any(
            path == config or path.startswith(f"{config}/") for config in normalized_configs
        ):
            continue
        selected.append(path)
    return sorted(set(selected))


def select_local_paths(input_folder: Path, *, configs: list[str], path_glob: str) -> list[str]:
    """Select deterministic local input paths.

    Args:
        input_folder: Local root containing source files.
        configs: Optional top-level directory prefixes.
        path_glob: fnmatch expression applied to each root-relative path.

    Returns:
        Sorted, unique root-relative paths.
    """
    normalized_configs = [config.strip("/") for config in configs]
    selected = []
    for path in input_folder.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(input_folder).as_posix()
        if not fnmatch.fnmatch(relative_path, path_glob):
            continue
        if normalized_configs and not any(
            relative_path == config or relative_path.startswith(f"{config}/") for config in normalized_configs
        ):
            continue
        selected.append(relative_path)
    return sorted(set(selected))


def infer_input_format(paths: list[str]) -> str:
    """Infer one supported input format from selected paths.

    Args:
        paths: Selected source-root-relative paths.

    Returns:
        ``parquet`` or ``jsonl``.

    Raises:
        ValueError: If a path is unsupported or the selection mixes formats.
    """
    formats = set()
    for path in paths:
        if path.endswith(".parquet"):
            formats.add("parquet")
        elif path.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
            formats.add("jsonl")
        else:
            raise ValueError(f"Unsupported input file extension: {path}")
    if len(formats) != 1:
        raise ValueError(f"Each run must select exactly one input format; found {sorted(formats)}")
    return formats.pop()


def derive_execution_shape(*, file_count: int, requested_tasks: int | None, requested_workers: int) -> tuple[int, int]:
    """Derive task and worker counts without assigning empty file shards.

    Args:
        file_count: Number of selected Parquet files.
        requested_tasks: Explicit task count, or None for automatic selection.
        requested_workers: Requested local worker count.

    Returns:
        Effective task and worker counts.

    Raises:
        ValueError: If counts are non-positive or explicit tasks exceed the input file count.
    """
    if file_count <= 0:
        raise ValueError("At least one Parquet input file is required")
    if requested_workers <= 0:
        raise ValueError("--workers must be positive")
    if requested_tasks is not None and requested_tasks <= 0:
        raise ValueError("--tasks must be positive")
    if requested_tasks is not None and requested_tasks > file_count:
        raise ValueError(f"--tasks={requested_tasks} exceeds the {file_count} selected Parquet files")
    tasks = (
        requested_tasks
        if requested_tasks is not None
        else min(file_count, requested_workers * AUTO_TASKS_PER_WORKER)
    )
    workers = min(requested_workers, tasks)
    return tasks, workers


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def renderer_metadata(name: str, seed: int, max_chars: int | None) -> dict[str, object]:
    """Return immutable renderer metadata for a preprocessing manifest.

    Args:
        name: Renderer selected on the command line.
        seed: Global renderer seed.
        max_chars: Maximum rendered characters per Stack v3 document chunk.

    Returns:
        Renderer name and, for Stack v3, its seed and source hash.
    """
    if name == "identity":
        if max_chars is not None:
            raise ValueError("--renderer-max-chars is only valid with --text-renderer=stack-v3")
        return {"name": name}
    if name == "stack-v3":
        if max_chars is None or max_chars <= 0:
            raise ValueError("--text-renderer=stack-v3 requires a positive --renderer-max-chars")
        if not STACK_V3_RENDERER_PATH.is_file():
            raise FileNotFoundError(STACK_V3_RENDERER_PATH)
        return {
            "name": name,
            "seed": seed,
            "max_chars": max_chars,
            "source_path": STACK_V3_RENDERER_RELATIVE_PATH,
            "source_sha256": sha256_file(STACK_V3_RENDERER_PATH),
        }
    raise ValueError(f"Unsupported text renderer: {name}")


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically write JSON data.

    Args:
        path: Destination JSON file.
        value: JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output_file:
        temporary_path = Path(output_file.name)
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(path)


def write_paths_atomic(path: Path, paths: list[str]) -> None:
    """Atomically write one repository-relative input path per line.

    Args:
        path: Destination path-list file.
        paths: Sorted repository-relative paths.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output_file:
        temporary_path = Path(output_file.name)
        for input_path in paths:
            output_file.write(f"{input_path}\n")
    temporary_path.replace(path)


def read_exact(input_file: BinaryIO, size: int) -> bytes:
    """Read exactly the requested number of bytes.

    Args:
        input_file: Open binary stream.
        size: Required byte count.

    Returns:
        Bytes read.

    Raises:
        ValueError: If the stream ends early.
    """
    value = input_file.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of index file; wanted {size} bytes, got {len(value)}")
    return value


def indexed_dataset_totals(index_paths: list[Path], *, require_mapping: bool) -> IndexedDatasetTotals:
    """Validate Megatron indexed datasets and aggregate their counts.

    Args:
        index_paths: Megatron .idx files whose matching .bin files will be checked.
        require_mapping: Require and validate one source-key Parquet sidecar per shard.

    Returns:
        Aggregate document, sequence, token, byte, and shard counts.
    """
    documents = 0
    sequences = 0
    tokens = 0
    output_bytes = 0
    mapping_rows = 0
    mapping_bytes = 0
    for index_path in sorted(index_paths):
        binary_path = index_path.with_suffix(".bin")
        if not binary_path.is_file():
            raise FileNotFoundError(binary_path)
        with index_path.open("rb") as input_file:
            if read_exact(input_file, len(INDEX_HEADER)) != INDEX_HEADER:
                raise ValueError(f"Invalid indexed-dataset header in {index_path}")
            version = struct.unpack("<Q", read_exact(input_file, 8))[0]
            if version != 1:
                raise ValueError(f"Unsupported indexed-dataset version {version} in {index_path}")
            dtype_code = struct.unpack("<B", read_exact(input_file, 1))[0]
            if dtype_code not in DTYPE_ITEM_SIZES:
                raise ValueError(f"Unsupported dtype code {dtype_code} in {index_path}")
            sequence_count = struct.unpack("<Q", read_exact(input_file, 8))[0]
            document_index_count = struct.unpack("<Q", read_exact(input_file, 8))[0]
            length_bytes = read_exact(input_file, sequence_count * 4)
            token_count = sum(value[0] for value in struct.iter_unpack("<i", length_bytes))
            read_exact(input_file, sequence_count * 8)
            read_exact(input_file, document_index_count * 8)
            if input_file.read(1):
                raise ValueError(f"Unexpected trailing bytes in {index_path}")

        expected_binary_bytes = token_count * DTYPE_ITEM_SIZES[dtype_code]
        if binary_path.stat().st_size != expected_binary_bytes:
            raise ValueError(f"{binary_path} has {binary_path.stat().st_size} bytes; expected {expected_binary_bytes}")
        documents += max(0, document_index_count - 1)
        sequences += sequence_count
        tokens += token_count
        output_bytes += index_path.stat().st_size + binary_path.stat().st_size
        if require_mapping:
            import pyarrow.parquet as pq

            mapping_path = index_path.with_suffix(".mapping.parquet")
            if not mapping_path.is_file():
                raise FileNotFoundError(mapping_path)
            parquet_file = pq.ParquetFile(mapping_path)
            missing_columns = MAPPING_COLUMNS.difference(parquet_file.schema_arrow.names)
            if missing_columns:
                raise ValueError(f"Missing columns in {mapping_path}: {sorted(missing_columns)}")
            shard_mapping_rows = parquet_file.metadata.num_rows
            shard_documents = max(0, document_index_count - 1)
            if shard_mapping_rows != shard_documents:
                raise ValueError(f"{mapping_path} has {shard_mapping_rows} rows; expected {shard_documents} documents")
            mapping_rows += shard_mapping_rows
            mapping_bytes += mapping_path.stat().st_size
            output_bytes += mapping_path.stat().st_size
    return IndexedDatasetTotals(
        documents,
        sequences,
        tokens,
        output_bytes,
        len(index_paths),
        mapping_rows,
        mapping_bytes,
    )


def package_versions(package_names: list[str]) -> dict[str, str]:
    """Return installed versions for selected packages.

    Args:
        package_names: Distribution names.

    Returns:
        Package version strings, or ``not-installed``.
    """
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def environment_metadata() -> dict[str, object]:
    """Collect reproducibility metadata for the current process.

    Returns:
        Host, CPU, Python, platform, and package metadata.
    """
    return {
        "available_cpus": available_cpu_count(),
        "hostname": socket.gethostname(),
        "packages": package_versions(["datatrove", "huggingface-hub", "pyarrow", "tokenizers"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def prepare_new_run(args: argparse.Namespace) -> dict[str, object]:
    """Resolve immutable inputs, download the tokenizer, and write a run manifest.

    Args:
        args: Parsed CLI arguments for a new run.

    Returns:
        Production preprocessing manifest.
    """
    from huggingface_hub import HfApi, snapshot_download
    from tokenizers import Tokenizer

    if args.output_folder.exists() and any(args.output_folder.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output folder without --resume: {args.output_folder}")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_folder / "metadata"
    tokenizer_dir = metadata_dir / "tokenizer"
    paths_file = metadata_dir / "input_paths.txt"

    api = HfApi()
    if args.input_folder is not None:
        input_folder = args.input_folder.expanduser().resolve()
        if not input_folder.is_dir():
            raise NotADirectoryError(input_folder)
        input_paths = select_local_paths(input_folder, configs=args.config, path_glob=args.path_glob)
        if not input_paths:
            raise ValueError(
                f"No files matched input_folder={str(input_folder)!r}, "
                f"configs={args.config!r}, glob={args.path_glob!r}"
            )
        input_format = infer_input_format(input_paths)
        source_bytes = sum((input_folder / path).stat().st_size for path in input_paths)
        dataset_manifest = {
            "configs": args.config,
            "id": input_folder.name,
            "input_format": input_format,
            "kind": "local",
            "path_glob": args.path_glob,
            "root": str(input_folder),
            "selected_bytes": source_bytes,
            "selected_files": len(input_paths),
        }
    else:
        dataset_info = api.dataset_info(args.dataset, revision=args.dataset_revision, files_metadata=True)
        dataset_revision = dataset_info.sha
        repository_paths = [sibling.rfilename for sibling in dataset_info.siblings]
        input_paths = select_parquet_paths(repository_paths, configs=args.config, path_glob=args.path_glob)
        if not input_paths:
            raise ValueError(
                f"No Parquet files matched dataset={args.dataset!r}, configs={args.config!r}, glob={args.path_glob!r}"
            )
        selected_set = set(input_paths)
        source_bytes = sum(
            int(sibling.size or 0) for sibling in dataset_info.siblings if sibling.rfilename in selected_set
        )
        dataset_manifest = {
            "configs": args.config,
            "id": args.dataset,
            "input_format": "parquet",
            "kind": "huggingface",
            "path_glob": args.path_glob,
            "requested_revision": args.dataset_revision,
            "resolved_revision": dataset_revision,
            "root": f"hf://datasets/{args.dataset}@{dataset_revision}/",
            "selected_bytes": source_bytes,
            "selected_files": len(input_paths),
        }

    input_format = str(dataset_manifest["input_format"])
    if args.text_renderer == "stack-v3":
        if args.input_folder is None:
            raise ValueError("--text-renderer=stack-v3 requires --input-folder")
        if input_format != "parquet":
            raise ValueError("--text-renderer=stack-v3 requires Parquet input")
        if args.id_column is not None:
            raise ValueError("--text-renderer=stack-v3 uses source filename/row IDs; do not set --id-column")
    renderer = renderer_metadata(args.text_renderer, args.renderer_seed, args.renderer_max_chars)

    local_tokenizer = Path(args.tokenizer).expanduser()
    if local_tokenizer.is_dir():
        local_tokenizer = local_tokenizer / "tokenizer.json"
    if local_tokenizer.is_file():
        local_tokenizer = local_tokenizer.resolve()
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_tokenizer, tokenizer_dir / "tokenizer.json")
        tokenizer_revision = "local"
        tokenizer_source = str(local_tokenizer)
    else:
        tokenizer_info = api.model_info(args.tokenizer, revision=args.tokenizer_revision)
        tokenizer_revision = tokenizer_info.sha
        snapshot_download(
            repo_id=args.tokenizer,
            revision=tokenizer_revision,
            local_dir=tokenizer_dir,
            allow_patterns=["tokenizer.json"],
        )
        tokenizer_source = args.tokenizer
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos_id = tokenizer.token_to_id(args.eos_token)
    if eos_id is None:
        raise ValueError(f"EOS token {args.eos_token!r} is not present in {args.tokenizer}")

    tasks, workers = derive_execution_shape(
        file_count=len(input_paths),
        requested_tasks=args.tasks,
        requested_workers=args.workers,
    )
    write_paths_atomic(paths_file, input_paths)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_manifest,
        "input_paths_file": "metadata/input_paths.txt",
        "input_paths_sha256": sha256_file(paths_file),
        "output": {
            "data_dir": "data",
            "data_prefixes_file": "data_prefixes.txt",
            "document_mapping": {
                "columns": sorted(MAPPING_COLUMNS),
                "enabled": args.text_renderer == "stack-v3",
                "format": "parquet",
                "suffix": ".mapping.parquet",
            },
            "save_filename": args.save_filename,
        },
        "pipeline": {
            "effective_workers": workers,
            "id_column": args.id_column,
            "reader_batch_size": args.reader_batch_size,
            "tasks": tasks,
            "text_column": args.text_column,
            "tokenizer_batch_size": args.tokenizer_batch_size,
        },
        "renderer": renderer,
        "schema_version": 2,
        "tokenizer": {
            "eos_id": eos_id,
            "eos_token": args.eos_token,
            "id": args.tokenizer,
            "json_path": "metadata/tokenizer/tokenizer.json",
            "json_sha256": sha256_file(tokenizer_json),
            "requested_revision": args.tokenizer_revision,
            "resolved_revision": tokenizer_revision,
            "source": tokenizer_source,
        },
    }
    write_json_atomic(args.output_folder / "manifest.json", manifest)
    return manifest


def load_resumable_run(args: argparse.Namespace) -> dict[str, object]:
    """Load and validate the immutable manifest for a resumed run.

    Args:
        args: Parsed CLI arguments containing resume-time expectations.

    Returns:
        Existing production preprocessing manifest.
    """
    manifest_path = args.output_folder / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"--resume requires {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest["dataset"]
    pipeline = manifest["pipeline"]
    output = manifest["output"]
    tokenizer = manifest["tokenizer"]
    stored_renderer = manifest.get("renderer", {"name": "identity"})
    requested_renderer = renderer_metadata(args.text_renderer, args.renderer_seed, args.renderer_max_chars)
    requested_kind = "local" if args.input_folder is not None else "huggingface"
    stored_kind = dataset.get("kind", "huggingface")
    expected = {
        "source_kind": (stored_kind, requested_kind),
        "configs": (dataset["configs"], args.config),
        "path_glob": (dataset["path_glob"], args.path_glob),
        "text_column": (pipeline["text_column"], args.text_column),
        "id_column": (pipeline["id_column"], args.id_column),
        "reader_batch_size": (pipeline["reader_batch_size"], args.reader_batch_size),
        "tokenizer_batch_size": (pipeline["tokenizer_batch_size"], args.tokenizer_batch_size),
        "save_filename": (output["save_filename"], args.save_filename),
        "tokenizer": (tokenizer["id"], args.tokenizer),
        "eos_token": (tokenizer["eos_token"], args.eos_token),
        "text_renderer": (stored_renderer.get("name", "identity"), requested_renderer["name"]),
    }
    if requested_renderer["name"] == "stack-v3":
        expected["renderer_seed"] = (stored_renderer.get("seed"), requested_renderer["seed"])
        expected["renderer_max_chars"] = (
            stored_renderer.get("max_chars"),
            requested_renderer["max_chars"],
        )
        expected["renderer_source_sha256"] = (
            stored_renderer.get("source_sha256"),
            requested_renderer["source_sha256"],
        )
    if requested_kind == "local":
        requested_root = str(args.input_folder.expanduser().resolve())
        expected["input_folder"] = (dataset.get("root"), requested_root)
    else:
        expected["dataset"] = (dataset["id"], args.dataset)
    mismatches = [name for name, (stored, requested) in expected.items() if stored != requested]
    if args.tasks is not None and args.tasks != pipeline["tasks"]:
        mismatches.append("tasks")
    if mismatches:
        raise ValueError(f"Resume arguments do not match manifest fields: {sorted(mismatches)}")

    paths_file = args.output_folder / manifest["input_paths_file"]
    tokenizer_json = args.output_folder / tokenizer["json_path"]
    if sha256_file(paths_file) != manifest["input_paths_sha256"]:
        raise ValueError(f"Input path-list hash does not match manifest: {paths_file}")
    if sha256_file(tokenizer_json) != tokenizer["json_sha256"]:
        raise ValueError(f"Tokenizer hash does not match manifest: {tokenizer_json}")
    return manifest


def validate_outputs(
    *,
    data_dir: Path,
    save_filename: str,
    tasks: int,
    require_mapping: bool,
) -> IndexedDatasetTotals:
    """Validate all expected output shards.

    Args:
        data_dir: Directory containing Megatron .bin/.idx pairs.
        save_filename: Filename prefix configured in the tokenizer writer.
        tasks: Expected output shard count.
        require_mapping: Require an aligned mapping sidecar for each shard.

    Returns:
        Aggregate indexed-dataset totals.
    """
    index_paths = sorted(data_dir.glob(f"{save_filename}_*_tokens.idx"))
    if len(index_paths) != tasks:
        raise RuntimeError(f"Expected {tasks} output indexes, found {len(index_paths)} under {data_dir}")
    return indexed_dataset_totals(index_paths, require_mapping=require_mapping)


def write_data_prefixes(*, output_folder: Path, data_dir: Path, save_filename: str) -> None:
    """Write movable, output-folder-relative Megatron dataset prefixes.

    Args:
        output_folder: Root of the preprocessing output.
        data_dir: Directory containing the indexed datasets.
        save_filename: Filename prefix configured in the tokenizer writer.
    """
    prefixes = [
        str(index_path.with_suffix("").relative_to(output_folder))
        for index_path in sorted(data_dir.glob(f"{save_filename}_*_tokens.idx"))
    ]
    write_paths_atomic(output_folder / "data_prefixes.txt", prefixes)


def run_pipeline(args: argparse.Namespace, manifest: dict[str, object]) -> None:
    """Execute or resume the source-to-Megatron pipeline.

    Args:
        args: Parsed CLI arguments.
        manifest: Immutable production preprocessing manifest.
    """
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.tokens import MegatronDocumentTokenizer

    if __package__:
        from .datatrove_text_reader import StackV3ParquetReader, TextJsonlReader, TextParquetReader
        from .mapped_megatron_tokenizer import MappedMegatronDocumentTokenizer
    else:
        from datatrove_text_reader import StackV3ParquetReader, TextJsonlReader, TextParquetReader
        from mapped_megatron_tokenizer import MappedMegatronDocumentTokenizer

    dataset = manifest["dataset"]
    pipeline = manifest["pipeline"]
    output = manifest["output"]
    renderer = manifest.get("renderer", {"name": "identity"})
    tokenizer = manifest["tokenizer"]
    require_mapping = bool(output.get("document_mapping", {}).get("enabled", False))
    tasks = int(pipeline["tasks"])
    workers = min(args.workers, tasks)
    if workers < args.workers:
        logger.warning(
            "Using %d workers instead of requested %d because only %d input tasks are available",
            workers,
            args.workers,
            tasks,
        )

    data_dir = args.output_folder / output["data_dir"]
    logs_dir = args.output_folder / "logs"
    paths_file = args.output_folder / manifest["input_paths_file"]
    tokenizer_json = args.output_folder / tokenizer["json_path"]
    result_path = args.output_folder / "result.json"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and result_path.is_file():
        totals = validate_outputs(
            data_dir=data_dir,
            save_filename=output["save_filename"],
            tasks=tasks,
            require_mapping=require_mapping,
        )
        write_data_prefixes(
            output_folder=args.output_folder,
            data_dir=data_dir,
            save_filename=output["save_filename"],
        )
        logger.info("Run is already complete and valid: %d documents, %d tokens", totals.documents, totals.tokens)
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if renderer.get("name") == "stack-v3":
        reader = StackV3ParquetReader(
            data_folder=dataset.get("root", dataset.get("remote_root")),
            paths_file=str(paths_file),
            batch_size=int(pipeline["reader_batch_size"]),
            dataset_name=str(dataset["id"]),
            seed=int(renderer["seed"]),
            max_chars=int(renderer["max_chars"]),
        )
    else:
        reader_class = TextParquetReader if dataset.get("input_format", "parquet") == "parquet" else TextJsonlReader
        reader = reader_class(
            data_folder=dataset.get("root", dataset.get("remote_root")),
            paths_file=str(paths_file),
            batch_size=int(pipeline["reader_batch_size"]),
            text_key=pipeline["text_column"],
            id_key=pipeline["id_column"],
        )
    tokenizer_kwargs = {
        "output_folder": str(data_dir),
        "save_filename": output["save_filename"],
        "tokenizer_name_or_path": str(tokenizer_json),
        "eos_token": tokenizer["eos_token"],
        "batch_size": int(pipeline["tokenizer_batch_size"]),
    }
    tokenizer_step = (
        MappedMegatronDocumentTokenizer(mapping_output_folder=data_dir, **tokenizer_kwargs)
        if require_mapping
        else MegatronDocumentTokenizer(**tokenizer_kwargs)
    )
    executor = LocalPipelineExecutor(
        pipeline=[
            reader,
            tokenizer_step,
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(logs_dir),
        skip_completed=args.resume,
    )
    logger.info(
        "Processing %d %s files with tasks=%d workers=%d from %s",
        dataset.get("selected_files", dataset.get("selected_parquet_files")),
        dataset.get("input_format", "parquet"),
        tasks,
        workers,
        dataset.get("root", dataset.get("remote_root")),
    )
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    executor.run()
    wall_seconds = time.perf_counter() - start
    totals = validate_outputs(
        data_dir=data_dir,
        save_filename=output["save_filename"],
        tasks=tasks,
        require_mapping=require_mapping,
    )
    write_data_prefixes(
        output_folder=args.output_folder,
        data_dir=data_dir,
        save_filename=output["save_filename"],
    )
    throughput = {
        "documents_per_second": None if args.resume else totals.documents / wall_seconds,
        "tokens_per_second": None if args.resume else totals.tokens / wall_seconds,
    }
    result = {
        "execution_mode": "resume" if args.resume else "new",
        "environment": environment_metadata(),
        "input": {
            "dataset": dataset["id"],
            "bytes": dataset.get("selected_bytes", dataset.get("selected_parquet_bytes")),
            "files": dataset.get("selected_files", dataset.get("selected_parquet_files")),
            "format": dataset.get("input_format", "parquet"),
            "kind": dataset.get("kind", "huggingface"),
            "resolved_revision": dataset.get("resolved_revision"),
        },
        "parameters": {
            "eos_id": tokenizer["eos_id"],
            "eos_token": tokenizer["eos_token"],
            "tasks": tasks,
            "text_renderer": renderer,
            "workers": workers,
        },
        "result": {
            **asdict(totals),
            **throughput,
            "wall_seconds": wall_seconds,
        },
        "started_at": started_at,
        "tool": "datatrove",
    }
    write_json_atomic(result_path, result)
    if args.resume:
        logger.info(
            "Resume completed: validated %d documents and %d tokens; this invocation took %.1f seconds",
            totals.documents,
            totals.tokens,
            wall_seconds,
        )
    else:
        logger.info(
            "Completed: %d documents, %d tokens in %.1f seconds (%.0f tokens/s)",
            totals.documents,
            totals.tokens,
            wall_seconds,
            totals.tokens / wall_seconds,
        )


def main() -> None:
    """Prepare immutable metadata and run DataTrove preprocessing."""
    args = parse_args()
    if args.reader_batch_size <= 0 or args.tokenizer_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    # A sibling lock also covers the interval before the output directory exists.
    # Lustre flock must be cluster-wide on every client used for this output.
    lock_path = args.output_folder.parent / f".{args.output_folder.name}.datatrove.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("Waiting for DataTrove output lock: %s", args.output_folder)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not args.resume and (args.output_folder / "manifest.json").is_file():
            logger.info("Existing run found after acquiring output lock; resuming %s", args.output_folder)
            args.resume = True
        manifest = load_resumable_run(args) if args.resume else prepare_new_run(args)
        run_pipeline(args, manifest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
