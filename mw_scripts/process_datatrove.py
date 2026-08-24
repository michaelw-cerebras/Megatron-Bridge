"""Stream Parquet files from a Hugging Face dataset into Megatron indexed datasets.

The source dataset and tokenizer revisions are resolved to immutable commits. Input
Parquet files are read directly through Hugging Face's fsspec filesystem; they are
not materialized as JSONL or copied into the output directory.

Usage examples:
python process_datatrove.py  \
    --dataset openbmb/Ultra-FineWeb \
    --path-glob 'data/ultrafineweb_l1_en_hq/*.parquet' \
    --text-column content \
    --output-folder /cb/cold/michaelw/ultrafineweb_l1_en_hq/ \
    --workers 96

"""

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
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


@dataclass(frozen=True)
class IndexedDatasetTotals:
    """Aggregate metadata from Megatron indexed-dataset shards."""

    documents: int
    sequences: int
    tokens: int
    output_bytes: int
    shards: int


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
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repository ID.")
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
        help="fnmatch pattern applied to repository-relative paths after config filtering.",
    )
    parser.add_argument("--text-column", default="text", help="Parquet column containing document text.")
    parser.add_argument(
        "--id-column",
        help="Optional document ID column. Defaults to stable IDs derived from source path and row number.",
    )
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b", help="Hugging Face tokenizer repository ID.")
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
        help="Stable output task/shard count. Defaults to min(input files, 2 * workers).",
    )
    parser.add_argument("--reader-batch-size", type=int, default=1000, help="Parquet reader batch size.")
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
    tasks = requested_tasks if requested_tasks is not None else min(file_count, requested_workers * 2)
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


def indexed_dataset_totals(index_paths: list[Path]) -> IndexedDatasetTotals:
    """Validate Megatron indexed datasets and aggregate their counts.

    Args:
        index_paths: Megatron .idx files whose matching .bin files will be checked.

    Returns:
        Aggregate document, sequence, token, byte, and shard counts.
    """
    documents = 0
    sequences = 0
    tokens = 0
    output_bytes = 0
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
            raise ValueError(
                f"{binary_path} has {binary_path.stat().st_size} bytes; expected {expected_binary_bytes}"
            )
        documents += max(0, document_index_count - 1)
        sequences += sequence_count
        tokens += token_count
        output_bytes += index_path.stat().st_size + binary_path.stat().st_size
    return IndexedDatasetTotals(documents, sequences, tokens, output_bytes, len(index_paths))


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

    tokenizer_info = api.model_info(args.tokenizer, revision=args.tokenizer_revision)
    tokenizer_revision = tokenizer_info.sha
    snapshot_download(
        repo_id=args.tokenizer,
        revision=tokenizer_revision,
        local_dir=tokenizer_dir,
        allow_patterns=["tokenizer.json"],
    )
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
        "dataset": {
            "configs": args.config,
            "id": args.dataset,
            "path_glob": args.path_glob,
            "remote_root": f"hf://datasets/{args.dataset}@{dataset_revision}/",
            "requested_revision": args.dataset_revision,
            "resolved_revision": dataset_revision,
            "selected_parquet_bytes": source_bytes,
            "selected_parquet_files": len(input_paths),
        },
        "input_paths_file": "metadata/input_paths.txt",
        "input_paths_sha256": sha256_file(paths_file),
        "output": {
            "data_dir": "data",
            "data_prefixes_file": "data_prefixes.txt",
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
        "schema_version": 1,
        "tokenizer": {
            "eos_id": eos_id,
            "eos_token": args.eos_token,
            "id": args.tokenizer,
            "json_path": "metadata/tokenizer/tokenizer.json",
            "json_sha256": sha256_file(tokenizer_json),
            "requested_revision": args.tokenizer_revision,
            "resolved_revision": tokenizer_revision,
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
    expected = {
        "dataset": (dataset["id"], args.dataset),
        "configs": (dataset["configs"], args.config),
        "path_glob": (dataset["path_glob"], args.path_glob),
        "text_column": (pipeline["text_column"], args.text_column),
        "id_column": (pipeline["id_column"], args.id_column),
        "reader_batch_size": (pipeline["reader_batch_size"], args.reader_batch_size),
        "tokenizer_batch_size": (pipeline["tokenizer_batch_size"], args.tokenizer_batch_size),
        "save_filename": (output["save_filename"], args.save_filename),
        "tokenizer": (tokenizer["id"], args.tokenizer),
        "eos_token": (tokenizer["eos_token"], args.eos_token),
    }
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


def validate_outputs(*, data_dir: Path, save_filename: str, tasks: int) -> IndexedDatasetTotals:
    """Validate all expected output shards.

    Args:
        data_dir: Directory containing Megatron .bin/.idx pairs.
        save_filename: Filename prefix configured in the tokenizer writer.
        tasks: Expected output shard count.

    Returns:
        Aggregate indexed-dataset totals.
    """
    index_paths = sorted(data_dir.glob(f"{save_filename}_*_tokens.idx"))
    if len(index_paths) != tasks:
        raise RuntimeError(f"Expected {tasks} output indexes, found {len(index_paths)} under {data_dir}")
    return indexed_dataset_totals(index_paths)


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
    """Execute or resume the remote-Parquet-to-Megatron pipeline.

    Args:
        args: Parsed CLI arguments.
        manifest: Immutable production preprocessing manifest.
    """
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.tokens import MegatronDocumentTokenizer

    if __package__:
        from .datatrove_text_reader import TextParquetReader
    else:
        from datatrove_text_reader import TextParquetReader

    dataset = manifest["dataset"]
    pipeline = manifest["pipeline"]
    output = manifest["output"]
    tokenizer = manifest["tokenizer"]
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
        totals = validate_outputs(data_dir=data_dir, save_filename=output["save_filename"], tasks=tasks)
        write_data_prefixes(
            output_folder=args.output_folder,
            data_dir=data_dir,
            save_filename=output["save_filename"],
        )
        logger.info("Run is already complete and valid: %d documents, %d tokens", totals.documents, totals.tokens)
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    executor = LocalPipelineExecutor(
        pipeline=[
            TextParquetReader(
                data_folder=dataset["remote_root"],
                paths_file=str(paths_file),
                batch_size=int(pipeline["reader_batch_size"]),
                text_key=pipeline["text_column"],
                id_key=pipeline["id_column"],
            ),
            MegatronDocumentTokenizer(
                output_folder=str(data_dir),
                save_filename=output["save_filename"],
                tokenizer_name_or_path=str(tokenizer_json),
                eos_token=tokenizer["eos_token"],
                batch_size=int(pipeline["tokenizer_batch_size"]),
            ),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(logs_dir),
        skip_completed=args.resume,
    )
    logger.info(
        "Processing %d remote Parquet files with tasks=%d workers=%d from %s",
        dataset["selected_parquet_files"],
        tasks,
        workers,
        dataset["remote_root"],
    )
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    executor.run()
    wall_seconds = time.perf_counter() - start
    totals = validate_outputs(data_dir=data_dir, save_filename=output["save_filename"], tasks=tasks)
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
            "parquet_bytes": dataset["selected_parquet_bytes"],
            "parquet_files": dataset["selected_parquet_files"],
            "resolved_revision": dataset["resolved_revision"],
        },
        "parameters": {
            "eos_id": tokenizer["eos_id"],
            "eos_token": tokenizer["eos_token"],
            "tasks": tasks,
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
    manifest = load_resumable_run(args) if args.resume else prepare_new_run(args)
    run_pipeline(args, manifest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
