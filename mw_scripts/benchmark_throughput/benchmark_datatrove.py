"""
Benchmark DataTrove Parquet-to-Megatron tokenization throughput.

Usage:
python benchmark_datatrove.py \
--manifest /cb/home/michaelw/ws/Megatron-Bridge/mw_scripts/benchmark_throughput/benchmark_data/manifest.json \
--run-dir /cb/home/michaelw/ws//Megatron-Bridge/mw_scripts/benchmark_throughput/benchmark_data/runs/datatrove_w16 \
--tasks 192 \
--workers 16 \
--discard-output


for WORKERS in 48 64 96; do
  python benchmark_datatrove.py \
    --manifest /cb/home/michaelw/ws/Megatron-Bridge/mw_scripts/benchmark_throughput/benchmark_data/manifest.json \
    --run-dir "/cb/home/michaelw/ws/Megatron-Bridge/mw_scripts/benchmark_throughput/benchmark_data/runs/datatrove_w${WORKERS}_96c" \
    --tasks 192 \
    --workers "$WORKERS" \
    --discard-output
done

"""

import argparse
import logging
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import MegatronDocumentTokenizer
from tokenizers import Tokenizer

from benchmark_utils import (
    directory_size,
    environment_metadata,
    indexed_dataset_totals,
    load_manifest,
    resolve_manifest_path,
    sha256_file,
    totals_as_dict,
    write_json_atomic,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest from prepare_benchmark_data.py.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Unique directory for this benchmark result.")
    parser.add_argument("--workers", type=int, required=True, help="Concurrent DataTrove local worker processes.")
    parser.add_argument(
        "--tasks",
        type=int,
        help="Fixed number of DataTrove tasks; defaults to the prepared Parquet shard count.",
    )
    parser.add_argument("--reader-batch-size", type=int, default=1000, help="Parquet reader batch size.")
    parser.add_argument("--tokenizer-batch-size", type=int, default=1000, help="Tokenizer encode batch size.")
    parser.add_argument("--eos-token", default="<|return|>", help="End-of-document token.")
    parser.add_argument(
        "--discard-output",
        action="store_true",
        help="Delete only this run's newly created .bin/.idx output after recording metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Run DataTrove and record end-to-end throughput metrics."""
    args = parse_args()
    manifest = load_manifest(args.manifest)
    parquet_dir = resolve_manifest_path(args.manifest, manifest["parquet_dir"])
    tokenizer_dir = resolve_manifest_path(args.manifest, manifest["tokenizer_dir"])
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not parquet_dir.is_dir():
        raise FileNotFoundError(parquet_dir)
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    tasks = args.tasks if args.tasks is not None else int(manifest["num_shards"])
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {parquet_dir}")
    if tasks <= 0 or args.workers <= 0 or args.workers > tasks:
        raise ValueError("Require 1 <= --workers <= --tasks")
    if tasks > len(parquet_files):
        raise ValueError(f"Requested {tasks} tasks for only {len(parquet_files)} Parquet files")
    if args.reader_batch_size <= 0 or args.tokenizer_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to reuse benchmark run directory: {args.run_dir}")

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos_id = tokenizer.token_to_id(args.eos_token)
    if eos_id is None:
        raise ValueError(f"EOS token {args.eos_token!r} is not present in the tokenizer")
    expected_tokenizer_hash = manifest.get("tokenizer_json_sha256")
    if expected_tokenizer_hash is not None and sha256_file(tokenizer_json) != expected_tokenizer_hash:
        raise ValueError(f"Tokenizer hash does not match manifest: {tokenizer_json}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.run_dir.mkdir(parents=True)
    output_dir = args.run_dir / "output"
    logs_dir = args.run_dir / "logs"
    output_dir.mkdir()
    input_bytes = directory_size(parquet_dir)
    logger.info(
        "Running DataTrove with tasks=%d workers=%d over %d Parquet shards; EOS=%r (%d)",
        tasks,
        args.workers,
        len(parquet_files),
        args.eos_token,
        eos_id,
    )

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=str(parquet_dir),
                glob_pattern="*.parquet",
                batch_size=args.reader_batch_size,
                read_metadata=False,
                text_key="text",
                id_key="id",
            ),
            MegatronDocumentTokenizer(
                output_folder=str(output_dir),
                save_filename="datatrove",
                tokenizer_name_or_path=str(tokenizer_json),
                eos_token=args.eos_token,
                batch_size=args.tokenizer_batch_size,
            ),
        ],
        tasks=tasks,
        workers=args.workers,
        logging_dir=str(logs_dir),
    )
    started_at = datetime.now(UTC).isoformat()
    benchmark_start = time.perf_counter()
    executor.run()
    wall_seconds = time.perf_counter() - benchmark_start

    index_paths = sorted(output_dir.glob("datatrove_*_tokens.idx"))
    if len(index_paths) != tasks:
        raise RuntimeError(f"Expected {tasks} output indexes, found {len(index_paths)}")
    totals = indexed_dataset_totals(index_paths)
    result = {
        "benchmark_scope": "local_parquet_read+tokenize+indexed_dataset_write",
        "environment": environment_metadata(["datatrove", "datasets", "pyarrow", "tokenizers"]),
        "input": {
            "bytes": input_bytes,
            "manifest": str(args.manifest.resolve()),
            "parquet_shards": len(parquet_files),
            "rows": manifest.get("rows"),
        },
        "parameters": {
            "eos_token": args.eos_token,
            "reader_batch_size": args.reader_batch_size,
            "tasks": tasks,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "tokenizers_parallelism": os.environ["TOKENIZERS_PARALLELISM"],
            "workers": args.workers,
        },
        "result": {
            **totals_as_dict(totals),
            "documents_per_second": totals.documents / wall_seconds,
            "tokens_per_second": totals.tokens / wall_seconds,
            "wall_seconds": wall_seconds,
        },
        "started_at": started_at,
        "tool": "datatrove",
    }
    result_path = args.run_dir / "result.json"
    write_json_atomic(result_path, result)
    logger.info(
        "DataTrove completed: %d documents, %d tokens in %.3f seconds (%.0f tokens/s)",
        totals.documents,
        totals.tokens,
        wall_seconds,
        totals.tokens / wall_seconds,
    )
    logger.info("Wrote benchmark metrics to %s", result_path)
    if args.discard_output:
        shutil.rmtree(output_dir)
        logger.info("Discarded this run's tokenized output; logs and result.json were retained")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
