 
"""
Benchmark Megatron-LM JSONL preprocessing throughput.

Usage:
python /workspace/ws/Megatron-Bridge/mw_scripts/benchmark_throughput/benchmark_megatron.py \
  --manifest /workspace/ws/Megatron-Bridge/mw_scripts/benchmark_throughput/benchmark_data/manifest.json \
  --preprocess-script /workspace/ws/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py \
  --run-dir /workspace/ws/Megatron-Bridge/mw_scripts/benchmark_throughput/benchmark_data/runs/megatron_w16 \
  --workers 16 \
  --mode tokenization-only \
  --discard-output
"""

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import pyarrow.parquet as pq
from transformers import AutoTokenizer

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


def find_preprocess_script() -> Path:
    """Find Megatron-LM's checked-in preprocessing entry point.

    Returns:
        Located path or the expected checkout-relative fallback path.
    """
    relative_path = Path("3rdparty/Megatron-LM/tools/preprocess_data.py")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return candidate
    return relative_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest from prepare_benchmark_data.py.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Unique directory for this benchmark result.")
    parser.add_argument("--workers", type=int, required=True, help="Megatron tokenizer worker processes.")
    parser.add_argument(
        "--mode",
        choices=("tokenization-only", "end-to-end"),
        default="tokenization-only",
        help="Reuse prepared JSONL or include local Parquet-to-JSONL conversion.",
    )
    parser.add_argument(
        "--preprocess-script",
        type=Path,
        default=find_preprocess_script(),
        help="Path to Megatron-LM tools/preprocess_data.py inside the container.",
    )
    parser.add_argument("--temp-dir", type=Path, help="Node-local directory for end-to-end temporary JSONL.")
    parser.add_argument("--parquet-batch-size", type=int, default=1000, help="Parquet-to-JSONL batch size.")
    parser.add_argument("--log-interval", type=int, default=100_000, help="Megatron progress logging interval.")
    parser.add_argument("--eos-token", default="<|return|>", help="Expected GPT-OSS EOS token.")
    parser.add_argument(
        "--discard-output",
        action="store_true",
        help="Delete only this run's newly created .bin/.idx output after recording metrics.",
    )
    return parser.parse_args()


def materialize_jsonl(parquet_dir: Path, output_file: TextIO, *, batch_size: int) -> tuple[int, int]:
    """Convert prepared local Parquet shards into Megatron's JSONL input.

    Args:
        parquet_dir: Directory containing prepared Parquet shards.
        output_file: Open JSONL output stream.
        batch_size: Parquet batch size.

    Returns:
        Written and skipped-empty document counts.
    """
    rows_written = 0
    empty_rows_skipped = 0
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text:
                    empty_rows_skipped += 1
                    continue
                if not isinstance(text, str):
                    raise TypeError(f"Expected text string, found {type(text).__name__}")
                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                rows_written += 1
    return rows_written, empty_rows_skipped


def run_megatron(
    args: argparse.Namespace,
    *,
    input_path: Path,
    tokenizer_dir: Path,
    output_prefix: Path,
) -> None:
    """Invoke the checked-in Megatron-LM preprocessing script.

    Args:
        args: Parsed command-line arguments.
        input_path: JSONL input path.
        tokenizer_dir: Pinned Hugging Face tokenizer directory.
        output_prefix: Megatron output prefix before key/level suffixes.
    """
    command = [
        sys.executable,
        str(args.preprocess_script),
        "--input",
        str(input_path),
        "--output-prefix",
        str(output_prefix),
        "--json-keys",
        "text",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer_dir),
        "--tokenizer-hf-no-include-special-tokens",
        "--append-eod",
        "--workers",
        str(args.workers),
        "--log-interval",
        str(args.log_interval),
    ]
    logger.info("Running: %s", shlex.join(command))
    subprocess.run(command, check=True)


def benchmark_with_input(
    args: argparse.Namespace,
    *,
    input_path: Path,
    tokenizer_dir: Path,
    output_prefix: Path,
) -> float:
    """Time Megatron preprocessing for an existing JSONL input.

    Args:
        args: Parsed command-line arguments.
        input_path: JSONL input path.
        tokenizer_dir: Pinned tokenizer directory.
        output_prefix: Megatron output prefix.

    Returns:
        Tokenization wall time in seconds.
    """
    tokenization_start = time.perf_counter()
    run_megatron(args, input_path=input_path, tokenizer_dir=tokenizer_dir, output_prefix=output_prefix)
    return time.perf_counter() - tokenization_start


def main() -> None:
    """Run Megatron preprocessing and record throughput metrics."""
    args = parse_args()
    manifest = load_manifest(args.manifest)
    parquet_dir = resolve_manifest_path(args.manifest, manifest["parquet_dir"])
    tokenizer_dir = resolve_manifest_path(args.manifest, manifest["tokenizer_dir"])
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not parquet_dir.is_dir():
        raise FileNotFoundError(parquet_dir)
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    if not args.preprocess_script.is_file():
        raise FileNotFoundError(args.preprocess_script)
    if args.workers <= 0 or args.parquet_batch_size <= 0 or args.log_interval <= 0:
        raise ValueError("Worker count, batch size, and log interval must be positive")
    if args.temp_dir is not None and not args.temp_dir.is_dir():
        raise FileNotFoundError(args.temp_dir)
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to reuse benchmark run directory: {args.run_dir}")

    expected_tokenizer_hash = manifest.get("tokenizer_json_sha256")
    if expected_tokenizer_hash is not None and sha256_file(tokenizer_json) != expected_tokenizer_hash:
        raise ValueError(f"Tokenizer hash does not match manifest: {tokenizer_json}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.eos_token != args.eos_token or tokenizer.eos_token_id is None:
        raise ValueError(
            f"Expected EOS {args.eos_token!r}; found token={tokenizer.eos_token!r}, id={tokenizer.eos_token_id}"
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.run_dir.mkdir(parents=True)
    output_dir = args.run_dir / "output"
    output_dir.mkdir()
    output_prefix = output_dir / "megatron"
    expected_index_path = output_dir / "megatron_text_document.idx"
    prepared_jsonl_value = manifest.get("jsonl_file")
    started_at = datetime.now(UTC).isoformat()
    materialization_seconds = 0.0
    materialized_rows: int | None = None
    empty_rows_skipped: int | None = None
    input_bytes = directory_size(parquet_dir)

    benchmark_start = time.perf_counter()
    if args.mode == "tokenization-only":
        if not isinstance(prepared_jsonl_value, str):
            raise ValueError("Manifest has no prepared JSONL; rerun preparation without --no-write-jsonl")
        input_path = resolve_manifest_path(args.manifest, prepared_jsonl_value)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        input_bytes = input_path.stat().st_size
        tokenization_seconds = benchmark_with_input(
            args,
            input_path=input_path,
            tokenizer_dir=tokenizer_dir,
            output_prefix=output_prefix,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="megatron-benchmark-", dir=args.temp_dir) as temporary_dir:
            input_path = Path(temporary_dir) / "input.jsonl"
            materialization_start = time.perf_counter()
            with input_path.open("x", encoding="utf-8") as output_file:
                materialized_rows, empty_rows_skipped = materialize_jsonl(
                    parquet_dir,
                    output_file,
                    batch_size=args.parquet_batch_size,
                )
            materialization_seconds = time.perf_counter() - materialization_start
            logger.info(
                "Materialized %d rows (%d bytes) in %.3f seconds; skipped %d empty rows",
                materialized_rows,
                input_path.stat().st_size,
                materialization_seconds,
                empty_rows_skipped,
            )
            tokenization_seconds = benchmark_with_input(
                args,
                input_path=input_path,
                tokenizer_dir=tokenizer_dir,
                output_prefix=output_prefix,
            )
    wall_seconds = time.perf_counter() - benchmark_start

    if not expected_index_path.is_file():
        raise FileNotFoundError(expected_index_path)
    totals = indexed_dataset_totals([expected_index_path])
    result = {
        "benchmark_scope": (
            "jsonl_tokenize+indexed_dataset_write"
            if args.mode == "tokenization-only"
            else "local_parquet_to_jsonl+tokenize+indexed_dataset_write"
        ),
        "environment": environment_metadata(
            ["datasets", "megatron-core", "pyarrow", "tokenizers", "transformers"]
        ),
        "input": {
            "bytes": input_bytes,
            "manifest": str(args.manifest.resolve()),
            "materialized_rows": materialized_rows,
            "parquet_shards": len(list(parquet_dir.glob("*.parquet"))),
            "rows": manifest.get("rows"),
        },
        "parameters": {
            "eos_token": args.eos_token,
            "log_interval": args.log_interval,
            "mode": args.mode,
            "parquet_batch_size": args.parquet_batch_size,
            "tokenizers_parallelism": os.environ["TOKENIZERS_PARALLELISM"],
            "workers": args.workers,
        },
        "result": {
            **totals_as_dict(totals),
            "documents_per_second": totals.documents / wall_seconds,
            "empty_rows_skipped": empty_rows_skipped,
            "materialization_seconds": materialization_seconds,
            "tokenization_seconds": tokenization_seconds,
            "tokenization_tokens_per_second": totals.tokens / tokenization_seconds,
            "tokens_per_second": totals.tokens / wall_seconds,
            "wall_seconds": wall_seconds,
        },
        "started_at": started_at,
        "tool": "megatron",
    }
    result_path = args.run_dir / "result.json"
    write_json_atomic(result_path, result)
    logger.info(
        "Megatron completed: %d documents, %d tokens in %.3f seconds (%.0f tokens/s end-to-end)",
        totals.documents,
        totals.tokens,
        wall_seconds,
        totals.tokens / wall_seconds,
    )
    logger.info("Wrote benchmark metrics to %s", result_path)
    if args.discard_output:
        shutil.rmtree(output_dir)
        logger.info("Discarded this run's tokenized output; result.json was retained")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
