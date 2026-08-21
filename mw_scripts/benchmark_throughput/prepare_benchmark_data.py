"""
Stream a pinned HF dataset and create balanced local benchmark shards.

Usage:
    python prepare_benchmark_data.py \
  --work-dir benchmark_data \
  --num-shards 192
"""

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import TextIO

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

from benchmark_utils import directory_size, sha256_file, write_json_atomic


logger = logging.getLogger(__name__)
TOKENIZER_FILES = [
    "chat_template.jinja",
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="nvidia/Nemotron-Pretraining-Legal-v1",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--config",
        default="Nemotron-Pretraining-Legal-eCFR-QA",
        help="Dataset configuration to stream.",
    )
    parser.add_argument("--dataset-revision", default="main", help="Dataset revision or commit.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--text-column", default="text", help="Column containing document text.")
    parser.add_argument("--id-column", default="uuid", help="Column containing a stable document ID.")
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b", help="Tokenizer repository.")
    parser.add_argument("--tokenizer-revision", default="main", help="Tokenizer revision or commit.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Benchmark input and metadata directory.")
    parser.add_argument("--num-shards", type=int, default=192, help="Number of balanced local Parquet shards.")
    parser.add_argument("--reader-batch-size", type=int, default=1000, help="HF streaming reader batch size.")
    parser.add_argument(
        "--writer-batch-size",
        type=int,
        default=256,
        help="Rows buffered per output shard before a Parquet write.",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "snappy", "zstd"),
        default="zstd",
        help="Compression for local benchmark Parquet shards.",
    )
    parser.add_argument(
        "--write-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also prepare the reusable JSONL input required by Megatron-LM.",
    )
    return parser.parse_args()


class BalancedParquetWriters:
    """Round-robin rows into a fixed number of balanced Parquet files."""

    def __init__(
        self,
        output_dir: Path,
        *,
        num_shards: int,
        batch_size: int,
        compression: str,
    ) -> None:
        """Initialize the balanced writers.

        Args:
            output_dir: Directory for output Parquet files.
            num_shards: Number of output files.
            batch_size: Rows buffered per file before writing.
            compression: PyArrow Parquet compression codec.
        """
        self.output_dir = output_dir
        self.num_shards = num_shards
        self.batch_size = batch_size
        self.compression = None if compression == "none" else compression
        self.schema = pa.schema([pa.field("id", pa.large_string()), pa.field("text", pa.large_string())])
        self.id_buffers: list[list[str]] = [[] for _ in range(num_shards)]
        self.text_buffers: list[list[str]] = [[] for _ in range(num_shards)]
        self.writers: list[pq.ParquetWriter | None] = [None] * num_shards
        self.rows_per_shard = [0] * num_shards

    def _writer(self, shard: int) -> pq.ParquetWriter:
        """Return the writer for a shard, creating it lazily.

        Args:
            shard: Zero-based shard number.

        Returns:
            Open Parquet writer.
        """
        writer = self.writers[shard]
        if writer is None:
            output_path = self.output_dir / f"part_{shard:05d}.parquet"
            writer = pq.ParquetWriter(
                output_path,
                self.schema,
                compression=self.compression,
                use_dictionary=False,
            )
            self.writers[shard] = writer
        return writer

    def _flush(self, shard: int) -> None:
        """Flush one shard buffer.

        Args:
            shard: Zero-based shard number.
        """
        if not self.id_buffers[shard]:
            return
        table = pa.table(
            {
                "id": self.id_buffers[shard],
                "text": self.text_buffers[shard],
            },
            schema=self.schema,
        )
        self._writer(shard).write_table(table)
        self.id_buffers[shard].clear()
        self.text_buffers[shard].clear()

    def add(self, *, document_id: str, text: str, row_number: int) -> None:
        """Add one document to its balanced output shard.

        Args:
            document_id: Stable document identifier.
            text: Document text.
            row_number: Zero-based output row number used for round-robin assignment.
        """
        shard = row_number % self.num_shards
        self.id_buffers[shard].append(document_id)
        self.text_buffers[shard].append(text)
        self.rows_per_shard[shard] += 1
        if len(self.id_buffers[shard]) >= self.batch_size:
            self._flush(shard)

    def close(self) -> None:
        """Flush and close all Parquet writers."""
        for shard in range(self.num_shards):
            self._flush(shard)
        for writer in self.writers:
            if writer is not None:
                writer.close()


def stream_and_write(
    args: argparse.Namespace,
    *,
    dataset_revision: str,
    parquet_dir: Path,
    jsonl_file: TextIO | None,
) -> tuple[int, int, list[int]]:
    """Stream HF rows into balanced Parquet shards and optional JSONL.

    Args:
        args: Parsed command-line arguments.
        dataset_revision: Immutable dataset commit.
        parquet_dir: Temporary output directory for Parquet shards.
        jsonl_file: Optional open JSONL output stream.

    Returns:
        Written rows, skipped empty rows, and per-shard row counts.
    """
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        revision=dataset_revision,
        streaming=True,
    )
    features = set(dataset.features or [])
    if args.text_column not in features:
        raise ValueError(f"Dataset config has no {args.text_column!r} column; features={sorted(features)}")
    if args.id_column not in features:
        logger.warning("Dataset has no %r column; generated row IDs will be used", args.id_column)

    writers = BalancedParquetWriters(
        parquet_dir,
        num_shards=args.num_shards,
        batch_size=args.writer_batch_size,
        compression=args.compression,
    )
    rows_written = 0
    empty_rows_skipped = 0
    try:
        for batch in dataset.iter(batch_size=args.reader_batch_size):
            batch_length = len(batch[args.text_column])
            batch_ids = batch.get(args.id_column)
            for row_index in range(batch_length):
                text = batch[args.text_column][row_index]
                if not text:
                    empty_rows_skipped += 1
                    continue
                if not isinstance(text, str):
                    raise TypeError(f"Expected text string, found {type(text).__name__}")
                raw_id = batch_ids[row_index] if batch_ids is not None else None
                document_id = str(raw_id) if raw_id is not None else f"row-{rows_written:012d}"
                writers.add(document_id=document_id, text=text, row_number=rows_written)
                if jsonl_file is not None:
                    jsonl_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                rows_written += 1
                if rows_written % 100_000 == 0:
                    logger.info("Prepared %d documents", rows_written)
    finally:
        writers.close()
    return rows_written, empty_rows_skipped, writers.rows_per_shard


def main() -> None:
    """Pin inputs and build local Parquet/JSONL benchmark artifacts."""
    args = parse_args()
    if args.num_shards <= 0 or args.reader_batch_size <= 0 or args.writer_batch_size <= 0:
        raise ValueError("Shard and batch sizes must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work_dir / "manifest.json"
    parquet_dir = args.work_dir / "input_parquet"
    jsonl_path = args.work_dir / "input.jsonl"
    tokenizer_dir = args.work_dir / "tokenizer"
    for output_path in (manifest_path, parquet_dir):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    if args.write_jsonl and jsonl_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {jsonl_path}")

    api = HfApi()
    dataset_info = api.dataset_info(args.dataset, revision=args.dataset_revision, files_metadata=True)
    dataset_revision = dataset_info.sha
    tokenizer_revision = api.model_info(args.tokenizer, revision=args.tokenizer_revision).sha
    source_parquet_files = sorted(
        sibling.rfilename
        for sibling in dataset_info.siblings
        if sibling.rfilename.startswith(f"{args.config}/") and sibling.rfilename.endswith(".parquet")
    )
    if not source_parquet_files:
        raise RuntimeError(f"No Parquet files found for {args.config} in {args.dataset}@{dataset_revision}")
    logger.info("Pinned dataset revision %s and tokenizer revision %s", dataset_revision, tokenizer_revision)
    logger.info("Source config contains %d remote Parquet shards", len(source_parquet_files))

    snapshot_download(
        repo_id=args.tokenizer,
        revision=tokenizer_revision,
        local_dir=tokenizer_dir,
        allow_patterns=TOKENIZER_FILES,
    )
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)

    preparation_start = time.perf_counter()
    temporary_jsonl_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="gpt-oss-benchmark-", dir=args.work_dir) as temporary_dir_name:
        temporary_dir = Path(temporary_dir_name)
        temporary_parquet_dir = temporary_dir / "input_parquet"
        temporary_parquet_dir.mkdir()
        temporary_jsonl_path = temporary_dir / "input.jsonl" if args.write_jsonl else None
        if temporary_jsonl_path is None:
            rows_written, empty_rows_skipped, rows_per_shard = stream_and_write(
                args,
                dataset_revision=dataset_revision,
                parquet_dir=temporary_parquet_dir,
                jsonl_file=None,
            )
        else:
            with temporary_jsonl_path.open("x", encoding="utf-8") as jsonl_file:
                rows_written, empty_rows_skipped, rows_per_shard = stream_and_write(
                    args,
                    dataset_revision=dataset_revision,
                    parquet_dir=temporary_parquet_dir,
                    jsonl_file=jsonl_file,
                )
        if rows_written < args.num_shards:
            raise ValueError(f"Only {rows_written} rows were written for {args.num_shards} requested shards")
        temporary_parquet_dir.replace(parquet_dir)
        if temporary_jsonl_path is not None:
            temporary_jsonl_path.replace(jsonl_path)

    preparation_seconds = time.perf_counter() - preparation_start
    manifest = {
        "config": args.config,
        "dataset": args.dataset,
        "dataset_revision": dataset_revision,
        "empty_rows_skipped": empty_rows_skipped,
        "id_column": args.id_column,
        "jsonl_bytes": jsonl_path.stat().st_size if args.write_jsonl else None,
        "jsonl_file": jsonl_path.name if args.write_jsonl else None,
        "num_shards": args.num_shards,
        "parquet_bytes": directory_size(parquet_dir),
        "parquet_compression": args.compression,
        "parquet_dir": parquet_dir.name,
        "preparation_seconds": preparation_seconds,
        "rows": rows_written,
        "rows_per_shard_max": max(rows_per_shard),
        "rows_per_shard_min": min(rows_per_shard),
        "source_parquet_files": source_parquet_files,
        "split": args.split,
        "text_column": args.text_column,
        "tokenizer": args.tokenizer,
        "tokenizer_dir": tokenizer_dir.name,
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "tokenizer_revision": tokenizer_revision,
        "version": 1,
    }
    write_json_atomic(manifest_path, manifest)
    logger.info(
        "Prepared %d rows in %d balanced shards (min=%d, max=%d) in %.3f seconds",
        rows_written,
        args.num_shards,
        min(rows_per_shard),
        max(rows_per_shard),
        preparation_seconds,
    )
    logger.info("Wrote benchmark manifest to %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
