# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Stream Hugging Face Parquet data into Megatron format with DataTrove.

Usage:
python tokenize_datatrove.py \
  --manifest raw_data/manifest.json \
  --tokenizer-json raw_data/tokenizer/tokenizer.json \
  --output-dir raw_data/datatrove \
  --tasks 1 \
  --workers 1
"""

import argparse
import json
import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import MegatronDocumentTokenizer
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest produced by prepare_input.py.")
    parser.add_argument("--tokenizer-json", type=Path, required=True, help="Pinned tokenizer.json file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="DataTrove output directory.")
    parser.add_argument("--eos-token", default="<|return|>", help="End-of-document token.")
    parser.add_argument("--read-batch-size", type=int, default=1000, help="Streaming reader batch size.")
    parser.add_argument("--tokenizer-batch-size", type=int, default=1000, help="Tokenizer batch size.")
    parser.add_argument("--tasks", type=int, default=1, help="Number of DataTrove output shards.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent local DataTrove tasks.")
    parser.add_argument("--resume", action="store_true", help="Skip ranks marked complete in the logging directory.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    """Load and minimally validate the parity manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"configs", "dataset", "dataset_revision", "id_column", "split", "text_column"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    return manifest


def main() -> None:
    """Run a resumable, production-shaped streaming DataTrove pipeline."""
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if not args.tokenizer_json.is_file():
        raise FileNotFoundError(args.tokenizer_json)
    if args.read_batch_size <= 0 or args.tokenizer_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.tasks <= 0 or args.workers <= 0 or args.workers > args.tasks:
        raise ValueError("Require 1 <= --workers <= --tasks")

    existing_outputs = list(args.output_dir.glob("datatrove_*_tokens.bin")) + list(
        args.output_dir.glob("datatrove_*_tokens.idx")
    )
    if existing_outputs and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing DataTrove outputs: {existing_outputs[:3]}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    eos_id = tokenizer.token_to_id(args.eos_token)
    if eos_id is None:
        raise ValueError(f"EOS token {args.eos_token!r} is not present in the tokenizer")
    logger.info("Using EOS token %r with ID %d", args.eos_token, eos_id)

    readers = [
        HuggingFaceDatasetReader(
            dataset=manifest["dataset"],
            dataset_options={
                "name": config,
                "revision": manifest["dataset_revision"],
                "split": manifest["split"],
            },
            streaming=True,
            batch_size=args.read_batch_size,
            text_key=manifest["text_column"],
            id_key=manifest["id_column"],
        )
        for config in manifest["configs"]
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    executor = LocalPipelineExecutor(
        pipeline=[
            *readers,
            MegatronDocumentTokenizer(
                output_folder=str(args.output_dir),
                save_filename="datatrove",
                tokenizer_name_or_path=str(args.tokenizer_json),
                eos_token=args.eos_token,
                batch_size=args.tokenizer_batch_size,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=str(args.output_dir / "logs"),
        skip_completed=args.resume,
    )
    executor.run()
    logger.info("Wrote %d DataTrove shard(s) under %s", args.tasks, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
