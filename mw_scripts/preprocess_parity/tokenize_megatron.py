# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Stream HF rows to temporary JSONL and invoke Megatron-LM preprocessing.

Usage:
python tokenize_megatron.py \
  --manifest raw_data/manifest.json \
  --tokenizer-dir raw_data/tokenizer \
  --preprocess-script /workspace/ws/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py \
  --output-dir raw_data/megatron \
  --temp-dir /tmp \
  --workers 8
"""

import argparse
import json
import logging
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def find_preprocess_script() -> Path:
    """Find the checked-in Megatron-LM preprocessing script from this file's ancestors."""
    relative_path = Path("3rdparty/Megatron-LM/tools/preprocess_data.py")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return candidate
    return relative_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest produced by prepare_input.py.")
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="Pinned HF tokenizer directory.")
    parser.add_argument(
        "--preprocess-script",
        type=Path,
        default=find_preprocess_script(),
        help="Path to Megatron-LM tools/preprocess_data.py inside the container.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Megatron output directory.")
    parser.add_argument("--temp-dir", type=Path, help="Local directory for temporary JSONL; defaults to TMPDIR.")
    parser.add_argument("--eos-token", default="<|return|>", help="Expected GPT-OSS end-of-document token.")
    parser.add_argument("--workers", type=int, default=8, help="Megatron tokenizer worker processes.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    """Load and minimally validate the parity manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"configs", "dataset", "dataset_revision", "split", "text_column"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    return manifest


def materialize_jsonl(manifest: dict, output_path: Path) -> tuple[int, int]:
    """Stream every pinned HF config into the JSONL format required by Megatron-LM."""
    rows_written = 0
    empty_rows_skipped = 0
    with output_path.open("x", encoding="utf-8") as output_file:
        for config in manifest["configs"]:
            logger.info("Streaming config %s", config)
            dataset = load_dataset(
                manifest["dataset"],
                config,
                split=manifest["split"],
                revision=manifest["dataset_revision"],
                streaming=True,
            )
            for row in dataset:
                text = row[manifest["text_column"]]
                if not text:
                    empty_rows_skipped += 1
                    continue
                if not isinstance(text, str):
                    raise TypeError(
                        f"Expected string in {manifest['text_column']!r}, found {type(text).__name__}"
                    )
                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                rows_written += 1
    return rows_written, empty_rows_skipped


def run_megatron(args: argparse.Namespace, input_path: Path) -> None:
    """Run the checked-in Megatron-LM preprocessing entry point."""
    output_prefix = args.output_dir / "megatron"
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
        str(args.tokenizer_dir),
        "--tokenizer-hf-no-include-special-tokens",
        "--append-eod",
        "--workers",
        str(args.workers),
        "--log-interval",
        "100",
    ]
    logger.info("Running: %s", shlex.join(command))
    subprocess.run(command, check=True)


def main() -> None:
    """Create Megatron's required temporary JSONL, preprocess it, then delete it."""
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if not (args.tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(args.tokenizer_dir / "tokenizer.json")
    if not args.preprocess_script.is_file():
        raise FileNotFoundError(args.preprocess_script)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.temp_dir is not None and not args.temp_dir.is_dir():
        raise FileNotFoundError(args.temp_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.eos_token != args.eos_token:
        raise ValueError(f"Expected EOS token {args.eos_token!r}, found {tokenizer.eos_token!r}")
    if tokenizer.eos_token_id is None:
        raise ValueError(f"EOS token {args.eos_token!r} does not have a token ID")
    logger.info(
        "Megatron --append-eod will append EOS token %r with ID %d",
        tokenizer.eos_token,
        tokenizer.eos_token_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_prefix = args.output_dir / "megatron_text_document"
    for suffix in (".bin", ".idx"):
        if expected_prefix.with_suffix(suffix).exists():
            raise FileExistsError(f"Refusing to overwrite {expected_prefix.with_suffix(suffix)}")

    with tempfile.TemporaryDirectory(prefix="megatron-preprocess-", dir=args.temp_dir) as temporary_dir:
        input_path = Path(temporary_dir) / "input.jsonl"
        conversion_start = time.perf_counter()
        rows_written, empty_rows_skipped = materialize_jsonl(manifest, input_path)
        conversion_seconds = time.perf_counter() - conversion_start
        logger.info(
            "Materialized %d rows (%d bytes) in %.3f seconds; skipped %d empty rows",
            rows_written,
            input_path.stat().st_size,
            conversion_seconds,
            empty_rows_skipped,
        )

        preprocessing_start = time.perf_counter()
        run_megatron(args, input_path)
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        logger.info("Megatron preprocessing completed in %.3f seconds", preprocessing_seconds)

    logger.info("Temporary JSONL was removed")
    logger.info("Megatron output prefix: %s", expected_prefix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
