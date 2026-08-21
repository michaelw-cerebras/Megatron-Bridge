# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Pin all Nemotron sample configs and the GPT-OSS tokenizer for parity.

Usage: 

python prepare_input.py \
  --work-dir raw_data
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="nvidia/Nemotron-Pretraining-Dataset-sample",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument("--dataset-revision", default="main", help="Dataset revision or commit.")
    parser.add_argument("--split", default="train", help="Dataset split used by every config.")
    parser.add_argument("--text-column", default="text", help="Column containing document text.")
    parser.add_argument("--id-column", default="id", help="Column containing the document ID.")
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b", help="Tokenizer repository.")
    parser.add_argument("--tokenizer-revision", default="main", help="Tokenizer revision or commit.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory for parity metadata and outputs.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Resolve immutable revisions and write a manifest without downloading dataset shards."""
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work_dir / "manifest.json"
    tokenizer_dir = args.work_dir / "tokenizer"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {manifest_path}")

    api = HfApi()
    dataset_info = api.dataset_info(args.dataset, revision=args.dataset_revision, files_metadata=True)
    dataset_sha = dataset_info.sha
    tokenizer_sha = api.model_info(args.tokenizer, revision=args.tokenizer_revision).sha
    logger.info("Resolved dataset revision %s to %s", args.dataset_revision, dataset_sha)
    logger.info("Resolved tokenizer revision %s to %s", args.tokenizer_revision, tokenizer_sha)

    parquet_files = sorted(
        sibling.rfilename for sibling in dataset_info.siblings if sibling.rfilename.endswith(".parquet")
    )
    discovered_configs = sorted({path.split("/", maxsplit=1)[0] for path in parquet_files if "/" in path})
    if not parquet_files or not discovered_configs:
        raise RuntimeError(f"No configured Parquet shards found in {args.dataset}@{dataset_sha}")

    configs = []
    excluded_configs = {}
    for config in discovered_configs:
        dataset = load_dataset(
            args.dataset,
            config,
            split=args.split,
            revision=dataset_sha,
            streaming=True,
        )
        features = sorted(dataset.features) if dataset.features is not None else []
        if args.text_column not in features:
            excluded_configs[config] = {
                "features": features,
                "reason": f"missing {args.text_column!r} column",
            }
            continue
        configs.append(config)

    snapshot_download(
        repo_id=args.tokenizer,
        revision=tokenizer_sha,
        local_dir=tokenizer_dir,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ],
    )
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Tokenizer snapshot does not contain {tokenizer_json.name}")

    manifest = {
        "configs": configs,
        "dataset": args.dataset,
        "dataset_revision": dataset_sha,
        "excluded_configs": excluded_configs,
        "id_column": args.id_column,
        "parquet_files": parquet_files,
        "split": args.split,
        "text_column": args.text_column,
        "tokenizer": args.tokenizer,
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "tokenizer_revision": tokenizer_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Pinned all %d text configs and %d Parquet shards", len(configs), len(parquet_files))
    for config, details in excluded_configs.items():
        logger.warning("Excluded non-text config %s: %s", config, details["reason"])
    logger.info("No dataset shards were persisted locally")
    logger.info("Wrote reproducibility manifest to %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
