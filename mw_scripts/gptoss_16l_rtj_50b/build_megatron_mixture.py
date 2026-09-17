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

"""Expand this experiment's source weights into a Megatron shard-level data-args file."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse mixture expansion arguments."""
    parser = argparse.ArgumentParser(
        description="Build a token-proportional Megatron data blend from source-level weights."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--mixture-config", required=True, type=Path)
    parser.add_argument("--output-data-args", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--train-tokens",
        type=int,
        help="Override train_tokens from the mixture config when calculating audit statistics.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from path."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace path with UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_name = temporary_file.name
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _load_source_weights(config: dict[str, Any], config_path: Path) -> dict[str, Decimal]:
    """Validate and return positive source weights whose sum is one."""
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version in {config_path}: {config.get('schema_version')!r}")
    raw_weights = config.get("source_weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(f"source_weights must be a non-empty object in {config_path}")

    weights: dict[str, Decimal] = {}
    for source, raw_weight in raw_weights.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"Invalid source name in {config_path}: {source!r}")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"Weight for {source!r} must be numeric")
        weight = Decimal(str(raw_weight))
        if weight <= 0:
            raise ValueError(f"Weight for {source!r} must be positive")
        weights[source] = weight

    total_weight = sum(weights.values(), Decimal(0))
    if not math.isclose(float(total_weight), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Source weights sum to {total_weight}, expected 1.0")
    return weights


def _resolve_prefixes(dataset_root: Path, prefixes_file: Path) -> list[Path]:
    """Resolve and validate dataset prefixes listed in a source manifest."""
    if not prefixes_file.is_file():
        raise FileNotFoundError(prefixes_file)
    prefixes = [
        (dataset_root / line.strip()).resolve()
        for line in prefixes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prefixes:
        raise ValueError(f"No dataset prefixes found in {prefixes_file}")
    for prefix in prefixes:
        if any(character.isspace() for character in str(prefix)):
            raise ValueError(f"Megatron data prefixes cannot contain whitespace: {prefix}")
        for suffix in (".bin", ".idx"):
            dataset_file = Path(f"{prefix}{suffix}")
            if not dataset_file.is_file():
                raise FileNotFoundError(dataset_file)
    return prefixes


def build_mixture(
    *,
    dataset_root: Path,
    mixture_config_path: Path,
    train_tokens_override: int | None,
) -> tuple[str, dict[str, Any]]:
    """Build native Megatron data args and an auditable resolved manifest."""
    dataset_root = dataset_root.expanduser().resolve()
    mixture_config_path = mixture_config_path.expanduser().resolve()
    dataset_manifest_path = dataset_root / "manifest.json"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(dataset_manifest_path)
    if not mixture_config_path.is_file():
        raise FileNotFoundError(mixture_config_path)

    dataset_manifest = _read_json(dataset_manifest_path)
    mixture_config = _read_json(mixture_config_path)
    source_weights = _load_source_weights(mixture_config, mixture_config_path)
    dataset_sources = dataset_manifest.get("sources")
    if not isinstance(dataset_sources, dict):
        raise ValueError(f"Missing sources object in {dataset_manifest_path}")

    train_tokens = train_tokens_override
    if train_tokens is None:
        raw_train_tokens = mixture_config.get("train_tokens")
        if not isinstance(raw_train_tokens, int) or isinstance(raw_train_tokens, bool):
            raise TypeError(f"train_tokens must be an integer in {mixture_config_path}")
        train_tokens = raw_train_tokens
    if train_tokens < 1:
        raise ValueError("train_tokens must be positive")

    data_args_lines: list[str] = []
    resolved_sources: dict[str, Any] = {}
    total_shards = 0
    token_width: int | None = None
    resolved_weight_sum = Decimal(0)

    for source, source_weight in source_weights.items():
        source_metadata = dataset_sources.get(source)
        if not isinstance(source_metadata, dict):
            raise ValueError(f"Source {source!r} is not present in {dataset_manifest_path}")
        tokenized = source_metadata.get("tokenized")
        if not isinstance(tokenized, dict) or not isinstance(tokenized.get("tokens"), int):
            raise ValueError(f"Source {source!r} has no tokenized token count")
        available_tokens = int(tokenized["tokens"])
        if available_tokens < 1:
            raise ValueError(f"Source {source!r} has no tokens")

        raw_prefixes_file = source_metadata.get("data_prefixes_file")
        if not isinstance(raw_prefixes_file, str):
            raise ValueError(f"Source {source!r} has no data_prefixes_file")
        prefixes = _resolve_prefixes(dataset_root, dataset_root / raw_prefixes_file)
        expected_shards = tokenized.get("shards")
        if expected_shards != len(prefixes):
            raise ValueError(
                f"Source {source!r} has {len(prefixes)} prefixes, but its manifest declares {expected_shards} shards"
            )

        total_bin_bytes = sum(Path(f"{prefix}.bin").stat().st_size for prefix in prefixes)
        source_token_width, remainder = divmod(total_bin_bytes, available_tokens)
        if remainder or source_token_width < 1:
            raise ValueError(
                f"Cannot infer an integral token width for {source!r}: "
                f"bin_bytes={total_bin_bytes}, tokens={available_tokens}"
            )
        if token_width is None:
            token_width = source_token_width
        elif source_token_width != token_width:
            raise ValueError(
                f"Inconsistent token widths: {source!r} uses {source_token_width} bytes, expected {token_width}"
            )

        resolved_shards: list[dict[str, Any]] = []
        source_shard_tokens = 0
        source_shard_weight = Decimal(0)
        for prefix in prefixes:
            bin_size = Path(f"{prefix}.bin").stat().st_size
            shard_tokens, shard_remainder = divmod(bin_size, token_width)
            if shard_remainder or shard_tokens < 1:
                raise ValueError(f"Invalid token storage size for {prefix}.bin: {bin_size}")
            shard_weight = source_weight * Decimal(shard_tokens) / Decimal(available_tokens)
            data_args_lines.extend((format(shard_weight, ".17g"), str(prefix)))
            resolved_shards.append(
                {
                    "prefix": str(prefix),
                    "tokens": shard_tokens,
                    "weight": float(shard_weight),
                }
            )
            source_shard_tokens += shard_tokens
            source_shard_weight += shard_weight

        if source_shard_tokens != available_tokens:
            raise ValueError(
                f"Source {source!r} shard tokens sum to {source_shard_tokens}, expected {available_tokens}"
            )
        if not math.isclose(float(source_shard_weight), float(source_weight), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Shard weights for {source!r} do not sum to the requested source weight")

        target_tokens = Decimal(train_tokens) * source_weight
        resolved_sources[source] = {
            "weight": float(source_weight),
            "available_tokens": available_tokens,
            "target_train_tokens": int(target_tokens),
            "expected_epochs": float(target_tokens / Decimal(available_tokens)),
            "shard_count": len(prefixes),
            "shards": resolved_shards,
        }
        total_shards += len(prefixes)
        resolved_weight_sum += source_shard_weight
        LOGGER.info(
            "%s: weight=%.4f target_tokens=%d available_tokens=%d expected_epochs=%.3f shards=%d",
            source,
            float(source_weight),
            int(target_tokens),
            available_tokens,
            float(target_tokens / Decimal(available_tokens)),
            len(prefixes),
        )

    if not math.isclose(float(resolved_weight_sum), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Resolved shard weights sum to {resolved_weight_sum}, expected 1.0")

    data_args_content = "\n".join(data_args_lines) + "\n"
    resolved_manifest = {
        "schema_version": 1,
        "name": mixture_config.get("name"),
        "dataset_root": str(dataset_root),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "mixture_config": str(mixture_config_path),
        "mixture_config_sha256": _sha256(mixture_config_path),
        "data_args_sha256": hashlib.sha256(data_args_content.encode("utf-8")).hexdigest(),
        "train_tokens": train_tokens,
        "token_width_bytes": token_width,
        "source_count": len(resolved_sources),
        "shard_count": total_shards,
        "weight_sum": float(resolved_weight_sum),
        "sources": resolved_sources,
    }
    return data_args_content, resolved_manifest


def main() -> None:
    """Build and write the requested mixture artifacts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_args_content, resolved_manifest = build_mixture(
        dataset_root=args.dataset_root,
        mixture_config_path=args.mixture_config,
        train_tokens_override=args.train_tokens,
    )
    _atomic_write(args.output_data_args.expanduser().resolve(), data_args_content)
    _atomic_write(
        args.output_manifest.expanduser().resolve(),
        json.dumps(resolved_manifest, indent=2, sort_keys=True) + "\n",
    )
    LOGGER.info(
        "Wrote %d weighted shard prefixes to %s",
        resolved_manifest["shard_count"],
        args.output_data_args.expanduser().resolve(),
    )
    LOGGER.info("Wrote resolved mixture manifest to %s", args.output_manifest.expanduser().resolve())


if __name__ == "__main__":
    main()
