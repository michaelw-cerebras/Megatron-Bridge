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

"""Expand full-path CodeSign source weights into Megatron shard weights."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture-config", required=True, type=Path)
    parser.add_argument("--output-data-args", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--train-tokens",
        type=int,
        help="Override train_tokens from the mixture config for audit statistics.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file."""
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


def _load_source_weights(config: dict[str, Any], config_path: Path) -> dict[Path, tuple[Decimal, Decimal]]:
    """Load enabled source weights and return raw and normalized values."""
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version in {config_path}: {config.get('schema_version')!r}")
    if config.get("normalize_weights") is not True:
        raise ValueError(f"{config_path} must set normalize_weights to true")

    raw_sources = config.get("source_weights")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError(f"source_weights must be a non-empty object in {config_path}")

    enabled: dict[Path, Decimal] = {}
    for raw_path, raw_weight in raw_sources.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Invalid source path in {config_path}: {raw_path!r}")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"Weight for {raw_path!r} must be numeric")
        weight = Decimal(str(raw_weight))
        if not weight.is_finite() or weight < 0:
            raise ValueError(f"Weight for {raw_path!r} must be finite and nonnegative")
        if weight == 0:
            continue
        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            source_path = config_path.parent / source_path
        source_path = source_path.absolute()
        if source_path in enabled:
            raise ValueError(f"Duplicate resolved source path: {source_path}")
        enabled[source_path] = weight

    if not enabled:
        raise ValueError(f"No sources are enabled in {config_path}. Set one or more source_weights values above zero.")
    total = sum(enabled.values(), Decimal(0))
    return {source: (weight, weight / total) for source, weight in enabled.items()}


def _resolve_prefixes(source_root: Path) -> list[Path]:
    """Resolve and validate a source's Megatron data prefixes."""
    prefixes_path = source_root / "data_prefixes.txt"
    if not prefixes_path.is_file():
        raise FileNotFoundError(prefixes_path)
    prefixes: list[Path] = []
    for line in prefixes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw_prefix = Path(line.strip())
        prefix = raw_prefix if raw_prefix.is_absolute() else source_root / raw_prefix
        prefix = prefix.absolute()
        if any(character.isspace() for character in str(prefix)):
            raise ValueError(f"Megatron data prefixes cannot contain whitespace: {prefix}")
        for suffix in (".bin", ".idx"):
            dataset_file = Path(f"{prefix}{suffix}")
            if not dataset_file.is_file():
                raise FileNotFoundError(dataset_file)
        prefixes.append(prefix)
    if not prefixes:
        raise ValueError(f"No data prefixes found in {prefixes_path}")
    if len(prefixes) != len(set(prefixes)):
        raise ValueError(f"Duplicate data prefixes found in {prefixes_path}")
    return prefixes


def _load_tokenizer(source_root: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Load tokenizer metadata from a source or its hierarchy manifest."""
    tokenizer = manifest.get("tokenizer")
    manifest_path = source_root / "manifest.json"
    if isinstance(tokenizer, dict):
        return tokenizer, manifest_path

    hierarchy_path = source_root.parent / "hierarchy_manifest.json"
    if not hierarchy_path.is_file():
        raise ValueError(f"No tokenizer metadata in {manifest_path} and no {hierarchy_path}")
    hierarchy = _read_json(hierarchy_path)
    tokenizer = hierarchy.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError(f"Missing tokenizer metadata in {hierarchy_path}")
    return tokenizer, hierarchy_path


def _validate_tokenizer(tokenizer: dict[str, Any], metadata_path: Path) -> tuple[int, str]:
    """Return the stable tokenizer identity used to compare all sources."""
    eos_id = tokenizer.get("eos_id")
    digest = tokenizer.get("json_sha256")
    if not isinstance(eos_id, int) or isinstance(eos_id, bool):
        raise ValueError(f"Invalid tokenizer eos_id in {metadata_path}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Invalid tokenizer json_sha256 in {metadata_path}")
    return eos_id, digest


def _load_shard_tokens(
    source_root: Path,
    manifest: dict[str, Any],
    prefixes: list[Path],
) -> tuple[list[int], int, int | None]:
    """Load exact per-shard token counts for regular and zero-copy sources."""
    raw_shards = manifest.get("shards")
    if isinstance(raw_shards, list):
        tokens_by_prefix: dict[Path, int] = {}
        for shard in raw_shards:
            if not isinstance(shard, dict):
                raise TypeError(f"Invalid shard entry in {source_root / 'manifest.json'}")
            raw_prefix = shard.get("prefix")
            tokens = shard.get("tokens")
            if (
                not isinstance(raw_prefix, str)
                or not isinstance(tokens, int)
                or isinstance(tokens, bool)
                or tokens < 1
            ):
                raise ValueError(f"Invalid shard metadata in {source_root / 'manifest.json'}: {shard!r}")
            shard_prefix = Path(raw_prefix)
            if not shard_prefix.is_absolute():
                shard_prefix = source_root / shard_prefix
            tokens_by_prefix[shard_prefix.absolute()] = tokens
        if set(prefixes) != set(tokens_by_prefix):
            raise ValueError(f"Shard prefixes disagree between manifest.json and data_prefixes.txt in {source_root}")
        shard_tokens = [tokens_by_prefix[prefix] for prefix in prefixes]
        declared_tokens = manifest.get("tokens")
        if not isinstance(declared_tokens, int) or isinstance(declared_tokens, bool):
            raise ValueError(f"Missing total tokens in {source_root / 'manifest.json'}")
        if sum(shard_tokens) != declared_tokens:
            raise ValueError(f"Shard tokens sum to {sum(shard_tokens)}, expected {declared_tokens} in {source_root}")
        return shard_tokens, declared_tokens, None

    result_path = source_root / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = _read_json(result_path).get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Missing result object in {result_path}")
    total_tokens = result.get("tokens")
    if not isinstance(total_tokens, int) or isinstance(total_tokens, bool) or total_tokens < 1:
        raise ValueError(f"Invalid token count in {result_path}")
    total_bin_bytes = sum(Path(f"{prefix}.bin").stat().st_size for prefix in prefixes)
    token_width, remainder = divmod(total_bin_bytes, total_tokens)
    if remainder or token_width not in (1, 2, 4, 8):
        raise ValueError(
            f"Cannot infer token width for {source_root}: bin_bytes={total_bin_bytes}, tokens={total_tokens}"
        )
    shard_tokens: list[int] = []
    for prefix in prefixes:
        bin_bytes = Path(f"{prefix}.bin").stat().st_size
        tokens, shard_remainder = divmod(bin_bytes, token_width)
        if shard_remainder or tokens < 1:
            raise ValueError(f"Invalid token storage size for {prefix}.bin: {bin_bytes}")
        shard_tokens.append(tokens)
    if sum(shard_tokens) != total_tokens:
        raise ValueError(f"Shard tokens sum to {sum(shard_tokens)}, expected {total_tokens} in {source_root}")
    return shard_tokens, total_tokens, token_width


def build_mixture(
    *,
    mixture_config_path: Path,
    train_tokens_override: int | None,
) -> tuple[str, dict[str, Any]]:
    """Build Megatron data args and an auditable resolved mixture manifest."""
    mixture_config_path = mixture_config_path.expanduser().resolve()
    if not mixture_config_path.is_file():
        raise FileNotFoundError(mixture_config_path)
    mixture_config = _read_json(mixture_config_path)
    source_weights = _load_source_weights(mixture_config, mixture_config_path)

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
    tokenizer_identity: tuple[int, str] | None = None
    resolved_tokenizer: dict[str, Any] | None = None
    resolved_weight_sum = Decimal(0)
    total_shards = 0

    for source_root, (raw_weight, source_weight) in source_weights.items():
        manifest_path = source_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = _read_json(manifest_path)
        tokenizer, tokenizer_metadata_path = _load_tokenizer(source_root, manifest)
        identity = _validate_tokenizer(tokenizer, tokenizer_metadata_path)
        if tokenizer_identity is None:
            tokenizer_identity = identity
            resolved_tokenizer = tokenizer
        elif identity != tokenizer_identity:
            raise ValueError(
                f"Tokenizer mismatch for {source_root}: eos/sha={identity}, expected {tokenizer_identity}"
            )

        prefixes = _resolve_prefixes(source_root)
        shard_tokens, available_tokens, token_width = _load_shard_tokens(source_root, manifest, prefixes)
        source_shards: list[dict[str, Any]] = []
        source_shard_weight = Decimal(0)
        for prefix, tokens in zip(prefixes, shard_tokens):
            shard_weight = source_weight * Decimal(tokens) / Decimal(available_tokens)
            data_args_lines.extend((format(shard_weight, ".17g"), str(prefix)))
            source_shards.append({"prefix": str(prefix), "tokens": tokens, "weight": float(shard_weight)})
            source_shard_weight += shard_weight
        if not math.isclose(float(source_shard_weight), float(source_weight), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Shard weights for {source_root} do not sum to the normalized source weight")

        target_tokens = Decimal(train_tokens) * source_weight
        resolved_sources[str(source_root)] = {
            "raw_weight": float(raw_weight),
            "weight": float(source_weight),
            "available_tokens": available_tokens,
            "target_train_tokens": int(target_tokens),
            "expected_epochs": float(target_tokens / Decimal(available_tokens)),
            "documents": manifest.get("documents"),
            "storage": manifest.get("storage", "regular-bin-idx"),
            "token_width_bytes": token_width,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "shard_count": len(prefixes),
            "shards": source_shards,
        }
        total_shards += len(prefixes)
        resolved_weight_sum += source_shard_weight
        LOGGER.info(
            "%s: raw_weight=%s normalized_weight=%.6f available_tokens=%d target_tokens=%d shards=%d",
            source_root,
            raw_weight,
            float(source_weight),
            available_tokens,
            int(target_tokens),
            len(prefixes),
        )

    if not math.isclose(float(resolved_weight_sum), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Resolved shard weights sum to {resolved_weight_sum}, expected 1.0")
    if resolved_tokenizer is None:
        raise AssertionError("At least one tokenizer must have been resolved")

    data_args_content = "\n".join(data_args_lines) + "\n"
    resolved_manifest = {
        "schema_version": 1,
        "name": mixture_config.get("name"),
        "description": mixture_config.get("description"),
        "mixture_config": str(mixture_config_path),
        "mixture_config_sha256": _sha256(mixture_config_path),
        "data_args_sha256": hashlib.sha256(data_args_content.encode("utf-8")).hexdigest(),
        "train_tokens": train_tokens,
        "source_count": len(resolved_sources),
        "shard_count": total_shards,
        "weight_sum": float(resolved_weight_sum),
        "tokenizer": resolved_tokenizer,
        "sources": resolved_sources,
    }
    return data_args_content, resolved_manifest


def main() -> None:
    """Build and write mixture artifacts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_args, resolved_manifest = build_mixture(
        mixture_config_path=args.mixture_config,
        train_tokens_override=args.train_tokens,
    )
    output_data_args = args.output_data_args.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    _atomic_write(output_data_args, data_args)
    _atomic_write(output_manifest, json.dumps(resolved_manifest, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Wrote %d weighted shard prefixes to %s", resolved_manifest["shard_count"], output_data_args)
    LOGGER.info("Wrote resolved mixture manifest to %s", output_manifest)


if __name__ == "__main__":
    main()
