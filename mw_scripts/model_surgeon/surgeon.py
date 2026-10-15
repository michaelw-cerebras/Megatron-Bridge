# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive a surgery variant of a Hugging Face checkpoint.

Graph-rearrangement variants do not change any tensor, so this clones the
Hugging Face directory unchanged and records which computation graph the clone
needs. The clone is what training, evaluation, and export all point at.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from megatron.bridge.models.surgery.manifest import (
    ARCHITECTURE_FIELD,
    MANIFEST_NAME,
    build_manifest,
    legacy_architecture,
    manifest_variant,
    read_manifest,
)


LOGGER = logging.getLogger(__name__)
PIPELINE_HF_CACHE = Path(__file__).resolve().parents[1] / "mellum2_3xh200/.cache/huggingface/hub"


@dataclass(frozen=True)
class FileRecord:
    """Validation record for one cloned Hugging Face file."""

    path: str
    size: int
    verification: str


@dataclass(frozen=True)
class VerificationResult:
    """Exact weight verification result for a derived checkpoint."""

    variant_id: str
    parameter_tensors: int
    weight_shards: int
    same_inode_shards: int
    sha256_shards: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        if candidate == candidate.parent:
            raise ValueError(f"Cannot find an existing parent for {path}")
        candidate = candidate.parent
    return candidate


def _same_filesystem(first: Path, second: Path) -> bool:
    return _nearest_existing_parent(first).stat().st_dev == _nearest_existing_parent(second).stat().st_dev


def _validate_hf_source(source_dir: Path) -> tuple[dict[str, Any], list[Path], int]:
    """Validate that a directory is a sharded Hugging Face checkpoint we can clone.

    Args:
        source_dir: Candidate Hugging Face directory.

    Returns:
        The parsed config, the sorted file list, and the weight-map entry count.

    Raises:
        ValueError: If required files, the weight map, or shards are invalid.
    """
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("Source must contain config.json and model.safetensors.index.json")
    config = _read_json(config_path)
    if not isinstance(config.get("model_type"), str):
        raise ValueError("Source config.json has no model_type")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not all(isinstance(name, str) for name in architectures):
        raise ValueError("Source config.json has no architectures list")
    if ARCHITECTURE_FIELD in config:
        raise ValueError(
            "Source config already carries a surgery marker; run surgery on the original checkpoint, "
            "not on a derived one"
        )

    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Safetensors index has no weight_map")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise ValueError("Safetensors weight_map must contain string parameter and shard names")
    shard_names = sorted(set(weight_map.values()))
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in shard_names):
        raise ValueError("Safetensors index contains an unsafe or unsupported shard path")
    missing = [name for name in shard_names if not (source_dir / name).is_file()]
    if missing:
        raise ValueError("Safetensors index references missing shards: " + ", ".join(missing))

    files = sorted(path for path in source_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"No Hugging Face files found in {source_dir}")
    return config, files, len(weight_map)


def check_bridge_support(architectures: list[str]) -> None:
    """Fail early when Megatron Bridge has no bridge registered for this family.

    Surgery only rearranges a graph Bridge already knows how to build. Without a
    registered bridge the derived checkpoint is unusable, and the failure would
    otherwise surface much later during a multi-node launch.

    Args:
        architectures: ``architectures`` from the source ``config.json``.

    Raises:
        ValueError: If no registered bridge matches any listed architecture.
    """
    from megatron.bridge import AutoBridge

    supported = set(AutoBridge.list_supported_models())
    if not supported.intersection(architectures):
        raise ValueError(
            f"Megatron Bridge has no registered bridge for architectures {architectures}. "
            "Add model support first, or pass --no-check-bridge to clone anyway."
        )


def resolve_hf_source(
    source: str,
    *,
    target_dir: Path,
    cache_dir: Path | None,
    revision: str | None,
) -> Path:
    """Resolve a local HF directory or download a Hub model snapshot.

    Args:
        source: Local directory or Hugging Face repository ID.
        target_dir: Output directory used to derive a same-filesystem cache.
        cache_dir: Optional Hugging Face cache root.
        revision: Optional Hub branch, tag, or commit.

    Returns:
        Absolute local snapshot directory.

    Raises:
        ValueError: If a local source path exists but is not a directory.
    """
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve(strict=True)
    if local.exists():
        raise ValueError(f"Local Hugging Face source is not a directory: {local}")

    from huggingface_hub import snapshot_download

    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = Path(hf_home).expanduser() / "hub"
        elif PIPELINE_HF_CACHE.is_dir() and _same_filesystem(PIPELINE_HF_CACHE, target_dir.parent):
            cache_dir = PIPELINE_HF_CACHE
        else:
            cache_dir = target_dir.expanduser().resolve(strict=False).parent / ".cache/huggingface/hub"
    snapshot = snapshot_download(
        repo_id=source,
        revision=revision,
        cache_dir=str(cache_dir.expanduser()),
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"],
    )
    return Path(snapshot).resolve(strict=True)


def _clone_file(source: Path, target: Path, *, copy_mode: str) -> FileRecord:
    resolved_source = source.resolve(strict=True)
    if copy_mode == "hardlink":
        try:
            os.link(resolved_source, target)
        except OSError as error:
            if error.errno == errno.EXDEV:
                raise ValueError(
                    "Hard links require source cache and output on the same filesystem; "
                    "omit --cache-dir or use --copy-mode copy"
                ) from error
            raise
        if not os.path.samefile(resolved_source, target):
            raise RuntimeError(f"Hard-link verification failed for {source}")
        verification = "same-inode"
    else:
        shutil.copy2(resolved_source, target)
        source_digest = _sha256(resolved_source)
        target_digest = _sha256(target)
        if source_digest != target_digest:
            raise RuntimeError(f"SHA-256 verification failed for {source}")
        verification = f"sha256:{source_digest}"
    if resolved_source.stat().st_size != target.stat().st_size:
        raise RuntimeError(f"Size verification failed for {source}")
    return FileRecord(path=source.name, size=target.stat().st_size, verification=verification)


def clone_hf_checkpoint(
    source_dir: Path,
    target_dir: Path,
    *,
    variant_id: str,
    params: dict[str, Any] | None = None,
    copy_mode: str = "hardlink",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Clone a Hugging Face checkpoint and record the surgery variant it needs.

    Args:
        source_dir: Existing Hugging Face safetensors directory.
        target_dir: New Hugging Face directory for the derived variant.
        variant_id: Stable variant identifier such as ``"v1"``.
        params: Variant parameters; ``None`` uses the variant defaults.
        copy_mode: ``hardlink`` for zero-copy same-filesystem weights, or
            ``copy`` for an independent SHA-256-verified clone.
        dry_run: Validate and report without writing anything.

    Returns:
        The surgery manifest.

    Raises:
        ValueError: If paths, the variant, or checkpoint contents are invalid.
        FileExistsError: If the target already exists.
    """
    from megatron.bridge.models.surgery import describe_added_parameters, get_variant

    if copy_mode not in {"hardlink", "copy"}:
        raise ValueError("copy_mode must be 'hardlink' or 'copy'")
    variant = get_variant(variant_id)
    resolved_params = variant.resolve_params(params)

    source_dir = source_dir.resolve(strict=True)
    target_dir = target_dir.resolve(strict=False)
    if not source_dir.is_dir():
        raise ValueError(f"Source Hugging Face path is not a directory: {source_dir}")
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite target: {target_dir}")
    if source_dir == target_dir or source_dir in target_dir.parents or target_dir in source_dir.parents:
        raise ValueError("Source and target directories must not contain one another")

    config, source_files, parameter_count = _validate_hf_source(source_dir)
    total_bytes = sum(path.resolve(strict=True).stat().st_size for path in source_files)
    added = describe_added_parameters(variant, resolved_params, config)
    manifest = build_manifest(
        variant_id=variant.id,
        variant_slug=variant.slug,
        equation=variant.equation,
        params=resolved_params,
        source_hf_directory=str(source_dir),
        model_type=str(config["model_type"]),
        parameter_tensors=parameter_count,
        copy_mode=copy_mode,
        file_count=len(source_files),
        total_bytes=total_bytes,
        adds_parameters=added,
    )
    if dry_run:
        return manifest

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir.with_name(f".{target_dir.name}.incomplete-{os.getpid()}")
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary target already exists: {temporary_dir}")
    try:
        temporary_dir.mkdir()
        records: list[dict[str, Any]] = []
        for source in source_files:
            if source.name == "config.json":
                continue
            # Only immutable weight shards need zero-copy links. Copy the small
            # tokenizer/index metadata so editing the target cannot corrupt a hub blob.
            file_mode = copy_mode if source.suffix == ".safetensors" else "copy"
            record = _clone_file(source, temporary_dir / source.name, copy_mode=file_mode)
            records.append(asdict(record))
            LOGGER.info("Cloned %s", source.name)

        target_config = dict(config)
        target_config[ARCHITECTURE_FIELD] = legacy_architecture(variant.id)
        target_config["megatron_bridge_surgery_variant"] = variant.id
        config_target = temporary_dir / "config.json"
        config_target.write_text(json.dumps(target_config, indent=2, sort_keys=True) + "\n")
        records.append(
            asdict(
                FileRecord(
                    path="config.json",
                    size=config_target.stat().st_size,
                    verification=f"derived:added-surgery-marker-{variant.id}",
                )
            )
        )
        manifest["files"] = sorted(records, key=lambda record: str(record["path"]))
        (temporary_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_dir.rename(target_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return manifest


def verify_checkpoint(source_dir: Path, target_dir: Path) -> VerificationResult:
    """Verify that a derived checkpoint has exactly the source weights.

    Same-inode shards are exact by construction. Independently copied shards are
    compared by SHA-256. The safetensors weight maps must also match, which
    proves that every parameter resolves to the same byte-identical shard. This
    is the invariant the whole toolkit rests on: a surgery variant stays
    interchangeable with the original checkpoint.

    Args:
        source_dir: Original Hugging Face checkpoint directory.
        target_dir: Derived Hugging Face checkpoint directory.

    Returns:
        Counts describing the successful exact verification.

    Raises:
        ValueError: If metadata or any weight shard differs.
    """
    source_dir = source_dir.expanduser().resolve(strict=True)
    target_dir = target_dir.expanduser().resolve(strict=True)
    _, _, parameter_count = _validate_hf_source(source_dir)

    manifest = read_manifest(target_dir)
    variant_id, _ = manifest_variant(manifest)
    target_config = _read_json(target_dir / "config.json")
    if target_config.get(ARCHITECTURE_FIELD) != legacy_architecture(variant_id):
        raise ValueError("Target config surgery marker does not match its manifest variant")

    source_index = _read_json(source_dir / "model.safetensors.index.json")
    target_index = _read_json(target_dir / "model.safetensors.index.json")
    source_weight_map = source_index.get("weight_map")
    if source_weight_map != target_index.get("weight_map"):
        raise ValueError("Source and target safetensors weight maps differ")
    if not isinstance(source_weight_map, dict):
        raise ValueError("Source safetensors index has no weight_map")

    shard_names = sorted(set(source_weight_map.values()))
    same_inode_shards = 0
    sha256_shards = 0
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            raise ValueError("Safetensors weight map contains a non-string shard name")
        source_shard = source_dir / shard_name
        target_shard = target_dir / shard_name
        if not target_shard.is_file():
            raise ValueError(f"Target is missing weight shard: {shard_name}")
        if os.path.samefile(source_shard, target_shard):
            same_inode_shards += 1
        elif _sha256(source_shard) == _sha256(target_shard):
            sha256_shards += 1
        else:
            raise ValueError(f"Weight shard differs from source: {shard_name}")

    if manifest.get("parameter_tensors") != parameter_count:
        raise ValueError("Manifest parameter count does not match the safetensors index")
    return VerificationResult(
        variant_id=variant_id,
        parameter_tensors=parameter_count,
        weight_shards=len(shard_names),
        same_inode_shards=same_inode_shards,
        sha256_shards=sha256_shards,
    )


def parse_params(raw: str | None) -> dict[str, Any] | None:
    """Parse the ``--params`` JSON object.

    Args:
        raw: JSON text, or ``None`` to use the variant defaults.

    Returns:
        The parsed parameters, or ``None``.

    Raises:
        ValueError: If the text is not a JSON object.
    """
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"--params must be a JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("--params must be a JSON object")
    return parsed


def main() -> None:
    """Parse arguments and clone the Hugging Face checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Hugging Face model ID or local HF directory")
    parser.add_argument("--output", type=Path, required=True, help="New derived Hugging Face directory")
    parser.add_argument("--variant", required=True, help="Surgery variant id, for example v1")
    parser.add_argument(
        "--params",
        help='JSON variant parameters, for example \'{"layers": {"kind": "every_k", "k": 4}}\'',
    )
    parser.add_argument("--revision", help="Optional Hugging Face branch, tag, or commit")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache root")
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink reuses immutable files; copy creates an independent SHA-256-verified clone",
    )
    parser.add_argument(
        "--no-check-bridge",
        dest="check_bridge",
        action="store_false",
        help="Skip the check that Megatron Bridge has a registered bridge for this model family",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = parse_params(args.params)
    source_dir = resolve_hf_source(
        args.source,
        target_dir=args.output,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    LOGGER.info("Resolved %s to %s", args.source, source_dir)
    if args.check_bridge:
        check_bridge_support(list(_read_json(source_dir / "config.json").get("architectures", [])))

    manifest = clone_hf_checkpoint(
        source_dir,
        args.output,
        variant_id=args.variant,
        params=params,
        copy_mode=args.copy_mode,
        dry_run=args.dry_run,
    )
    variant = manifest["variant"]
    LOGGER.info(
        "%s %s:%s layers=%s over %d parameter tensors in %d files, %.2f GiB",
        "Validated" if args.dry_run else "Created",
        variant["id"],
        variant["slug"],
        variant["params"].get("layers"),
        manifest["parameter_tensors"],
        manifest["file_count"],
        int(manifest["total_bytes"]) / 1024**3,
    )
    if not args.dry_run:
        result = verify_checkpoint(source_dir, args.output)
        LOGGER.info(
            "Verified %s: %d tensors across %d exact weight shards (%d same-inode, %d SHA-256)",
            result.variant_id,
            result.parameter_tensors,
            result.weight_shards,
            result.same_inode_shards,
            result.sha256_shards,
        )


if __name__ == "__main__":
    main()
