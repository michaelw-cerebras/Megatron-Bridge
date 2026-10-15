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

"""Surgery manifest: the contract between a checkpoint and the graph it needs.

A surgically rearranged checkpoint has byte-identical weights and identical
state-dict keys to the original, so nothing in the weights themselves says which
computation graph to run. The manifest is the only record, and every consumer
(training, evaluation harnesses, export) must read it rather than guessing from
the directory name.

Schema v2 keys the graph on a stable variant id. The v1 ``target_architecture``
string is still written and still read so existing checkpoints and any consumer
already matching on it keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_NAME = "architecture_surgery.json"
ARCHITECTURE_FIELD = "megatron_bridge_architecture"
SCHEMA_VERSION = 2
ARTIFACT_FORMAT = "huggingface_safetensors"

# v1 wrote a single architecture string. Keep both directions of the mapping so
# checkpoints produced before the registry existed still resolve to a variant.
LEGACY_ARCHITECTURE_BY_VARIANT: dict[str, str] = {
    "v0": "original",
    "v1": "parallel_attention_moe",
}
VARIANT_BY_LEGACY_ARCHITECTURE: dict[str, str] = {value: key for key, value in LEGACY_ARCHITECTURE_BY_VARIANT.items()}


def legacy_architecture(variant_id: str) -> str:
    """Return the v1-compatible architecture string for a variant.

    Args:
        variant_id: Stable variant identifier such as ``"v1"``.

    Returns:
        The legacy string, or ``"surgery_<id>"`` for variants added after v1.
    """
    return LEGACY_ARCHITECTURE_BY_VARIANT.get(variant_id, f"surgery_{variant_id}")


def build_manifest(
    *,
    variant_id: str,
    variant_slug: str,
    equation: str,
    params: dict[str, Any],
    source_hf_directory: str,
    model_type: str,
    parameter_tensors: int,
    copy_mode: str,
    file_count: int,
    total_bytes: int,
    adds_parameters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a schema-v2 surgery manifest.

    Args:
        variant_id: Stable variant identifier such as ``"v1"``.
        variant_slug: Short human-readable variant label.
        equation: The rewritten per-layer computation.
        params: Canonical variant parameters, including the layer selector.
        source_hf_directory: Resolved original Hugging Face directory.
        model_type: ``model_type`` from the source ``config.json``.
        parameter_tensors: Number of entries in the safetensors weight map.
        copy_mode: ``"hardlink"`` or ``"copy"``.
        file_count: Number of files cloned into the target directory.
        total_bytes: Total size of the source files.
        adds_parameters: Descriptions of weights the variant introduces that do
            not exist in the source checkpoint. ``None`` for variants that only
            rearrange the graph.

    Returns:
        The manifest dictionary, with an empty ``files`` list to be filled in.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_format": ARTIFACT_FORMAT,
        "variant": {
            "id": variant_id,
            "slug": variant_slug,
            "equation": equation,
            "params": params,
        },
        # v1 compatibility: consumers that matched on this string still work.
        "target_architecture": legacy_architecture(variant_id),
        "model_type": model_type,
        # Every tensor inherited from the source is byte-identical. A variant
        # may still add new tensors; those are listed in adds_parameters.
        "weights_identical_to_source": True,
        "adds_parameters": adds_parameters or [],
        "parameter_tensors": parameter_tensors,
        "copy_mode": copy_mode,
        "megatron_random_initialization": "disabled_by_hf_import",
        "source_hf_directory": source_hf_directory,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "files": [],
    }


def read_manifest(hf_dir: Path) -> dict[str, Any]:
    """Read and validate a surgery manifest from a Hugging Face directory.

    Accepts both schema v1 and v2 and always returns a v2-shaped dictionary.

    Args:
        hf_dir: Directory containing ``architecture_surgery.json``.

    Returns:
        A manifest with a populated ``variant`` block.

    Raises:
        ValueError: If the manifest is missing, unreadable, or unrecognized.
    """
    path = hf_dir / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"Hugging Face directory is missing its surgery manifest: {path}")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid surgery manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Surgery manifest must be a JSON object: {path}")

    version = manifest.get("schema_version")
    if manifest.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported surgery artifact format in {path}")
    if version == SCHEMA_VERSION:
        variant = manifest.get("variant")
        if not isinstance(variant, dict) or not isinstance(variant.get("id"), str):
            raise ValueError(f"Surgery manifest {path} has no variant identifier")
        return manifest
    if version == 1:
        architecture = manifest.get("target_architecture")
        variant_id = VARIANT_BY_LEGACY_ARCHITECTURE.get(str(architecture))
        if variant_id is None:
            raise ValueError(f"Surgery manifest {path} has unrecognized architecture {architecture!r}")
        upgraded = dict(manifest)
        # v1 only ever rewrote every layer, so the selector is unambiguous.
        upgraded["variant"] = {
            "id": variant_id,
            "slug": "parallel" if variant_id == "v1" else "original",
            "params": {"layers": {"kind": "all" if variant_id == "v1" else "none"}},
        }
        return upgraded
    raise ValueError(f"Unsupported surgery manifest schema_version {version!r} in {path}")


def manifest_variant(manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract the variant id and parameters from a normalized manifest.

    Args:
        manifest: Manifest returned by :func:`read_manifest`.

    Returns:
        The variant identifier and its parameter dictionary.

    Raises:
        ValueError: If the variant block is malformed.
    """
    variant = manifest.get("variant")
    if not isinstance(variant, dict):
        raise ValueError("Surgery manifest has no variant block")
    variant_id = variant.get("id")
    params = variant.get("params", {})
    if not isinstance(variant_id, str) or not isinstance(params, dict):
        raise ValueError("Surgery manifest variant block is malformed")
    return variant_id, params
