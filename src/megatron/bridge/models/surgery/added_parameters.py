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

"""Describe the weights a parameter-adding variant introduces.

This lands in the surgery manifest, which is what downstream consumers read to
learn what a derived checkpoint contains beyond the original tensors. Each
variant must be described explicitly: emitting another variant's tensor names
would be worse than emitting nothing, because the manifest is trusted.
"""

from __future__ import annotations

import math
from typing import Any

from megatron.bridge.models.surgery.compensatory_mlp import CORRECTION_ATTRIBUTE
from megatron.bridge.models.surgery.cross_gate import GATE_ATTRIBUTE


def _entry(name: str, shape: list[int], layers: int, params: Any, initialization: str) -> dict[str, Any]:
    return {
        "hf_parameter": name,
        "shape": shape,
        "per_layer_elements": math.prod(shape) if shape else 1,
        "layers": layers,
        "initialization": initialization,
        "init_seed": params.get("init_seed"),
        "trainable": True,
    }


def describe_added_parameters(variant: Any, params: Any, hf_config: Any) -> list[dict[str, Any]]:
    """Describe the per-layer tensors a variant adds to the original checkpoint.

    Args:
        variant: The resolved :class:`Variant`.
        params: Its canonical parameters.
        hf_config: The source Hugging Face ``config.json`` contents.

    Returns:
        One entry per added weight pattern, empty for graph-only variants.

    Raises:
        ValueError: If a variant declares added parameters but has no
            description here, which would otherwise produce a manifest that
            misreports the checkpoint's contents.
    """
    if not getattr(variant, "adds_parameters", False):
        return []
    hidden = int(hf_config["hidden_size"])
    layers = int(hf_config["num_hidden_layers"])

    if variant.id == "v1_1":
        initialization = str(params.get("init", "random"))
        prefix = f"model.layers.*.{CORRECTION_ATTRIBUTE}"
        return [
            _entry(f"{prefix}.norm.weight", [hidden], layers, params, initialization),
            _entry(f"{prefix}.up_proj.weight", [hidden, hidden], layers, params, initialization),
            _entry(f"{prefix}.down_proj.weight", [hidden, hidden], layers, params, initialization),
        ]

    if variant.id in {"v3_2", "v3_3"}:
        prefix = f"model.layers.*.{GATE_ATTRIBUTE}"
        if params.get("gate") == "scalar":
            # One learnable scalar per branch; zero so v3_2/v3_3 starts at plain v3.
            return [_entry(f"{prefix}.alpha", [], layers, params, "zero")]
        inner = max(1, int(hidden * float(params.get("gate_dim_ratio", 0.25))))
        return [
            _entry(f"{prefix}.v_proj.weight", [inner, hidden], layers, params, "normal"),
            _entry(f"{prefix}.u_proj.weight", [inner, hidden], layers, params, "normal"),
            # Zeroed so the gate contributes nothing before training.
            _entry(f"{prefix}.w_proj.weight", [hidden, inner], layers, params, "zero"),
        ]

    raise ValueError(
        f"Variant {variant.id} declares added parameters but has no manifest description; "
        "add one rather than letting the manifest misreport the checkpoint"
    )
