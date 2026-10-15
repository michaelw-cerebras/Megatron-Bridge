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

"""The small trainable correction branch added by variant v4.

Parallelizing a block drops the term ``m(x + a(x)) - m(x)``: the MLP no longer
sees attention's update. ``c_theta`` is a small MLP trained to put that term
back, while every original weight stays frozen.

It is built from Megatron's own ``MLP`` so tensor parallelism, the gated
activation, and the Hugging Face weight layout all match the model's existing
MLPs. That keeps export to ``gate_proj`` / ``up_proj`` / ``down_proj`` on the
standard mapping path instead of needing a bespoke one.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from megatron.core.transformer.mlp import MLP, MLPSubmodules


CORRECTION_ATTRIBUTE = "compensatory_mlp"
INIT_SCHEMES = ("random", "zero_output")


def build_compensatory_mlp(config: Any, *, expansion_factor: int = 1) -> MLP:
    """Build the correction branch for one layer.

    Args:
        config: The layer's transformer config.
        expansion_factor: Hidden multiplier for the correction MLP. ``1`` keeps
            it small, which is the point of the variant.

    Returns:
        An ``MLP`` shaped like the model's own MLPs but ``expansion_factor``
        times the hidden size wide.

    Raises:
        ValueError: If the Transformer Engine linear layers are unavailable.
    """
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )
    except ImportError as error:  # pragma: no cover - depends on the container
        raise ValueError(f"Model surgery variant v4 requires Transformer Engine linear layers: {error}") from None

    correction_config = copy.copy(config)
    correction_config.ffn_hidden_size = config.hidden_size * expansion_factor
    # The branch is dense even in an MoE model: it corrects the residual, not
    # the routing. Clearing the MoE fields keeps MLP from building experts.
    correction_config.num_moe_experts = None
    correction_config.moe_ffn_hidden_size = None

    return MLP(
        correction_config,
        MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )


def initialize_compensatory_mlp(
    correction: MLP,
    config: Any,
    *,
    scheme: str,
    seed: int,
    layer_number: int,
) -> None:
    """Initialize the correction branch explicitly and reproducibly.

    Megatron is told to skip initialization so the Hugging Face import can own
    every original weight. The correction branch has no Hugging Face
    counterpart on the first launch, so nothing would otherwise write to it and
    it would train from uninitialized memory.

    Args:
        correction: The branch to initialize, in place.
        config: The layer's transformer config, for the init standard deviation.
        scheme: ``"random"`` for the normal initialization used elsewhere in the
            model, or ``"zero_output"`` to zero the output projection so the
            branch contributes nothing at step 0.
        seed: Base seed; combined with ``layer_number`` so layers differ but a
            rerun reproduces the same weights.
        layer_number: One-based layer number, used to decorrelate layers.

    Raises:
        ValueError: If the scheme is unknown.
    """
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"Unknown v4 correction init scheme {scheme!r}; expected one of {list(INIT_SCHEMES)}")

    # Draw from an explicitly seeded generator rather than Megatron's
    # init_method_* helpers: those use the global RNG, so the result would
    # depend on whatever else consumed randomness first and a rerun would not
    # reproduce the same branch.
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003 * layer_number)
    input_std = config.init_method_std
    # Match how Megatron scales a residual-output projection, so the branch's
    # initial contribution has the same magnitude as any other sublayer's.
    output_std = config.init_method_std / math.sqrt(2.0 * config.num_layers)

    for name, parameter in correction.named_parameters():
        is_output_projection = "linear_fc2" in name.split(".")
        if is_output_projection and scheme == "zero_output":
            sample = torch.zeros(parameter.shape, dtype=torch.float32)
        else:
            sample = torch.normal(
                0.0,
                output_std if is_output_projection else input_std,
                size=tuple(parameter.shape),
                generator=generator,
            )
        with torch.no_grad():
            parameter.copy_(sample.to(device=parameter.device, dtype=parameter.dtype))


def freeze_except_compensatory(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze every parameter except the correction branches.

    Args:
        model: The raw ``GPTModel`` before mixed-precision or DDP wrapping.

    Returns:
        The number of trainable and frozen parameters, for logging.
    """
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        # Match a path component, not a substring: the correction branch may sit
        # at the root of whatever module is being walked, so a ".name." test
        # would miss it and silently freeze the only trainable weights.
        if CORRECTION_ATTRIBUTE in name.split("."):
            parameter.requires_grad = True
            trainable += parameter.numel()
        else:
            parameter.requires_grad = False
            frozen += parameter.numel()
    return trainable, frozen


def describe_added_parameters(variant: Any, params: Any, hf_config: Any) -> list[dict[str, Any]]:
    """Describe the weights a variant introduces, for the surgery manifest.

    Args:
        variant: The resolved :class:`Variant`.
        params: Its canonical parameters.
        hf_config: The source Hugging Face ``config.json`` contents.

    Returns:
        One entry per added weight pattern, empty for graph-only variants.
    """
    if not getattr(variant, "adds_parameters", False):
        return []
    hidden = int(hf_config["hidden_size"])
    layers = int(hf_config["num_hidden_layers"])
    width = hidden * int(params.get("expansion_factor", 1))
    return [
        {
            "hf_parameter": f"model.layers.*.{CORRECTION_ATTRIBUTE}.{name}.weight",
            "shape": shape,
            "per_layer_elements": shape[0] * shape[1],
            "layers": layers,
            "initialization": params.get("init", "random"),
            "init_seed": params.get("init_seed"),
            "trainable": True,
        }
        for name, shape in (
            ("gate_proj", [width, hidden]),
            ("up_proj", [width, hidden]),
            ("down_proj", [hidden, width]),
        )
    ]
