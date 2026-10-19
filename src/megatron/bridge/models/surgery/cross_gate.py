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

"""Gated cross-state injection between two parallel decoders (variant v3_2/v3_3).

In the adjacent-decoder-parallel layout each branch's MLP only sees its own
attention. These gates let a branch's MLP also see the *other* branch's
attention output, by a learnable amount::

    g(x, h) = gate applied to the other branch's attention output (h - x)

Both forms start at exactly zero, so training begins from the unmodified v3
graph and the model has to earn any mixing it does.

Two directions exist and they are not equivalent. Pairing decoder ``i`` with
``i+1``, injecting ``a_i`` into ``m_{i+1}`` **restores** information the
original sequential model had and parallelization destroyed. Injecting
``a_{i+1}`` into ``m_i`` creates a path the original model never had.
"""

from __future__ import annotations

import math
from typing import Any

import torch


GATE_ATTRIBUTE = "cross_gate"
GATE_KINDS = ("scalar", "nonlinear")
DIRECTIONS = ("both", "forward", "backward")


class ScalarCrossGate(torch.nn.Module):
    """``g(x, h) = alpha * (h - x)`` with one learnable scalar.

    The cheapest possible mixing: one parameter per branch. Note that with the
    injection applied before the MLP's norm, ``alpha`` is not directly readable
    as a mixing strength, because the norm rescales the sum.
    """

    def __init__(self) -> None:
        """Create the gate with ``alpha = 0``, so it starts as a no-op."""
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(()))

    def forward(self, block_input: torch.Tensor, other_attention: torch.Tensor) -> torch.Tensor:
        """Scale the other branch's attention output.

        Args:
            block_input: The block input ``x``; unused by this form.
            other_attention: The other branch's attention output ``h - x``.

        Returns:
            The injection term to add to this branch's MLP input.
        """
        del block_input
        return self.alpha * other_attention


class NonlinearCrossGate(torch.nn.Module):
    """``g(x, h) = W[V(h - x) * SiLU(Ux)]``.

    The other branch's attention output is projected, then modulated
    elementwise by a SiLU-gated projection of the block input, so how much of
    the other branch leaks in depends on the token itself.

    ``U`` and ``V`` are column-parallel and ``W`` is row-parallel, so the
    elementwise product happens in the sharded inner dimension and tensor
    parallelism works exactly as it does for an MLP.
    """

    def __init__(self, config: Any, inner_size: int) -> None:
        """Build the gate with ``W`` zeroed, so it starts as a no-op.

        Args:
            config: The layer's transformer config.
            inner_size: Inner dimension ``d`` of the gate.

        Raises:
            ValueError: If the Transformer Engine linear layers are unavailable.
        """
        super().__init__()
        try:
            from megatron.core.extensions.transformer_engine import (
                TEColumnParallelLinear,
                TERowParallelLinear,
            )
        except ImportError as error:  # pragma: no cover - depends on the container
            raise ValueError(f"Model surgery variant v3_2/v3_3 requires Transformer Engine linear layers: {error}") from None

        common = {
            "config": config,
            "init_method": config.init_method,
            "bias": False,
            "skip_bias_add": True,
            "is_expert": False,
        }
        self.v_proj = TEColumnParallelLinear(
            config.hidden_size, inner_size, gather_output=False, skip_weight_param_allocation=False, **common
        )
        self.u_proj = TEColumnParallelLinear(
            config.hidden_size, inner_size, gather_output=False, skip_weight_param_allocation=False, **common
        )
        self.w_proj = TERowParallelLinear(
            inner_size, config.hidden_size, input_is_parallel=True, **{**common, "init_method": config.init_method}
        )

    def forward(self, block_input: torch.Tensor, other_attention: torch.Tensor) -> torch.Tensor:
        """Modulate the other branch's attention output by the block input.

        Args:
            block_input: The block input ``x``.
            other_attention: The other branch's attention output ``h - x``.

        Returns:
            The injection term to add to this branch's MLP input.

        Raises:
            ValueError: If a projection returns a bias this form cannot absorb.
        """
        projected, projected_bias = self.v_proj(other_attention)
        modulator, modulator_bias = self.u_proj(block_input)
        if projected_bias is not None or modulator_bias is not None:
            raise ValueError("Model surgery variant v3_2/v3_3 requires bias-free cross-gate projections")
        output, output_bias = self.w_proj(projected * torch.nn.functional.silu(modulator))
        if output_bias is not None:
            raise ValueError("Model surgery variant v3_2/v3_3 requires a bias-free cross-gate output")
        return output


def gate_inner_size(config: Any, ratio: float) -> int:
    """Return the gate's inner dimension for a hidden-size ratio.

    Args:
        config: The layer's transformer config.
        ratio: Fraction of ``hidden_size`` to use.

    Returns:
        The inner dimension, at least 1.

    Raises:
        ValueError: If the ratio is not positive.
    """
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio <= 0:
        raise ValueError(f"Model surgery variant v3_2/v3_3 needs a positive gate_dim_ratio, got {ratio!r}")
    return max(1, int(config.hidden_size * ratio))


def build_cross_gate(config: Any, *, kind: str, gate_dim_ratio: float) -> torch.nn.Module:
    """Build one branch's cross-injection gate.

    Args:
        config: The layer's transformer config.
        kind: ``"scalar"`` or ``"nonlinear"``.
        gate_dim_ratio: Inner dimension as a fraction of hidden size; unused
            by the scalar form.

    Returns:
        The gate module.

    Raises:
        ValueError: If the kind is unknown.
    """
    if kind == "scalar":
        return ScalarCrossGate()
    if kind == "nonlinear":
        return NonlinearCrossGate(config, gate_inner_size(config, gate_dim_ratio))
    raise ValueError(f"Unknown v3_2/v3_3 cross-gate kind {kind!r}; expected one of {list(GATE_KINDS)}")


def initialize_cross_gate(gate: torch.nn.Module, config: Any, *, seed: int, layer_number: int) -> None:
    """Initialize a gate so that it starts as an exact no-op.

    Megatron runs with ``perform_initialization=False`` so the Hugging Face
    import owns every original weight; nothing would otherwise write to these
    tensors. The output projection is zeroed in both forms, so v3_2/v3_3 begins from
    the unmodified v3 graph; the other projections are drawn from an explicitly
    seeded generator so a rerun reproduces the same gate.

    Args:
        gate: The gate to initialize, in place.
        config: The layer's transformer config, for the init standard deviation.
        seed: Base seed, combined with the layer number.
        layer_number: One-based global layer number.
    """
    if isinstance(gate, ScalarCrossGate):
        with torch.no_grad():
            gate.alpha.zero_()
        return

    generator = torch.Generator(device="cpu").manual_seed(seed + 2_000_003 * layer_number)
    std = config.init_method_std / math.sqrt(2.0 * config.num_layers)
    for name, parameter in gate.named_parameters():
        if "w_proj" in name.split("."):
            sample = torch.zeros(parameter.shape, dtype=torch.float32)
        else:
            sample = torch.normal(0.0, std, size=tuple(parameter.shape), generator=generator)
        with torch.no_grad():
            parameter.copy_(sample.to(device=parameter.device, dtype=parameter.dtype))


def gate_is_active(direction: str, *, position: int, group_size: int) -> bool:
    """Decide whether the branch at ``position`` receives an injection.

    Pairing decoder ``i`` (position 0) with ``i+1`` (position 1):

    - ``forward`` injects ``a_i`` into ``m_{i+1}``, restoring what the original
      sequential model had.
    - ``backward`` injects ``a_{i+1}`` into ``m_i``, a path the original never had.

    Args:
        direction: ``"both"``, ``"forward"`` or ``"backward"``.
        position: Zero-based position of this branch in its group.
        group_size: Number of branches in the group.

    Returns:
        Whether this branch's gate contributes.

    Raises:
        ValueError: If the direction is unknown.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown v3_2/v3_3 direction {direction!r}; expected one of {list(DIRECTIONS)}")
    if direction == "both":
        return True
    if direction == "forward":
        return position == group_size - 1
    return position == 0
