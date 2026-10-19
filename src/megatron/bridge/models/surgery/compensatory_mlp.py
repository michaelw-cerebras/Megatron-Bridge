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

"""Ivan's trainable compensatory branch used by surgery variant v1_1."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


CORRECTION_ATTRIBUTE = "compensatory_mlp"
INIT_SCHEMES = ("random",)


class CompensatoryMLP(nn.Module):
    """Compute ``down_proj(SiLU(up_proj(RMSNorm(x))))`` with TP linears."""

    def __init__(self, config: Any) -> None:
        """Build the fixed-width, non-gated correction branch.

        Args:
            config: The layer's Megatron transformer config.
        """
        super().__init__()
        try:
            from megatron.core.extensions.transformer_engine import (
                TEColumnParallelLinear,
                TENorm,
                TERowParallelLinear,
            )
        except ImportError as error:  # pragma: no cover - depends on the container
            raise ValueError(f"Model surgery variant v1_1 requires Transformer Engine: {error}") from None

        hidden = config.hidden_size
        self.norm = TENorm(config, hidden, eps=config.layernorm_epsilon)
        self.linear_fc1 = TEColumnParallelLinear(
            hidden,
            hidden,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            name="compensatory_up_proj",
        )
        self.linear_fc2 = TERowParallelLinear(
            hidden,
            hidden,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            name="compensatory_down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Apply the normalized non-gated SiLU MLP."""
        intermediate, input_bias = self.linear_fc1(self.norm(hidden_states))
        if input_bias is not None:
            raise ValueError("Variant v1_1 requires a bias-free compensatory up projection")
        output, output_bias = self.linear_fc2(torch.nn.functional.silu(intermediate))
        if output_bias is not None:
            raise ValueError("Variant v1_1 requires a bias-free compensatory down projection")
        return output, None


def build_compensatory_mlp(config: Any) -> CompensatoryMLP:
    """Build Ivan's hidden-width compensatory MLP for one layer."""
    return CompensatoryMLP(config)


def initialize_compensatory_mlp(
    correction: nn.Module,
    config: Any,
    *,
    scheme: str,
    seed: int,
    layer_number: int,
) -> None:
    """Initialize the branch like an HF module plus RMSNorm.

    Both linear matrices use ``initializer_range`` rather than zeroing or
    residual-depth scaling. An explicit per-layer generator is necessary
    because Bridge disables Megatron initialization while importing HF weights.
    """
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"Unknown v1_1 correction init scheme {scheme!r}; expected one of {list(INIT_SCHEMES)}")

    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003 * layer_number)
    with torch.no_grad():
        for name, parameter in correction.named_parameters():
            if name == "norm.weight":
                parameter.fill_(1.0)
                continue
            sample = torch.normal(
                0.0,
                config.init_method_std,
                size=tuple(parameter.shape),
                generator=generator,
            )
            parameter.copy_(sample.to(device=parameter.device, dtype=parameter.dtype))


def freeze_except_compensatory(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze every parameter except Ivan's newly added branches."""
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        if CORRECTION_ATTRIBUTE in name.split("."):
            parameter.requires_grad = True
            trainable += parameter.numel()
        else:
            parameter.requires_grad = False
            frozen += parameter.numel()
    return trainable, frozen
