# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mellum layer specifications using stock Megatron attention and MoE kernels."""

import copy
import math
from functools import partial

import torch
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.router import TopKRouter

from megatron.bridge.models.gpt_provider import GPTModelProvider, transformer_engine_layer_spec


def yarn_frequency_ratio(config: GPTModelProvider) -> torch.Tensor:
    """Return YaRN/default frequency ratios using HF's correction convention."""
    dim = config.kv_channels
    original = config.yarn_original_max_position_embeddings
    base = config.rotary_base
    low = dim * math.log(original / (config.yarn_beta_fast * 2 * math.pi)) / (2 * math.log(base))
    high = dim * math.log(original / (config.yarn_beta_slow * 2 * math.pi)) / (2 * math.log(base))
    if config.yarn_correction_range_round_to_int:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
    ratio = 1 - ramp + ramp / config.yarn_rotary_scaling_factor
    return torch.cat((ratio, ratio))


class MellumSelfAttention(SelfAttention):
    """Select the layer's RoPE frequencies and attention concentration factor.

    The GPT model supplies ordinary RoPE angles, already partitioned for CP.
    Full-attention layers multiply these by the YaRN frequency ratio. Scaling
    is applied by Megatron's existing rotary implementation using a private
    attention config, leaving sliding-attention layers at concentration 1.
    """

    def __init__(self, config: GPTModelProvider, submodules: object, layer_number: int, **kwargs: object) -> None:
        layer_config = copy.copy(config)
        is_sliding = bool(config.window_attn_skip_freq[layer_number - 1])
        if is_sliding:
            layer_config.yarn_rotary_scaling_factor = 1.0
        super().__init__(config=layer_config, submodules=submodules, layer_number=layer_number, **kwargs)
        ratio = torch.ones(config.kv_channels) if is_sliding else yarn_frequency_ratio(config)
        self.register_buffer("rotary_frequency_ratio", ratio, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        rotary_pos_emb: torch.Tensor | tuple | None = None,
        **kwargs: object,
    ) -> tuple:
        """Transform incoming angles before stock attention applies RoPE."""
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = tuple(
                    angle * self.rotary_frequency_ratio if angle is not None else None for angle in rotary_pos_emb
                )
            else:
                rotary_pos_emb = rotary_pos_emb * self.rotary_frequency_ratio
        return super().forward(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb, **kwargs)


class MellumTopKRouter(TopKRouter):
    """Choose experts in FP32 after the checkpoint's BF16 gate projection.

    BF16 and FP32 top-k kernels can resolve tied scores differently. HF selects
    experts on FP32 probabilities, so do not run top-k directly on BF16 logits.
    Keep Megatron's auxiliary losses and cast final routing weights back to the
    gate output dtype, matching the published model.
    """

    def routing(
        self, logits: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route using stock normalized top-k and auxiliary-loss machinery."""
        probs, routing_map = super().routing(logits.float(), padding_mask=padding_mask)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
        return probs.to(logits.dtype), routing_map


def mellum_layer_spec(config: GPTModelProvider) -> ModuleSpec:
    """Create the original serial attention/MoE block with Mellum RoPE."""
    spec = transformer_engine_layer_spec(config)
    spec.submodules.self_attention.module = MellumSelfAttention
    mlp = spec.submodules.mlp
    moe_submodules = mlp.keywords["submodules"] if isinstance(mlp, partial) else mlp.submodules
    moe_submodules.router = MellumTopKRouter
    return spec
