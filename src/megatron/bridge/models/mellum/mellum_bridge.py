# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversion of the original Mellum2 architecture and published checkpoints."""

import math

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mellum.modeling_mellum.hf_reference import register_hf_fallback
from megatron.bridge.models.mellum.modeling_mellum.layer_specs import mellum_layer_spec
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge


register_hf_fallback()
AutoMapping.register_module_type("MellumTopKRouter", "replicated")


@MegatronModelBridge.register_bridge(source="MellumForCausalLM", target=GPTModel, model_type="mellum")
class MellumBridge(MegatronModelBridge):
    """HF ↔ Megatron bridge for serial Mellum2 models with hybrid attention."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Map architecture fields and configure per-layer RoPE and MoE."""
        hf = hf_pretrained.config
        if hf.hidden_act != "silu" or not hf.norm_topk_prob:
            raise ValueError("Mellum currently requires SiLU and normalized top-k routing")
        if hf.attention_bias or hf.tie_word_embeddings:
            raise ValueError("Mellum currently requires bias-free attention and untied embeddings")
        if len(hf.layer_types) != hf.num_hidden_layers or set(hf.layer_types) - {
            "sliding_attention",
            "full_attention",
        }:
            raise ValueError("Invalid Mellum attention layer_types")
        if hf.mlp_layer_types != ["sparse"] * hf.num_hidden_layers:
            raise ValueError("This bring-up supports the all-MoE Mellum2 checkpoint")
        local = hf.rope_parameters["sliding_attention"]
        full = hf.rope_parameters["full_attention"]
        if local["rope_type"] != "default" or full["rope_type"] != "yarn":
            raise ValueError("Expected default sliding RoPE and YaRN full-attention RoPE")
        if local["rope_theta"] != full["rope_theta"]:
            raise ValueError("Different local/global RoPE bases are not supported yet")
        if local.get("partial_rotary_factor", 1.0) != 1.0 or full.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("Partial rotary dimensions require a separate Mellum variant")

        provider = super().provider_bridge(hf_pretrained)
        provider.normalization = "RMSNorm"
        provider.layernorm_zero_centered_gamma = False
        provider.gated_linear_unit = True
        provider.activation_func = torch.nn.functional.silu
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True
        provider.kv_channels = hf.head_dim
        provider.position_embedding_type = "rope"
        provider.rotary_base = full["rope_theta"]
        provider.rotary_percent = 1.0
        provider.window_size = (hf.sliding_window - 1, 0)
        provider.window_attn_skip_freq = [int(kind == "sliding_attention") for kind in hf.layer_types]
        provider.transformer_layer_spec = mellum_layer_spec
        provider.yarn_rotary_scaling_factor = float(full["factor"])
        provider.yarn_original_max_position_embeddings = full["original_max_position_embeddings"]
        provider.yarn_beta_fast = full.get("beta_fast", 32.0)
        provider.yarn_beta_slow = full.get("beta_slow", 1.0)
        provider.yarn_correction_range_round_to_int = full.get("truncate", True)
        factor = provider.yarn_rotary_scaling_factor
        concentration = full.get("attention_factor", 1 + 0.1 * math.log(factor))
        if factor <= 1:
            raise ValueError("Mellum YaRN currently requires factor > 1")
        provider.yarn_mscale = (concentration - 1) / (0.1 * math.log(factor))
        provider.yarn_mscale_all_dim = 0.0
        # Fused RoPE paths may omit the concentration factor; keep the baseline explicit.
        provider.apply_rope_fusion = False
        provider.fused_single_qkv_rope = False
        provider.flash_decode = False
        provider.moe_grouped_gemm = True
        provider.moe_ffn_hidden_size = hf.moe_intermediate_size
        provider.moe_router_pre_softmax = True  # HF selects on FP32 softmax, then normalizes top-k.
        provider.moe_router_score_function = "softmax"
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = hf.router_aux_loss_coef
        # Match HF's BF16 gate projection; routing softmax itself uses FP32.
        provider.moe_router_dtype = None
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = False
        provider.moe_router_fusion = False
        provider.autocast_dtype = torch.bfloat16
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Reuse per-expert mappings, plus surgery variant v4's correction branch.

        The correction mappings are always registered. Conversion tasks are built
        by walking Megatron's parameters, so for a model without the branch these
        patterns simply never match, and importing a checkpoint that lacks the
        Hugging Face side logs a warning and skips rather than failing. That
        keeps one registry correct for every variant.
        """
        base = Qwen3MoEBridge.mapping_registry(self)
        return MegatronMappingRegistry(
            *base.mappings,
            GatedMLPMapping(
                megatron_param="decoder.layers.*.compensatory_mlp.linear_fc1.weight",
                gate="model.layers.*.compensatory_mlp.gate_proj.weight",
                up="model.layers.*.compensatory_mlp.up_proj.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.compensatory_mlp.linear_fc2.weight",
                hf_param="model.layers.*.compensatory_mlp.down_proj.weight",
            ),
        )

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Preserve hybrid attention, head dimension, and YaRN on HF export."""
        result = super().megatron_to_hf_config(provider)
        factor = provider.yarn_rotary_scaling_factor
        variant = getattr(provider, "surgery_variant_id", None)
        if variant is not None:
            # Without these an exported checkpoint would load into the original
            # sequential graph and silently produce wrong evaluation numbers.
            result["megatron_bridge_surgery_variant"] = variant
            result["compensatory_mlp_expansion_factor"] = getattr(provider, "surgery_correction_expansion_factor", 1)
        result.update(
            max_position_embeddings=max(
                provider.seq_length, math.ceil(factor * provider.yarn_original_max_position_embeddings)
            ),
            head_dim=provider.kv_channels,
            layer_types=["sliding_attention" if flag else "full_attention" for flag in provider.window_attn_skip_freq],
            mlp_layer_types=["sparse"] * provider.num_layers,
            sliding_window=provider.window_size[0] + 1,
            use_sliding_window=True,
            norm_topk_prob=True,
            router_aux_loss_coef=provider.moe_aux_loss_coeff,
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": provider.rotary_base},
                "full_attention": {
                    "rope_type": "yarn",
                    "rope_theta": provider.rotary_base,
                    "factor": factor,
                    "original_max_position_embeddings": provider.yarn_original_max_position_embeddings,
                    "beta_fast": provider.yarn_beta_fast,
                    "beta_slow": provider.yarn_beta_slow,
                    "truncate": provider.yarn_correction_range_round_to_int,
                    "attention_factor": 1 + 0.1 * provider.yarn_mscale * math.log(factor),
                },
            },
        )
        result.pop("rope_scaling", None)
        return result
