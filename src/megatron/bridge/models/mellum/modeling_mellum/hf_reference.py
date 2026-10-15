# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF compatibility reference for images without native Mellum support.

Matches the published per-expert checkpoint layout and the computations in
Transformers' modeling_mellum.py. This deliberately uses an unfused, uncached
reference implementation for conversion/parity; training uses Megatron-Core.
"""

import copy

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention, Qwen3MoeRMSNorm


# Graphs this reference can reproduce. Cross-layer variants (v2, v3, v3_1)
# rearrange across layer boundaries and are NOT expressible in this per-layer
# class; loading one must fail rather than quietly run the sequential graph and
# report wrong evaluation numbers.
SUPPORTED_SURGERY_VARIANTS = frozenset({"v0", "v1", "v4"})
_LEGACY_ARCHITECTURE_TO_VARIANT = {"original": "v0", "parallel_attention_moe": "v1"}


def resolve_surgery_variant(config: "PretrainedConfig") -> str:
    """Return the surgery variant a checkpoint's config asks for.

    Args:
        config: A loaded Mellum configuration.

    Returns:
        The variant id; ``"v0"`` when the config carries no surgery marker.

    Raises:
        ValueError: If the config names a variant this reference cannot run.
    """
    variant = getattr(config, "megatron_bridge_surgery_variant", None)
    if variant is None:
        legacy = getattr(config, "megatron_bridge_architecture", None)
        variant = _LEGACY_ARCHITECTURE_TO_VARIANT.get(str(legacy), "v0") if legacy else "v0"
    if variant not in SUPPORTED_SURGERY_VARIANTS:
        raise ValueError(
            f"This Mellum reference implements {sorted(SUPPORTED_SURGERY_VARIANTS)} but the checkpoint "
            f"requests {variant!r}. Running it here would silently evaluate a different computation graph."
        )
    return variant


class MellumConfig(PretrainedConfig):
    """Compatibility configuration retaining the published Mellum fields."""

    model_type = "mellum"

    def __init__(self, **kwargs: object) -> None:
        # Base PretrainedConfig validates a single RoPE dictionary before it has
        # retained custom layer_types. Preserve Mellum's nested dictionaries.
        rope_parameters = kwargs.pop("rope_parameters", None)
        # Default explicitly rather than relying on getattr at build time, so the
        # value is serialized and an exported checkpoint fully describes its own
        # graph to whoever loads it next.
        expansion = kwargs.pop("compensatory_mlp_expansion_factor", 1)
        super().__init__(**kwargs)
        self.rope_parameters = rope_parameters
        self.compensatory_mlp_expansion_factor = expansion

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None) -> None:
        # Transformers 5.8.1's inherited validator reads original context from
        # self.rope_parameters even when validating one nested layer type.
        original = self.rope_parameters
        self.rope_parameters = rope_parameters
        try:
            super()._validate_yarn_rope_parameters(rope_parameters, ignore_keys=ignore_keys)
        finally:
            self.rope_parameters = original


class _MLP(nn.Module):
    def __init__(self, config: MellumConfig, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _MoE(nn.Module):
    def __init__(self, config: MellumConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([_MLP(config, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.topk = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        hidden = hidden.reshape(-1, shape[-1])
        logits = self.gate(hidden)
        probs, indices = torch.softmax(logits.float(), dim=-1).topk(self.topk, dim=-1)
        if self.normalize:
            probs = probs / probs.sum(dim=-1, keepdim=True)
        probs = probs.to(logits.dtype)
        output = torch.zeros_like(hidden)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == index)
            output.index_add_(0, rows, expert(hidden[rows]) * probs[rows, slots, None])
        return output.reshape(shape)


class _Layer(nn.Module):
    def __init__(self, config: MellumConfig, index: int) -> None:
        super().__init__()
        attention_config = copy.copy(config)
        attention_config.sliding_window = (
            config.sliding_window if config.layer_types[index] == "sliding_attention" else None
        )
        self.self_attn = Qwen3MoeAttention(attention_config, index)
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = (
            _MoE(config) if config.mlp_layer_types[index] == "sparse" else _MLP(config, config.intermediate_size)
        )
        self.surgery_variant = resolve_surgery_variant(config)
        if self.surgery_variant == "v4":
            factor = config.compensatory_mlp_expansion_factor
            self.compensatory_mlp = _MLP(config, config.hidden_size * factor)

    def forward(
        self, hidden: torch.Tensor, *, mask: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        attention, _ = self.self_attn(self.input_layernorm(hidden), rotary, mask)
        if self.surgery_variant == "v0":
            hidden = hidden + attention
            return hidden + self.mlp(self.post_attention_layernorm(hidden))
        # v1 and v4 share the residual: the MLP reads the block input, not the
        # post-attention state.
        output = hidden + attention + self.mlp(self.post_attention_layernorm(hidden))
        if self.surgery_variant == "v4":
            # Third branch, no norm, reading the same residual as the other two.
            output = output + self.compensatory_mlp(hidden)
        return output


class _Model(nn.Module):
    def __init__(self, config: MellumConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_Layer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MellumForCausalLM(PreTrainedModel, GenerationMixin):
    """Uncached HF reference with the original Mellum residual structure.

    Use ``use_cache=False`` for generation. The implementation exists only for
    installations lacking native Mellum; it preserves checkpoint key names.
    """

    config_class = MellumConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _no_split_modules = ["_Layer"]

    def __init__(self, config: MellumConfig) -> None:
        super().__init__(config)
        self.model = _Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.generation_config.use_cache = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding table."""
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """Return the language-model projection."""
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: object) -> dict:
        """Recompute the full prefix for reference generation."""
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask"), "use_cache": False}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: object = None,
        use_cache: bool = False,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        """Compute logits using per-layer masks and HF's RoPE definitions."""
        if past_key_values is not None or use_cache:
            raise ValueError("The compatibility reference requires use_cache=False and no past_key_values")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        if position_ids is None:
            position_ids = positions[None, :]
        rotary = {}
        masks = {}
        for kind in set(self.config.layer_types):
            params = self.config.rope_parameters[kind]
            if params["rope_type"] == "default":
                inv_freq = 1.0 / (
                    params["rope_theta"]
                    ** (torch.arange(0, self.config.head_dim, 2, device=hidden.device).float() / self.config.head_dim)
                )
                scale = 1.0
            else:
                inv_freq, scale = ROPE_INIT_FUNCTIONS[params["rope_type"]](self.config, hidden.device, layer_type=kind)
            frequencies = position_ids.float()[..., None] * inv_freq[None, None, :]
            angles = torch.cat((frequencies, frequencies), dim=-1)
            rotary[kind] = ((angles.cos() * scale).to(hidden.dtype), (angles.sin() * scale).to(hidden.dtype))
            allowed = positions[None, :] <= positions[:, None]
            if kind == "sliding_attention":
                allowed &= positions[None, :] > positions[:, None] - self.config.sliding_window
            mask = torch.zeros((hidden.shape[0], 1, length, length), device=hidden.device, dtype=hidden.dtype)
            mask.masked_fill_(~allowed[None, None, :, :], torch.finfo(hidden.dtype).min)
            if attention_mask is not None:
                mask.masked_fill_(~attention_mask[:, None, None, :].bool(), torch.finfo(hidden.dtype).min)
            masks[kind] = mask
        for index, layer in enumerate(self.model.layers):
            kind = self.config.layer_types[index]
            hidden = layer(hidden, mask=masks[kind], rotary=rotary[kind])
        logits = self.lm_head(self.model.norm(hidden))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1)
            )
        return CausalLMOutputWithPast(logits=logits, loss=loss)


def register_hf_fallback() -> None:
    """Register the reference only when Transformers lacks Mellum support."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "mellum" not in CONFIG_MAPPING:
        AutoConfig.register("mellum", MellumConfig)
        AutoModelForCausalLM.register(MellumConfig, MellumForCausalLM)
