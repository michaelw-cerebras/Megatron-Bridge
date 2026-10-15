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

"""Variant v1: run attention and the MLP/MoE from the same pre-residual input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

from megatron.bridge.models.surgery.sublayers import check_layer_support, run_attention, run_mlp, sublayers_of


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext


VARIANT_ID = "v1"


class ParallelAttentionMLPLayer(TransformerLayer):
    """Compute ``x + attention(N_a(x)) + mlp(N_m(x))`` instead of chaining the sublayers.

    This changes only the computation graph. The module tree and every
    state-dict key stay identical to the stock ``TransformerLayer``, so a
    checkpoint remains interchangeable with the unmodified model.

    The layer works for both dense MLP and MoE submodules, and for specs where
    either normalization is fused into ``linear_qkv`` / ``linear_fc1`` and the
    corresponding ``*_layernorm`` module is an ``IdentityOp``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the layer and reject execution features this graph has not validated.

        Args:
            *args: Forwarded to ``TransformerLayer``.
            **kwargs: Forwarded to ``TransformerLayer``.
        """
        super().__init__(*args, **kwargs)
        check_layer_support(self, variant_id=VARIANT_ID)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | tuple | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context: "BaseInferenceContext | None" = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        *,
        inference_params: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``x + attention(N_a(x)) + mlp(N_m(x))`` and the unchanged context.

        Args:
            hidden_states: Pre-residual input with shape ``[sequence, batch, hidden]``.
            attention_mask: Self-attention mask.
            context: Unchanged decoder context placeholder.
            context_mask: Must be unused because this variant has no cross-attention.
            rotary_pos_emb: Rotary angles supplied by the model.
            rotary_pos_cos: Optional precomputed rotary cosine.
            rotary_pos_sin: Optional precomputed rotary sine.
            rotary_pos_cos_sin: Optional combined rotary cosine and sine.
            attention_bias: Optional attention score bias.
            inference_context: Inference cache state.
            packed_seq_params: Packed-sequence metadata.
            sequence_len_offset: Inference sequence offset.
            padding_mask: Padding mask used by MoE routing losses.
            inference_params: Deprecated alias for ``inference_context``.

        Returns:
            The parallel residual output and the unchanged context.

        Raises:
            ValueError: If a cross-attention mask or fused TP inference is used.
        """
        if context_mask is not None:
            raise ValueError("Model surgery variant v1 does not support cross-attention context masks")
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            raise ValueError("Model surgery variant v1 does not support fused TP inference")

        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        sublayers = sublayers_of(self)
        attention_output = run_attention(
            sublayers,
            hidden_states,
            variant_id=VARIANT_ID,
            attention_kwargs={
                "attention_mask": attention_mask,
                "inference_context": inference_context,
                "rotary_pos_emb": rotary_pos_emb,
                "rotary_pos_cos": rotary_pos_cos,
                "rotary_pos_sin": rotary_pos_sin,
                "rotary_pos_cos_sin": rotary_pos_cos_sin,
                "attention_bias": attention_bias,
                "packed_seq_params": packed_seq_params,
                "sequence_len_offset": sequence_len_offset,
            },
        )
        mlp_output = run_mlp(sublayers, hidden_states, variant_id=VARIANT_ID, padding_mask=padding_mask)

        output = residual + attention_output + mlp_output
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
