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

"""Variant v1_1: parallel attention/MLP plus a trainable compensatory MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

from megatron.bridge.models.surgery.compensatory_mlp import (
    CORRECTION_ATTRIBUTE,
    build_compensatory_mlp,
)
from megatron.bridge.models.surgery.sublayers import check_layer_support, run_attention, run_mlp, sublayers_of


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext


VARIANT_ID = "v1_1"


class CompensatedParallelLayer(TransformerLayer):
    """Compute ``x + a(N_a(x)) + m(N_m(x)) + c_theta(x)``.

    This is :class:`~megatron.bridge.models.surgery.parallel_layer.ParallelAttentionMLPLayer`
    with a third branch. All three read the same pre-residual input, so the
    correction branch stays co-schedulable with the other two — it never waits
    on attention.

    Unlike every other variant, this one **adds parameters**. The original
    weights keep their keys, shapes and values; the correction branch appears
    alongside them under ``compensatory_mlp``. The branch is the only trainable
    part of the model, frozen into place by the surgery pre-wrap hook.

    The correction branch follows Ivan's implementation: it first applies its
    own RMSNorm, then a hidden-width, non-gated SiLU MLP.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the layer, its correction branch, and reject unsupported features.

        Args:
            *args: Forwarded to ``TransformerLayer``.
            **kwargs: Forwarded to ``TransformerLayer``.
        """
        super().__init__(*args, **kwargs)
        check_layer_support(self, variant_id=VARIANT_ID)
        setattr(self, CORRECTION_ATTRIBUTE, build_compensatory_mlp(self.config))

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
        """Return ``x + a(N_a(x)) + m(N_m(x)) + c_theta(x)`` and the unchanged context.

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
            The compensated residual output and the unchanged context.

        Raises:
            ValueError: If a cross-attention mask, fused TP inference, or a
                bias-producing correction branch is encountered.
        """
        if context_mask is not None:
            raise ValueError("Model surgery variant v1_1 does not support cross-attention context masks")
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            raise ValueError("Model surgery variant v1_1 does not support fused TP inference")

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

        # Third branch: RMSNorm -> up projection -> SiLU -> down projection.
        correction_output, correction_bias = apply_module(getattr(self, CORRECTION_ATTRIBUTE))(hidden_states)
        if correction_bias is not None:
            raise ValueError("Model surgery variant v1_1 requires a bias-free correction branch")

        output = residual + attention_output + mlp_output + correction_output
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
