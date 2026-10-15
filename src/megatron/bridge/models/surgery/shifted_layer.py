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

"""Variant v2: shifted attention/MLP overlap across adjacent layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

from megatron.bridge.models.surgery.sublayers import (
    DecoderSublayers,
    check_layer_support,
    run_attention,
    run_mlp,
    sublayers_of,
)


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext


VARIANT_ID = "v2"
PLAN_ATTRIBUTE = "_surgery_shift_plan"


@dataclass(frozen=True)
class ShiftPlan:
    """This layer's position in a shifted run.

    Attributes:
        previous_mlp: Sublayers of the preceding layer in the run, whose MLP this
            layer executes. ``None`` for the layer that opens the run.
        closes_run: Whether this layer also applies its own MLP, which closes the
            run so that every MLP in the run runs exactly once.
    """

    previous_mlp: DecoderSublayers | None
    closes_run: bool


class ShiftedAttentionMLPLayer(TransformerLayer):
    """Run layer ``i``'s MLP alongside layer ``i+1``'s attention on a shared input.

    For a run of layers ``s..e``::

        u_s     = x     + a_s(N_s^a(x))
        u_i     = u_i-1 + m_i-1(N_i-1^m(u_i-1)) + a_i(N_i^a(u_i-1))   for s < i <= e
        output  = u_e   + m_e(N_e^m(u_e))

    Every attention and every MLP in the run executes exactly once, so no weight
    is duplicated or dropped and the state dict is unchanged. A run of length one
    reduces to the original sequential layer.

    The cross-layer reference is installed after construction by
    ``megatron.bridge.models.surgery.wiring``; an unwired layer refuses to run.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the layer and reject unsupported execution features.

        Args:
            *args: Forwarded to ``TransformerLayer``.
            **kwargs: Forwarded to ``TransformerLayer``.
        """
        super().__init__(*args, **kwargs)
        check_layer_support(self, variant_id=VARIANT_ID)

    @property
    def shift_plan(self) -> ShiftPlan:
        """Return the wiring installed by the surgery pre-wrap hook.

        Returns:
            The layer's shifted-run plan.

        Raises:
            RuntimeError: If the model was built without running the hook, which
                would otherwise silently drop or double-count sublayers.
        """
        plan = self.__dict__.get(PLAN_ATTRIBUTE)
        if plan is None:
            raise RuntimeError(
                "Model surgery variant v2 layer is unwired. Build the model through "
                "megatron.bridge.models.surgery.apply_surgery so the pre-wrap hook runs."
            )
        return plan

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
        """Apply this layer's attention together with the previous layer's MLP.

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
            The shifted residual output and the unchanged context.

        Raises:
            ValueError: If a cross-attention mask or fused TP inference is used.
        """
        if context_mask is not None:
            raise ValueError("Model surgery variant v2 does not support cross-attention context masks")
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            raise ValueError("Model surgery variant v2 does not support fused TP inference")

        plan = self.shift_plan
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        attention_kwargs = {
            "attention_mask": attention_mask,
            "inference_context": inference_context,
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "rotary_pos_cos_sin": rotary_pos_cos_sin,
            "attention_bias": attention_bias,
            "packed_seq_params": packed_seq_params,
            "sequence_len_offset": sequence_len_offset,
        }

        update = run_attention(
            sublayers_of(self),
            hidden_states,
            variant_id=VARIANT_ID,
            attention_kwargs=attention_kwargs,
        )
        if plan.previous_mlp is not None:
            update = update + run_mlp(
                plan.previous_mlp,
                hidden_states,
                variant_id=VARIANT_ID,
                padding_mask=padding_mask,
            )
        output = residual + update

        if plan.closes_run:
            output = output + run_mlp(
                sublayers_of(self),
                output,
                variant_id=VARIANT_ID,
                padding_mask=padding_mask,
            )

        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
