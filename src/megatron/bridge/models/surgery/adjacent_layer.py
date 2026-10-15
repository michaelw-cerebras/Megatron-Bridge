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

"""Variant v3: adjacent whole decoders applied in parallel to a shared input."""

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
)


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext


VARIANT_ID = "v3"
PLAN_ATTRIBUTE = "_surgery_adjacent_plan"


@dataclass(frozen=True)
class AdjacentPlan:
    """This layer's role inside a parallel decoder group.

    Attributes:
        group: Sublayers of every decoder in the group, in original order, when
            this layer joins the group's outputs. ``None`` when this layer is a
            pass-through whose sublayers are executed by the joining layer.
    """

    group: tuple[DecoderSublayers, ...] | None


class AdjacentDecoderParallelLayer(TransformerLayer):
    """Apply a group of complete decoders to the same input and sum their deltas.

    For a group of decoders ``B_1..B_g``, each with
    ``delta_i(u) = a_i(u) + m_i(u + a_i(u))``::

        y = x + sum_i delta_i(x)
          = B_1(x) + ... + B_g(x) - (g - 1) * x

    Attention and MLP stay sequential *within* each decoder; only the decoders
    themselves become parallel. Earlier members of a group pass their input
    through untouched and the final member executes every member's sublayers, so
    each module still runs exactly once and the state dict is unchanged. A group
    of size one reduces to the original sequential decoder.

    The cross-layer references are installed after construction by
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
    def adjacent_plan(self) -> AdjacentPlan:
        """Return the wiring installed by the surgery pre-wrap hook.

        Returns:
            The layer's group plan.

        Raises:
            RuntimeError: If the model was built without running the hook, which
                would otherwise silently drop whole decoders from the graph.
        """
        plan = self.__dict__.get(PLAN_ATTRIBUTE)
        if plan is None:
            raise RuntimeError(
                "Model surgery variant v3 layer is unwired. Build the model through "
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
        """Pass through, or run the whole group and sum its deltas.

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
            The group output (or the untouched input) and the unchanged context.

        Raises:
            ValueError: If a cross-attention mask or fused TP inference is used.
        """
        if context_mask is not None:
            raise ValueError("Model surgery variant v3 does not support cross-attention context masks")
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            raise ValueError("Model surgery variant v3 does not support fused TP inference")

        plan = self.adjacent_plan
        if plan.group is None:
            # A later member of this group executes this layer's sublayers.
            return hidden_states, context

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

        output = residual
        for member in plan.group:
            attention_output = run_attention(
                member,
                hidden_states,
                variant_id=VARIANT_ID,
                attention_kwargs=attention_kwargs,
            )
            mlp_output = run_mlp(
                member,
                hidden_states + attention_output,
                variant_id=VARIANT_ID,
                padding_mask=padding_mask,
            )
            output = output + attention_output + mlp_output

        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
