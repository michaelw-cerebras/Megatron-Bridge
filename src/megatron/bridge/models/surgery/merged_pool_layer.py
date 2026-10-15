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

"""Variant v3_1: parallel decoders whose expert pools are merged into one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

from megatron.bridge.models.surgery.merged_moe import (
    apply_joint_aux_loss,
    compute_merged_routing,
    log_pool_balance,
    run_moe_with_routing,
)
from megatron.bridge.models.surgery.sublayers import DecoderSublayers, check_layer_support, run_attention


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext


VARIANT_ID = "v3_1"
PLAN_ATTRIBUTE = "_surgery_merged_pool_plan"


@dataclass(frozen=True)
class MergedPoolPlan:
    """This layer's role inside a merged-pool decoder group.

    Attributes:
        group: Sublayers of every decoder in the group, in original order, when
            this layer joins the group. ``None`` when this layer is a
            pass-through whose sublayers are executed by the joining layer.
    """

    group: tuple[DecoderSublayers, ...] | None


class MergedPoolDecoderLayer(TransformerLayer):
    """Run a group's attentions in parallel, then one MoE over the merged pool.

    For a group of decoders ``1..g``::

        z = x + sum_i a_i(N_i^a(x))
        logits = concat_i router_i(N_i^m(z))             # g * E experts
        p      = softmax(logits); top-(g * k); renormalize once
        y      = z + sum_j p_j * expert_j(N_owner(j)^m(z))

    Compared with :class:`~megatron.bridge.models.surgery.adjacent_layer.AdjacentDecoderParallelLayer`,
    the attentions are unchanged but the per-decoder MoEs become a single pool.
    A token still activates ``g * k`` experts, so total parameters and active
    expert FLOPs match, but it may now spend that budget unevenly across pools.

    Every router, norm and expert keeps its own weights and its own place in the
    module tree, so the state dict is identical to the original model.
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
    def merged_pool_plan(self) -> MergedPoolPlan:
        """Return the wiring installed by the surgery pre-wrap hook.

        Returns:
            The layer's group plan.

        Raises:
            RuntimeError: If the model was built without running the hook.
        """
        plan = self.__dict__.get(PLAN_ATTRIBUTE)
        if plan is None:
            raise RuntimeError(
                "Model surgery variant v3_1 layer is unwired. Build the model through "
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
        """Pass through, or run the group's attentions and its merged MoE.

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
            padding_mask: Unsupported here; joint routing has no padding path yet.
            inference_params: Deprecated alias for ``inference_context``.

        Returns:
            The group output (or the untouched input) and the unchanged context.

        Raises:
            ValueError: If a cross-attention mask, a padding mask, or fused TP
                inference is used.
        """
        if context_mask is not None:
            raise ValueError("Model surgery variant v3_1 does not support cross-attention context masks")
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            raise ValueError("Model surgery variant v3_1 does not support fused TP inference")

        plan = self.merged_pool_plan
        if plan.group is None:
            # A later member of this group executes this layer's sublayers.
            return hidden_states, context
        if padding_mask is not None:
            # The merged selection would have to mask padded tokens consistently
            # across both pools; rejecting is safer than silently skewing routing.
            raise ValueError("Model surgery variant v3_1 does not support padding masks")

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

        # Every attention reads the block input; their updates join the residual
        # before the merged MoE sees it.
        shared = residual
        for member in plan.group:
            shared = shared + run_attention(
                member,
                hidden_states,
                variant_id=VARIANT_ID,
                attention_kwargs=attention_kwargs,
            )

        moe_layers = [member.mlp for member in plan.group]
        normed = [member.mlp_norm(shared) for member in plan.group]
        routing = compute_merged_routing(moe_layers, normed, variant_id=VARIANT_ID)
        if self.training and torch.is_grad_enabled():
            routing = apply_joint_aux_loss(routing, moe_layers)
            log_pool_balance(routing, moe_layers)

        output = shared
        for index, moe_layer in enumerate(moe_layers):
            output = output + run_moe_with_routing(
                moe_layer,
                normed[index],
                routing.probs[index],
                routing.routing_map[index],
            )

        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
