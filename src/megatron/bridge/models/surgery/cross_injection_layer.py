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

"""Variant v3_2/v3_3: parallel decoders with gated cross-state injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

from megatron.bridge.models.surgery.cross_gate import GATE_ATTRIBUTE, build_cross_gate
from megatron.bridge.models.surgery.sublayers import DecoderSublayers, check_layer_support, run_attention


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext


VARIANT_ID = "v3_2/v3_3"
PLAN_ATTRIBUTE = "_surgery_cross_injection_plan"
BRANCHES = 2


@dataclass(frozen=True)
class CrossInjectionPlan:
    """This layer's role inside a cross-injected decoder pair.

    Attributes:
        group: Sublayers of both decoders, in original order, when this layer
            joins the pair. ``None`` for the pass-through member.
        gates: Each branch's gate, aligned with ``group``. ``None`` entries are
            branches the configured direction disables.
    """

    group: tuple[DecoderSublayers, ...] | None
    gates: tuple[torch.nn.Module | None, ...] = ()


class CrossInjectedParallelLayer(TransformerLayer):
    """Two parallel decoders whose MLPs each see a gated share of the other's attention.

    For a pair with ``h_i = x + a_i(N^a_i(x))``::

        g_1 = gate_1(x, a_2)
        g_2 = gate_2(x, a_1)
        o   = h_1 + m_1(N^m_1(h_1 + g_1)) + h_2 + m_2(N^m_2(h_2 + g_2)) - x

    The injection enters **before** each MLP's own norm, so the norm rescales
    the sum and large gate values stay bounded.

    Both gates start at exactly zero, so the model begins at the plain
    adjacent-parallel graph and must learn any mixing. Exactly two branches are
    supported: the formulation is pairwise and there is no defined way to
    combine three or more sources.

    Cross-layer references and gate initialization are installed after
    construction by ``megatron.bridge.models.surgery.wiring``; an unwired layer
    refuses to run.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the layer and its gate, and reject unsupported features.

        Args:
            *args: Forwarded to ``TransformerLayer``.
            **kwargs: Forwarded to ``TransformerLayer``.
        """
        super().__init__(*args, **kwargs)
        check_layer_support(self, variant_id=VARIANT_ID)
        setattr(
            self,
            GATE_ATTRIBUTE,
            build_cross_gate(
                self.config,
                kind=getattr(self.config, "surgery_gate_kind", "scalar"),
                gate_dim_ratio=getattr(self.config, "surgery_gate_dim_ratio", 0.25),
            ),
        )

    @property
    def cross_injection_plan(self) -> CrossInjectionPlan:
        """Return the wiring installed by the surgery pre-wrap hook.

        Returns:
            The layer's pair plan.

        Raises:
            RuntimeError: If the model was built without running the hook.
        """
        plan = self.__dict__.get(PLAN_ATTRIBUTE)
        if plan is None:
            raise RuntimeError(
                "Model surgery variant v3_2/v3_3 layer is unwired. Build the model through "
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
        """Pass through, or run the pair with gated cross-injection.

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
            The pair output (or the untouched input) and the unchanged context.

        Raises:
            ValueError: If a cross-attention mask, fused TP inference, or a
                bias-producing MLP is encountered.
        """
        if context_mask is not None:
            raise ValueError("Model surgery variant v3_2/v3_3 does not support cross-attention context masks")
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            raise ValueError("Model surgery variant v3_2/v3_3 does not support fused TP inference")

        plan = self.cross_injection_plan
        if plan.group is None:
            # The joining member of this pair executes this layer's sublayers.
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

        # Every attention reads the block input, so both are available before
        # any MLP runs. That is what makes the cross-injection possible at all.
        attentions = [
            run_attention(member, hidden_states, variant_id=VARIANT_ID, attention_kwargs=attention_kwargs)
            for member in plan.group
        ]

        output = residual + sum(attentions)
        for index, member in enumerate(plan.group):
            mlp_input = hidden_states + attentions[index]
            gate = plan.gates[index] if index < len(plan.gates) else None
            if gate is not None:
                # The other branch's attention output, gated. With two branches
                # "the other" is unambiguous; more would need a combining rule.
                other = attentions[1 - index]
                mlp_input = mlp_input + gate(hidden_states, other)
            normed = apply_module(member.mlp_norm)(mlp_input)
            if isinstance(normed, tuple):
                raise ValueError("Model surgery variant v3_2/v3_3 requires an MLP norm that returns a single tensor")
            mlp_output, mlp_bias = apply_module(member.mlp)(normed, padding_mask=padding_mask)
            if mlp_bias is not None:
                raise ValueError("Model surgery variant v3_2/v3_3 requires a bias-free MLP output")
            output = output + mlp_output

        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
