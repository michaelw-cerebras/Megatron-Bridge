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

"""Joint routing over the union of several layers' expert pools.

Two MoE layers normally route independently: each token takes top-k of its own
E experts, so a group of two spends a fixed k+k budget. Merging the pools lets a
token take top-2k from all 2E experts, so the split can be 12+4 or 6+10 instead
of a fixed 8+8. Total parameters and active expert FLOPs are unchanged; only the
routing becomes dynamic.

Each layer keeps its own router weights, its own norm, and its own experts, so
the state dict is untouched. Only the *selection* is joint. The expert dispatch
still runs per layer through that layer's own token dispatcher, which is what
keeps this compatible with expert parallelism.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    switch_load_balancing_loss_func,
)


# Router behaviours the joint selection reimplements. Anything outside this set
# would silently change the selection semantics, so it is rejected instead.
_REQUIRED_ROUTER_SETTINGS: tuple[tuple[str, Any], ...] = (
    ("moe_token_dispatcher_type", "alltoall"),
    ("moe_router_pre_softmax", True),
    ("moe_router_score_function", "softmax"),
    ("moe_router_num_groups", None),
    ("moe_router_group_topk", None),
    ("moe_router_topk_scaling_factor", None),
    ("moe_expert_capacity_factor", None),
    ("moe_z_loss_coeff", None),
    ("moe_input_jitter_eps", None),
    ("moe_router_enable_expert_bias", False),
    ("moe_router_force_load_balancing", False),
    ("moe_latent_size", None),
    ("moe_shared_expert_intermediate_size", None),
)


@dataclass(frozen=True)
class MergedRouting:
    """Result of one joint top-2k selection over the union of two expert pools.

    Attributes:
        probs: Per-pool dense probabilities, each ``[num_tokens, experts_per_pool]``.
            Already globally renormalized; a pool's slice does **not** sum to 1.
        routing_map: Per-pool boolean selection masks of the same shape.
        scores: Full softmax over the merged pool, ``[num_tokens, total_experts]``.
        merged_routing_map: Full selection mask, ``[num_tokens, total_experts]``.
        selected_per_pool: Number of (token, expert) selections taken from each pool.
    """

    probs: tuple[torch.Tensor, ...]
    routing_map: tuple[torch.Tensor, ...]
    scores: torch.Tensor
    merged_routing_map: torch.Tensor
    selected_per_pool: tuple[torch.Tensor, ...]


def check_merged_pool_support(moe_layers: Sequence[Any], *, variant_id: str) -> None:
    """Reject router or MoE settings the joint selection does not reimplement.

    Args:
        moe_layers: The ``MoELayer`` instances whose pools are merged.
        variant_id: Variant identifier used in error messages.

    Raises:
        ValueError: If a layer is not a router-based MoE layer, the pools differ
            in size, or any unsupported router feature is enabled.
    """
    for index, layer in enumerate(moe_layers):
        if not hasattr(layer, "router") or not hasattr(layer, "token_dispatcher"):
            raise ValueError(
                f"Model surgery variant {variant_id} needs every grouped layer to use a routed MoE; "
                f"layer {index} is {type(layer).__name__}"
            )

    config = moe_layers[0].config
    experts = {layer.config.num_moe_experts for layer in moe_layers}
    topks = {layer.router.topk for layer in moe_layers}
    if len(experts) != 1 or len(topks) != 1:
        raise ValueError(
            f"Model surgery variant {variant_id} needs identical pools; got num_experts={experts}, topk={topks}"
        )

    unsupported = [
        f"{name} must be {wanted!r}, got {getattr(config, name, None)!r}"
        for name, wanted in _REQUIRED_ROUTER_SETTINGS
        if getattr(config, name, None) != wanted
    ]
    if unsupported:
        raise ValueError(
            f"Model surgery variant {variant_id} cannot merge expert pools under this router config: "
            + "; ".join(unsupported)
        )


def compute_merged_routing(
    moe_layers: Sequence[Any],
    normed_inputs: Sequence[torch.Tensor],
    *,
    variant_id: str,
) -> MergedRouting:
    """Select top-(pools x k) experts jointly across every pool.

    Each layer's own router scores the shared token state through that layer's
    own MLP norm. The logits are concatenated and a single softmax plus a single
    top-k runs over the union, so a token is free to spend its whole budget on
    one pool.

    Args:
        moe_layers: ``MoELayer`` instances contributing their experts, in order.
        normed_inputs: Each layer's normed view of the shared input, ``[s, b, h]``.
        variant_id: Variant identifier used in error messages.

    Returns:
        The joint selection, already split per pool.
    """
    per_pool_experts = moe_layers[0].config.num_moe_experts
    merged_topk = moe_layers[0].router.topk * len(moe_layers)
    compute_dtype = normed_inputs[0].dtype

    # Score in fp32 regardless of the parameter dtype: top-k over 2E entries is
    # more tie-prone than over E, and ties resolved differently from the original
    # model would change expert selection for no modelling reason.
    logits = torch.cat(
        [
            layer.router.gating(normed).reshape(-1, per_pool_experts).float()
            for layer, normed in zip(moe_layers, normed_inputs)
        ],
        dim=-1,
    )
    scores = torch.softmax(logits, dim=-1)

    top_scores, top_indices = torch.topk(scores, k=merged_topk, dim=-1)
    # Renormalize exactly once, over the merged selection. Letting each pool
    # renormalize its own slice afterwards would restore the fixed k+k split this
    # variant exists to remove.
    top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)

    merged_probs = torch.zeros_like(scores).scatter(1, top_indices, top_scores)
    merged_routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(
        1, top_indices, torch.ones_like(top_scores, dtype=torch.bool)
    )

    probs = merged_probs.to(compute_dtype).split(per_pool_experts, dim=-1)
    routing_map = merged_routing_map.split(per_pool_experts, dim=-1)
    selected_per_pool = tuple(pool_map.sum() for pool_map in routing_map)
    return MergedRouting(
        probs=tuple(probs),
        routing_map=tuple(routing_map),
        scores=scores,
        merged_routing_map=merged_routing_map,
        selected_per_pool=selected_per_pool,
    )


def apply_joint_aux_loss(routing: MergedRouting, moe_layers: Sequence[Any]) -> MergedRouting:
    """Attach one load-balancing loss computed over the merged pool.

    Keeping each layer's own aux loss would push each pool toward balance within
    its own experts, which directly opposes the dynamic split this variant is
    testing. One loss over all pools balances the union instead.

    Args:
        routing: The joint selection to attach the loss to.
        moe_layers: The grouped ``MoELayer`` instances, in order.

    Returns:
        The routing with the aux-loss autograd hook attached to each pool's probs.
    """
    router = moe_layers[0].router
    coefficient = router.get_aux_loss_coeff("aux_loss")
    if coefficient == 0:
        return routing

    merged_topk = router.topk * len(moe_layers)
    total_experts = routing.scores.shape[-1]
    num_tokens = routing.scores.shape[0]
    aux_loss = switch_load_balancing_loss_func(
        probs=routing.scores,
        tokens_per_expert=routing.merged_routing_map.sum(dim=0),
        total_num_tokens=num_tokens,
        topk=merged_topk,
        num_experts=total_experts,
        moe_aux_loss_coeff=coefficient,
    )
    # Use Megatron's standard name so the existing training logger includes the
    # joint loss in console, TensorBoard, and W&B output. Record the same group
    # value at every member layer: the logger averages over all MoE layers, not
    # over merged groups, so recording only the joining layer would halve the
    # reported value for group_size=2.
    for moe_layer in moe_layers:
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / coefficient,
            moe_layer.router.layer_number,
            moe_layer.config.num_layers,
        )
    # Scale through the first pool's probs only: the loss is already joint, and
    # attaching it to every pool would multiply its gradient by the pool count.
    scaled_first = MoEAuxLossAutoScaler.apply(routing.probs[0], aux_loss)
    return MergedRouting(
        probs=(scaled_first, *routing.probs[1:]),
        routing_map=routing.routing_map,
        scores=routing.scores,
        merged_routing_map=routing.merged_routing_map,
        selected_per_pool=routing.selected_per_pool,
    )


def log_pool_balance(routing: MergedRouting, moe_layers: Sequence[Any]) -> None:
    """Record how the merged budget was split between pools.

    A value of ``1 / num_pools`` means the group reproduced the original fixed
    split; movement away from it is the model actually using the freedom this
    variant grants, which is the headline measurement of the experiment.

    Args:
        routing: The joint selection.
        moe_layers: The grouped ``MoELayer`` instances, in order.
    """
    total = sum(routing.selected_per_pool)
    for index, selected in enumerate(routing.selected_per_pool):
        fraction = selected.float() / total.float().clamp_min(1.0)
        # See apply_joint_aux_loss: duplicate the group metric over its member
        # layer slots so Megatron's per-MoE-layer aggregation preserves 0.5 as
        # the balanced value for two equal pools.
        for moe_layer in moe_layers:
            save_to_aux_losses_tracker(
                f"merged_pool_frac_{index}",
                fraction,
                moe_layer.router.layer_number,
                moe_layer.config.num_layers,
            )


def run_moe_with_routing(
    moe_layer: Any,
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
) -> torch.Tensor:
    """Run one MoE layer's dispatch and experts using an externally chosen routing.

    This mirrors ``MoELayer.forward`` with ``route()`` replaced by the caller's
    selection. Dispatch still happens inside the layer, so expert parallelism and
    the layer's configured token dispatcher are untouched.

    Args:
        moe_layer: The ``MoELayer`` to run.
        hidden_states: Normed input for this layer, ``[s, b, h]``.
        probs: Dense probabilities for this layer's experts, ``[num_tokens, E]``.
        routing_map: Boolean selection mask of the same shape.

    Returns:
        This layer's contribution to the merged output.

    Raises:
        ValueError: If the layer produces an MLP bias, which the merged sum
            cannot represent.
    """
    shared_expert_output = moe_layer.shared_experts_compute(hidden_states)
    # MCore's dropless all-to-all fast path assumes every token has exactly
    # config.moe_router_topk selections in this layer. A merged pool deliberately
    # violates that assumption: one pool can receive anywhere from zero to g*k
    # selections for a token. The dispatcher already has a dynamic-size path,
    # selected when a capacity factor is present, which sizes the permutation
    # from routing_map.sum() without applying capacity unless drop_and_pad is on.
    # Select that path only for this preprocess call, then restore the shared
    # TransformerConfig before communication or expert compute.
    dispatcher_config = getattr(moe_layer.token_dispatcher, "config", None)
    if dispatcher_config is None:
        # Lightweight test doubles do not carry MCore dispatcher config.
        dispatch_input, dispatch_probs = moe_layer.preprocess(hidden_states, probs, routing_map)
    else:
        original_capacity_factor = dispatcher_config.moe_expert_capacity_factor
        dispatcher_config.moe_expert_capacity_factor = 1.0
        try:
            dispatch_input, dispatch_probs = moe_layer.preprocess(hidden_states, probs, routing_map)
        finally:
            dispatcher_config.moe_expert_capacity_factor = original_capacity_factor
    dispatched, dispatched_probs = moe_layer.dispatch(dispatch_input, dispatch_probs)
    output, mlp_bias = moe_layer.routed_experts_compute(dispatched, dispatched_probs)
    if mlp_bias is not None:
        raise ValueError("Merged expert pools require a bias-free MoE output")
    output = moe_layer.combine(output)
    return moe_layer.postprocess(output, shared_expert_output)
