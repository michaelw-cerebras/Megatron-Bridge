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

"""Install cross-layer references on a built decoder.

Variants v2 and v3 execute one layer's sublayers inside a neighbour's forward.
That wiring cannot live in the layer spec, because a spec describes a layer in
isolation and knows nothing about its siblings. It is installed instead on the
constructed model, through Megatron Bridge's ``pre_wrap_hook``, which runs on
the raw ``GPTModel`` before mixed-precision and DDP wrapping.

References are stored in ``__dict__`` rather than as module attributes:
``torch.nn.Module.__setattr__`` would register a second owner for each module
and duplicate it in the state dict, breaking the invariant that a surgery
checkpoint stays byte-identical to the original.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from megatron.bridge.models.surgery.adjacent_layer import (
    PLAN_ATTRIBUTE as ADJACENT_PLAN_ATTRIBUTE,
)
from megatron.bridge.models.surgery.adjacent_layer import (
    AdjacentDecoderParallelLayer,
    AdjacentPlan,
)
from megatron.bridge.models.surgery.compensatory_mlp import (
    CORRECTION_ATTRIBUTE,
    freeze_except_compensatory,
    initialize_compensatory_mlp,
)
from megatron.bridge.models.surgery.cross_gate import (
    GATE_ATTRIBUTE,
    gate_is_active,
    initialize_cross_gate,
)
from megatron.bridge.models.surgery.cross_injection_layer import (
    BRANCHES,
    CrossInjectedParallelLayer,
    CrossInjectionPlan,
)
from megatron.bridge.models.surgery.cross_injection_layer import (
    PLAN_ATTRIBUTE as CROSS_PLAN_ATTRIBUTE,
)
from megatron.bridge.models.surgery.merged_moe import check_merged_pool_support
from megatron.bridge.models.surgery.merged_pool_layer import (
    PLAN_ATTRIBUTE as MERGED_POOL_PLAN_ATTRIBUTE,
)
from megatron.bridge.models.surgery.merged_pool_layer import (
    MergedPoolDecoderLayer,
    MergedPoolPlan,
)
from megatron.bridge.models.surgery.shifted_layer import (
    PLAN_ATTRIBUTE as SHIFT_PLAN_ATTRIBUTE,
)
from megatron.bridge.models.surgery.shifted_layer import (
    ShiftedAttentionMLPLayer,
    ShiftPlan,
)
from megatron.bridge.models.surgery.sublayers import sublayers_of
from megatron.bridge.models.surgery.variants import resolve_surgery_layers


LOGGER = logging.getLogger(__name__)


def _decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError(f"Model surgery wiring needs a decoder with layers, got {type(model).__name__}")
    return layers


def _wire_shifted(layers: Sequence[torch.nn.Module], selected: Sequence[int]) -> None:
    for position, index in enumerate(selected):
        layer = layers[index]
        if not isinstance(layer, ShiftedAttentionMLPLayer):
            raise ValueError(f"Layer {index} is {type(layer).__name__}, expected a v2 layer")
        previous = None if position == 0 else sublayers_of(layers[selected[position - 1]])
        layer.__dict__[SHIFT_PLAN_ATTRIBUTE] = ShiftPlan(
            previous_mlp=previous,
            closes_run=position == len(selected) - 1,
        )


def _wire_adjacent(layers: Sequence[torch.nn.Module], selected: Sequence[int], group_size: int) -> None:
    for start in range(0, len(selected), group_size):
        group = selected[start : start + group_size]
        members = tuple(sublayers_of(layers[index]) for index in group)
        for position, index in enumerate(group):
            layer = layers[index]
            if not isinstance(layer, AdjacentDecoderParallelLayer):
                raise ValueError(f"Layer {index} is {type(layer).__name__}, expected a v3 layer")
            # Only the last member joins; the earlier ones pass their input
            # through so every member sees the same tensor.
            layer.__dict__[ADJACENT_PLAN_ATTRIBUTE] = AdjacentPlan(
                group=members if position == len(group) - 1 else None
            )


def _wire_merged_pool(layers: Sequence[torch.nn.Module], selected: Sequence[int], group_size: int) -> None:
    for start in range(0, len(selected), group_size):
        group = selected[start : start + group_size]
        members = tuple(sublayers_of(layers[index]) for index in group)
        check_merged_pool_support([member.mlp for member in members], variant_id="v3_1")
        for position, index in enumerate(group):
            layer = layers[index]
            if not isinstance(layer, MergedPoolDecoderLayer):
                raise ValueError(f"Layer {index} is {type(layer).__name__}, expected a v3_1 layer")
            layer.__dict__[MERGED_POOL_PLAN_ATTRIBUTE] = MergedPoolPlan(
                group=members if position == len(group) - 1 else None
            )


def _prepare_compensatory(
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    selected: frozenset[int],
    params: Mapping[str, Any],
) -> None:
    """Initialize v1_1's correction branches and freeze every original weight.

    Megatron is told to skip initialization so the Hugging Face import owns all
    original weights; nothing would otherwise write to the new branch. Layers
    are matched by their global ``layer_number`` rather than list position, so
    this stays correct under pipeline parallelism.
    """
    prepared = []
    for layer in layers:
        correction = getattr(layer, CORRECTION_ATTRIBUTE, None)
        if correction is None:
            continue
        # layer_number is 1-based and global; the selector is 0-based.
        if (layer.layer_number - 1) not in selected:
            raise ValueError(f"Layer {layer.layer_number} has a correction branch but was not selected for v1_1")
        initialize_compensatory_mlp(
            correction,
            layer.config,
            scheme=str(params["init"]),
            seed=int(params["init_seed"]),
            layer_number=layer.layer_number,
        )
        prepared.append(layer.layer_number)

    trainable, frozen = freeze_except_compensatory(model)
    LOGGER.info(
        "Variant v1_1: initialized %d correction branches (%s, seed %s); %d trainable / %d frozen parameters",
        len(prepared),
        params["init"],
        params["init_seed"],
        trainable,
        frozen,
    )
    if trainable == 0:
        raise ValueError("Variant v1_1 froze every parameter; the correction branches were not found")


def _wire_cross_injection(
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    selected: Sequence[int],
    params: Mapping[str, Any],
) -> None:
    """Pair adjacent decoders, initialize their gates, and optionally freeze."""
    direction = str(params["direction"])
    for start in range(0, len(selected), BRANCHES):
        pair = selected[start : start + BRANCHES]
        members = tuple(sublayers_of(layers[index]) for index in pair)
        gates: list[torch.nn.Module | None] = []
        for position, index in enumerate(pair):
            layer = layers[index]
            if not isinstance(layer, CrossInjectedParallelLayer):
                raise ValueError(f"Layer {index} is {type(layer).__name__}, expected a v3_2/v3_3 layer")
            gate = getattr(layer, GATE_ATTRIBUTE)
            initialize_cross_gate(
                gate,
                layer.config,
                seed=int(params["init_seed"]),
                layer_number=layer.layer_number,
            )
            gates.append(gate if gate_is_active(direction, position=position, group_size=len(pair)) else None)

        for position, index in enumerate(pair):
            layers[index].__dict__[CROSS_PLAN_ATTRIBUTE] = CrossInjectionPlan(
                group=members if position == len(pair) - 1 else None,
                gates=tuple(gates),
            )

    if params.get("freeze_original", False):
        trainable = 0
        frozen = 0
        for name, parameter in model.named_parameters():
            if GATE_ATTRIBUTE in name.split("."):
                parameter.requires_grad = True
                trainable += parameter.numel()
            else:
                parameter.requires_grad = False
                frozen += parameter.numel()
        if trainable == 0:
            raise ValueError("Variant v3_2/v3_3 froze every parameter; the cross gates were not found")
        LOGGER.info("Variant v3_2/v3_3: %d trainable / %d frozen parameters", trainable, frozen)
    LOGGER.info(
        "Wired v3_2/v3_3 cross-injection across %d pairs (gate=%s, direction=%s)",
        len(selected) // BRANCHES,
        params.get("gate"),
        direction,
    )


def wire_model(model: torch.nn.Module, *, variant: str, params: Mapping[str, Any] | None = None) -> None:
    """Install a variant's cross-layer references on one built model.

    Args:
        model: Raw ``GPTModel`` (not yet wrapped for mixed precision or DDP).
        variant: Stable variant identifier such as ``"v2"``.
        params: Variant parameters; ``None`` uses the variant defaults.

    Raises:
        ValueError: If the variant needs no wiring, the decoder does not expose
            the expected number of layers, or a layer has the wrong class.
    """
    resolved, resolved_params, selected = resolve_surgery_layers(model.config, variant=variant, params=params)
    if not resolved.needs_wiring:
        raise ValueError(f"Variant {resolved.id} does not use cross-layer wiring")
    if not selected:
        return

    layers = _decoder_layers(model)
    if resolved.id == "v1_1":
        # Layer-local: no cross-layer references, so pipeline stages are fine.
        _prepare_compensatory(model, layers, selected, resolved_params)
        return
    # Cross-layer variants require PP=1, so local index == global layer index.
    # Verify rather than assume: a mismatch would wire the wrong neighbours.
    if len(layers) != model.config.num_layers:
        raise ValueError(
            f"Variant {resolved.id} expects all {model.config.num_layers} layers on one stage, "
            f"but this model holds {len(layers)}"
        )

    ordered = sorted(selected)
    if resolved.id == "v2":
        _wire_shifted(layers, ordered)
    elif resolved.id in {"v3_2", "v3_3"}:
        _wire_cross_injection(model, layers, ordered, resolved_params)
    elif resolved.id == "v3_1":
        _wire_merged_pool(layers, ordered, int(resolved_params["group_size"]))
    else:
        _wire_adjacent(layers, ordered, int(resolved_params["group_size"]))
    LOGGER.info("Wired surgery variant %s across layers %s", resolved.id, ordered)


def make_pre_wrap_hook(
    *,
    variant: str,
    params: Mapping[str, Any] | None = None,
) -> Callable[[list[torch.nn.Module]], list[torch.nn.Module]]:
    """Build a Bridge pre-wrap hook that wires a variant's cross-layer references.

    Args:
        variant: Stable variant identifier such as ``"v2"``.
        params: Variant parameters; ``None`` uses the variant defaults.

    Returns:
        A hook suitable for ``ModelProviderMixin.register_pre_wrap_hook``.
    """

    def hook(model: list[torch.nn.Module]) -> list[torch.nn.Module]:
        """Wire every model chunk in place and return the list unchanged.

        Args:
            model: Model chunks produced by ``get_model``.

        Returns:
            The same list, with cross-layer references installed.

        Raises:
            ValueError: If more than one chunk is present, which means virtual
                pipeline parallelism is active and the layer indices would not
                be global.
        """
        if len(model) != 1:
            raise ValueError(
                f"Surgery variant {variant} requires a single model chunk (pipeline_model_parallel_size=1), "
                f"got {len(model)}"
            )
        wire_model(model[0], variant=variant, params=params)
        return model

    return hook
