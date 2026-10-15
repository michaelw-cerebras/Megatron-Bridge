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

"""Registry of model-surgery variants.

Identifiers are append-only. A variant that turns out to be wrong is marked
``retired``; its number is never reused and never renumbered, because manifests,
checkpoint directories, and result keys reference it forever.

The parallelization ratio is a *parameter* of a variant (``layers``), not a
separate variant, so a single variant can be swept across ratios.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.surgery.adjacent_layer import AdjacentDecoderParallelLayer
from megatron.bridge.models.surgery.compensated_layer import CompensatedParallelLayer
from megatron.bridge.models.surgery.merged_pool_layer import MergedPoolDecoderLayer
from megatron.bridge.models.surgery.parallel_layer import ParallelAttentionMLPLayer
from megatron.bridge.models.surgery.requirements import check_config_requirements
from megatron.bridge.models.surgery.selectors import SelectorSpec, normalize_selector, resolve_selector
from megatron.bridge.models.surgery.shifted_layer import ShiftedAttentionMLPLayer


_LAYER_LOCAL_REQUIREMENTS: tuple[str, ...] = (
    "pre_norm_residual",
    "no_hidden_dropout",
    "no_mlp_chunking",
    "no_cuda_graphs",
    "no_fused_tp_inference",
    "no_selective_recompute",
    "no_cpu_offloading",
    "non_local_attention_backend",
)
# Variants that execute one layer's sublayers inside a neighbour's forward.
_CROSS_LAYER_REQUIREMENTS: tuple[str, ...] = _LAYER_LOCAL_REQUIREMENTS + (
    "no_pipeline_parallel",
    "no_expert_parallel_overlap",
)


@dataclass(frozen=True)
class Variant:
    """One graph-rearrangement variant.

    Attributes:
        id: Stable append-only identifier such as ``"v1"``.
        slug: Short human-readable label used in plots and tables.
        summary: One-line description of the rearrangement.
        equation: The rewritten per-layer computation.
        layer_module: ``TransformerLayer`` subclass applied to selected layers,
            or ``None`` for the identity variant.
        config_requirements: Names of preconditions in ``CONFIG_REQUIREMENTS``.
        default_params: Parameter defaults merged under caller-supplied params.
        selection_rule: ``"any"``, ``"contiguous"``, or ``"contiguous_groups"``.
        needs_wiring: Whether the model needs a post-build step (cross-layer
            references, or initializing and freezing added parameters).
        adds_parameters: Whether the variant introduces weights that do not
            exist in the original checkpoint.
        status: ``"active"`` or ``"retired"``.
    """

    id: str
    slug: str
    summary: str
    equation: str
    layer_module: type | None
    config_requirements: tuple[str, ...] = ()
    default_params: Mapping[str, Any] = field(default_factory=dict)
    selection_rule: str = "any"
    needs_wiring: bool = False
    adds_parameters: bool = False
    status: str = "active"

    def resolve_params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Merge caller parameters over the variant defaults and validate them.

        Args:
            params: Caller-supplied parameters, or ``None`` for defaults only.

        Returns:
            The canonical parameter dictionary recorded in the manifest.

        Raises:
            ValueError: If an unknown parameter is supplied, ``layers`` is
                malformed, or ``group_size`` is not a positive integer.
        """
        merged = dict(self.default_params)
        for key, value in (params or {}).items():
            if key not in merged:
                raise ValueError(
                    f"Variant {self.id} has no parameter {key!r}; known parameters: {sorted(merged) or 'none'}"
                )
            merged[key] = value
        if "layers" in merged:
            merged["layers"] = normalize_selector(merged["layers"])
        if "expansion_factor" in merged:
            factor = merged["expansion_factor"]
            if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
                raise ValueError(f"Variant {self.id} requires an integer expansion_factor >= 1, got {factor!r}")
        if "init" in merged:
            from megatron.bridge.models.surgery.compensatory_mlp import INIT_SCHEMES

            if merged["init"] not in INIT_SCHEMES:
                raise ValueError(f"Variant {self.id} init must be one of {list(INIT_SCHEMES)}, got {merged['init']!r}")
        if "group_size" in merged:
            group_size = merged["group_size"]
            if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
                raise ValueError(f"Variant {self.id} requires an integer group_size >= 1, got {group_size!r}")
        return merged

    def validate_selection(self, selected: frozenset[int], params: Mapping[str, Any]) -> None:
        """Check that the resolved layer set is expressible by this variant.

        Args:
            selected: Resolved zero-based global layer indices.
            params: Canonical variant parameters.

        Raises:
            ValueError: If the selection violates the variant's selection rule.
        """
        if not selected or self.selection_rule == "any":
            return
        ordered = sorted(selected)
        if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
            raise ValueError(
                f"Variant {self.id} needs a contiguous run of layers because it couples neighbours; got {ordered}"
            )
        if self.selection_rule == "contiguous_groups":
            group_size = int(params["group_size"])
            if len(ordered) % group_size != 0:
                raise ValueError(
                    f"Variant {self.id} needs the selected run length ({len(ordered)}) to be a multiple "
                    f"of group_size ({group_size}); a partial group has no defined behaviour"
                )

    def make_layer_spec(self, base_spec: ModuleSpec) -> ModuleSpec:
        """Derive the rewritten layer spec from the model family's own base spec.

        Args:
            base_spec: Spec the model family would otherwise use for this layer.

        Returns:
            A spec with the same submodules but the variant's layer class.

        Raises:
            ValueError: If the variant does not rewrite individual layers.
        """
        if self.layer_module is None:
            raise ValueError(f"Variant {self.id} does not rewrite individual layers")
        return ModuleSpec(
            module=self.layer_module,
            params=base_spec.params,
            submodules=base_spec.submodules,
            metainfo=base_spec.metainfo,
        )


_DEFAULT_LAYERS: SelectorSpec = "all"

VARIANTS: dict[str, Variant] = {
    "v0": Variant(
        id="v0",
        slug="original",
        summary="Unmodified model. Baseline arm and the target of the degenerate-equivalence test.",
        equation="h = x + a(N_a(x)); y = h + m(N_m(h))",
        layer_module=None,
        default_params={"layers": "none"},
    ),
    "v1": Variant(
        id="v1",
        slug="parallel",
        summary="Attention and MLP/MoE read the same pre-residual input within a layer.",
        equation="y = x + a(N_a(x)) + m(N_m(x))",
        layer_module=ParallelAttentionMLPLayer,
        config_requirements=_LAYER_LOCAL_REQUIREMENTS,
        default_params={"layers": _DEFAULT_LAYERS},
    ),
    "v2": Variant(
        id="v2",
        slug="shifted",
        summary="Layer i's MLP runs alongside layer i+1's attention on a shared residual.",
        equation="u_i = u_{i-1} + m_{i-1}(N^m_{i-1}(u_{i-1})) + a_i(N^a_i(u_{i-1}))",
        layer_module=ShiftedAttentionMLPLayer,
        config_requirements=_CROSS_LAYER_REQUIREMENTS,
        default_params={"layers": _DEFAULT_LAYERS},
        selection_rule="contiguous",
        needs_wiring=True,
    ),
    "v3": Variant(
        id="v3",
        slug="adjacent",
        summary="Adjacent whole decoders read the same input; their deltas are summed.",
        equation="y = x + sum_i delta_i(x), delta_i(u) = a_i(u) + m_i(u + a_i(u))",
        layer_module=AdjacentDecoderParallelLayer,
        config_requirements=_CROSS_LAYER_REQUIREMENTS,
        default_params={"layers": _DEFAULT_LAYERS, "group_size": 2},
        selection_rule="contiguous_groups",
        needs_wiring=True,
    ),
    "v3_1": Variant(
        id="v3_1",
        slug="merged-pool",
        summary="Parallel decoders whose expert pools merge into one global top-(g*k) selection.",
        equation="z = x + sum_i a_i(N^a_i(x)); y = z + top_{g*k} over concat_i router_i(N^m_i(z))",
        layer_module=MergedPoolDecoderLayer,
        config_requirements=_CROSS_LAYER_REQUIREMENTS,
        default_params={"layers": _DEFAULT_LAYERS, "group_size": 2},
        selection_rule="contiguous_groups",
        needs_wiring=True,
    ),
    "v4": Variant(
        id="v4",
        slug="compensated",
        summary="Parallel attention/MLP plus a small trainable correction branch; originals frozen.",
        equation="y = x + a(N_a(x)) + m(N_m(x)) + c_theta(x)",
        layer_module=CompensatedParallelLayer,
        config_requirements=_LAYER_LOCAL_REQUIREMENTS,
        default_params={
            "layers": _DEFAULT_LAYERS,
            "expansion_factor": 1,
            "init": "zero_output",
            "init_seed": 1234,
        },
        # The hook initializes the new branch and freezes everything else; it is
        # not cross-layer wiring, but it is still a required post-build step.
        needs_wiring=True,
        adds_parameters=True,
    ),
}


def get_variant(variant_id: str) -> Variant:
    """Look up a registered variant.

    Args:
        variant_id: Stable variant identifier such as ``"v1"``.

    Returns:
        The registered variant.

    Raises:
        ValueError: If the identifier is unknown or the variant is retired.
    """
    try:
        variant = VARIANTS[variant_id]
    except KeyError:
        raise ValueError(f"Unknown surgery variant {variant_id!r}; known: {sorted(VARIANTS)}") from None
    if variant.status != "active":
        raise ValueError(f"Surgery variant {variant_id} is {variant.status} and must not be used for new runs")
    return variant


def list_variants() -> list[Variant]:
    """Return every registered variant in identifier order."""
    return [VARIANTS[key] for key in sorted(VARIANTS)]


def resolve_surgery_layers(
    config: Any,
    *,
    variant: str,
    params: Mapping[str, Any] | None = None,
) -> tuple[Variant, dict[str, Any], frozenset[int]]:
    """Resolve a variant, its parameters, and the global layers it rewrites.

    Args:
        config: Transformer config or model provider.
        variant: Stable variant identifier such as ``"v1"``.
        params: Variant parameters; ``None`` uses the variant defaults.

    Returns:
        The variant, its canonical parameters, and the zero-based global layer
        indices it rewrites.

    Raises:
        ValueError: If the variant, its parameters, or the config are invalid.
    """
    resolved = get_variant(variant)
    resolved_params = resolved.resolve_params(params)
    selected = resolve_selector(resolved_params.get("layers", "none"), config=config)
    if selected and resolved.layer_module is None:
        raise ValueError(f"Variant {resolved.id} rewrites no layers but its selector matched {len(selected)} layers")
    resolved.validate_selection(selected, resolved_params)
    if selected:
        check_config_requirements(config, resolved.config_requirements, variant_id=resolved.id)
    return resolved, resolved_params, selected
