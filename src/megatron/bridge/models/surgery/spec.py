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

"""Wrap a model family's own layer spec with a surgery variant.

``wrap_spec`` is the single injection point that makes surgery model-agnostic:
the family keeps full ownership of what a layer *is* (attention variant, MoE
layout, per-layer RoPE), and surgery only decides which layers get the rewritten
computation graph.

``apply_surgery`` is the entry point callers should use. It configures the spec
and, for variants that couple neighbouring layers, also registers the pre-wrap
hook that installs the cross-layer references.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_block import (
    LayerNormImpl,
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.surgery.selectors import selector_ratio
from megatron.bridge.models.surgery.variants import resolve_surgery_layers
from megatron.bridge.models.surgery.wiring import make_pre_wrap_hook


BaseSpec = ModuleSpec | Callable[[Any], ModuleSpec]
BlockSpecBuilder = Callable[..., TransformerBlockSubmodules]


def wrap_spec(
    base_spec: BaseSpec,
    *,
    variant: str,
    params: Mapping[str, Any] | None = None,
) -> BlockSpecBuilder:
    """Return a block-spec builder that applies ``variant`` to the selected layers.

    The result is assignable to ``GPTModelProvider.transformer_layer_spec``. It
    accepts ``vp_stage`` so Megatron slices the per-layer specs correctly under
    pipeline and virtual-pipeline parallelism.

    Prefer :func:`apply_surgery`, which also installs cross-layer wiring when the
    variant needs it.

    Args:
        base_spec: The model family's layer spec, or a callable building it from
            the config.
        variant: Stable variant identifier such as ``"v1"``.
        params: Variant parameters; ``None`` uses the variant defaults.

    Returns:
        A callable ``(config, vp_stage=None) -> TransformerBlockSubmodules``.
    """

    def build_surgery_block_spec(config: Any, vp_stage: int | None = None) -> TransformerBlockSubmodules:
        """Build this pipeline stage's per-layer specs for the surgery variant.

        Args:
            config: Transformer config supplied by Megatron.
            vp_stage: Virtual pipeline stage, or ``None`` when unused.

        Returns:
            Per-layer specs for the local pipeline stage plus the final norm.
        """
        # ModuleSpec defines __call__ (it builds the module), so a plain
        # callable() check would construct a layer instead of reusing the spec.
        resolved_base = base_spec if isinstance(base_spec, ModuleSpec) else base_spec(config)
        resolved, _, selected = resolve_surgery_layers(config, variant=variant, params=params)
        surgical_spec = resolved.make_layer_spec(resolved_base) if selected else resolved_base

        layer_specs = [surgical_spec if index in selected else resolved_base for index in range(config.num_layers)]
        if getattr(config, "pipeline_model_parallel_size", 1) > 1:
            # layer_specs is local to this pipeline stage: entry i is global layer
            # i + offset. Slicing wrong would build every layer on every stage.
            offset = get_transformer_layer_offset(config, vp_stage)
            num_local_layers = get_num_layers_to_build(config, vp_stage)
        else:
            # A single stage owns every layer. Computing this directly keeps spec
            # building independent of torch.distributed initialization.
            offset = 0
            num_local_layers = config.num_layers
        # TransformerBlockSubmodules.layer_norm defaults to None, which would
        # silently drop the decoder's final norm. Megatron supplies LayerNormImpl
        # itself only on the plain-ModuleSpec path.
        return TransformerBlockSubmodules(
            layer_specs=layer_specs[offset : offset + num_local_layers],
            layer_norm=LayerNormImpl,
        )

    return build_surgery_block_spec


def describe_surgery(
    config: Any,
    *,
    variant: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize what a variant would do to a model, for logging and manifests.

    Args:
        config: Transformer config or model provider.
        variant: Stable variant identifier such as ``"v1"``.
        params: Variant parameters; ``None`` uses the variant defaults.

    Returns:
        A JSON-serializable description including the resolved layer indices and
        the parallelization ratio ``p``.
    """
    resolved, resolved_params, selected = resolve_surgery_layers(config, variant=variant, params=params)
    return {
        "id": resolved.id,
        "slug": resolved.slug,
        "equation": resolved.equation,
        "params": resolved_params,
        "num_layers": config.num_layers,
        "rewritten_layers": sorted(selected),
        "parallelization_ratio": selector_ratio(selected, config.num_layers),
    }


def apply_surgery(
    provider: Any,
    *,
    base_spec: BaseSpec,
    variant: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Configure a model provider to build and wire a surgery variant.

    Sets the provider's ``transformer_layer_spec`` and, for variants that couple
    neighbouring layers, registers the pre-wrap hook that installs the
    cross-layer references on the constructed model.

    Args:
        provider: Model provider to configure in place.
        base_spec: The model family's layer spec, or a callable building it.
        variant: Stable variant identifier such as ``"v1"``.
        params: Variant parameters; ``None`` uses the variant defaults.

    Returns:
        The description returned by :func:`describe_surgery`, suitable for logging.
    """
    described = describe_surgery(provider, variant=variant, params=params)
    resolved, resolved_params, selected = resolve_surgery_layers(provider, variant=variant, params=params)
    provider.transformer_layer_spec = wrap_spec(base_spec, variant=variant, params=params)
    if selected:
        # Recorded on the provider so HF export can describe the graph a
        # reloaded checkpoint must rebuild.
        provider.surgery_variant_id = resolved.id
    if resolved.id == "v4" and selected:
        # The layer reads this at construction time to size its correction branch.
        provider.surgery_correction_expansion_factor = int(resolved_params["expansion_factor"])
    if resolved.id == "v3_1" and selected:
        provider.moe_additional_logging_metrics = tuple(
            f"merged_pool_frac_{index}" for index in range(int(resolved_params["group_size"]))
        )
    if resolved.needs_wiring and selected:
        provider.register_pre_wrap_hook(make_pre_wrap_hook(variant=variant, params=params))
    return described
