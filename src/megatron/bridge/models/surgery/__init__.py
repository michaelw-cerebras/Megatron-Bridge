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

"""Model-agnostic decoder-graph surgery.

Every variant here keeps the state-dict keys and tensor bytes identical to the
unmodified model; only the computation graph changes. See
``mw_scripts/model_surgeon/README.md`` for the research workflow built on top.
"""

from megatron.bridge.models.surgery.added_parameters import describe_added_parameters
from megatron.bridge.models.surgery.adjacent_layer import AdjacentDecoderParallelLayer, AdjacentPlan
from megatron.bridge.models.surgery.compensated_layer import CompensatedParallelLayer
from megatron.bridge.models.surgery.compensatory_mlp import (
    CORRECTION_ATTRIBUTE,
    build_compensatory_mlp,
    freeze_except_compensatory,
    initialize_compensatory_mlp,
)
from megatron.bridge.models.surgery.cross_gate import (
    GATE_ATTRIBUTE,
    NonlinearCrossGate,
    ScalarCrossGate,
    build_cross_gate,
    gate_is_active,
    initialize_cross_gate,
)
from megatron.bridge.models.surgery.cross_injection_layer import (
    CrossInjectedParallelLayer,
    CrossInjectionPlan,
)
from megatron.bridge.models.surgery.manifest import (
    ARCHITECTURE_FIELD,
    MANIFEST_NAME,
    VARIANT_ALIASES,
    build_manifest,
    canonical_variant_id,
    legacy_architecture,
    manifest_variant,
    read_manifest,
    resolve_variant_reference,
)
from megatron.bridge.models.surgery.merged_moe import MergedRouting, compute_merged_routing
from megatron.bridge.models.surgery.merged_pool_layer import MergedPoolDecoderLayer, MergedPoolPlan
from megatron.bridge.models.surgery.parallel_layer import ParallelAttentionMLPLayer
from megatron.bridge.models.surgery.requirements import CONFIG_REQUIREMENTS, check_config_requirements
from megatron.bridge.models.surgery.selectors import normalize_selector, resolve_selector, selector_ratio
from megatron.bridge.models.surgery.shifted_layer import ShiftedAttentionMLPLayer, ShiftPlan
from megatron.bridge.models.surgery.spec import apply_surgery, describe_surgery, wrap_spec
from megatron.bridge.models.surgery.sublayers import DecoderSublayers, sublayers_of
from megatron.bridge.models.surgery.variants import (
    VARIANTS,
    Variant,
    get_variant,
    list_variants,
    resolve_surgery_layers,
)
from megatron.bridge.models.surgery.wiring import make_pre_wrap_hook, wire_model


__all__ = [
    "ARCHITECTURE_FIELD",
    "CONFIG_REQUIREMENTS",
    "MANIFEST_NAME",
    "VARIANT_ALIASES",
    "VARIANTS",
    "CORRECTION_ATTRIBUTE",
    "GATE_ATTRIBUTE",
    "AdjacentDecoderParallelLayer",
    "CompensatedParallelLayer",
    "CrossInjectedParallelLayer",
    "CrossInjectionPlan",
    "NonlinearCrossGate",
    "ScalarCrossGate",
    "AdjacentPlan",
    "DecoderSublayers",
    "MergedPoolDecoderLayer",
    "MergedPoolPlan",
    "MergedRouting",
    "ParallelAttentionMLPLayer",
    "ShiftPlan",
    "ShiftedAttentionMLPLayer",
    "Variant",
    "apply_surgery",
    "build_compensatory_mlp",
    "build_cross_gate",
    "build_manifest",
    "canonical_variant_id",
    "check_config_requirements",
    "compute_merged_routing",
    "describe_added_parameters",
    "describe_surgery",
    "freeze_except_compensatory",
    "gate_is_active",
    "initialize_cross_gate",
    "initialize_compensatory_mlp",
    "get_variant",
    "legacy_architecture",
    "list_variants",
    "make_pre_wrap_hook",
    "manifest_variant",
    "normalize_selector",
    "read_manifest",
    "resolve_selector",
    "resolve_surgery_layers",
    "resolve_variant_reference",
    "selector_ratio",
    "sublayers_of",
    "wire_model",
    "wrap_spec",
]
