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

"""Declarative preconditions a model config must satisfy for a surgery variant.

A rewritten computation graph is only correct under assumptions the stock
``TransformerLayer`` does not advertise. Every variant names the assumptions it
relies on so an unsupported model fails at spec-build time with an explanatory
error instead of silently training the wrong graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfigRequirement:
    """One named precondition evaluated against a transformer config."""

    name: str
    description: str
    predicate: Callable[[Any], bool]


def _attr(config: Any, name: str, default: Any) -> Any:
    """Read a config attribute, treating absence as the feature being disabled."""
    return getattr(config, name, default)


CONFIG_REQUIREMENTS: dict[str, ConfigRequirement] = {
    "pre_norm_residual": ConfigRequirement(
        name="pre_norm_residual",
        description="residual must branch from the un-normalized layer input",
        predicate=lambda c: not _attr(c, "apply_residual_connection_post_layernorm", False),
    ),
    "no_hidden_dropout": ConfigRequirement(
        name="no_hidden_dropout",
        description="hidden_dropout must be 0.0",
        predicate=lambda c: _attr(c, "hidden_dropout", 0.0) == 0.0,
    ),
    "no_mlp_chunking": ConfigRequirement(
        name="no_mlp_chunking",
        description="mlp_chunks_for_prefill and mlp_chunks_for_training must be 1",
        predicate=lambda c: _attr(c, "mlp_chunks_for_prefill", 1) == 1 and _attr(c, "mlp_chunks_for_training", 1) == 1,
    ),
    "no_cuda_graphs": ConfigRequirement(
        name="no_cuda_graphs",
        description="cuda_graph_impl must be 'none'",
        predicate=lambda c: _attr(c, "cuda_graph_impl", "none") == "none",
    ),
    "no_fused_tp_inference": ConfigRequirement(
        name="no_fused_tp_inference",
        description="inference_fuse_tp_communication must be disabled",
        predicate=lambda c: not _attr(c, "inference_fuse_tp_communication", False),
    ),
    "no_selective_recompute": ConfigRequirement(
        name="no_selective_recompute",
        description="recompute_granularity must not be 'selective'",
        predicate=lambda c: _attr(c, "recompute_granularity", None) != "selective",
    ),
    "no_cpu_offloading": ConfigRequirement(
        name="no_cpu_offloading",
        description="cpu_offloading must be disabled",
        predicate=lambda c: not _attr(c, "cpu_offloading", False),
    ),
    # GPTModelProvider.provide() rewrites core_attention for the local backend by
    # reaching through `spec.submodules`. Surgery hands Megatron a
    # TransformerBlockSubmodules instead, which has no such attribute, so the
    # rewrite would be skipped without warning.
    "non_local_attention_backend": ConfigRequirement(
        name="non_local_attention_backend",
        description="attention_backend must not be 'local' (surgery returns a block spec)",
        predicate=lambda c: str(getattr(_attr(c, "attention_backend", None), "name", "")).lower() != "local",
    ),
    # Cross-layer variants run one layer's MLP inside a neighbour's forward. Across
    # a pipeline boundary those two layers live on different stages, so the graph
    # cannot be expressed without relocating weights -- which would break the
    # byte-identical-checkpoint invariant. Fail instead of silently training a
    # different graph at stage boundaries.
    "no_pipeline_parallel": ConfigRequirement(
        name="no_pipeline_parallel",
        description="pipeline_model_parallel_size must be 1 for cross-layer variants",
        predicate=lambda c: (
            (_attr(c, "pipeline_model_parallel_size", 1) or 1) <= 1
            and not _attr(c, "virtual_pipeline_model_parallel_size", None)
        ),
    ),
    # The expert-parallel overlap scheduler assumes each layer's MoE runs inside
    # that layer's forward. Cross-layer variants break that assumption.
    "no_expert_parallel_overlap": ConfigRequirement(
        name="no_expert_parallel_overlap",
        description="overlap_moe_expert_parallel_comm must be disabled for cross-layer variants",
        predicate=lambda c: not _attr(c, "overlap_moe_expert_parallel_comm", False),
    ),
}


def check_config_requirements(config: Any, requirements: Iterable[str], *, variant_id: str) -> None:
    """Raise if the config violates any named requirement.

    Args:
        config: Transformer config or model provider being surgically rewritten.
        requirements: Requirement names registered in ``CONFIG_REQUIREMENTS``.
        variant_id: Variant identifier used in the error message.

    Raises:
        KeyError: If a requirement name is not registered.
        ValueError: If the config violates one or more requirements.
    """
    violated: list[str] = []
    for name in requirements:
        requirement = CONFIG_REQUIREMENTS[name]
        if not requirement.predicate(config):
            violated.append(f"{requirement.name} ({requirement.description})")
    if violated:
        raise ValueError(
            f"Model surgery variant {variant_id} cannot run on this config; unmet requirements: " + "; ".join(violated)
        )
