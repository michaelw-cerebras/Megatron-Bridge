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

"""Declarative selection of which decoder layers a surgery variant rewrites.

The selector is the parallelization-ratio knob. It is stored verbatim in the
surgery manifest, so every form here must be JSON-serializable and must resolve
to the same layer set on every rank and on every resume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SelectorSpec = str | Mapping[str, Any]

_SELECTOR_KINDS = frozenset({"all", "none", "indices", "every_k", "config_list"})


def normalize_selector(selector: SelectorSpec) -> dict[str, Any]:
    """Expand a selector shorthand into its canonical dictionary form.

    Args:
        selector: ``"all"``, ``"none"``, or a mapping with a ``kind`` key.

    Returns:
        A new canonical selector dictionary.

    Raises:
        ValueError: If the selector shorthand or ``kind`` is unknown.
    """
    if isinstance(selector, str):
        if selector not in {"all", "none"}:
            raise ValueError(f"Unknown layer selector shorthand {selector!r}; use 'all', 'none', or a mapping")
        return {"kind": selector}
    if not isinstance(selector, Mapping):
        raise ValueError(f"Layer selector must be a string or mapping, got {type(selector).__name__}")
    kind = selector.get("kind")
    if kind not in _SELECTOR_KINDS:
        raise ValueError(f"Unknown layer selector kind {kind!r}; expected one of {sorted(_SELECTOR_KINDS)}")
    return dict(selector)


def _resolve_indices(selector: Mapping[str, Any], num_layers: int) -> frozenset[int]:
    indices = selector.get("indices")
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("Layer selector 'indices' must be a list of integers")
    resolved: set[int] = set()
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"Layer index must be an integer, got {index!r}")
        if not 0 <= index < num_layers:
            raise ValueError(f"Layer index {index} is outside the model's {num_layers} layers")
        resolved.add(index)
    return frozenset(resolved)


def _resolve_every_k(selector: Mapping[str, Any], num_layers: int) -> frozenset[int]:
    k = selector.get("k")
    offset = selector.get("offset", 0)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"Layer selector 'k' must be an integer >= 1, got {k!r}")
    if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset < k:
        raise ValueError(f"Layer selector 'offset' must be an integer in [0, {k}), got {offset!r}")
    return frozenset(range(offset, num_layers, k))


def _resolve_config_list(selector: Mapping[str, Any], num_layers: int, config: object) -> frozenset[int]:
    field = selector.get("field")
    if not isinstance(field, str):
        raise ValueError("Layer selector 'field' must be the name of a per-layer config attribute")
    if not hasattr(config, field):
        raise ValueError(f"Model config has no per-layer field {field!r} for this layer selector")
    values = getattr(config, field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Config field {field!r} must be a per-layer sequence, got {type(values).__name__}")
    if len(values) != num_layers:
        raise ValueError(f"Config field {field!r} has {len(values)} entries but the model has {num_layers} layers")
    if "equals" in selector:
        wanted = selector["equals"]
        return frozenset(index for index, value in enumerate(values) if value == wanted)
    return frozenset(index for index, value in enumerate(values) if value)


def resolve_selector(selector: SelectorSpec, *, config: object) -> frozenset[int]:
    """Resolve a selector to the global, zero-based layer indices it rewrites.

    Args:
        selector: Selector shorthand or canonical mapping.
        config: Transformer config exposing ``num_layers`` and, for the
            ``config_list`` kind, the referenced per-layer field.

    Returns:
        The zero-based global layer indices the variant applies to.

    Raises:
        ValueError: If the selector is malformed or inconsistent with the config.
    """
    normalized = normalize_selector(selector)
    num_layers = getattr(config, "num_layers", None)
    if not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("Model config must define a positive 'num_layers' before resolving a layer selector")

    kind = normalized["kind"]
    if kind == "all":
        return frozenset(range(num_layers))
    if kind == "none":
        return frozenset()
    if kind == "indices":
        return _resolve_indices(normalized, num_layers)
    if kind == "every_k":
        return _resolve_every_k(normalized, num_layers)
    return _resolve_config_list(normalized, num_layers, config)


def selector_ratio(selected: frozenset[int], num_layers: int) -> float:
    """Return the parallelization ratio ``p`` recorded alongside experiment results.

    Args:
        selected: Resolved layer indices.
        num_layers: Total decoder layers in the model.

    Returns:
        Fraction of layers the variant rewrites, in ``[0, 1]``.

    Raises:
        ValueError: If ``num_layers`` is not positive.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    return len(selected) / num_layers
