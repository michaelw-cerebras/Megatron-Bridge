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

"""Shared sublayer invocation for surgery variants.

Every variant rewires the *same* attention and MLP calls; only the operands and
the order change. Keeping the calls here means a variant's forward reads as the
equation it implements, and the guard rails are stated once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.typed_torch import apply_module


@dataclass(frozen=True)
class DecoderSublayers:
    """The four modules a decoder layer contributes to a rewritten graph.

    Holding these in a plain dataclass rather than assigning them as module
    attributes is deliberate: ``torch.nn.Module.__setattr__`` would register a
    second owner for each module and duplicate it in the state dict, which is
    exactly the invariant surgery must not break.
    """

    attention_norm: torch.nn.Module
    attention: torch.nn.Module
    mlp_norm: torch.nn.Module
    mlp: torch.nn.Module


def sublayers_of(layer: torch.nn.Module) -> DecoderSublayers:
    """Collect a built decoder layer's sublayers as a non-owning reference.

    Args:
        layer: A constructed ``TransformerLayer``.

    Returns:
        References to the layer's attention and MLP modules and their norms.
    """
    return DecoderSublayers(
        attention_norm=layer.input_layernorm,
        attention=layer.self_attention,
        mlp_norm=layer.pre_mlp_layernorm,
        mlp=layer.mlp,
    )


def _single_tensor(value: object, *, variant_id: str, what: str) -> torch.Tensor:
    if isinstance(value, tuple):
        raise ValueError(f"Model surgery variant {variant_id} requires {what} to return a single tensor")
    return value  # type: ignore[return-value]  # non-tuple norm output is a Tensor


def run_attention(
    sublayers: DecoderSublayers,
    hidden_states: torch.Tensor,
    *,
    variant_id: str,
    attention_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Normalize and run self-attention, returning only the residual update.

    Args:
        sublayers: Modules to run; may belong to a different layer than the caller.
        hidden_states: Input to normalize, shape ``[sequence, batch, hidden]``.
        variant_id: Variant identifier used in error messages.
        attention_kwargs: Stock attention keyword arguments, threaded unchanged.

    Returns:
        The attention output.

    Raises:
        ValueError: If the norm returns a tuple or attention produces a bias.
    """
    normed = _single_tensor(
        apply_module(sublayers.attention_norm)(hidden_states),
        variant_id=variant_id,
        what="an attention norm",
    )
    output, bias = sublayers.attention(normed, **attention_kwargs)
    if bias is not None:
        raise ValueError(f"Model surgery variant {variant_id} requires bias-free attention output")
    return output


def run_mlp(
    sublayers: DecoderSublayers,
    hidden_states: torch.Tensor,
    *,
    variant_id: str,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Normalize and run the MLP or MoE, returning only the residual update.

    Args:
        sublayers: Modules to run; may belong to a different layer than the caller.
        hidden_states: Input to normalize, shape ``[sequence, batch, hidden]``.
        variant_id: Variant identifier used in error messages.
        padding_mask: Padding mask used by MoE routing losses.

    Returns:
        The MLP output.

    Raises:
        ValueError: If the norm returns a tuple or the MLP produces a bias.
    """
    normed = _single_tensor(
        apply_module(sublayers.mlp_norm)(hidden_states),
        variant_id=variant_id,
        what="an MLP norm",
    )
    output, bias = apply_module(sublayers.mlp)(normed, padding_mask=padding_mask)
    if bias is not None:
        raise ValueError(f"Model surgery variant {variant_id} requires a bias-free MLP output")
    return output


def check_layer_support(layer: torch.nn.Module, *, variant_id: str) -> None:
    """Reject execution features a rewritten layer has not been validated against.

    These are instance attributes derived by ``TransformerLayer.__init__`` from
    the config, so they can only be checked once the layer exists.

    Args:
        layer: The constructed layer.
        variant_id: Variant identifier used in the error message.

    Raises:
        ValueError: If the layer enables an unsupported feature.
    """
    unsupported: list[str] = []
    if layer.hidden_dropout != 0.0:
        unsupported.append("hidden_dropout")
    if not isinstance(layer.cross_attention, IdentityOp):
        unsupported.append("cross_attention")
    if layer.recompute_input_layernorm or layer.recompute_pre_mlp_layernorm or layer.recompute_mlp:
        unsupported.append("selective_recompute")
    if layer.offload_attn_norm or layer.offload_mlp_norm:
        unsupported.append("activation_offloading")
    if unsupported:
        raise ValueError(
            f"Model surgery variant {variant_id} has not validated these execution features: " + ", ".join(unsupported)
        )
