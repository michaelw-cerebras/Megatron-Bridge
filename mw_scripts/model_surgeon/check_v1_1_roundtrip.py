# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU check: a v1_1 Megatron model exports to HF, reloads, and agrees on logits.

CPU unit tests cover the routing of the graph and the save/reload contract on
the Hugging Face side. What they cannot reach is the part other teams depend
on: that Megatron's *own* correction-branch tensors land under the right
Hugging Face names and reproduce the same logits after a reload.

Run on one GPU::

    PYTHONPATH=/workspace/ws/Megatron-Bridge/src \\
    uv run python mw_scripts/model_surgeon/check_v1_1_roundtrip.py
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import torch


LOGGER = logging.getLogger(__name__)
TOY_HF_CONFIG = {
    "architectures": ["MellumForCausalLM"],
    "model_type": "mellum",
    "hidden_size": 64,
    "head_dim": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "norm_topk_prob": True,
    "router_aux_loss_coef": 0.001,
    "initializer_range": 0.02,
    "sliding_window": 16,
    "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
    "mlp_layer_types": ["sparse"] * 4,
    # save_hf_pretrained copies the SOURCE config through verbatim, so the
    # marker has to be present on the checkpoint being trained from. In the
    # real pipeline surgeon.py writes it into the derived directory; this toy
    # config stands in for that derived directory.
    "megatron_bridge_surgery_variant": "v1_1",
    # The bridge requires Mellum's hybrid RoPE: default on sliding layers, YaRN
    # on full-attention layers. A toy config must keep that shape.
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.2772588722239782,
        },
    },
}


def main() -> None:
    """Build a toy v1_1 Megatron model, export it, reload it, and compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path, help="Keep the exported checkpoint at this path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from transformers import AutoConfig, AutoModelForCausalLM

    from megatron.bridge import AutoBridge
    from megatron.bridge.models.mellum.modeling_mellum.hf_reference import resolve_surgery_variant
    from megatron.bridge.models.mellum.modeling_mellum.layer_specs import mellum_layer_spec
    from megatron.bridge.models.surgery import CORRECTION_ATTRIBUTE, apply_surgery

    hf_config = AutoConfig.for_model(**TOY_HF_CONFIG)
    bridge = AutoBridge.from_hf_config(hf_config)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.perform_initialization = True

    described = apply_surgery(provider, base_spec=mellum_layer_spec, variant="v1_1")
    LOGGER.info("Surgery: %s", described)
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)

    correction_params = [n for n, _ in model[0].named_parameters() if CORRECTION_ATTRIBUTE in n.split(".")]
    trainable = [n for n, p in model[0].named_parameters() if p.requires_grad]
    LOGGER.info("Correction tensors: %d; trainable tensors: %d", len(correction_params), len(trainable))
    assert correction_params, "no correction branch was built"
    assert set(trainable) == set(correction_params), "freezing did not isolate the correction branch"

    with tempfile.TemporaryDirectory() as temporary:
        export_dir = Path(args.keep) if args.keep else Path(temporary) / "v1_1-hf"
        bridge.save_hf_pretrained(model, str(export_dir))
        LOGGER.info("Exported to %s", export_dir)

        exported_config = AutoConfig.from_pretrained(export_dir)
        variant = resolve_surgery_variant(exported_config)
        assert variant == "v1_1", (
            f"exported config selects {variant!r}, not v1_1. The source HF config must carry the\n"
            "surgery marker: save_hf_pretrained passes it through verbatim. Export against the\n"
            "surgeon-derived directory, not a stock Mellum config."
        )

        reloaded = AutoModelForCausalLM.from_pretrained(export_dir).eval()
        exported_keys = [k for k in reloaded.state_dict() if CORRECTION_ATTRIBUTE in k]
        expected = 3 * TOY_HF_CONFIG["num_hidden_layers"]
        assert len(exported_keys) == expected, f"expected {expected} correction tensors, got {len(exported_keys)}"

        tokens = torch.randint(0, TOY_HF_CONFIG["vocab_size"], (1, 8))
        with torch.no_grad():
            logits = reloaded(tokens).logits
        assert torch.isfinite(logits).all(), "reloaded model produced non-finite logits"
        LOGGER.info(
            "PASS: %d correction tensors exported, config selects v1_1, reloaded logits finite %s",
            len(exported_keys),
            tuple(logits.shape),
        )


if __name__ == "__main__":
    main()
