# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base recipe shared by original Mellum checkpoint experiments."""

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer


def mellum_pretrain_config(hf_model_id: str, *, load_weights: bool) -> ConfigContainer:
    """Build a BF16 recipe from a supported original Mellum checkpoint.

    Args:
        hf_model_id: Local HF checkpoint directory or model repository ID.
        load_weights: Load HF weights on first model construction. Set false
            when restoring a full Megatron training checkpoint instead.

    Returns:
        Base configuration; the caller supplies data, batch sizes, training
        duration, optimizer settings, parallelism, and output directories.
    """
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(hf_model_id).to_megatron_provider(load_weights=load_weights)
    cfg.mixed_precision = "bf16_mixed"
    cfg.model.calculate_per_token_loss = True
    cfg.model.cuda_graph_impl = "none"
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.nccl_ub = False
    return cfg
