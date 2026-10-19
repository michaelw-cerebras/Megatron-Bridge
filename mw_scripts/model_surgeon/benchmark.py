# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a checkpoint-free single-node Mellum parallel-architecture benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent / "mellum2_3xh200"
sys.path.insert(0, str(PIPELINE_DIR))

from configuration import config_digest, validate_snapshot_files  # noqa: E402
from train import build_config  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mixture-snapshot", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--variant", default="v1", help="Surgery variant id recorded in --model-dir's manifest")
    parser.add_argument(
        "--data-shards",
        type=int,
        default=8,
        help="Evenly sampled real-data shards used by every performance candidate",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, help="Optional peak learning-rate override")
    parser.add_argument("--min-lr", type=float, help="Optional minimum learning-rate override")
    parser.add_argument("--lr-warmup-iters", type=int, help="Optional warmup-iteration override")
    parser.add_argument("--precision", choices=("bf16", "fp8-current"), default="bf16")
    parser.add_argument("--dispatcher", choices=("alltoall", "deepep"), default="alltoall")
    parser.add_argument("--moe-permute-fusion", action="store_true")
    parser.add_argument("--ep-overlap", action="store_true")
    parser.add_argument("--dp-overlap", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build a controlled benchmark config and enter distributed training."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size != 8:
        raise ValueError(f"This benchmark requires exactly one 8-GPU node, got WORLD_SIZE={world_size}")
    if args.iterations < 2:
        raise ValueError("--iterations must be at least 2")
    if args.data_shards < 1:
        raise ValueError("--data-shards must be positive")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.min_lr is not None and args.min_lr < 0:
        raise ValueError("--min-lr must be non-negative")
    if args.lr_warmup_iters is not None and args.lr_warmup_iters < 0:
        raise ValueError("--lr-warmup-iters must be non-negative")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    if not args.mixture_snapshot.is_file():
        raise FileNotFoundError(args.mixture_snapshot)
    if args.ep_overlap and args.ep == 1:
        raise ValueError("EP overlap requires EP > 1")
    if args.ep_overlap and args.variant in {"v2", "v3", "v3_1", "v3_2", "v3_3"}:
        raise ValueError(f"{args.variant} executes MoE blocks across layer boundaries and forbids EP overlap")

    mixture = json.loads(args.mixture_snapshot.read_text())
    validate_snapshot_files(mixture)
    all_shards = mixture["shards"]
    if args.data_shards > len(all_shards):
        raise ValueError(f"Requested {args.data_shards} data shards, but only {len(all_shards)} are available")
    # Spread selection across the resolved mixture rather than favoring one
    # adjacent dataset family. Equal benchmark weights make this deterministic.
    selected_indices = [index * len(all_shards) // args.data_shards for index in range(args.data_shards)]
    mixture["shards"] = [copy.deepcopy(all_shards[index]) for index in selected_indices]
    for shard in mixture["shards"]:
        shard["weight"] = 1.0 / args.data_shards
    settings = copy.deepcopy(mixture["config"])
    settings.pop("wandb", None)
    # The derived directory's manifest decides the graph; this only has to name the
    # same variant so train.py's config/checkpoint agreement check passes.
    settings["model"] = {"hf_path": str(args.model_dir.resolve()), "architecture": args.variant}
    settings["train"]["micro_batch_size"] = args.micro_batch_size
    if args.lr is not None:
        settings["optimizer"]["lr"] = args.lr
    if args.min_lr is not None:
        settings["optimizer"]["min_lr"] = args.min_lr
    if args.lr_warmup_iters is not None:
        settings["scheduler"]["lr_warmup_iters"] = args.lr_warmup_iters
    settings["parallelism"] = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": args.cp,
        "expert_model_parallel_size": args.ep,
        "expert_tensor_parallel_size": args.etp,
    }
    settings["run"] = {
        "directory": str(args.run_dir.resolve()),
        "save_interval": 1_000_000,
        "keep_last_checkpoints": 1,
        "log_interval": 1,
        "resume_checkpoint": None,
        "exit_interval": args.iterations,
    }
    mixture["hf_snapshot"] = str(args.model_dir.resolve())
    mixture["config_digest"] = config_digest(settings)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "metadata").mkdir(exist_ok=True)
    cfg = build_config(settings, mixture, world_size=world_size)
    args.dataset_cache.mkdir(parents=True, exist_ok=True)
    cfg.dataset.path_to_cache = str(args.dataset_cache.resolve())
    cfg.checkpoint.save = None
    cfg.checkpoint.save_interval = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.logger.log_interval = 1
    cfg.logger.throughput_window_size = 3

    if args.precision == "fp8-current":
        cfg.mixed_precision = "bf16_with_fp8_current_scaling_mixed"
        cfg.model.moe_router_padding_for_fp8 = True
    else:
        cfg.mixed_precision = "bf16_mixed"
        cfg.model.moe_router_padding_for_fp8 = False

    if args.dispatcher == "deepep":
        from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend

        apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend="deepep")
    else:
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_permute_fusion = args.moe_permute_fusion

    if args.ep_overlap or args.dp_overlap:
        from megatron.bridge.training.comm_overlap import CommOverlapConfig

        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
            overlap_grad_reduce=True if args.dp_overlap else None,
            overlap_param_gather=True if args.dp_overlap else None,
            overlap_moe_expert_parallel_comm=args.ep_overlap,
            delay_wgrad_compute=False,
        )
        cfg.model.moe_shared_expert_overlap = False

    benchmark = {
        "model_dir": str(args.model_dir.resolve()),
        "variant": args.variant,
        "world_size": world_size,
        "iterations": args.iterations,
        "data_shards": args.data_shards,
        "data_prefixes": [shard["prefix"] for shard in mixture["shards"]],
        "global_batch_size": cfg.train.global_batch_size,
        "sequence_length": cfg.model.seq_length,
        "tp": args.tp,
        "cp": args.cp,
        "ep": args.ep,
        "etp": args.etp,
        "micro_batch_size": args.micro_batch_size,
        "lr": settings["optimizer"]["lr"],
        "min_lr": settings["optimizer"]["min_lr"],
        "lr_warmup_iters": settings["scheduler"]["lr_warmup_iters"],
        "precision": args.precision,
        "dispatcher": args.dispatcher,
        "moe_permute_fusion": args.moe_permute_fusion,
        "ep_overlap": args.ep_overlap,
        "dp_overlap": args.dp_overlap,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        (args.run_dir / "benchmark_config.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
        LOGGER.info("Benchmark configuration: %s", benchmark)

    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
