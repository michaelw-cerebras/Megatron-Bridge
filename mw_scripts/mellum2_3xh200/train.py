# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continue pretraining a loaded Mellum checkpoint using a validated YAML."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

from configuration import (
    MixtureSnapshot,
    TrainingSettings,
    config_digest,
    prepare_mixture,
    read_config,
    validate_parallelism,
    validate_snapshot_files,
    write_json_atomic,
)
from wandb_logging import apply_wandb_config, check_wandb_access, prepare_wandb_environment


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


LOGGER = logging.getLogger(__name__)


def build_config(settings: TrainingSettings, mixture: MixtureSnapshot, *, world_size: int) -> "ConfigContainer":
    """Construct a BF16 continued-pretraining recipe with explicit resume semantics."""
    from megatron.bridge.recipes.mellum import mellum_pretrain_config

    resume = settings["run"]["resume_checkpoint"]
    if resume is not None and not (Path(resume) / "latest_checkpointed_iteration.txt").is_file():
        raise ValueError("resume_checkpoint must be a Megatron checkpoint root with its iteration tracker")
    if mixture["config_digest"] != config_digest(settings):
        raise ValueError("Mixture snapshot is stale; rerun --prepare after changing YAML")
    if (
        resume is None
        and (Path(settings["run"]["directory"]) / "checkpoints/latest_checkpointed_iteration.txt").exists()
    ):
        raise ValueError("Existing checkpoints require explicit resume_checkpoint or a new run.directory")
    cfg = mellum_pretrain_config(mixture["hf_snapshot"], load_weights=resume is None)
    cfg.dist.distributed_timeout_minutes = 180
    if settings["dataset"]["eod_id"] >= cfg.model.vocab_size:
        raise ValueError("Dataset EOD is outside the model vocabulary")
    accum = validate_parallelism(settings, world_size=world_size, experts=cfg.model.num_moe_experts)
    for name, value in settings["parallelism"].items():
        if not hasattr(cfg.model, name):
            raise ValueError(f"Unknown model parallelism field: {name}")
        setattr(cfg.model, name, value)
    cfg.model.sequence_parallel = cfg.model.tensor_model_parallel_size > 1
    cfg.train.global_batch_size = settings["train"]["global_batch_size"]
    cfg.train.micro_batch_size = settings["train"]["micro_batch_size"]
    seq = settings["train"]["sequence_length"]
    if seq > cfg.model.seq_length:
        raise ValueError("Training sequence exceeds the checkpoint context length")
    cfg.model.seq_length = seq
    cfg.dataset.sequence_length = seq
    cfg.train.train_iters = math.ceil(settings["train"]["train_tokens"] / (seq * cfg.train.global_batch_size))
    cfg.train.train_samples = None
    cfg.train.exit_interval = settings["run"]["exit_interval"]
    cfg.rng.seed = settings["train"]["seed"]
    cfg.dataset.random_seed = settings["train"]["seed"]
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.null_tokenizer_eod_id = settings["dataset"]["eod_id"]
    cfg.tokenizer.null_tokenizer_pad_id = settings["dataset"]["eod_id"]
    cfg.dataset.data_path = [item for shard in mixture["shards"] for item in (str(shard["weight"]), shard["prefix"])]
    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.split = "1,0,0"
    # The attention backend supplies the causal mask. An unused S x S mask
    # cached per shard in every DataLoader worker can exhaust host RAM.
    cfg.dataset.create_attention_mask = False
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.fast_cache_load = False
    cfg.dataset.defer_npy_index_mmap = False
    cfg.dataset.num_workers = settings["dataset"]["num_workers"]
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0
    for name, value in settings["optimizer"].items():
        if not hasattr(cfg.optimizer, name):
            raise ValueError(f"Unknown optimizer field: {name}")
        setattr(cfg.optimizer, name, value)
    cfg.scheduler.lr_decay_style = settings["scheduler"]["lr_decay_style"]
    cfg.scheduler.lr_warmup_iters = settings["scheduler"]["lr_warmup_iters"]
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.start_weight_decay = cfg.optimizer.weight_decay
    cfg.scheduler.end_weight_decay = cfg.optimizer.weight_decay
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.scheduler.use_checkpoint_opt_param_scheduler = resume is not None
    directory = Path(settings["run"]["directory"])
    cfg.checkpoint.save = str(directory / "checkpoints")
    cfg.checkpoint.load = resume
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.load_optim = resume is not None
    cfg.checkpoint.load_rng = resume is not None
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.finetune = False
    cfg.checkpoint.save_interval = settings["run"]["save_interval"]
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.async_save = False
    cfg.checkpoint.fully_parallel_save = True
    cfg.dataset.path_to_cache = str(directory / "dataset_cache")
    cfg.logger.tensorboard_dir = str(directory / "tensorboard")
    cfg.logger.log_interval = settings["run"]["log_interval"]
    cfg.logger.log_throughput = True
    cfg.logger.log_throughput_to_tensorboard = True
    cfg.logger.throughput_window_size = 20
    cfg.logger.save_config_filepath = str(directory / "metadata" / "ConfigContainer.yaml")
    apply_wandb_config(cfg.logger, settings)
    LOGGER.info(
        "world=%d accumulation=%d train_iters=%d scheduled_tokens=%d HF_load=%s",
        world_size,
        accum,
        cfg.train.train_iters,
        cfg.train.train_iters * seq * cfg.train.global_batch_size,
        resume is None,
    )
    return cfg


def main() -> None:
    """Prepare metadata, validate configuration, or enter distributed training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch-info", action="store_true")
    mode.add_argument("--wandb-check", action="store_true")
    parser.add_argument("--world-size", type=int, help="Explicit GPU count for dry-run; training uses WORLD_SIZE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = read_config(args.config)
    directory = Path(settings["run"]["directory"])
    snapshot = directory / "metadata" / "resolved_mixture.json"
    if args.launch_info:
        # Shell reads exactly one machine-readable line, never evals its contents.
        import sys

        sys.stdout.write(str(directory) + "\n")
        return
    if args.wandb_check:
        check_wandb_access(settings)
        return
    if args.prepare:
        from huggingface_hub import snapshot_download

        import megatron.bridge.models.mellum  # noqa: F401 -- registers HF compatibility config

        if (
            settings["run"]["resume_checkpoint"] is None
            and (directory / "checkpoints" / "latest_checkpointed_iteration.txt").exists()
        ):
            raise ValueError("Checkpoints already exist; set resume_checkpoint or choose a new run.directory")
        if snapshot.exists():
            mixture = json.loads(snapshot.read_text())
            if mixture["config_digest"] != config_digest(settings):
                raise ValueError("Run already has a different immutable mixture/config; choose a new run.directory")
            validate_snapshot_files(mixture)
        else:
            if not Path(settings["model"]["hf_path"]).is_dir():
                hf_snapshot = snapshot_download(
                    settings["model"]["hf_path"], allow_patterns=["*.json", "*.safetensors", "*.jinja"]
                )
            else:
                hf_snapshot = str(Path(settings["model"]["hf_path"]).resolve())
            mixture = prepare_mixture(settings, hf_snapshot=hf_snapshot)
        write_json_atomic(snapshot, mixture)
        LOGGER.info("Resolved %d shards into %s", len(mixture["shards"]), snapshot)
        return
    if not snapshot.is_file():
        raise ValueError("No prepared data snapshot; complete launch.sh --prepare before --dry-run or --train")
    mixture = json.loads(snapshot.read_text())
    validate_snapshot_files(mixture)
    if mixture.get("reserved_ids_not_checked") and int(os.environ.get("RANK", "0")) == 0:
        LOGGER.warning(
            "This run skipped the renamed reserved-ID scan for %d tokenizer manifests; absence is UNVERIFIED",
            len(mixture["reserved_ids_not_checked"]),
        )
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else args.world_size
    if world_size is None:
        parser.error("Use launch.sh or supply --world-size for dry-run; distributed training sets WORLD_SIZE")
    cfg = build_config(settings, mixture, world_size=world_size)
    if args.dry_run:
        cfg.to_yaml(directory / "metadata" / "ConfigContainer.yaml")
        return
    prepare_wandb_environment(settings, rank=int(os.environ["RANK"]), world_size=world_size)
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
