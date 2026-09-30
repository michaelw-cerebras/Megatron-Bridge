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

"""Continue GPT-OSS 20B pretraining on a resolved CodeSign mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_ROOT = REPO_ROOT / "scripts" / "performance"
if str(PERFORMANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PERFORMANCE_ROOT))

from utils.utils import get_perf_optimized_recipe  # noqa: E402

from megatron.bridge.training.gpt_step import forward_step  # noqa: E402
from megatron.bridge.training.pretrain import pretrain  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse continual-pretraining arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True, type=Path)
    checkpoint_source = parser.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument("--pretrained-checkpoint", type=Path)
    checkpoint_source.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--data-args-path", required=True, type=Path)
    parser.add_argument("--resolved-mixture", required=True, type=Path)
    parser.add_argument("--train-tokens", required=True, type=_positive_int)
    parser.add_argument("--seq-length", type=_positive_int, default=8192)
    parser.add_argument("--micro-batch-size", type=_positive_int, default=1)
    parser.add_argument("--global-batch-size", type=_positive_int, default=128)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=2)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--min-lr", type=float, default=2e-7)
    parser.add_argument("--lr-warmup-iters", type=_nonnegative_int, default=1000)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--save-interval", type=_positive_int, default=1000)
    parser.add_argument("--keep-last-checkpoints", type=_positive_int, default=2)
    parser.add_argument("--tensorboard-dir", required=True, type=Path)
    parser.add_argument("--log-interval", type=_positive_int, default=10)
    parser.add_argument("--config-output", required=True, type=Path)
    parser.add_argument("--enable-nccl-ub", action="store_true")
    parser.add_argument(
        "--disable-symmetric-nccl-ub-registration",
        action="store_true",
        help="Use conventional local NCCL user-buffer registration instead of NCCL windows.",
    )
    parser.add_argument("--dryrun", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_data_args(
    hf_model: Path,
    data_args_path: Path,
    resolved_mixture_path: Path,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Validate tokenizer identity and weighted Megatron data prefixes."""
    data_args_path = data_args_path.expanduser().resolve()
    resolved_mixture_path = resolved_mixture_path.expanduser().resolve()
    if not data_args_path.is_file():
        raise FileNotFoundError(data_args_path)
    if not resolved_mixture_path.is_file():
        raise FileNotFoundError(resolved_mixture_path)
    resolved = _read_json(resolved_mixture_path)
    tokenizer = resolved.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError(f"Missing tokenizer metadata in {resolved_mixture_path}")
    tokenizer_json = hf_model / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    expected_digest = tokenizer.get("json_sha256")
    actual_digest = _sha256(tokenizer_json)
    if expected_digest != actual_digest:
        raise ValueError(
            f"Tokenized data used tokenizer SHA-256 {expected_digest}; {tokenizer_json} has {actual_digest}"
        )

    expected_data_args_digest = resolved.get("data_args_sha256")
    actual_data_args_digest = _sha256(data_args_path)
    if expected_data_args_digest != actual_data_args_digest:
        raise ValueError(
            f"Data args SHA-256 {actual_data_args_digest} does not match resolved mixture {expected_data_args_digest}"
        )

    data_args = data_args_path.read_text(encoding="utf-8").split()
    if not data_args or len(data_args) % 2:
        raise ValueError(f"Expected weight/prefix pairs in {data_args_path}")
    weights: list[float] = []
    prefixes: list[str] = []
    for index in range(0, len(data_args), 2):
        raw_weight = data_args[index]
        prefix = data_args[index + 1]
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"Invalid blend weight {raw_weight!r} in {data_args_path}") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Blend weights must be finite and positive, got {raw_weight!r}")
        for suffix in (".bin", ".idx"):
            dataset_file = Path(f"{prefix}{suffix}")
            if not dataset_file.is_file():
                raise FileNotFoundError(dataset_file)
        weights.append(weight)
        prefixes.append(prefix)
    if len(prefixes) != len(set(prefixes)):
        raise ValueError(f"Duplicate dataset prefixes found in {data_args_path}")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Blend weights sum to {sum(weights):.17g}, expected 1.0")
    return data_args, resolved, tokenizer


def _validate_parallelism(cfg: Any, *, micro_batch_size: int, global_batch_size: int) -> None:
    """Validate recipe parallel dimensions against the launched world size."""
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    tp = cfg.model.tensor_model_parallel_size or 1
    pp = cfg.model.pipeline_model_parallel_size or 1
    cp = cfg.model.context_parallel_size or 1
    ep = cfg.model.expert_model_parallel_size or 1
    model_parallel = tp * pp * cp
    if world_size % model_parallel:
        raise ValueError(f"WORLD_SIZE={world_size} must be divisible by TP*PP*CP={model_parallel}")
    data_parallel = world_size // model_parallel
    if data_parallel % ep:
        raise ValueError(f"DP={data_parallel} must be divisible by EP={ep}")
    if global_batch_size % (micro_batch_size * data_parallel):
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"micro_batch_size*DP={micro_batch_size * data_parallel}"
        )


def build_config(args: argparse.Namespace) -> Any:
    """Compose CodeSign continual-pretraining settings onto the B200 recipe."""
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("--min-lr must be nonnegative and must not exceed --lr")
    if args.disable_symmetric_nccl_ub_registration and not args.enable_nccl_ub:
        raise ValueError("--disable-symmetric-nccl-ub-registration requires --enable-nccl-ub")

    hf_model = args.hf_model.expanduser().resolve()
    hf_config_path = hf_model / "config.json"
    if not hf_config_path.is_file():
        raise FileNotFoundError(hf_config_path)
    hf_config = _read_json(hf_config_path)
    vocab_size = int(hf_config["vocab_size"])
    eos_id = int(hf_config["eos_token_id"])
    pad_id = int(hf_config["pad_token_id"])

    data_args, resolved_mixture, tokenizer = _load_data_args(
        hf_model,
        args.data_args_path,
        args.resolved_mixture,
    )
    if resolved_mixture.get("train_tokens") != args.train_tokens:
        raise ValueError(
            f"Resolved mixture was built for {resolved_mixture.get('train_tokens')} tokens, "
            f"but training requested {args.train_tokens}"
        )
    if int(tokenizer["eos_id"]) != eos_id:
        raise ValueError(f"Dataset EOD id {tokenizer['eos_id']} does not match model EOS id {eos_id}")
    if max(eos_id, pad_id) >= vocab_size:
        raise ValueError(f"Special-token ids must be below vocab_size={vocab_size}")

    tokens_per_iteration = args.seq_length * args.global_batch_size
    train_iters = math.ceil(args.train_tokens / tokens_per_iteration)
    scheduled_tokens = train_iters * tokens_per_iteration
    if args.lr_warmup_iters >= train_iters:
        raise ValueError("--lr-warmup-iters must be smaller than the computed train iterations")

    os.environ["GPT_OSS_20B_HF_MODEL"] = str(hf_model)
    cfg = get_perf_optimized_recipe(
        model_family_name="gpt_oss",
        model_recipe_name="gpt_oss_20b",
        train_task="pretrain",
        gpu="b200",
        compute_dtype="fp8_mx",
        mock=False,
        config_variant="v1",
    )
    _validate_parallelism(cfg, micro_batch_size=args.micro_batch_size, global_batch_size=args.global_batch_size)

    cfg.model.vocab_size = vocab_size
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = vocab_size
    cfg.tokenizer.null_tokenizer_eod_id = eos_id
    cfg.tokenizer.null_tokenizer_pad_id = pad_id

    cfg.model.seq_length = args.seq_length
    cfg.dataset.sequence_length = args.seq_length
    cfg.dataset.data_path = data_args
    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.split = "1,0,0"
    cfg.dataset.fast_cache_load = False
    cfg.dataset.defer_npy_index_mmap = False
    cfg.dataset.num_workers = args.num_workers
    cfg.dataset.num_dataset_builder_threads = max(1, args.num_workers)
    if args.dataset_cache_dir is not None:
        cfg.dataset.path_to_cache = str(args.dataset_cache_dir.expanduser().resolve())

    cfg.train.train_iters = train_iters
    cfg.train.train_samples = None
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0

    cfg.optimizer.lr = args.lr
    cfg.optimizer.min_lr = args.min_lr
    cfg.scheduler.lr_warmup_iters = args.lr_warmup_iters
    cfg.scheduler.lr_decay_iters = train_iters
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.override_opt_param_scheduler = True

    tensorboard_dir = args.tensorboard_dir.expanduser().resolve()
    config_output = args.config_output.expanduser().resolve()
    cfg.logger.log_interval = args.log_interval
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = str(tensorboard_dir)
    cfg.logger.tensorboard_log_interval = args.log_interval
    cfg.logger.log_timers_to_tensorboard = True
    cfg.logger.log_throughput_to_tensorboard = True
    cfg.logger.save_config_filepath = str(config_output)

    save_dir = args.save_dir.expanduser().resolve()
    cfg.checkpoint.save = str(save_dir)
    cfg.checkpoint.save_interval = args.save_interval
    cfg.checkpoint.most_recent_k = args.keep_last_checkpoints
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_load = True
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.async_save = False
    if args.pretrained_checkpoint is not None:
        pretrained_checkpoint = args.pretrained_checkpoint.expanduser().resolve()
        if not pretrained_checkpoint.is_dir():
            raise FileNotFoundError(pretrained_checkpoint)
        cfg.checkpoint.pretrained_checkpoint = str(pretrained_checkpoint)
        cfg.checkpoint.load = None
        cfg.checkpoint.load_optim = False
        cfg.checkpoint.load_rng = False
        cfg.checkpoint.finetune = True
        mode = "initialize-from-pretrained"
    else:
        if args.resume_checkpoint is None:
            raise AssertionError("A checkpoint source is required")
        resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(resume_checkpoint)
        cfg.checkpoint.pretrained_checkpoint = None
        cfg.checkpoint.load = str(resume_checkpoint)
        cfg.checkpoint.load_optim = True
        cfg.checkpoint.load_rng = True
        cfg.checkpoint.finetune = False
        mode = "full-state-resume"

    cfg.ddp.nccl_ub = args.enable_nccl_ub
    cfg.ddp.disable_symmetric_registration = args.disable_symmetric_nccl_ub_registration

    if int(os.environ.get("RANK", "0")) == 0:
        LOGGER.info(
            "Training plan: mode=%s requested_tokens=%d scheduled_tokens=%d train_iters=%d "
            "tokens_per_iteration=%d sources=%s weighted_shards=%d",
            mode,
            args.train_tokens,
            scheduled_tokens,
            train_iters,
            tokens_per_iteration,
            resolved_mixture.get("source_count"),
            len(data_args) // 2,
        )
        LOGGER.info(
            "System preset: gpt_oss_20b/b200/fp8_mx/v1 TP=%s PP=%s CP=%s EP=%s",
            cfg.model.tensor_model_parallel_size,
            cfg.model.pipeline_model_parallel_size,
            cfg.model.context_parallel_size,
            cfg.model.expert_model_parallel_size,
        )
        LOGGER.info(
            "Tokenizer: vocab=%d eod=%d pad=%d sha256=%s", vocab_size, eos_id, pad_id, tokenizer["json_sha256"]
        )
        LOGGER.info("Checkpoint directory: %s; retaining latest %d", save_dir, args.keep_last_checkpoints)
        LOGGER.info("TensorBoard directory: %s", tensorboard_dir)
    return cfg


def main() -> None:
    """Build the combined recipe and launch training."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = build_config(args)
    config_output = args.config_output.expanduser().resolve()
    config_output.parent.mkdir(parents=True, exist_ok=True)
    if args.dryrun:
        cfg.to_yaml(config_output)
        LOGGER.info("Dry-run ConfigContainer saved to %s", config_output)
        return
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
