import argparse
import inspect
import json
from pathlib import Path
from unittest.mock import patch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.gpt_oss import gpt_oss as gpt_oss_recipe
from megatron.bridge.training.callbacks import CallbackContext, CallbackManager
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


GPT_OSS_TOKENIZER = "openai/gpt-oss-20b"


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


def _build_gpt_oss_20b_config(hf_model: str) -> ConfigContainer:
    """Build an offline GPT-OSS config across old and new recipe APIs."""
    recipe = gpt_oss_recipe.gpt_oss_20b_pretrain_config
    if "hf_model_id" in inspect.signature(recipe).parameters:
        return recipe(hf_model_id=hf_model)

    # Megatron Bridge v0.5.0 hard-codes the online model ID in its no-argument
    # recipe. Redirect only that lookup to the already available local config.
    local_bridge = AutoBridge.from_hf_pretrained(hf_model)
    with patch.object(gpt_oss_recipe.AutoBridge, "from_hf_pretrained", return_value=local_bridge):
        return recipe()


def parse_args() -> argparse.Namespace:
    """Parse GPT-OSS continual-pretraining arguments."""
    parser = argparse.ArgumentParser(description="Continue pretraining GPT-OSS 20B on Megatron indexed data.")
    parser.add_argument(
        "--hf-model",
        required=True,
        help="Local Hugging Face directory containing the GPT-OSS config.json.",
    )
    checkpoint_source = parser.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        help="Imported model-only Megatron checkpoint used for a fresh continual-pretraining run.",
    )
    checkpoint_source.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Training checkpoint directory used to restore model, optimizer, RNG, and iteration state.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Preprocessing output root containing manifest.json, data_prefixes.txt, and data/.",
    )
    parser.add_argument("--train-iters", type=_positive_int, default=1)
    parser.add_argument("--seq-length", type=_positive_int, default=512)
    parser.add_argument("--global-batch-size", type=_positive_int, default=4)
    parser.add_argument("--micro-batch-size", type=_positive_int, default=1)
    parser.add_argument("--tensor-parallel", type=_positive_int, default=2)
    parser.add_argument("--pipeline-parallel", type=_positive_int, default=1)
    parser.add_argument("--expert-parallel", type=_positive_int, default=8)
    parser.add_argument("--expert-tensor-parallel", type=_positive_int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-dataset-shards",
        type=_positive_int,
        help="Use only the first N dataset shards; useful for short smoke tests.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-iters", type=_nonnegative_int, default=0)
    parser.add_argument("--eval-interval", type=_positive_int)
    parser.add_argument("--eval-iters", type=_positive_int, default=20)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the loaded checkpoint at iteration 0.",
    )
    parser.add_argument("--log-interval", type=_positive_int, default=1)
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        help="Optional TensorBoard event directory. Prefer node-local storage for long runs.",
    )
    parser.add_argument(
        "--no-full-recompute",
        action="store_true",
        help="Disable full activation recomputation. Full recomputation is enabled by default.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Optional output directory for continual-pretraining checkpoints.",
    )
    parser.add_argument(
        "--save-interval",
        type=_positive_int,
        help="Checkpoint interval. Defaults to train-iters when --save-dir is provided.",
    )
    return parser.parse_args()


def load_dataset(dataset_root: Path, max_shards: int | None) -> tuple[list[str], int]:
    """Validate Megatron dataset prefixes and return them with the tokenizer EOD id."""
    dataset_root = dataset_root.expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    prefixes_path = dataset_root / "data_prefixes.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not prefixes_path.is_file():
        raise FileNotFoundError(prefixes_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = manifest["tokenizer"]
    if tokenizer["id"] != GPT_OSS_TOKENIZER:
        raise ValueError(
            f"Dataset used tokenizer {tokenizer['id']!r}; expected {GPT_OSS_TOKENIZER!r} for GPT-OSS training."
        )

    prefixes = [
        str((dataset_root / line.strip()).resolve())
        for line in prefixes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_shards is not None:
        prefixes = prefixes[:max_shards]
    if not prefixes:
        raise ValueError(f"No dataset prefixes found in {prefixes_path}")

    for prefix in prefixes:
        for suffix in (".bin", ".idx"):
            dataset_file = Path(f"{prefix}{suffix}")
            if not dataset_file.is_file():
                raise FileNotFoundError(dataset_file)

    return prefixes, int(tokenizer["eos_id"])


def _write_eval_only_metrics_to_tensorboard(context: CallbackContext) -> None:
    """Write eval-only validation metrics that Bridge otherwise omits from TensorBoard."""
    writer = context.state.tensorboard_logger
    if writer is None or context.total_loss_dict is None:
        return

    for key, value in context.total_loss_dict.items():
        writer.add_scalar(f"{key} validation", value.item(), context.state.train_state.step)
        writer.add_scalar(
            f"{key} validation vs samples",
            value.item(),
            context.state.train_state.consumed_train_samples,
        )
    writer.flush()


def main() -> None:
    """Load GPT-OSS 20B weights and continue pretraining on indexed data."""
    args = parse_args()
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must not exceed --lr")
    if args.lr_warmup_iters >= args.train_iters:
        raise ValueError("--lr-warmup-iters must be smaller than --train-iters")

    pretrained_checkpoint = None
    if args.pretrained_checkpoint is not None:
        pretrained_checkpoint = args.pretrained_checkpoint.expanduser().resolve()
        if not pretrained_checkpoint.is_dir():
            raise FileNotFoundError(pretrained_checkpoint)

    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(resume_checkpoint)
        tracker = resume_checkpoint / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise FileNotFoundError(f"Resume checkpoint has no tracker file: {tracker}")

    cfg = _build_gpt_oss_20b_config(args.hf_model)

    # Full GPT-OSS 20B architecture from config.json with single-node MoE parallelism.
    cfg.model.tensor_model_parallel_size = args.tensor_parallel
    cfg.model.pipeline_model_parallel_size = args.pipeline_parallel
    cfg.model.expert_model_parallel_size = args.expert_parallel
    cfg.model.expert_tensor_parallel_size = args.expert_tensor_parallel
    cfg.model.sequence_parallel = args.tensor_parallel > 1
    cfg.model.seq_length = args.seq_length

    if not args.no_full_recompute:
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = "uniform"
        cfg.model.recompute_num_layers = 1

    prefixes, eos_id = load_dataset(args.dataset_root, args.max_dataset_shards)
    cfg.dataset.data_path = prefixes
    cfg.dataset.sequence_length = args.seq_length
    cfg.dataset.num_workers = args.num_workers
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.null_tokenizer_eod_id = eos_id

    if resume_checkpoint is None:
        # Initialize from imported model-only weights and create fresh optimizer/RNG state.
        assert pretrained_checkpoint is not None
        cfg.checkpoint.pretrained_checkpoint = str(pretrained_checkpoint)
        cfg.checkpoint.load = None
        cfg.checkpoint.load_optim = False
        cfg.checkpoint.load_rng = False
        cfg.checkpoint.finetune = True
    else:
        # Continue the same training run, including optimizer, scheduler, RNG, and consumed samples.
        cfg.checkpoint.pretrained_checkpoint = None
        cfg.checkpoint.load = str(resume_checkpoint)
        cfg.checkpoint.load_optim = True
        cfg.checkpoint.load_rng = True
        cfg.checkpoint.finetune = False
    cfg.checkpoint.async_save = False

    save_dir = args.save_dir
    if save_dir is None and resume_checkpoint is not None:
        save_dir = resume_checkpoint
    if save_dir is None:
        cfg.checkpoint.save = None
        cfg.checkpoint.save_interval = None
    else:
        cfg.checkpoint.save = str(save_dir.expanduser().resolve())
        cfg.checkpoint.save_interval = args.save_interval or args.train_iters

    cfg.train.train_iters = args.train_iters
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size
    cfg.validation.eval_interval = args.eval_interval
    cfg.validation.eval_iters = args.eval_iters if args.eval_interval is not None or args.eval_only else 0
    cfg.validation.skip_train = args.eval_only
    cfg.logger.log_interval = args.log_interval
    cfg.logger.tensorboard_log_interval = args.log_interval
    if args.tensorboard_dir is not None:
        cfg.logger.tensorboard_dir = str(args.tensorboard_dir.expanduser().resolve())

    cfg.optimizer.lr = args.lr
    cfg.optimizer.min_lr = args.min_lr
    cfg.scheduler.lr_warmup_iters = args.lr_warmup_iters
    cfg.scheduler.lr_decay_iters = args.train_iters

    callbacks = None
    if args.eval_only:
        callbacks = CallbackManager()
        callbacks.register("on_eval_end", _write_eval_only_metrics_to_tensorboard)

    pretrain(config=cfg, forward_step_func=forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()
