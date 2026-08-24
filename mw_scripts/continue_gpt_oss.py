# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import json
from pathlib import Path

from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_20b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


GPT_OSS_TOKENIZER = "openai/gpt-oss-20b"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse GPT-OSS continual-pretraining arguments."""
    parser = argparse.ArgumentParser(description="Continue pretraining GPT-OSS 20B on Megatron indexed data.")
    parser.add_argument(
        "--hf-model",
        required=True,
        help="Local Hugging Face directory containing the GPT-OSS config.json.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        required=True,
        type=Path,
        help="Imported Megatron checkpoint directory used to initialize model weights.",
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


def main() -> None:
    """Load GPT-OSS 20B weights and continue pretraining on indexed data."""
    args = parse_args()
    pretrained_checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    if not pretrained_checkpoint.is_dir():
        raise FileNotFoundError(pretrained_checkpoint)

    cfg = gpt_oss_20b_pretrain_config(hf_model_id=args.hf_model)

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

    # Initialize from imported model-only weights and create fresh optimizer/RNG state.
    cfg.checkpoint.pretrained_checkpoint = str(pretrained_checkpoint)
    cfg.checkpoint.load = None
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.checkpoint.finetune = True
    cfg.checkpoint.async_save = False
    if args.save_dir is None:
        cfg.checkpoint.save = None
        cfg.checkpoint.save_interval = None
    else:
        cfg.checkpoint.save = str(args.save_dir.expanduser().resolve())
        cfg.checkpoint.save_interval = args.save_interval or args.train_iters

    cfg.train.train_iters = args.train_iters
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1

    cfg.optimizer.lr = args.lr
    cfg.optimizer.min_lr = args.min_lr
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = args.train_iters

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
