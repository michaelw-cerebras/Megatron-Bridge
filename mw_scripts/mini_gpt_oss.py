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


def parse_args() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser(description="Run a tiny GPT-OSS pretraining smoke test.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Preprocessing output root containing manifest.json, data_prefixes.txt, and data/.",
    )
    parser.add_argument(
        "--hf-model",
        default=GPT_OSS_TOKENIZER,
        help=(
            "Hugging Face model identifier or local directory containing config.json. "
            "Use a local directory in air-gapped environments."
        ),
    )
    parser.add_argument("--train-iters", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--expert-parallel", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-dataset-shards",
        type=int,
        help=(
            "Use only the first N dataset shards. This is useful for tiny smoke tests whose "
            "sample count is too small to assign a positive blend weight to every shard."
        ),
    )
    return parser.parse_args()


def load_dataset(dataset_root: Path) -> tuple[list[str], int]:
    """Load and validate movable Megatron dataset prefixes and tokenizer metadata."""
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
    if not prefixes:
        raise ValueError(f"No dataset prefixes found in {prefixes_path}")
    for prefix in prefixes:
        for suffix in (".bin", ".idx"):
            dataset_file = Path(f"{prefix}{suffix}")
            if not dataset_file.is_file():
                raise FileNotFoundError(dataset_file)

    return prefixes, int(tokenizer["eos_id"])


def main() -> None:
    """Run the GPT-OSS smoke test with mock or pre-tokenized data."""
    args = parse_args()
    cfg = gpt_oss_20b_pretrain_config(hf_model_id=args.hf_model)

    # Parallelism. The defaults target a single 2-GPU node.
    cfg.model.tensor_model_parallel_size = args.tensor_parallel
    cfg.model.pipeline_model_parallel_size = args.pipeline_parallel
    cfg.model.expert_model_parallel_size = args.expert_parallel
    cfg.model.sequence_parallel = args.tensor_parallel > 1

    # Tiny GPT-OSS model.
    cfg.model.num_layers = 4
    cfg.model.hidden_size = 512
    cfg.model.num_attention_heads = 8
    cfg.model.num_query_groups = 2
    cfg.model.kv_channels = 64

    cfg.model.ffn_hidden_size = 1024
    cfg.model.num_moe_experts = 8
    cfg.model.moe_router_topk = 2
    cfg.model.moe_ffn_hidden_size = 1024

    # Stock GPT-OSS-style alternating SWA/full attention.
    cfg.model.window_size = (127, 0)
    cfg.model.window_attn_skip_freq = 2

    # Sequence length and optional pre-tokenized Megatron data.
    cfg.model.seq_length = args.seq_length
    cfg.dataset.sequence_length = args.seq_length
    cfg.dataset.num_workers = args.num_workers
    if args.dataset_root is not None:
        prefixes, eos_id = load_dataset(args.dataset_root)
        if args.max_dataset_shards is not None:
            if args.max_dataset_shards < 1:
                raise ValueError("--max-dataset-shards must be at least 1")
            prefixes = prefixes[: args.max_dataset_shards]
        cfg.dataset.data_path = prefixes
        cfg.tokenizer.vocab_size = cfg.model.vocab_size
        cfg.tokenizer.null_tokenizer_eod_id = eos_id

    # Tiny training run. Disable validation and checkpoint I/O for a pure smoke test.
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.train_iters = args.train_iters
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = None

    # Keep scheduler bounds consistent with the shortened run.
    cfg.optimizer.lr = 3e-4
    cfg.optimizer.min_lr = 3e-5
    cfg.scheduler.lr_warmup_iters = min(1, args.train_iters)
    cfg.scheduler.lr_decay_iters = args.train_iters

    pretrain(
        config=cfg,
        forward_step_func=forward_step,
    )


if __name__ == "__main__":
    main()
