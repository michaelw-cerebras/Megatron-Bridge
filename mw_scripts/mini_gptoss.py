import argparse
import inspect
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from unittest.mock import patch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider, transformer_engine_layer_spec
from megatron.bridge.recipes.gpt_oss import gpt_oss as gpt_oss_recipe
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


GPT_OSS_TOKENIZER = "openai/gpt-oss-20b"
GPT_OSS_PAD_TOKEN_ID = 199_999


def _build_biasless_shared_expert(
    *,
    config: TransformerConfig,
    submodules: MLPSubmodules,
    gate: bool,
    pg_collection: ProcessGroupCollection | None = None,
    name: str | None = None,
) -> SharedExpertMLP:
    """Build the shared expert without disabling bias in the rest of the layer."""
    shared_expert_config = deepcopy(config)
    shared_expert_config.add_bias_linear = False
    shared_expert_config.bias_activation_fusion = False
    return SharedExpertMLP(
        config=shared_expert_config,
        submodules=submodules,
        gate=gate,
        pg_collection=pg_collection,
        name=name,
    )


def _gpt_oss_transformer_layer_spec(config: GPTModelProvider) -> ModuleSpec:
    """Reuse the optimized TE spec while isolating the shared expert's bias limitation."""
    layer_spec = transformer_engine_layer_spec(config)
    moe_builder = layer_spec.submodules.mlp
    if not isinstance(moe_builder, partial):
        raise TypeError(f"Expected a partial MoE builder, got {type(moe_builder).__name__}")

    moe_submodules = moe_builder.keywords.get("submodules")
    if not isinstance(moe_submodules, MoESubmodules):
        raise TypeError(f"Expected MoESubmodules, got {type(moe_submodules).__name__}")

    shared_expert_builder = moe_submodules.shared_experts
    if not isinstance(shared_expert_builder, partial):
        raise TypeError(f"Expected a partial shared-expert builder, got {type(shared_expert_builder).__name__}")

    shared_expert_submodules = shared_expert_builder.keywords.get("submodules")
    if not isinstance(shared_expert_submodules, MLPSubmodules):
        raise TypeError(f"Expected shared-expert MLPSubmodules, got {type(shared_expert_submodules).__name__}")

    moe_submodules.shared_experts = partial(
        _build_biasless_shared_expert,
        submodules=shared_expert_submodules,
    )
    return layer_spec


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
    """Parse custom GPT-OSS benchmark arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the custom GPT-OSS pretraining architecture.")
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
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--expert-parallel", type=int, default=8)
    parser.add_argument("--expert-tensor-parallel", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        help="Optional TensorBoard event directory. Checkpoint saving remains disabled.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate after warmup.")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate after decay.")
    parser.add_argument(
        "--lr-warmup-iters",
        type=int,
        help="Warmup iterations. Defaults to 10%% of train-iters.",
    )
    parser.add_argument(
        "--lr-decay-iters",
        type=int,
        help="Learning-rate decay iterations. Defaults to train-iters.",
    )
    parser.add_argument(
        "--lr-decay-style",
        choices=("constant", "linear", "cosine", "inverse-square-root"),
        default="cosine",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp8-current", "mxfp8"),
        default="bf16",
        help="Use BF16, Hopper FP8 current scaling, or Blackwell MXFP8.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Enable full activation recomputation when the benchmark does not fit without it.",
    )
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
    """Benchmark the custom GPT-OSS model with mock or pre-tokenized data."""
    args = parse_args()
    if args.train_iters < 1:
        raise ValueError("--train-iters must be at least 1")
    lr_warmup_iters = args.lr_warmup_iters
    if lr_warmup_iters is None:
        lr_warmup_iters = args.train_iters // 10
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.train_iters
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("--min-lr must be nonnegative and must not exceed --lr")
    if lr_warmup_iters < 0:
        raise ValueError("--lr-warmup-iters must be nonnegative")
    if lr_decay_iters < 1:
        raise ValueError("--lr-decay-iters must be at least 1")
    if lr_warmup_iters >= lr_decay_iters:
        raise ValueError("--lr-warmup-iters must be smaller than --lr-decay-iters")

    cfg = _build_gpt_oss_20b_config(args.hf_model)

    # Parallelism. The defaults target a single 8-GPU node with one expert per rank group.
    cfg.model.tensor_model_parallel_size = args.tensor_parallel
    cfg.model.pipeline_model_parallel_size = args.pipeline_parallel
    cfg.model.expert_model_parallel_size = args.expert_parallel
    cfg.model.expert_tensor_parallel_size = args.expert_tensor_parallel
    cfg.model.sequence_parallel = args.tensor_parallel > 1

    # Custom GPT-OSS architecture. This is no longer a tiny model: it has roughly
    # 13B routed-expert parameters before embeddings, attention, and optimizer state.
    cfg.model.num_layers = 16
    cfg.model.hidden_size = 2880
    cfg.model.num_attention_heads = 32
    cfg.model.num_query_groups = 1
    cfg.model.kv_channels = 128

    cfg.model.ffn_hidden_size = 1440
    cfg.model.num_moe_experts = 64
    cfg.model.moe_router_topk = 16
    cfg.model.moe_ffn_hidden_size = 1440
    cfg.model.moe_shared_expert_intermediate_size = 1440
    cfg.model.moe_router_load_balancing_type = "aux_loss"
    cfg.model.moe_aux_loss_coeff = 0.9

    # GPT-OSS uses bias in attention, the router, and routed experts. Keep the
    # optimized TE layer spec, but give the MCore shared expert a local
    # bias-free config because SharedExpertMLP does not support linear bias.
    cfg.model.transformer_layer_spec = _gpt_oss_transformer_layer_spec
    cfg.model.add_bias_linear = True
    cfg.model.add_qkv_bias = True
    cfg.model.bias_activation_fusion = True

    # Every layer uses sliding attention. A None skip frequency applies the
    # configured window to every layer; (32767, 0) matches a 32768-token HF window.
    cfg.model.window_size = (32767, 0)
    cfg.model.window_attn_skip_freq = None

    # The stock GPT-OSS bridge currently overrides these with the public model's
    # factor=32 defaults, so restore the custom config values explicitly.
    cfg.model.rotary_base = 150000
    cfg.model.yarn_rotary_scaling_factor = 64.0
    cfg.model.yarn_original_max_position_embeddings = 4096
    cfg.model.yarn_beta_fast = 32.0
    cfg.model.yarn_beta_slow = 1.0
    cfg.model.yarn_correction_range_round_to_int = False

    if args.precision == "fp8-current":
        cfg.mixed_precision = "bf16_with_fp8_current_scaling_mixed"
        cfg.model.moe_router_padding_for_fp8 = True
    elif args.precision == "mxfp8":
        cfg.mixed_precision = "bf16_with_mxfp8_mixed"
        cfg.model.moe_router_padding_for_fp8 = True

    if args.recompute:
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = "uniform"
        cfg.model.recompute_num_layers = 1

    # Sequence length and optional pre-tokenized Megatron data.
    cfg.model.seq_length = args.seq_length
    cfg.dataset.sequence_length = args.seq_length
    cfg.dataset.num_workers = args.num_workers
    cfg.tokenizer.null_tokenizer_pad_id = GPT_OSS_PAD_TOKEN_ID
    if args.dataset_root is not None:
        prefixes, eos_id = load_dataset(args.dataset_root)
        if args.max_dataset_shards is not None:
            if args.max_dataset_shards < 1:
                raise ValueError("--max-dataset-shards must be at least 1")
            prefixes = prefixes[: args.max_dataset_shards]
        cfg.dataset.data_path = prefixes
        cfg.tokenizer.vocab_size = cfg.model.vocab_size
        cfg.tokenizer.null_tokenizer_eod_id = eos_id

    # Disable validation and checkpoint I/O so the measurement covers training only.
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.train_iters = args.train_iters
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = args.log_interval
    cfg.logger.log_throughput = True
    if args.tensorboard_dir is not None:
        cfg.logger.tensorboard_dir = str(args.tensorboard_dir.expanduser().resolve())
        cfg.logger.tensorboard_log_interval = args.log_interval
        cfg.logger.log_timers_to_tensorboard = True
        cfg.logger.log_throughput_to_tensorboard = True
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = None

    cfg.optimizer.lr = args.lr
    cfg.optimizer.min_lr = args.min_lr
    cfg.scheduler.lr_warmup_iters = lr_warmup_iters
    cfg.scheduler.lr_decay_iters = lr_decay_iters
    cfg.scheduler.lr_decay_style = args.lr_decay_style
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.override_opt_param_scheduler = True

    pretrain(
        config=cfg,
        forward_step_func=forward_step,
    )


if __name__ == "__main__":
    main()
