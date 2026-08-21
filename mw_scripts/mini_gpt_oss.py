from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_20b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def main():
    cfg = gpt_oss_20b_pretrain_config()

    # -------------------------
    # Parallelism: 2 GPUs
    # -------------------------
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 2
    cfg.model.sequence_parallel = False

    # -------------------------
    # Tiny GPT-OSS model
    # -------------------------
    cfg.model.num_layers = 4
    cfg.model.hidden_size = 512
    cfg.model.num_attention_heads = 8
    cfg.model.num_query_groups = 2
    cfg.model.kv_channels = 64

    cfg.model.ffn_hidden_size = 1024
    cfg.model.num_moe_experts = 8
    cfg.model.moe_router_topk = 2
    cfg.model.moe_ffn_hidden_size = 1024

    # Stock GPT-OSS-style alternating SWA/full attention
    cfg.model.window_size = (127, 0)
    cfg.model.window_attn_skip_freq = 2

    # -------------------------
    # Sequence length / mock data
    # -------------------------
    cfg.model.seq_length = 1024
    cfg.dataset.sequence_length = 1024
    cfg.dataset.mock = True

    # -------------------------
    # Tiny training run
    # -------------------------
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 2
    cfg.train.train_iters = 2

    # -------------------------
    # Optimizer / scheduler
    # -------------------------
    cfg.optimizer.lr = 3e-4
    cfg.optimizer.min_lr = 3e-5

    cfg.scheduler.lr_warmup_iters = 1
    cfg.scheduler.lr_decay_iters = 2

    pretrain(
        config=cfg,
        forward_step_func=forward_step,
    )


if __name__ == "__main__":
    main()
