# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight roundtrip and HF/Megatron forward parity for toy or real Mellum."""

import argparse
import hashlib
import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from megatron.bridge import AutoBridge


LOGGER = logging.getLogger(__name__)


def create_toy(path: Path) -> None:
    """Create a small checkpoint exercising full/sliding attention and nonstandard head size."""
    config = AutoConfig.for_model(
        "mellum",
        architectures=["MellumForCausalLM"],
        hidden_size=64,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        initializer_range=0.02,
        dtype="bfloat16",
        use_cache=False,
        bos_token_id=0,
        eos_token_id=28,
        pad_token_id=None,
        sliding_window=16,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        mlp_layer_types=["sparse"] * 4,
        rope_parameters={
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
    )
    torch.manual_seed(1234)
    AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").to(torch.bfloat16).save_pretrained(path)


def create_training_fixture(path: Path) -> None:
    """Write a tiny, isolated training/checkpoint fixture, never touching user data."""
    import numpy as np
    import yaml
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {"<|endoftext|>": 0, **{f"t{index}": index for index in range(1, 128)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="t1")),
        eos_token="<|endoftext|>",
        unk_token="t1",
    )
    tokenizer.save_pretrained(path)
    data = path / "synthetic_data"
    data.mkdir(exist_ok=True)
    prefix = data / "toy"
    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    for _ in range(64):
        builder.add_document(list(range(1, 128)) + [0], [128])
    builder.finalize(f"{prefix}.idx")
    token_path = path / "tokenizer.json"
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer": {
                    "eos_id": 0,
                    "json_path": "tokenizer.json",
                    "json_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
                }
            },
            indent=2,
        )
        + "\n"
    )
    settings = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    settings["model"]["hf_path"] = str(path)
    settings["train"].update(train_tokens=64 * 24 * 6, sequence_length=64, global_batch_size=24)
    settings["parallelism"]["expert_model_parallel_size"] = 4
    settings["scheduler"]["lr_warmup_iters"] = 1
    settings["dataset"].update(num_workers=0, subsets={str(prefix): 1})
    settings["run"].update(directory=str(path / "training_run"), save_interval=2, log_interval=1, exit_interval=2)
    if "wandb" in settings:
        settings["wandb"]["enabled"] = False  # Synthetic smoke must not create production tracking runs.
    (path / "smoke.local.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    LOGGER.info("Training smoke fixture: %s", path / "smoke.local.yaml")


def main() -> None:
    """Compare every exported tensor, logits beyond the sliding window, and CE loss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default="JetBrains/Mellum2-12B-A2.5B-Thinking")
    parser.add_argument("--toy-dir", type=Path)
    parser.add_argument("--toy-training-fixture", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1152)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--generate-tokens", type=int, default=0)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    try:
        if args.toy_dir is not None:
            if rank == 0:
                create_toy(args.toy_dir)
                if args.toy_training_fixture:
                    create_training_fixture(args.toy_dir)
            dist.barrier()
            source = str(args.toy_dir)
        else:
            source = args.hf_model
        bridge = AutoBridge.from_hf_pretrained(source, torch_dtype=torch.bfloat16)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = args.tp
        provider.expert_model_parallel_size = args.ep
        provider.expert_tensor_parallel_size = 1
        provider.sequence_parallel = args.tp > 1
        provider.parallel_output = False
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.finalize()
        provider.initialize_model_parallel(seed=1234)
        models = provider.provide_distributed_model(wrap_with_ddp=False)
        for model in models:
            model.eval()
        checked = 0
        mismatches = []
        for name, weight in bridge.export_hf_weights(models, cpu=True):
            if rank == 0:
                original = bridge.hf_pretrained.state[name]
                if not torch.equal(weight.float(), original.float()):
                    mismatches.append(name)
                checked += 1
        failure = torch.tensor([bool(mismatches)], device="cuda")
        dist.broadcast(failure, 0)
        if failure.item():
            raise RuntimeError(f"Weight roundtrip mismatches: {mismatches[:10]}")
        if args.toy_dir is not None:
            ids = torch.arange(64, device="cuda")[None, :] % provider.vocab_size
        else:
            tokenizer = AutoTokenizer.from_pretrained(source)
            text = "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n" * args.tokens
            ids = tokenizer(text, return_tensors="pt").input_ids[:, : args.tokens].cuda()
        positions = torch.arange(ids.shape[1], device="cuda")[None, :]
        with torch.no_grad():
            logits = models[0](input_ids=ids, position_ids=positions, attention_mask=None).float()
        if rank == 0:
            reference = (
                AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="sdpa")
                .cuda()
                .eval()
            )
            with torch.no_grad():
                expected = reference(ids, use_cache=False).logits.float()
            cosine = torch.nn.functional.cosine_similarity(logits.flatten(), expected.flatten(), dim=0).item()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, provider.vocab_size), ids[:, 1:].reshape(-1)
            )
            ref_loss = torch.nn.functional.cross_entropy(
                expected[:, :-1].reshape(-1, provider.vocab_size), ids[:, 1:].reshape(-1)
            )
            agreement = (logits.argmax(-1) == expected.argmax(-1)).float().mean().item()
            result = {
                "model": source,
                "tp": args.tp,
                "ep": args.ep,
                "tokens": ids.shape[1],
                "weights_exact": checked,
                "cosine": cosine,
                "max_logit_diff": (logits - expected).abs().max().item(),
                "argmax_agreement": agreement,
                "loss": loss.item(),
                "hf_loss": ref_loss.item(),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            LOGGER.info("Parity: %s", result)
            failure[0] = not (cosine >= args.min_cosine and abs(loss.item() - ref_loss.item()) < 0.02)
        dist.broadcast(failure, 0)
        if failure.item():
            raise RuntimeError("Forward parity failed; inspect the result JSON")
        if args.generate_tokens:
            if args.toy_dir is not None:
                raise ValueError("Generation sanity is intended for published models, not random toy weights")
            tokenizer = AutoTokenizer.from_pretrained(source)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Write a Python function fibonacci(n). Return Python code."}],
                tokenize=False,
                add_generation_prompt=True,
            )
            generated = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            prompt_length = generated.shape[1]
            with torch.no_grad():
                for _ in range(args.generate_tokens):
                    positions = torch.arange(generated.shape[1], device="cuda")[None, :]
                    output = models[0](input_ids=generated, position_ids=positions, attention_mask=None)
                    next_id = output[:, -1].argmax(-1, keepdim=True)
                    generated = torch.cat((generated, next_id), dim=1)
                    if next_id.item() == tokenizer.eos_token_id:
                        break
            if rank == 0:
                result["generation"] = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=False)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                LOGGER.info("Megatron generation: %s", result["generation"])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
