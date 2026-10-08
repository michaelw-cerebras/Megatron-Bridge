# Mellum2

`MellumBridge` supports the original all-MoE Mellum2 architecture, starting with
[`JetBrains/Mellum2-12B-A2.5B-Thinking`](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking).
It preserves per-head QK RMSNorm, explicit head dimensions, normalized top-k routing,
sliding/full attention patterns, and full-layer YaRN frequency/concentration scaling.

The implementation uses `GPTModelProvider` and a Mellum layer specification. It
does not alter Megatron-Core. Configs with unsupported attention, rotary, or MLP
variants fail explicitly. Parallel attention/MoE surgery is not implemented.

The published per-expert checkpoint layout reuses the Qwen3 MoE weight mappings.
On Transformers installations without Mellum, Bridge registers a compatibility
configuration and uncached HF reference. Native Mellum registrations take precedence.

The continued-pretraining recipe is `mellum_pretrain_config(hf_model_id, load_weights=...)`.
Use `load_weights=True` for first construction from HF; use false when restoring
an existing Megatron training checkpoint.

See [the training manual](../../../mw_scripts/mellum2_3xh200/user_manual.md) for
YAML mixtures, SSH/Docker launch, EFA checks, parity, checkpoint/resume, and limitations.

For a focused single-GPU conversion and forward-parity check:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 \
  mw_scripts/mellum2_3xh200/validate.py --output /tmp/mellum-parity.json
```

Dynamic batching/flash-decoding inference and precision recipes beyond BF16 have
not been validated. The baseline deliberately disables fused RoPE and CUDA graphs.
