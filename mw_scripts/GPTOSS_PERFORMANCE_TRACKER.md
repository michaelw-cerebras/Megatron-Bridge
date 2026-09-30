# GPT-OSS Performance Tracker

Last updated: 2026-09-19

This document records reproducible throughput experiments for the two GPT-OSS
workloads on one 8x B200 node. Add new measurements here rather than relying on
TensorBoard history or transient training logs.

## Comparison rules

- Report steady-state throughput as the mean of iterations 5--16 unless a row
  says otherwise. Exclude initialization and CUDA-graph capture iterations.
- Keep container, GPU count, sequence length, precision, routing mode, dataset
  type, and batch shape fixed when calling a result an optimization.
- A larger global batch size can increase tokens/s by amortizing optimizer and
  synchronization work, but it changes optimization semantics. Record it as a
  separate batch-size result, not as a kernel-only improvement.
- Do not compare 20B mock-data numbers directly with 16L RTJ-data numbers:
  the models, active MoE compute, parallelism, and input paths differ.

## Runtime context

| Item | Value |
| --- | --- |
| Hardware | 8x NVIDIA B200 (single node) |
| Main container | `nvcr.io/nvidia/nemo:26.06` (`megatron_bridge_mw`) |
| CUDA runtime shown by training logs | CUDA 13.2 |
| Transformer Engine / DeepEP seen by B200 preflight | TE 2.16.0, DeepEP 1.2.1 |
| Sequence length | 8192 |
| 16L data | RTJ 50B tokenized mixture, 771 weighted indexed-dataset prefixes |
| 20B performance data | Mock data |

## GPT-OSS-20B (official-style mock throughput)

Script: `mw_scripts/gptoss_20b_official_b200_mock_perf.sh`

The productive configuration was FP8-MX, MBS=2, EP=1, full-iteration CUDA
graph, and random initialization. Its key throughput lever was global batch
size. These are mock-data performance numbers, not convergence measurements.

| Global batch size | Approx. tokens/s | Interpretation |
| ---: | ---: | --- |
| 16 | ~210k | Initial optimized mock baseline. |
| 32 | ~232k | Larger accumulation improves amortization. |
| 64 | ~246k | `gptoss_20B_gbs64_mock_fp8mx_mbs2`; about 2.134 s/step. |
| 128 | ~255k | Continued batch-size scaling. |
| 256 | ~258k | Highest observed 20B mock number. |

### 20B conclusions

- The increase from roughly 210k to 258k tokens/s was primarily GBS scaling,
  not an isolated DeepEP or CUDA-graph gain.
- EP=2 with DeepEP plus Transformer Engine scoped graphs completed but was
  around 185k tokens/s, so it was not retained for this single-node 20B shape.
- The current stack reached the high-250k range on the official-style mock
  configuration. The public ~291k--300k reference remains a different software
  stack/recipe target rather than a validated apples-to-apples comparison.

## GPT-OSS 16L custom variant (real RTJ data)

Script: `mw_scripts/gptoss_16l_rtj_50b/gptoss_16L_rtj_50B_train.sh`

Architecture highlights: 16 layers, hidden size 2880, 64 routed experts,
top-16 routing, routed and shared expert FFN size 1440. Although it has fewer
layers than 20B, its active routed-expert compute proxy is larger because it
uses top-16 rather than top-4 routing. The initial EP=8 layout also performs
expert dispatch/combine across all eight GPUs.

### FLOP accounting: why 16L is not a smaller 20B workload

This section counts the math needed for **one token** at sequence length 8192.
It uses the resolved configurations saved with the measured runs:

- 20B mock run: 24 layers, hidden size 2880, top-k 4, expert FFN 2880,
  64 query heads of dimension 64, 8 KV heads, mock vocabulary 128256.
- 16L RTJ run: 16 layers, hidden size 2880, top-k 16, expert FFN 1440, one
  shared expert of width 1440, 32 query heads of dimension 128, one KV head,
  vocabulary 201088.

`GFLOPs` means one billion floating-point operations. A matrix multiplication
of an `M x K` matrix by a `K x N` matrix costs about `2 x M x K x N` FLOPs:
one multiplication and one addition for each of its `M x N` outputs. The
forward calculations below ignore small operations such as bias additions,
normalization, softmax, and activation functions. The training estimate is
approximately three times forward FLOPs because it includes forward, gradient
with respect to inputs, and gradient with respect to weights.

#### Routed and shared experts

A SwiGLU expert has three large matrix multiplications:

```text
hidden -> FFN
hidden -> FFN
FFN    -> hidden
```

Therefore one expert costs `6 x hidden_size x expert_FFN_size` forward FLOPs
per token. Top-k routing multiplies that cost by the number of selected
experts.

```text
20B routed experts
= 24 layers x 4 selected experts x 6 x 2880 x 2880
= 4.778 GFLOPs per token

16L routed experts
= 16 layers x 16 selected experts x 6 x 2880 x 1440
= 6.370 GFLOPs per token

16L shared expert
= 16 layers x 1 shared expert x 6 x 2880 x 1440
= 0.398 GFLOPs per token
```

The 16L routed-expert work alone is `6.370 / 4.778 = 1.333x` the 20B value.
After its shared expert is included, 16L expert work is `6.768 / 4.778 =
1.416x` the 20B value. Fewer layers do not compensate for routing to sixteen
routed experts instead of four.

#### Attention projections and attention core

Both models have query width 4096:

```text
20B query width = 64 query heads x 64 head dimension = 4096
16L query width = 32 query heads x 128 head dimension = 4096
```

The 20B KV width is `8 x 64 = 512`; the 16L KV width is `1 x 128 = 128`.
Q, K, V, and output projections are therefore:

```text
20B per layer = 2 x 2880 x (4096 + 512 + 512 + 4096)
              = 53.084 MFLOPs per token per layer
20B all layers = 24 x 53.084 = 1.274 GFLOPs per token

16L per layer = 2 x 2880 x (4096 + 128 + 128 + 4096)
              = 48.660 MFLOPs per token per layer
16L all layers = 16 x 48.660 = 0.779 GFLOPs per token
```

For the attention-score calculation and attention-value calculation together,
the forward cost is about `4 x context_tokens x query_width` per layer.

- At sequence length 8192, full causal attention has about 4096 prior tokens
  per query on average.
- The 16L window is 32768, larger than the tested sequence length, so all 16
  layers act as full attention for this benchmark.
- The 20B recipe alternates: 12 full-attention layers and 12 local-attention
  layers with a 128-token window (`window_attn_skip_freq=2`).

```text
16L attention core
= 16 x 4 x 4096 average context tokens x 4096 query width
= 1.074 GFLOPs per token

20B attention core
= 12 x 4 x 4096 x 4096       # full-attention layers
 + 12 x 4 x about-127 x 4096 # local-attention layers
= 0.830 GFLOPs per token
```

The 16L attention core is heavier, but it has fewer layers and much narrower
KV projections. Attention as a whole is slightly lighter for 16L; it is not
the main explanation for the throughput gap.

#### Output logits

The final output projection maps hidden size 2880 to the vocabulary. Its
forward cost is `2 x hidden_size x vocabulary_size` per token.

```text
20B mock logits = 2 x 2880 x 128256 = 0.739 GFLOPs per token
16L RTJ logits  = 2 x 2880 x 201088 = 1.158 GFLOPs per token
```

The real 16L vocabulary makes this output calculation `1.158 / 0.739 =
1.568x` larger. This is also why MBS=4 ran out of memory while attempting to
materialize logits of shape `(8192, 4, 201088)`.

#### Total forward and training FLOPs per token

| Forward component | 20B mock | 16L RTJ |
| --- | ---: | ---: |
| Routed experts | 4.778 GF | 6.370 GF |
| Shared expert | 0 | 0.398 GF |
| Attention projections | 1.274 GF | 0.779 GF |
| Attention core | 0.830 GF | 1.074 GF |
| Output logits | 0.739 GF | 1.158 GF |
| Router projection (small) | 0.004 GF | 0.006 GF |
| **Total forward FLOPs/token** | **7.625 GF** | **9.785 GF** |
| **Approximate training FLOPs/token (`3 x forward`)** | **22.875 GF** | **29.354 GF** |

Therefore the custom 16L run needs:

```text
29.354 / 22.875 = 1.283x
```

or about **28.3% more training FLOPs per token** than the 20B mock benchmark.
This agrees with the measured 16L log: `155.6k tokens/s x 29.354 GF/token / 8
GPUs` is about `571 TFLOPs/s/GPU`, matching the reported approximately 572
TFLOPs/s/GPU.

If 16L achieved the 20B peak's same per-GPU compute efficiency, its expected
throughput based on FLOPs alone would be roughly:

```text
260k tokens/s / 1.283 = about 203k tokens/s
```

The remaining difference from about 203k to the measured 155.6k is not a
model-FLOP difference. It is the execution-efficiency gap addressed by the
later sections: EP=4 dispatch/combine versus 20B EP=1, top-16 routing traffic,
scoped rather than full-iteration CUDA graphs, different GEMM/logit shapes,
and mock versus real data paths.

### Fixed-GBS=128 results

All successful rows below use real RTJ data, sequence length 8192, MBS=2,
MXFP8, random initialization, no full activation recompute, B200 optimized
profile, and `FORCE_ROUTER_LOAD_BALANCING=1` unless noted otherwise.

Logs and resolved launch configuration for each run live under
`/cb/home/michaelw/ws/training_runs/<run-name>/`.

| Run name | EP | DeepEP SMs | EP overlap | Mean step time | Tokens/s | Result |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `gptoss_16L_b200_ep8_mbs2_mxfp8_balanced_smoke` | 8 | 20 | No | 8.540 s | 122.8k | First MBS=2 MXFP8 baseline. |
| `gptoss_16L_b200_ep4_mbs2_mxfp8_balanced` | 4 | 20 | No | 7.554 s | 138.8k | Reducing EP communication group was a clear win. |
| `gptoss_16L_b200_ep4_mbs2_mxfp8_balanced_epoverlap_deepep16` | 4 | 16 | Yes | 7.274 s | 144.1k | Too few DeepEP SMs. |
| `gptoss_16L_b200_ep4_mbs2_mxfp8_balanced_epoverlap_retry` | 4 | 20 | Yes | 7.009 s | 149.6k | EP overlap adds 7.8% over EP=4 without overlap. |
| `gptoss_16L_b200_ep4_mbs2_mxfp8_balanced_epoverlap_deepep24` | 4 | 24 | Yes | 6.853 s | 153.0k | Better DeepEP resource balance. |
| `gptoss_16L_b200_ep4_mbs2_mxfp8_balanced_epoverlap_deepep28` | 4 | 28 | Yes | 6.782 s | 154.6k | Continued improvement. |
| `gptoss_16L_b200_ep4_mbs2_mxfp8_balanced_epoverlap_deepep32` | 4 | 32 | Yes | **6.739 s** | **155.6k** | Current best fixed-GBS result. |

### Batch and parallelism experiments

| Experiment | Outcome | Decision |
| --- | --- | --- |
| EP=1, MBS=2, GBS=128 | 10.217 s/step, 102.6k tokens/s; CUDA graph reserve peaked at 179 GiB. | Reject: replicated expert state leaves little memory headroom and is slower. |
| EP=4, MBS=4, GBS=128 | OOM while allocating 24.55 GiB FP32 logits of shape `(8192, 4, 201088)`. | Keep MBS=2 at this sequence length/vocabulary. |
| EP=4, MBS=2, GBS=256, DeepEP=20, overlap | 13.879 s/step, 151.1k tokens/s. | Only ~1.0% above the matching GBS=128/20-SM result; do not change training GBS solely for throughput. |

### Current recommended performance benchmark command

```bash
RUN_NAME=gptoss_16L_b200_perf_best \
TRAIN_TOKENS=16777216 \
LR_WARMUP_ITERS=1 \
MICRO_BATCH_SIZE=2 \
GLOBAL_BATCH_SIZE=128 \
EXPERT_PARALLEL=4 \
PRECISION=mxfp8 \
NO_FULL_RECOMPUTE=1 \
B200_OPTIMIZED=1 \
DEEP_EP_NUM_SMS=32 \
ENABLE_EP_OVERLAP=1 \
FORCE_ROUTER_LOAD_BALANCING=1 \
DISABLE_CHECKPOINTING=1 \
LOG_INTERVAL=1 \
./mw_scripts/gptoss_16l_rtj_50b/gptoss_16L_rtj_50B_train.sh
```

The launcher now defaults `DEEP_EP_NUM_SMS` to 32, so that variable is shown
for clarity rather than being required.

### Important 16L caveat: force-balanced routing

`FORCE_ROUTER_LOAD_BALANCING=1` is a performance-only setting. It gives each
expert a fixed token count so MXFP8 expert GEMM dimensions satisfy the required
32-element block alignment. With normal dynamic dropless top-16 routing,
MXFP8 failed on non-divisible expert-token shapes such as `(65776, 2880)`.

This changes router behavior from normal auxiliary-loss training. The 155.6k
result is therefore a valid systems benchmark, but not yet the default setting
for a convergence run. A normal-router training recipe should use the supported
precision path (for example `fp8-current`) unless MXFP8 padding/alignment can
be made semantics-preserving.

## Implementation changes retained in the 16L launcher

- `--enable-ep-overlap` now creates `CommOverlapConfig` before setting overlap
  fields. The prior path failed because the base recipe leaves `comm_overlap`
  unset.
- EP overlap deliberately leaves delayed expert wgrad off: the custom model's
  attention bias is incompatible with delayed wgrad plus TE attention graphs.
- The `--disable-deepep` A/B path explicitly selects standard `alltoall` and
  removes the flex backend.
- TE CUDA graph scopes omit `moe_preprocess` for standard all-to-all because
  its host synchronization is not graph-capturable. The DeepEP path uses
  `attn`, `moe_router`, and `moe_preprocess`.
- B200 profile defaults to 32 DeepEP SMs; this was verified by a non-training
  dry run.

## Suggested next experiments

1. Establish the best normal-router `fp8-current` configuration at EP=4 with
   overlap and 32 DeepEP SMs, then compare its convergence and throughput to
   the force-balanced MXFP8 benchmark.
2. Profile the 155.6k configuration to split remaining time among expert
   dispatch/combine, grouped GEMM, logits/loss, and host gaps before changing
   more knobs.
3. Test semantics-preserving MXFP8 expert-token padding/alignment if supported
   by a newer Megatron-Core/Transformer Engine stack.

## Entry template

```text
Date:
Model / script:
Data: real RTJ | mock
Container / MCore revision:
Precision and router mode:
TP / PP / EP / ETP:
MBS / GBS / sequence length:
CUDA graph / dispatcher / overlap / DeepEP SMs:
Steady-state window and result:
Peak GPU memory:
Outcome: retain | reject | needs follow-up
Log path:
```
