# Mellum2 parallel attention/MoE performance

This file records reproducible single-node H200 measurements for the derived
Mellum2 parallel architecture. It is intentionally separate from the production
training configuration.

## Twelve-case matrix (2026-10-14)

The production cases are `base-v0..v4` and `thinking-v0..v4`, including v3_1.
Base and Thinking
have the same architecture, so performance tuning is performed once per graph
and the result is shared by both source checkpoints.

| Graph | Current one-node setting | Evidence |
|---|---|---|
| v0 original | BF16, TP1/PP1/CP1/EP8/ETP1, all-to-all | Existing Mellum baseline |
| v1 parallel | BF16, TP1/PP1/CP1/EP4/ETP1, all-to-all, permute fusion | Measured below |
| v2 shifted | BF16, TP1/PP1/CP1/EP4/ETP1, all-to-all, permute fusion | Measured below |
| v3 adjacent | BF16, TP1/PP1/CP1/EP4/ETP1, all-to-all, permute fusion | Measured below |
| v3_1 merged pool | BF16, TP1/PP1/CP1/EP4/ETP1, all-to-all, permute fusion | Inherits v3 starting point |
| v4 compensated | BF16, TP1/PP1/CP1/EP4/ETP1, MBS2, all-to-all, permute fusion | Measured below |

The controlled v2/v3 sweep ran on GPU-31 after it became available. It compared
EP8 without fusion, EP4 without fusion, and EP4 with fusion for each graph while
holding every other variable fixed. v2/v3 prohibit PP and expert-parallel
overlap by construction.

## Method

- Hardware: one node with 8 NVIDIA H200 GPUs connected by NVLink.
- Sequence length: 8192.
- Global batch size: 192, matching the existing 50B-token run so its immutable
  dataset sampling cache can be reused.
- Dataset: eight real completed pretraining shards selected at evenly spaced
  positions from the immutable 9799-shard mixture. Every candidate uses the
  same subset and sampling cache, avoiding repeated full-mixture metadata scans.
- Checkpoint saving and W&B are disabled.
- Each candidate starts from the exact Hugging Face weights and performs real
  forward, backward, optimizer, and scheduler steps.
- Throughput is computed as `192 * 8192 / iteration_seconds` after warmup.

## Results (2026-10-09)

Steady means exclude the first two warmup iterations. Total throughput is
`tokens/s` across all eight GPUs.

| Checkpoint / candidate | Steady ms/step | tokens/s | Result |
|---|---:|---:|---|
| Thinking, BF16, TP1/EP8 baseline | 16,249 | 96,798 | Valid |
| Thinking, BF16, TP2/EP8 | 21,173 | 74,288 | Valid, slower |
| Thinking, BF16, TP4/EP8 | 39,959 | 39,362 | Valid, slower |
| Thinking, BF16, TP1/EP4 | 14,704 | 106,968 | Valid |
| Thinking, BF16, TP1/EP4 + permute fusion | **13,829** | **113,734** | **Best valid** |
| Thinking, BF16, TP1/EP4 + DeepEP | 15,651 | 100,494 | Valid, slower |
| Thinking, BF16, TP1/EP2 + permute fusion | 17,096 | 92,001 | Valid, slower |
| Thinking, BF16, TP2/EP4, MBS2 + permute fusion | 14,357 | 109,555 | Valid, slower |
| Base, BF16, TP1/EP4 + permute fusion | **13,895** | **113,197** | **Confirmed** |
| Thinking, FP8-current, TP1/EP4 | 13,805 | 113,931 | Invalid numerics |
| Thinking, BF16, TP1/EP4 + EP overlap | 14,629 at step 2 | n/a | Invalid numerics |
| Thinking, BF16, TP1/CP2/EP8 | n/a | n/a | Unsupported sliding-window CP path |
| Thinking, BF16, TP1/EP4, MBS2 + permute fusion | 13,463 at step 2 | n/a | OOM at step 3 |

The best valid Thinking result is 17.5% faster than the TP1/EP8 BF16 baseline.
Base and Thinking differ by only 0.5% under the winning configuration. Their
losses differ, as expected from their different weights, while every accepted
candidate had finite loss and gradients and no skipped or NaN iterations.

Peak allocated memory was 87.95 GiB/GPU for Thinking and 88.42 GiB/GPU for
Base under the winner. MBS2 reached roughly 134 GiB process usage and then
failed a 6 GiB allocation, so MBS1 is required for safe headroom.

### Rejected features

- FP8-current changed Thinking loss from the BF16 trajectory near 6.6 to about
  16.0 after the first optimizer step. Its speed matched, rather than exceeded,
  the BF16 winner.
- EP overlap changed the first-step loss from 6.605 to 1.751 and grad norm from
  about 21 to 1.46. The custom parallel block must not use that schedule until
  its semantics are understood and parity-tested.
- DeepEP was installed and ran correctly, but standard all-to-all was faster at
  this small single-node EP degree.
- CP2 was rejected by Megatron-Core because the default P2P context-parallel
  path does not support sliding-window attention.

## v2/v3 results (2026-10-13 PDT)

Steady means exclude iterations 1-2 and average iterations 3-8. Every candidate
completed all eight iterations with finite loss and gradients, zero skipped
iterations, and zero NaN iterations.

| Graph / candidate | Steady ms/step | tokens/s | Peak allocated / GPU | Result |
|---|---:|---:|---:|---|
| v2, EP8 | 15,465 | 101,706 | 81.25 GiB | Valid |
| v2, EP4 | 14,677 | 107,162 | 88.21 GiB | Valid |
| v2, EP4 + permute fusion | **13,748** | **114,409** | 88.21 GiB | **Best v2** |
| v3, EP8 | 16,821 | 93,507 | 82.67 GiB | Valid |
| v3, EP4 | 14,655 | 107,325 | 89.78 GiB | Valid |
| v3, EP4 + permute fusion | **13,789** | **114,068** | 89.78 GiB | **Best v3** |

For v2, the winner is 12.5% faster than EP8 and 6.8% faster than unfused EP4.
For v3, it is 22.0% faster than EP8 and 6.3% faster than unfused EP4. The v2
and v3 winners are also within 0.6% of the previously measured v1 winner, so
the production configs use the same EP4 + permute-fusion layout for all three
surgery graphs. Base was not repeated because its architecture is identical to
Thinking and the earlier v1 Base/Thinking confirmation differed by only 0.5%.

Megatron emitted an `AccumulateGrad` stream-mismatch performance warning for
v2/v3 because their graphs carry activations across adjacent layers. It did not
affect numerical validity, but it remains a possible future optimization area.

## v4 results (2026-10-14 PDT)

The first full-model smoke exposed an initialization issue: random correction
weights produced a non-finite forward loss at optimizer iteration 2. The
production default is therefore `zero_output`, which makes step 0 exactly equal
to v1 while retaining trainable input projections. Every accepted candidate
below completed eight iterations with finite loss and gradients and no skipped
or NaN iterations.

Peak learning rate was tested separately with a two-iteration warmup. `3e-4`,
`1e-4`, and `3e-5` all produced a non-finite forward loss immediately after
reaching peak. `3e-6` completed 12/12 iterations at peak with loss decreasing
from 6.308 to 4.049, so production v4 uses peak/min LR `3e-6`/`3e-7`.

| Candidate | Steady ms/step | tokens/s | Result |
|---|---:|---:|---|
| TP1/EP4, MBS1 + permute fusion | 29,788 | 52,800 | Valid, under-batched |
| TP1/EP4, MBS2 + permute fusion | **11,173** | **140,778** | **Best v4** |
| TP2/EP4, MBS2 + permute fusion | 13,412 | 117,274 | Valid, slower |
| TP1/EP4, MBS1, random init | n/a | n/a | Rejected: NaN at iteration 2 |

Because the original 12B parameters are frozen, v4 avoids their optimizer-state
memory and can safely use MBS2. This reduces gradient accumulation from 24 to 12
and makes v4 23.5% faster than the measured v1 winner. MBS3 was not attempted:
MBS2 already approached the practical H200 process-memory ceiling.

## Recommended single-node configuration

Use the parallel checkpoint for the desired model with:

```yaml
model:
  hf_path: /workspace/ml-olympus/codesign_pretrain/checkpoints/Mellum2-12B-A2.5B-Thinking-v2-shifted
  architecture: v2

train:
  sequence_length: 8192
  micro_batch_size: 1
  global_batch_size: 192

parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  context_parallel_size: 1
  expert_model_parallel_size: 4
  expert_tensor_parallel_size: 1
```

Set `performance.moe_permute_fusion: true`. Keep BF16, standard all-to-all,
activation recompute off, CUDA graphs off, and EP overlap off. The eight
production YAML files for v0-v3 use MBS1. For v4, use MBS2 with zero-output
initialization as encoded by its derived checkpoint; all twelve production YAML
files encode the graph-specific choices.
These measurements apply to one 8-H200 NVLink node; re-profile EP and
dispatcher settings when returning to multiple nodes.

Raw logs and per-run `benchmark_config.json` files are under:

```text
/cb/ml-olympus/codesign_pretrain/runs/mellum2_parallel_perf/
/cb/ml-olympus/codesign_pretrain/runs/mellum2_surgery_perf/
```
