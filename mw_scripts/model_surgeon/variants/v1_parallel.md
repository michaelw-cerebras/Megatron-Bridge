# v1 — parallel attention / MLP

**Slug:** `parallel` · **Scope:** within a layer · **Status:** active

## Computation

Original (pre-norm, sequential):

```text
h = x + a(N_a(x))
y = h + m(N_m(h))
```

v1 (parallel):

```text
y = x + a(N_a(x)) + m(N_m(x))
```

Both norms are kept and both keep their original weights. Attention, router,
expert, embedding and output tensors are untouched. There are no new
parameters and no removed parameters.

The change is that the MLP/MoE now reads `N_m(x)` instead of
`N_m(x + a(N_a(x)))` — it no longer waits on attention, which is what makes the
two sublayers co-schedulable.

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | which decoder layers are rewritten |

Ratios used so far:

- `"all"` → `p = 1.0` (Llama 4B, all 16 layers)
- `{"kind": "config_list", "field": "window_attn_skip_freq"}` → `p = 0.75` on
  Mellum2 (the 3 sliding-window layers per 4-layer cycle; the dense layer stays
  sequential)

## Preconditions

Declared in `CONFIG_REQUIREMENTS` and checked at spec-build time:

`pre_norm_residual`, `no_hidden_dropout`, `no_mlp_chunking`, `no_cuda_graphs`,
`no_fused_tp_inference`, `no_selective_recompute`, `no_cpu_offloading`,
`non_local_attention_backend`.

Checked again at layer construction: no cross-attention, no selective
layernorm/MLP recompute, no activation offloading. Checked at forward: both
sublayers must be bias-free and both norms must return a single tensor.

`non_local_attention_backend` is not a modelling constraint. `GPTModelProvider`
rewrites `core_attention` for the local backend by reaching through
`spec.submodules`; surgery hands Megatron a `TransformerBlockSubmodules`, which
has no such attribute, so the rewrite would be skipped without any warning.

## Measured results

From `parallel_mlp_attn.pdf`, 4B-token continued-pretraining budget, weights
rearranged from the pretrained checkpoint.

### Mellum2 12B-A2.5B, p = 0.75

| model | final train loss | eval mean |
|---|---|---|
| original | 1.336 | 61.57% |
| v1 (rearranged) | 1.387 | 56.32% |
| v1 (random init) | 3.248 | 34.03% |

Largest eval drops: BBH 51.92 → 40.01, ARC-Challenge 54.52 → 45.99.

The random-init arm is the control: it confirms the recovery comes from reusing
pretrained weights, not from the parallel structure being easier to fit.

### Llama 4B 16L, p = 1.0

| model | final train loss | eval mean |
|---|---|---|
| original | 1.501 | 55.05% |
| v1 (rearranged) | 1.590 | 51.15% |

## Observations

- Training loss recovers to within ~0.05–0.09 of the original inside a few
  hundred million tokens, but the downstream eval gap (4–5 points mean) does not
  close over the same budget. Loss recovery and capability recovery are not the
  same thing.
- The pattern holds across a dense model and an MoE model, and across p = 0.75
  and p = 1.0, which is what motivated making the ratio a swept parameter rather
  than a per-model hardcode.

## Open questions

- Does the eval gap keep shrinking with a larger recovery budget, or has it
  plateaued? Every point so far is at exactly 4B tokens.
- How does the gap scale with model size? Two points (4B dense, 12B MoE) is not
  a trend.
- Does post-training (SFT/RL) widen or close the gap? Untested. If it closes
  it, v1 stops being a lossy trade and becomes free inference headroom.
