# v3_1 — parallel decoders with a merged expert pool

**Slug:** `merged-pool` · **Scope:** across whole decoders · **Status:** active

A derivative of [v3](v3_adjacent.md). The attentions are arranged identically;
what changes is that the group's MoE layers stop being separate routers over
separate pools.

## Computation

For a group of `g` decoders (default `g = 2`), each with `E` experts and top-`k`:

```text
z      = x + Σ_i a_i(N^a_i(x))                       # attentions still parallel
logits = concat_i router_i(N^m_i(z))                 # g·E logits
p      = softmax(logits) → top-(g·k) → renormalize ONCE
y      = z + Σ_j p_j · expert_j(N^m_owner(j)(z))
```

On Mellum2 that is **128 experts, global top-16** instead of two independent
top-8-of-64 selections.

## Why this is interesting

v3 forces every token into a fixed `k + k` split: 8 experts from branch 1 and 8
from branch 2, regardless of what the token needs. v3_1 removes that constraint
— a token can take 12+4, 6+10, or 16+0.

**Total parameters and active expert FLOPs are unchanged.** A token still fires
`g·k` experts. The only thing that changes is that the budget is allocated
dynamically rather than being split in half by construction. That is the
hypothesis under test: is the fixed split costing anything?

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | the contiguous run of decoders to group |
| `group_size` | `2` | decoders per merged group (`g`) |

Same selection rule as v3: contiguous, length a multiple of `group_size`.

## Preconditions

All of v3's (`no_pipeline_parallel`, `no_expert_parallel_overlap`, plus v1's
layer-local set), and additionally the joint selection reimplements the router,
so it rejects any router feature it does not reproduce:

`moe_router_pre_softmax=True`, `moe_router_score_function="softmax"`, no expert
capacity factor, no group-limited routing, no router topk scaling, no z-loss,
no input jitter, no aux-loss-free expert bias, no forced load balancing, no
latent MoE, no shared experts. The token dispatcher must be `alltoall`. Pools
must be the same size with the same `k`.

Mellum2 satisfies all of these. Another model family that does not will fail at
wiring time with a named reason rather than routing differently in silence.

## Implementation notes

- **Renormalization happens exactly once**, over the merged top-`g·k`. Each
  pool therefore receives a probability slice that does *not* sum to 1 — that
  asymmetry is the dynamic budget. Letting either dispatcher renormalize its own
  slice would silently restore the fixed `k + k` split and quietly turn this arm
  back into v3. A unit test asserts the per-pool slices do not sum to 1.
- **Scoring is done in fp32** regardless of parameter dtype. Top-k over `g·E`
  entries is more tie-prone than over `E`, and ties broken differently from the
  original model would change expert selection for no modelling reason.
- **Dispatch stays per layer.** Each `MoELayer` is driven through its own
  `preprocess → dispatch → routed_experts_compute → combine` with its half of
  the routing, so expert parallelism is preserved. Unlike ordinary top-k
  routing, a pool can receive a variable number of selections per token. The
  implementation therefore requires the all-to-all dispatcher and selects its
  dynamic-size preprocessing path; DeepEP and all-gather are rejected.
- **Checkpoint compatibility is preserved.** Every router, norm and expert keeps
  its own weights in its own place in the module tree; only the selection is
  joint. The state dict is identical to the original model, so `surgeon.py`'s
  byte-exact clone and verification apply unchanged.

## Load balancing

The aux loss is computed **jointly over all `g·E` experts**, not per layer.
Keeping the two per-layer losses would push each pool toward balance within its
own experts, which directly opposes the cross-pool selection this variant
exists to test.

## Instrumentation

Each group logs `merged_pool_frac_<i>` through Megatron's aux-loss tracker: the
fraction of the group's selections that landed in pool `i`.

- `0.5` means the group reproduced v3's fixed split and the extra freedom is
  unused.
- Movement away from `0.5`, and especially *variance* across tokens, is the
  model learning to allocate its expert budget dynamically.

This is the headline measurement of the experiment, not the loss curve.

## Expected behaviour

Initial loss should be high: the merged softmax renormalizes over a distribution
the routers were never trained to produce jointly, so early selections will be
poor. The question is recovery speed. Ivan also suggested an initial
**router-only** training phase as a follow-up; that is a separate experiment and
is not implemented here.

## Status

Implemented, unit-tested, and smoke-tested for two complete training iterations
on 8 H200 GPUs with EP=4 and the all-to-all dispatcher. The smoke test exercised
HF loading, real-data dispatch, forward, backward, and optimizer update. Before
launching a long job, still confirm the pool-fraction metrics in W&B as described
in the README.
