# v3_3 — parallel decoders with nonlinear cross-state injection

**Slug:** `cross-nonlinear` · **Scope:** across a decoder pair · **Status:** active
**Adds parameters** (three projections per branch).

Same layout and the same residual algebra as
[v3_2](v3_2_cross_scalar.md) — read that first. The only difference is the gate.

## Computation

```text
g(x, h) = W[ V(h - x) * SiLU(Ux) ]        * is elementwise
```

with `h - x` being the other branch's attention output. Substituted into the
pair:

```text
h_i = x + a_i(N^a_i(x))
o   = h_1 + m_1(N^m_1(h_1 + W_1[V_1(a_2) * SiLU(U_1 x)]))
    + h_2 + m_2(N^m_2(h_2 + W_2[V_2(a_1) * SiLU(U_2 x)])) - x
```

Where [v3_2](v3_2_cross_scalar.md) mixes by a single constant, here *how much*
of the other branch leaks in is modulated per token and per channel by the
block input itself. A token whose `Ux` is near zero in some channel receives
almost none of the other branch there.

The injection enters before the MLP's norm, and the run must be even-length —
both as in v3_2, and for the same reasons.

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | contiguous, even-length run of decoders to pair |
| `gate` | `"nonlinear"` | fixed for v3_3 |
| `gate_dim_ratio` | `0.25` | inner dimension `d` as a fraction of `hidden_size` |
| `direction` | `"both"` | `"both"`, `"forward"`, `"backward"` |
| `init_seed` | `1234` | seeds `U` and `V`, combined with the layer number |
| `freeze_original` | `False` | train everything, so v3_3 stays comparable to v3 |

## Cost

`U: hidden -> d`, `V: hidden -> d`, `W: d -> hidden`, so `3 * d * hidden` per
gate and two gates per pair. On Mellum2 (`hidden = 2304`, 14 pairs):

| `gate_dim_ratio` | `d` | parameters | share of 2.5B active |
|---|---|---|---|
| **0.25** (default) | 576 | **111M** | **+4.5%** |
| 0.5 | 1152 | 223M | +8.9% |
| 1.0 | 2304 | 446M | +18% |

Sweeping `d` is a one-line config change. The default is a bottleneck on the
reasoning that a gate modulates rather than computes, so it should not need
full width — and it is cheap enough to get a signal from quickly. If v3_3
clearly beats v3_2, widening `d` tests whether the gap is capacity-bound.

## Tensor parallelism

`U` and `V` are column-parallel and `W` is row-parallel, so the elementwise
product happens in the sharded inner dimension and TP behaves exactly as it
does for an ordinary MLP. No custom communication.

## Initialization

`W` is **zeroed**, so `g = 0` and training starts from the unmodified v3 graph,
exactly as in v3_2. `U` and `V` are drawn from an explicitly seeded
`torch.Generator` with `init_method_std / sqrt(2 * num_layers)` — the scaling
Megatron uses for a residual-output projection.

The seeded generator is deliberate: Megatron's `init_method_*` helpers draw
from the global RNG, which would make the gate depend on whatever consumed
randomness first and would not reproduce on a rerun.

`U` and `V` must **not** be zeroed. With `W = 0` the output is already zero;
zeroing the inputs too would leave `U` and `V` with no gradient and the gate
could never start learning.

## Preconditions

Identical to [v3_2](v3_2_cross_scalar.md): v1's layer-local set plus
`no_pipeline_parallel` and `no_expert_parallel_overlap`.

## Verification

```bash
uv run python -m pytest tests/unit_tests/models/surgery/test_cross_injection.py -q
```

Shared with v3_2. The v3_3-specific cases cover `gate_dim_ratio` validation and
the manifest shapes (`v_proj` and `u_proj` as `[d, hidden]`, `w_proj` as
`[hidden, d]`, with only `w_proj` zeroed).

## Suggested order

Run [v3_2](v3_2_cross_scalar.md) first. It is 28 parameters against 111M, and
it answers the prior question — does cross-branch mixing help at all — without
confounding it with added capacity. If v3_2 moves nothing, v3_3's gain (if any)
is capacity, not mixing.

## Status

Model implementation complete and tested. **Not yet run.** Same outstanding
items as [v3_2](v3_2_cross_scalar.md): training configs, and Hugging Face
export pending a block-level HF forward for cross-layer variants.
