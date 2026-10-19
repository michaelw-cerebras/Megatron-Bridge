# v3_2 — parallel decoders with scalar cross-state injection

**Slug:** `cross-scalar` · **Scope:** across a decoder pair · **Status:** active
**Adds parameters** (one scalar per branch).

## Computation

Two adjacent decoders run in parallel, as in [v3](v3_adjacent.md), but each
MLP now also sees a learnable share of the *other* branch's attention output:

```text
h_i = x + a_i(N^a_i(x))                       for i in {1, 2}
g_1 = alpha_1 * (h_2 - x)      = alpha_1 * a_2
g_2 = alpha_2 * (h_1 - x)      = alpha_2 * a_1
o   = h_1 + m_1(N^m_1(h_1 + g_1))
    + h_2 + m_2(N^m_2(h_2 + g_2)) - x
```

The trailing `- x` compensates the double residual from `h_1` and `h_2`, so at
`alpha = 0` this is exactly v3.

### Two decisions the formula alone does not fix

**The injection enters before each MLP's own norm** — `m(N(h + g))`, not
`m(N(h) + g)`. The norm therefore rescales the sum, which bounds what a large
`alpha` can do. The cost is that `alpha` is not directly readable as a mixing
strength. A unit test pins this placement, because the other choice behaves
very differently and the difference would be invisible in a loss curve.

**Exactly two branches.** `g(x, h_other)` has one unambiguous "other" only in a
pair. Three or more would need a combining rule nobody has specified, so the
selector rejects an odd-length run rather than inventing one.

## The two directions are not the same thing

Pairing original decoder `i` with `i+1`:

| injection | what it is |
|---|---|
| `a_i -> m_{i+1}` (**forward**) | **Restoration.** In the original sequential model `m_{i+1}` saw `a_i`. Parallelization destroyed that; this gives it back. |
| `a_{i+1} -> m_i` (**backward**) | **A new pathway.** `m_i` ran *before* `a_{i+1}` in the original and never saw it. |

Only the forward direction repairs something. That asymmetry is worth an
ablation rather than an assumption, which is why `direction` is a parameter.

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | contiguous, even-length run of decoders to pair |
| `gate` | `"scalar"` | fixed for v3_2; see [v3_3](v3_3_cross_nonlinear.md) |
| `direction` | `"both"` | `"both"`, `"forward"` (`a_i -> m_{i+1}` only), `"backward"` |
| `init_seed` | `1234` | unused by the scalar gate, kept for parity with v3_3 |
| `freeze_original` | `False` | train everything, so v3_2 stays comparable to v3 |

## Cost

One scalar per branch: **28 parameters** on Mellum2 (14 pairs x 2). Effectively
free, which makes this the cheapest possible test of whether cross-branch
mixing helps at all.

## Initialization

`alpha` starts at exactly **0**, so training begins from the unmodified v3
graph and the model must earn any mixing it does. This also gives a free
degenerate-equivalence test: with the gate untouched, the forward pass must
equal v3 to the last bit.

Zero init does **not** mean zero gradient — `alpha` multiplies a real signal,
so it receives gradient from step one. A unit test asserts this, because a gate
that could never move would look like "mixing doesn't help".

## Preconditions

Cross-layer, so the same set as [v3](v3_adjacent.md): everything v1 requires,
plus `no_pipeline_parallel` (the paired decoders must live on one stage) and
`no_expert_parallel_overlap`.

## Relationship to v3_1

[v3_1](v3_1_merged_pool.md) already mixes, and more aggressively: its merged
MoE reads `z = x + a_1 + a_2`, so it sees **both** attention outputs,
unconditionally. v3_2 gives each separate MLP a *learnable* share instead.

That makes the three arms a clean decomposition:

| | mixing | expert pool |
|---|---|---|
| v3 | none | separate |
| v3_2 | learnable, partial | separate |
| v3_1 | forced, full | merged |

v3_1 against v3_2 isolates whether merging the pool is worth anything beyond
simply sharing the attention signal. On a **dense** model v3_2 is the only one
of the two that is even possible.

## Verification

```bash
uv run python -m pytest tests/unit_tests/models/surgery/test_cross_injection.py -q
```

Covers the forward arithmetic against hand-computed references, the
zero-gate-equals-v3 property, norm placement, each `direction` setting, the
even-length rule, freezing, gradient flow to the gate, and the manifest
description.

## Status

Model implementation complete and tested. **Not yet run.**

Outstanding: training configs (owned by the training agent) and Hugging Face
export. v3_2/v3_3 is cross-layer, so like v2/v3/v3_1 the HF reference deliberately
**refuses** to load it rather than silently running the sequential graph — see
the README's handoff section. Exporting the `cross_gate.*` tensors before a
block-level HF forward exists would produce a checkpoint that loads cleanly and
computes the wrong thing.
