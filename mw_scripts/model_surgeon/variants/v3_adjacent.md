# v3 — adjacent decoders in parallel

**Slug:** `adjacent` · **Scope:** across whole decoders · **Status:** active

## Computation

For a group of `g` adjacent decoders, each with
`delta_i(u) = a_i(u) + m_i(u + a_i(u))`:

```text
y = x + sum_i delta_i(x)
  = B_1(x) + ... + B_g(x) - (g - 1) * x
```

Attention and MLP stay **sequential inside** each decoder; only the decoders
themselves run in parallel. This is the most aggressive depth reduction of the
three: with `group_size=2` the effective sequential depth halves.

A group of size 1 reduces exactly to the original sequential decoder — the
degenerate check, and a unit test.

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | the contiguous run of decoders to group |
| `group_size` | `2` | decoders per parallel group |

The selected run must be contiguous and its length a multiple of `group_size`;
a partial group has no defined behaviour and is rejected rather than silently
left sequential.

## Preconditions

Identical to [v2](v2_shifted.md): all of v1's, plus `no_pipeline_parallel` and
`no_expert_parallel_overlap`.

## Implementation

Within a group, every decoder except the last is a **pass-through** that returns
its input untouched; the last member executes all members' sublayers against
that shared input and sums the deltas. This keeps each module running exactly
once with no per-forward mutable state, which would be unsafe under recompute or
graph capture.

Wiring and the state-dict invariant work exactly as in [v2](v2_shifted.md).

## Measured results

From `parallel_mlp_attn.pdf`, Llama 4B 16L, 2,000 optimizer steps, `group_size=2`:

| arm | final train loss |
|---|---|
| original sequential | lowest |
| v1 parallel | middle |
| v3 adjacent | highest |

v3's initial loss spikes to ~8.3 (vs ~6.7 for v1), and it stays above v1 for the
whole run. Halving the sequential depth costs more than parallelizing within a
layer, which is the expected ordering. No downstream evals yet.

## Open questions

- Is the gap recoverable at all at a larger token budget, or is depth doing
  something parallel width cannot replace?
- Does `group_size=2` on a *subset* of layers (say the last quarter, where
  representations change least) cost much less than applying it everywhere?
  That is the natural first sweep and it is already expressible via `layers`.
