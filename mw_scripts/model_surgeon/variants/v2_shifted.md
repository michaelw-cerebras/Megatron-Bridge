# v2 — shifted attention / MLP overlap

**Slug:** `shifted` · **Scope:** across adjacent layers · **Status:** active

## Computation

For a run of layers `s..e`:

```text
u_s    = x     + a_s(N_s^a(x))
u_i    = u_i-1 + m_i-1(N_i-1^m(u_i-1)) + a_i(N_i^a(u_i-1))     for s < i <= e
output = u_e   + m_e(N_e^m(u_e))
```

Layer `i`'s MLP and layer `i+1`'s attention read the same residual, so they are
co-schedulable — the same win as [v1](v1_parallel.md), but each attention still
sees the output of the MLP two positions back rather than nothing. That is the
reason v2 is expected to lose less than v1, and the slide deck's Llama 4B curve
(purple) sits between the original and plain parallel.

Every attention and every MLP in the run executes exactly once. The first layer
in the run runs attention only; the last also runs its own MLP to close the run.
Counting: attentions `s..e` (e-s+1) and MLPs `s..e` (e-s+1). No weight is
duplicated or dropped, and the state dict is unchanged.

A run of length 1 reduces exactly to the original sequential layer. That is the
cheapest correctness check on the schedule and it is a unit test.

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | the contiguous run of layers to shift |

The selection must be **contiguous** — the variant couples neighbours, so a
gap has no defined behaviour and is rejected. Shifting a prefix of the layers
is how `p` is swept.

## Preconditions

All of [v1](v1_parallel.md)'s, plus:

- `no_pipeline_parallel` — layer `i`'s MLP runs inside layer `i+1`'s forward. Across
  a pipeline boundary those layers live on different stages, so the graph cannot be
  built without relocating weights, which would break the byte-identical-checkpoint
  invariant. PP=1 only, enforced at spec-build time and again in the pre-wrap hook.
  Mellum2 already runs PP=1 (EP=8 carries the sharding).
- `no_expert_parallel_overlap` — the EP overlap scheduler assumes a layer's MoE runs
  in that layer's forward.

## Implementation

The cross-layer reference cannot live in the layer spec: a spec describes a
layer in isolation and knows nothing about its siblings. It is installed on the
built model through Bridge's `register_pre_wrap_hook`, which runs on the raw
`GPTModel` before mixed-precision and DDP wrapping
(`src/megatron/bridge/models/surgery/wiring.py`).

References are stored in the layer's `__dict__`, not as module attributes:
`torch.nn.Module.__setattr__` would register a second owner for each module and
duplicate it in the state dict. A unit test asserts the state-dict keys are
unchanged after wiring.

An unwired layer raises rather than running a partial graph.

## Measured results

From `parallel_mlp_attn.pdf`, Llama 4B 16L, first 1,066 optimizer steps, same
seed and schedule as the other arms:

| arm | early loss (step ~200) | note |
|---|---|---|
| original sequential | lowest | baseline |
| v2 shifted | between | closer to original than v1 |
| v1 parallel | highest | |

The slide's own caption notes the shifted curve is incomplete, so treat the
ordering as directional, not as a final number. No downstream evals yet.

## Open questions

- Does the eval gap follow the loss ordering (v2 better than v1), or does it
  collapse the way v1's did despite a near-original loss?
- v2's inference win is smaller than v1's (attention still waits on the
  previous MLP across the residual). Worth measuring throughput, not just loss,
  before concluding it is the better trade.
