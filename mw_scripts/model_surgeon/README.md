# model_surgeon

Rearrange a pretrained model's decoder graph, then measure what the
rearrangement costs and how fast it is recovered.

The unit of study is a **(variant, model, parallelization ratio, token budget)**
tuple. The point of this directory is that adding one more cell to that matrix
should be cheap, and that cells should be comparable to each other.

## The invariant everything rests on

> A graph-rearrangement variant keeps the state-dict keys **and the tensor bytes**
> identical to the original checkpoint. Only the computation graph changes.

Consequences worth knowing:

- A derived checkpoint is interchangeable with the original. Export to HF,
  checkpoint resume, and downstream RL all keep working with no special cases.
- Nothing in the weights says which graph to run. **The manifest is the only
  record.** Any consumer that guesses from a directory name will silently
  evaluate the wrong graph and produce plausible-looking numbers.
- Every variant must support an empty layer selector, and in that degenerate
  configuration must be indistinguishable from the original. That is the
  regression test every new variant inherits for free.

`v1_1` was the first variant to break this, and `v3_2`/`v3_3` also add weights
(gates rather than a correction branch). The weakened invariant they keep is
still useful — every tensor
inherited from the original is unchanged in key, shape and value, and the new
ones are enumerated in the manifest's `adds_parameters`. Each such variant
must describe its own tensors in `added_parameters.py`; one that declares
`adds_parameters` without a description raises, because a manifest naming
another variant's tensors is worse than no manifest at all. Variants that *edit*
existing tensors (dropping layers, merging experts, width surgery) would be a
third category and still do not exist.

## Variants

Canonical machine identifiers use underscores (`v1_1`, `v3_2`, `v3_3`); prose
may render them as v1.1, v3.2, and v3.3. The October 2026 naming cleanup kept
read compatibility for the historical artifact ids `v4`, `v5_1`, and `v5_2`.
New manifests never write those historical ids, while old run directories and
result keys remain unchanged for provenance.

| id | slug | scope | equation |
|----|------|-------|----------|
| `v0` | original | — | `h = x + a(N_a(x))`; `y = h + m(N_m(h))` |
| `v1` | parallel | within a layer | `y = x + a(N_a(x)) + m(N_m(x))` |
| `v2` | shifted | across adjacent layers | `u_i = u_{i-1} + m_{i-1}(N^m_{i-1}(u_{i-1})) + a_i(N^a_i(u_{i-1}))` |
| `v3` | adjacent | across whole decoders | `y = x + Σ_i δ_i(x)`, `δ_i(u) = a_i(u) + m_i(u + a_i(u))` |
| `v3_1` | merged-pool | across whole decoders | `z = x + Σ_i a_i(x)`; `y = z + ` global top-(g·k) over `concat_i router_i(z)` |
| `v1_1` | compensated | within a layer | `y = x + a(N_a(x)) + m(N_m(x)) + c_θ(x)` — **adds parameters** |
| `v3_2` | cross-scalar | across a decoder pair | v3 plus `m_i(N(h_i + α_i·a_other))` — **adds parameters** |
| `v3_3` | cross-nonlinear | across a decoder pair | same with `g = W[V(h−x) * SiLU(Ux)]` — **adds parameters** |

Per-variant notes live in `variants/`.

`v2`, `v3`, `v3_1`, `v3_2` and `v3_3` execute one layer's sublayers inside a neighbour's forward, which
is why they carry two extra preconditions: **`no_pipeline_parallel`** (PP=1
only — across a stage boundary the coupled layers live on different ranks, and
expressing the graph would require relocating weights) and
`no_expert_parallel_overlap`. Mellum2 already runs PP=1 with EP=8, so this is
not currently a restriction; lifting it would mean giving up the invariant
above.

### variant ≠ parameter

The parallelization ratio `p` is a **parameter** of a variant, not a separate
variant. Mellum2's "3 sliding-window blocks parallel, 1 dense block sequential"
and Llama's "all 16 layers parallel" are both `v1`, at `p=0.75` and `p=1.0`.
Keeping them as one variant is what makes a *gap vs p* curve possible.

Layer selectors:

| selector | meaning |
|---|---|
| `"all"` / `"none"` | every layer / no layer |
| `{"kind": "indices", "indices": [0, 3]}` | explicit zero-based global layers |
| `{"kind": "every_k", "k": 4, "offset": 0}` | every k-th layer |
| `{"kind": "config_list", "field": "window_attn_skip_freq"}` | layers where a per-layer config field is truthy |

`config_list` is how a model family's own structure drives selection without
the surgeon knowing anything about that family — `window_attn_skip_freq` for
Mellum, `moe_layer_freq` or `layer_types` elsewhere.

## Layout

| path | what it is |
|---|---|
| `surgeon.py` | CLI: clone a checkpoint and record its variant |
| `verify.py` | re-verify that a derived checkpoint's weights are byte-exact |
| `benchmark.py` | single-node throughput comparison |
| `check_v1_1_roundtrip.py` | GPU check: v1_1 Megatron → HF export → reload |
| `variants/` | one note per variant: motivation, equation, measured results |
| `results/` | loss curves and eval tables, keyed by variant and ratio |

The runtime lives in the package, not here, because layer specs must be
importable by training and by checkpoint resume:

| path | what it is |
|---|---|
| `src/megatron/bridge/models/surgery/variants.py` | the registry |
| `src/megatron/bridge/models/surgery/spec.py` | `apply_surgery` / `wrap_spec`, the injection points |
| `src/megatron/bridge/models/surgery/selectors.py` | layer selection |
| `src/megatron/bridge/models/surgery/requirements.py` | per-variant preconditions |
| `src/megatron/bridge/models/surgery/manifest.py` | the manifest contract |
| `src/megatron/bridge/models/surgery/sublayers.py` | shared attention/MLP invocation |
| `src/megatron/bridge/models/surgery/wiring.py` | post-build wiring for v2/v3/v3_1/v1_1/v3_2/v3_3 |
| `src/megatron/bridge/models/surgery/merged_moe.py` | joint routing over merged expert pools |
| `src/megatron/bridge/models/surgery/parallel_layer.py` | v1's layer |
| `src/megatron/bridge/models/surgery/shifted_layer.py` | v2's layer |
| `src/megatron/bridge/models/surgery/adjacent_layer.py` | v3's layer |
| `src/megatron/bridge/models/surgery/merged_pool_layer.py` | v3_1's layer |
| `src/megatron/bridge/models/surgery/compensated_layer.py` | v1_1's layer |
| `src/megatron/bridge/models/surgery/compensatory_mlp.py` | v1_1's correction branch, init and freeze |
| `src/megatron/bridge/models/surgery/cross_injection_layer.py` | the cross-injection layer |
| `src/megatron/bridge/models/surgery/cross_gate.py` | the scalar and nonlinear cross gates |
| `src/megatron/bridge/models/surgery/added_parameters.py` | manifest description of added weights |

## Model-agnostic by construction

`apply_surgery` decorates the model family's **own** layer spec:

```python
from megatron.bridge.models.surgery import apply_surgery
from megatron.bridge.models.mellum.modeling_mellum.layer_specs import mellum_layer_spec

apply_surgery(cfg.model, base_spec=mellum_layer_spec, variant="v2", params=params)
```

The family keeps ownership of what a layer *is* — attention variant, MoE
layout, per-layer RoPE. Surgery only decides which layers get the rewritten
graph. Nothing under `surgery/` mentions a model family, and no family needs a
`parallel_<family>_layer_spec`.

For `v2`/`v3`/`v3_1`/`v3_2`/`v3_3`, `apply_surgery` additionally registers a `pre_wrap_hook` that
installs each layer's reference to its neighbour's sublayers once the model is
built (a layer spec describes a layer in isolation and cannot express this).
Those references live in the layer's `__dict__` rather than as module
attributes, because `torch.nn.Module.__setattr__` would register a second owner
and duplicate the module in the state dict. A layer that was never wired raises
instead of running a partial graph.

Two guards make an unsupported model fail loudly instead of training a wrong
graph:

- **Bridge check.** `surgeon.py` refuses a model family with no registered
  Megatron Bridge. Bring the model up first (`adding-model-support`); surgery is
  downstream of that, not a substitute for it. Override with `--no-check-bridge`.
- **Capability probe.** Each variant declares preconditions (pre-norm residual,
  no hidden dropout, no selective recompute, no CUDA graphs, …) checked against
  the config at spec-build time. Preconditions are only enforced when the
  selector actually rewrites something.

## Run

Validate without writing:

```bash
python mw_scripts/model_surgeon/surgeon.py \
  --source JetBrains/Mellum2-12B-A2.5B-Thinking \
  --output /absolute/v1-hf-directory \
  --variant v1 --dry-run
```

Create the derived directory:

```bash
python mw_scripts/model_surgeon/surgeon.py \
  --source JetBrains/Mellum2-12B-A2.5B-Thinking \
  --output /absolute/v1-hf-directory \
  --variant v1
```

Restrict to the sliding-window layers (the slide deck's Mellum2 arm):

```bash
python mw_scripts/model_surgeon/surgeon.py \
  --source JetBrains/Mellum2-12B-A2.5B-Thinking \
  --output /absolute/v1-swa-hf-directory \
  --variant v1 \
  --params '{"layers": {"kind": "config_list", "field": "window_attn_skip_freq"}}'
```

`--source` takes a local directory or a Hub model ID. Hub models reuse
`HF_HOME` or the existing Mellum pipeline cache when it is on the same
filesystem as the output, otherwise `<output-parent>/.cache/huggingface/hub`.
A cache is only reused when same-filesystem, so the default hardlink mode stays
zero-copy. Use `--copy-mode copy` for a fully independent, SHA-256-verified
directory.

Every conversion verifies that the weight maps match and that every shard is
byte-exact (same inode for hardlinks, SHA-256 for copies). Re-check later:

```bash
python mw_scripts/model_surgeon/verify.py \
  --source JetBrains/Mellum2-12B-A2.5B-Thinking \
  --derived /absolute/v1-hf-directory
```

## Manifest

`architecture_surgery.json`, schema v2:

```json
{
  "schema_version": 2,
  "artifact_format": "huggingface_safetensors",
  "variant": {
    "id": "v1",
    "slug": "parallel",
    "equation": "y = x + a(N_a(x)) + m(N_m(x))",
    "params": {"layers": {"kind": "all"}}
  },
  "target_architecture": "parallel_attention_moe",
  "weights_identical_to_source": true
}
```

`target_architecture` is the schema-v1 field, still written and still read, so
checkpoints and consumers from before the registry keep working. New consumers
should key on `variant.id`.

**If you are integrating an evaluation harness:** read this manifest (or
`config.json`'s `megatron_bridge_surgery_variant`) and build the matching
forward graph. HF's own `modeling_*.py` implements the *original* sequential
graph, so loading a derived checkpoint into stock HF runs the wrong
computation — it will not error, it will just report wrong numbers.
`megatron.bridge.models.surgery.read_manifest` is the supported reader.

## Train a variant

Copy `mw_scripts/mellum2_3xh200/config.yaml` and set:

```yaml
model:
  hf_path: /container-visible/path/to/v1-hf-directory
  architecture: v1

run:
  directory: /absolute/v1-training-run
  resume_checkpoint: null

wandb:
  run_name: a-new-unique-run-name
```

`architecture` must name the same variant as the checkpoint's manifest;
training refuses a mismatched pairing. The legacy `parallel_attention_moe`
spelling still resolves to `v1`. After the first Megatron checkpoint, set
`resume_checkpoint` and keep `hf_path` pointing at the derived directory — its
manifest continues to identify the graph.

Keep the data mixture, batch size, optimizer, scheduler, and seed identical
across arms. Nothing enforces this yet; see Next below.

The first forward pass of a derived checkpoint is intentionally **not**
logit-equivalent to the original. The weights are identical, but each sublayer
now sees a different input.

## Tests

```bash
uv run python -m pytest tests/unit_tests/models/surgery/ -q
```

- `test_selectors.py` — layer selection and its failure modes
- `test_variants.py` — registry, `wrap_spec`, PP slicing, requirement probes,
  degenerate equivalence, v1 forward arithmetic
- `test_cross_layer_variants.py` — v2/v3 forward arithmetic against hand-written
  references, group-of-one degeneracy, wiring, state-dict invariance, PP rejection
- `test_merged_pool.py` — v3_1 joint routing against a hand-computed softmax, the
  `g·k` budget invariant, single-renormalization, router preconditions, gradient flow
- `test_compensated.py` — v1_1 forward, reproducible init, freezing, manifest
  `adds_parameters`
- `test_cross_injection.py` — v3_2/v3_3 forward against hand-computed references,
  zero-gate-equals-v3, norm placement, direction switch, gradient flow
- `test_v1_1_hf_export.py` — HF graph selection, v1_1 forward, save/reload parity,
  and refusal to load cross-layer variants as sequential
- `test_surgeon_cli.py` — clone and byte-exactness, manifest v1/v2, the
  training wire-up, config/checkpoint disagreement

## Handing a checkpoint to another team

Only `v0`, `v1` and `v1_1` can be loaded by Hugging Face. `hf_reference.py`
implements those three graphs and **raises** for `v2`/`v3`/`v3_1`/`v3_2`/`v3_3` rather than
quietly running the sequential graph — a checkpoint that loads cleanly and
reports wrong numbers is the failure mode worth guarding against.

Export against the **surgeon-derived directory**, never a stock config:
`save_hf_pretrained` copies the source `config.json` through verbatim, so the
`megatron_bridge_surgery_variant` marker has to already be on it.

## Before launching v3_1

v3_1 is the first variant that reimplements part of Megatron's router, so CPU
tests cannot cover everything. Run these on one node before committing a long job:

1. **A few steps at the real EP degree.** The merged routing hands each
   `MoELayer` a probability slice with many zeros; the per-layer token counts are
   now data-dependent, so the all-to-all is load-imbalanced in a way the other
   arms never exercise. v3_1 requires `moe_token_dispatcher_type: alltoall`;
   DeepEP and all-gather are intentionally rejected.
2. **Confirm `merged_pool_frac_0` appears in the logs** and is near 0.5 at step 0.
   If it is pinned at exactly 0.5 the merge is not doing anything; if it is 0 or 1
   something is wrong with the split.
3. **Check grad norms on both pools' routers are nonzero.** The joint aux loss is
   attached through the first pool's probs only; a wiring mistake there would
   silently stop one router from learning.
4. **Compare step time against the v3 arm.** Expert FLOPs are identical by
   construction, so a large gap means dispatch imbalance, not extra compute.

The initial 2026-10-14 hardware smoke completed two full iterations on one
8xH200 node at EP=4, including backward and optimizer update. This establishes
functional execution; it is not a throughput benchmark.

## Next

Not built yet, in rough priority order:

1. **Experiment specs** — a baseline config plus variant arms, with shared
   fields (seed, mixture, optimizer, schedule, token budget) that an arm cannot
   override. Today comparability is maintained by hand.
2. **`results/`** — structured loss and eval output keyed by
   `(model, variant, params, p, budget, seed)`, with the slide deck's figures
   rendered from it rather than assembled manually.
3. **Lift the PP=1 restriction on v2/v3**, if a model that cannot fit on one
   pipeline stage needs them. Would require either accepting a different graph
   at stage boundaries or giving up the byte-identical-checkpoint invariant —
   neither is free, so defer until a real model forces the question.
4. **Throughput measurement per variant.** The loss/eval cost is measured; the
   inference win that motivates the whole exercise is not. v2's win is
   structurally smaller than v1's, which matters for picking between them.
5. **Router-only warmup for v3_1.** Freeze everything except the two routers
   for an initial phase, so the merged selection stabilizes before the experts
   start chasing it. Suggested by Ivan; a separate arm, not a change to v3_1.
