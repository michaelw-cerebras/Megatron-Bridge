# v4 — parallel attention/MLP + compensatory MLP

**Slug:** `compensated` · **Scope:** within a layer · **Status:** active
**This is the first variant that adds parameters.**

## Computation

```text
y = x + a(N_a(x)) + m(N_m(x)) + c_theta(x)
```

Three branches read the same residual `x`. `c_theta` does **not** depend on the
attention output, so it stays co-schedulable with the other two — the
parallelism win of [v1](v1_parallel.md) is preserved.

Slide 14 shows no norm on the correction branch, and that is what is
implemented. Each layer has its own branch, so it adapts to that layer's
residual scale without needing one.

## Motivation

Parallelizing a block drops `m(x + a(x)) - m(x)`: the MLP stops seeing
attention's update. `c_theta` is trained to put that term back.

Slide 13 derives the correction as `c(x, a) ≈ m(x+a) - m(x)`, which depends on
`a` and is linear in it (`J_m(x)·a(x)`). Slide 14's diagram instead takes only
`x`, which is what makes it parallel. **We implement slide 14.** A branch
reading only `x` cannot represent the Jacobian-vector product; it learns an
average correction. The `c(x,a)` and learnable-gate forms are not implemented.

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | which layers get a correction branch |
| `expansion_factor` | `1` | branch hidden width as a multiple of `hidden_size` |
| `init` | `"zero_output"` | `"random"` or `"zero_output"` |
| `init_seed` | `1234` | combined with the layer number for reproducibility |

`zero_output` zeroes the output projection so the branch contributes nothing at
step 0 — the run then **starts exactly at v1** rather than at a perturbed
model. This is the production default: the random initialization produced a
non-finite forward loss at optimizer iteration 2 in the full Mellum2 8-GPU
smoke test. Random initialization remains available for controlled experiments.

## Cost on Mellum2

Gated MLP at `expansion_factor=1`, hidden 2304, 28 layers:

```
fc1 [2*2304, 2304] + fc2 [2304, 2304] = 15.9M per layer
28 layers                              = 446M
```

- **+3.7%** of total parameters (12B)
- **+18%** of *active* parameters (2.5B)

The branch is dense and always on, so for an efficiency-motivated study the
active-parameter number is the one that matters. Do not report only the +3.7%.

## Training

All original weights are frozen; only the correction branches train. The
pre-wrap hook initializes every branch then freezes the rest, and **raises if
zero parameters end up trainable** — a silent all-frozen model would otherwise
train nothing and look merely flat.

Despite optimizing only ~446M of 12B parameters, v4 must use a lower LR than
the full-finetune arms. With zero-output initialization, Adam's first update
moves millions of output-projection weights together; the unnormalized dense
residual can grow abruptly. Peak-LR GPU smokes rejected `3e-4`, `1e-4`, and
`3e-5` with a non-finite forward loss, while `3e-6` completed 12 finite
iterations at peak. The production config therefore uses `3e-6`.

v4 is layer-local, so unlike v2/v3/v3_1 it imposes **no pipeline-parallel
restriction**. The hook matches layers by global `layer_number`, not list
position, so it stays correct under PP.

### Initialization

Megatron runs with `perform_initialization=False` so the Hugging Face import
owns every original weight. Nothing would otherwise write to the correction
branch, so it is initialized explicitly, from a seeded `torch.Generator`
(**not** Megatron's `init_method_*` helpers, which draw from the global RNG and
would make a rerun irreproducible). Standard deviations match Megatron's:
`init_method_std` for `fc1`, `init_method_std / sqrt(2 * num_layers)` for `fc2`.

## Hugging Face export

The exported checkpoint is meant to be handed to other teams, so the whole
graph — not just the tensors — has to survive the trip.

- `hf_reference.py` implements v0, v1 and v4. **Cross-layer variants (v2, v3,
  v3_1) raise on load** rather than silently running the sequential graph.
- Correction weights export as
  `model.layers.*.compensatory_mlp.{gate_proj,up_proj,down_proj}.weight`
  through the standard `GatedMLPMapping` / `AutoMapping` path.
- `config.json` carries `megatron_bridge_surgery_variant` and
  `compensatory_mlp_expansion_factor`, so a reload rebuilds the right graph.

**Sharp edge:** `save_hf_pretrained` passes the *source* HF config through
verbatim; it does not rebuild it from the provider. So **always export against
the surgeon-derived directory**, which carries the marker. Exporting against a
stock Mellum config silently produces a checkpoint that reloads as v0 — it will
load cleanly and report plausible, wrong numbers. `megatron_to_hf_config` also
stamps the marker, which covers the `export_ckpt` path.

### Verification

```bash
# CPU: graph selection, forward, save/reload parity
uv run python -m pytest tests/unit_tests/models/surgery/test_v4_hf_export.py -q

# GPU: real Megatron modules -> HF export -> reload
PYTHONPATH=$PWD/src uv run python mw_scripts/model_surgeon/check_v4_roundtrip.py
```

The GPU check confirms 8 Megatron correction tensors become 12 Hugging Face
tensors (gate/up/down per layer), the exported config selects v4, and the
reloaded model produces finite logits.

## Expected behaviour

On Llama 4B (slide 15–16), training loss was the **worst** of the three arms
(~1.70 vs 1.60 parallel, 1.50 original) while downstream evals were the
**best** (55.2% vs 55.05% original, 51.15% parallel).

Loss and evals disagree here, and the likely reason is that training loss is
measured on the continued-pretraining mixture: frozen weights cannot adapt to
that distribution (higher loss) but also cannot forget the original
capabilities (better evals). **Do not rank this arm by training loss.**

## Status

Implemented and tested end to end. A full Mellum2 8-GPU run completed eight
optimizer iterations with finite losses using zero-output initialization. The
measured one-node winner is BF16, TP1/EP4, MBS2, all-to-all with permute fusion;
see `../PERFORMANCE.md`.
