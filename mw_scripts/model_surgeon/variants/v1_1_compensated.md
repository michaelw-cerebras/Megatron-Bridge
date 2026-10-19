# v1_1 — Ivan-style compensated parallel decoder

**Slug:** `compensated` · **Scope:** within a layer · **Status:** active

## Computation

```text
y = x + attention(norm1(x)) + MoE(norm2(x))
      + down_proj(SiLU(up_proj(compensatory_norm(x))))
```

The three branches read the same residual `x`. The compensatory branch matches
Ivan's self-contained implementation: it has its own RMSNorm, two bias-free
`hidden_size × hidden_size` projections, and a non-gated SiLU activation.

This replaces the earlier local v1_1 experiment. That implementation had no
correction norm, used a gated MLP, and zero-initialized its output projection;
its checkpoints and performance numbers are not compatible with this v1_1.

## Parameters and initialization

| name | default | meaning |
|---|---|---|
| `layers` | `"all"` | layers receiving a correction branch |
| `init` | `"random"` | HF-style normal initialization for both projections |
| `init_seed` | `1234` | deterministic per-layer initialization seed |

`compensatory_norm.weight` starts at one. Both linear weights use
`initializer_range` (`init_method_std` in Megatron), matching Hugging Face's
normal `post_init` behavior. There is no zero-output option.

For Mellum2 (`hidden_size=2304`, 28 layers), the branch adds approximately
297.3M matrix parameters plus 64,512 RMSNorm scalars. All original checkpoint
parameters are frozen; only these new parameters train. The pre-wrap hook
**raises if nothing ends up trainable**, since an all-frozen model would train
silently and merely look flat.

Report the cost as a share of *active* parameters, not total. The branch is
dense and always on, so on Mellum2 it is **+12% active** (297M on 2.5B) against
only +2.5% of the 12B total. For a study motivated by inference efficiency the
active figure is the one that matters.

Unlike [v2](v2_shifted.md), [v3](v3_adjacent.md) and [v3_2](v3_2_cross_scalar.md),
v1_1 is **layer-local**: it couples nothing across layer boundaries, so it imposes
no pipeline-parallel restriction. The hook matches layers by global
`layer_number` rather than list position and stays correct under PP.

## Hugging Face export

The added tensors round-trip as:

```text
model.layers.*.compensatory_mlp.norm.weight
model.layers.*.compensatory_mlp.up_proj.weight
model.layers.*.compensatory_mlp.down_proj.weight
```

`config.json` retains `megatron_bridge_surgery_variant: v1_1`, so the HF
compatibility implementation rebuilds the correct graph. Always export against
the surgeon-derived directory; a stock Mellum config selects v0.

## Verification

```bash
uv run python -m pytest \
  tests/unit_tests/models/surgery/test_compensated.py \
  tests/unit_tests/models/surgery/test_v1_1_hf_export.py -q

PYTHONPATH=$PWD/src uv run python mw_scripts/model_surgeon/check_v1_1_roundtrip.py
```

The GPU round-trip checks exact correction tensor coverage, graph selection,
freezing, HF reload, and finite logits.

## Interpreting the results

On Llama 4B (slides 15-16) the earlier compensatory arm had the **worst**
training loss of the three (~1.70 against 1.60 parallel and 1.50 original) and
the **best** downstream evals (55.2% against 55.05% original and 51.15%
parallel).

The two disagree for a structural reason: training loss is measured on the
continued-pretraining mixture, and frozen weights cannot adapt to that
distribution (higher loss) but also cannot forget the original capabilities
(better evals). **Do not rank this arm by training loss.** It is the clearest
case in the whole matrix of loss failing to predict downstream behaviour.
