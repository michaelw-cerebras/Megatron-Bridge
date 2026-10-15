# v0 — original

**Slug:** `original` · **Scope:** none · **Status:** active

The unmodified model. `v0` exists for two reasons, both of which matter more
than they look:

1. **Baseline arm.** Every comparison needs the original measured under the
   same seed, mixture, optimizer, schedule and budget as the variants. Running
   it through the same code path as the variants is what makes the numbers
   comparable, rather than "the original, from some earlier run".

2. **Degenerate-equivalence target.** `v0` rewrites no layers, so
   `wrap_spec(base, variant="v0")` must return exactly the family's own specs.
   Any variant with an empty layer selector must do the same. If that test ever
   fails, the surgery harness itself is changing the model and every result it
   has produced is suspect.

## Computation

```text
h = x + a(N_a(x))
y = h + m(N_m(h))
```

## Parameters

| name | default | meaning |
|---|---|---|
| `layers` | `"none"` | fixed; `v0` has no layer module to apply |

Setting a non-empty selector on `v0` is an error, not a no-op.
