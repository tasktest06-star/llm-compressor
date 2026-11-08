# AWQ Scale Search

Activation-aware Weight Quantization (AWQ) reduces quantization error by moving
magnitude out of the weights and into a per-channel scale that is folded into a
neighbouring operation. The scale itself is not derived in closed form — it is
chosen by a **grid search** over candidate scales, keeping whichever candidate
gives the lowest output error for that mapping.

This guide explains what that search sweeps, and how `duo_scaling` and `n_grid`
shape it.

## The candidate scales

For each mapping (one *smooth* layer feeding a set of *balance* layers), AWQ
collects the mean absolute activation per input channel, `x_mean`. Candidates are
then generated from a `ratio` in `[0.0, 1.0]`, which acts as an exponent:

| `duo_scaling` | Candidate scale |
|---|---|
| `False` | `x_mean ** ratio` |
| `True` | `x_mean ** ratio / w_mean ** (1 - ratio)` |

where `w_mean` is the mean of the balance layers' weights, normalized per
quantization chunk (tensor, channel, group, or block) according to their weight
scheme. Each candidate is then normalized by `sqrt(max * min)` so it neither
inflates nor shrinks the overall magnitude.

In other words, `ratio` interpolates how aggressively channel importance is
expressed in the scale. At `ratio = 0.0` with `duo_scaling` off, the scale is
`x_mean ** 0 = 1` for every channel — the **identity**, i.e. leaving the weights
untouched. That point is valuable as a baseline: if no scaling helps this
mapping, the search should be able to say so.

## `duo_scaling`

`duo_scaling` selects which statistics inform the scale.

- **`False`** — activations only. The grid sweeps `n_grid` ratios evenly across
  `[0.0, 1.0]`, and `ratio = 0.0` gives the identity.
- **`True`** (default) — activations *and* weights, as in the AWQ paper. Because
  the duo formula divides by `w_mean ** (1 - ratio)`, `ratio = 0.0` is no longer
  the identity, so the identity point `(0.0, duo_scaling=False)` is prepended
  explicitly and the remaining `n_grid - 1` points use the duo formula.
- **`"both"`** — hedges between the two. Half the budget (`n_grid // 2` ratios)
  is swept with `duo_scaling` off and the same ratios again with it on, so each
  formula gets an equal share of the points.

`"both"` is the setting used in several of the key-model examples: it costs no
more forward passes than the others for a given `n_grid`, and it avoids
committing to one formula for every mapping in the model.

## `n_grid`

`n_grid` is the number of candidates evaluated per mapping, and so is directly
proportional to AWQ's runtime. It defaults to `20`. Lowering it trades scale
quality for speed.

Because the ratios are spaced across the interval, the usable minimum depends on
`duo_scaling` — a grid needs at least two distinct ratios to be a search at all:

| `duo_scaling` | Minimum `n_grid` | Points produced |
|---|---|---|
| `False` | 2 | `n_grid` |
| `True` | 3 | `n_grid` (one identity + `n_grid - 1` duo) |
| `"both"` | 4 | `2 * (n_grid // 2)` |

A smaller value is rejected when the modifier is constructed, so a mistake
surfaces while the recipe is being validated rather than part-way through
calibration.

Note that `"both"` halves the budget before spacing it, so an odd `n_grid`
evaluates one fewer point than you asked for: `n_grid=21` sweeps 10 ratios twice,
for 20 points.

## Choosing values

- Leave `n_grid` at its default unless AWQ runtime is a problem. See
  [Runtime Performance](runtime.md) for what to expect.
- If you are cutting `n_grid` to save time, keep it well above the minimum —
  near the minimum the search has so few ratios that it is unlikely to beat the
  identity.
- `duo_scaling` is the setting worth experimenting with per model; `n_grid`
  mostly buys resolution.

## See also

- [Compression Schemes](compression_schemes.md) — the weight formats AWQ is
  typically paired with.
- [Observers](observers.md) — AWQ patches in `memoryless_minmax` observers for
  the duration of the grid search.
- [Recipes](recipes.md) — how to set these arguments in a recipe.
