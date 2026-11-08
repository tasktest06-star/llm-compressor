# #3453 — Implement AIMER

- **Issue:** [vllm-project/llm-compressor#3453](https://github.com/vllm-project/llm-compressor/issues/3453)
- **Labels:** `enhancement`, `good first issue`, `good follow-up issue`
- **Assignee:** `rohan9446` (assigned 2026-11-05)
- **Paper:** [arXiv 2603.18492](https://arxiv.org/html/2603.18492v1)
- **Approaches:** [A — calibration-free converter](3453-approach-a-converter.md) (recommended),
  [B — pruning modifier](3453-approach-b-modifier.md)

## What is being asked

The issue text is one line: implement `AIMERPruningModifier` per the paper,
inheriting from `ExpertSparsityBase`.

**That instruction has since been superseded.** On 2026-11-06 `kylesayrs`
commented:

> I'll also add that we should probably not even implement this as
> `AIMERPruningModifier`, but instead implement as a calibration-free converter,
> `AIMERExpertPruner(ExpertPruner)`. AIMER doesn't need calibration data, so it
> would be simpler and faster to use this interface

Anyone starting from the issue title alone will build the wrong thing. The two
designs are written up separately; **approach A is the one a maintainer asked
for.**

## The scoring rule

AIMER scores each expert from its weights alone — no calibration data, no
activations:

```
AIMER(e) = sum(|w|) / sqrt(N * sum(w^2))
```

over all elements of expert `e`'s projections (gate / up / down), where `N` is
their total element count. This is a normalized sparsity ratio: the ratio of the
L1 norm to the L2 norm, scaled so it is comparable across tensors of different
size.

**Score direction is the one easy thing to get wrong.** The reference
implementation prunes the *highest*-scoring experts. Both of LLM Compressor's
pruning interfaces prune the *lowest* scores first, so the score must be
negated (`saliency = -AIMER`) or the comparison inverted. Get this backwards and
the result is not a crash — it is a model pruned of exactly the experts it
should have kept, which looks like "AIMER performs badly" rather than a bug.

## Why it is a good first issue

The whole method is a closed-form function of the weights. There is no
calibration loop, no hooks, no activation cache, and no dataset. The work is
almost entirely about fitting an existing interface correctly.

## Verified code touchpoints

| Path | Relevance |
|---|---|
| `src/llmcompressor/entrypoints/converter/expert_pruner.py` | `ExpertPruner(Converter)`. Scores experts and rewrites the checkpoint. `from_pretrained(..., metric=...)` with `Literal["magnitude", "saliency", "layerwise_saliency", "count"]`. |
| `src/llmcompressor/entrypoints/converter/expert_pruner.py:630` | `_magnitude_scores()` — the closest model to copy: opens shards with `safe_open`, scores, returns `dict[router_name, Tensor]` in `float64`, preserving router order. |
| `src/llmcompressor/modifiers/pruning/expert_sparsity/base.py:48` | `ExpertPruningModifierBase(Modifier)` — the in-memory path approach B would use. |

Two facts about `ExpertPruner` that shape approach A:

- Scores are keyed **by router name**, and the returned dict must preserve
  router (layer) order.
- It already handles both **2D** per-expert tensors
  (`...experts.3.gate_proj.weight`) and **3D** stacked tensors
  (`...experts.gate_proj.weight`, shape `[num_experts, out, in]`). AIMER scoring
  must handle both, which the existing `metric="magnitude"` path does *not*
  exercise, because it reads only the router weight — a single 2D tensor
  regardless of expert layout. This is the main piece of genuinely new code.

## Open question from the assignee

`rohan9446` asked:

> the base currently skips layers with all-zero routed-token counts. Would you
> be open to a small overridable check for whether a layer has usable
> statistics, so AIMER can use weight statistics without repurposing `count`?

This only arises under approach B. The converter path has no notion of routed
token counts, which is a further argument for approach A — the question
dissolves rather than needing an API change.

## Recommendation

Take **approach A**. It is what the maintainer asked for, it needs no new
lifecycle hooks, and it avoids the `count`-statistics question entirely. Keep
approach B on file only in case a reviewer wants AIMER usable inside an
in-memory `oneshot` run alongside other modifiers.

## Shared acceptance criteria

Whichever approach:

1. Scores match the paper's formula, verified against a hand-computed value on a
   small tensor.
2. Pruning direction matches the reference (highest AIMER pruned first).
3. Both 2D and 3D expert layouts covered.
4. An end-to-end run on a small MoE checkpoint, with the retained-expert count
   asserted and the result loadable.
5. A comparison against `metric="magnitude"` on the same checkpoint — not
   necessarily a win, but reported.
