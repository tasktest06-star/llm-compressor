# Unassigned issues — explanations and fix plans

Every open issue on `vllm-project/llm-compressor` that is **unassigned** and
beginner-to-medium in scope, captured on **2026-11-08**.

Each issue has two pages: one explaining the issue in detail, one planning the
fix.

!!! info "Why not the `good first issue` label?"
    All five issues carrying that label are **already assigned**. The label is
    also attached to research-scale work (implement a paper, replace a
    tensor-walking abstraction, build a calibration pipeline) rather than
    beginner work — those are planned separately under
    `docs/developer/good-first-issues/` on the `good_first_issue` branch.

    The issues here are the actual unclaimed beginner-to-medium work: unlabelled
    bug reports, most with a reproducer and a proposed fix from the reporter.

## Summary

| Issue | Topic | Reproduced | Existing PR | Difficulty | Pages |
|---|---|---|---|---|---|
| **#3306** | Distributed calibration `IndexError` when samples > dataset | ✅ exact | none | ⭐ easiest | [issue](3306-issue.md) · [plan](3306-plan.md) |
| **#3301** | HIGGS applies the wrong scheme to split MoE experts | ✅ exact | none | ⭐⭐ | [issue](3301-issue.md) · [plan](3301-plan.md) |
| **#3303** | Frozen save makes a later compressed save retain dense weights | ✅ exact | none | ⭐⭐⭐ | [issue](3303-issue.md) · [plan](3303-plan.md) |
| **#3324** | Reusing a recipe in a second `oneshot` raises | ✅ exact | ⚠️ [#3353](https://github.com/vllm-project/llm-compressor/pull/3353) | ⭐⭐ | [issue](3324-issue.md) · [plan](3324-plan.md) |
| **#3337** | AWQ leaves the MoE router out of hybrid balance layers | ✅ source-level | ⚠️ [#3338](https://github.com/vllm-project/llm-compressor/pull/3338) | ⭐⭐ | [issue](3337-issue.md) · [plan](3337-plan.md) |
| **#3479** | Automatic batch size selection, better data collators | n/a (RFC) | none | ⭐⭐⭐⭐ | [issue](3479-issue.md) · [plan](3479-plan.md) |
| **#3468** | Consolidate and convert targets for vLLM | n/a (mirror) | none | ⭐⭐⭐⭐⭐ | [issue](3468-issue.md) · [plan](3468-plan.md) |

**15 pages: 7 issue explanations, 7 fix plans, this index.**

## Where to start

**#3306.** A clamp runs one statement too late, so every rank crashes whenever
`num_calibration_samples` exceeds the dataset — which it does by default, since
`oneshot` defaults to 512. CPU-only reproducer, CPU-only test, no GPU anywhere,
and the fix is verified below.

Then **#3301**: pure regex logic, reproducible in under a second with no ILP
solver and no model — unusually cheap for the HIGGS subsystem.

## Two have PRs already — review, don't rewrite

| Issue | PR | State | What to do |
|---|---|---|---|
| #3324 | [#3353](https://github.com/vllm-project/llm-compressor/pull/3353) | open since 2026-11-03 | Review it. Two contributors already worked this. |
| #3337 | [#3338](https://github.com/vllm-project/llm-compressor/pull/3338) | open since 2026-10-20 | Review it. **+4/−1** source lines against exceptional evidence. |

#3338 is the standout: a four-line fix, backed by routing analysis over 943k
tokens, paired confidence intervals on two reference models in two languages,
and a third-party published checkpoint showing the same signature — sitting
unmerged for three weeks. If you review one thing from this set, review that.

## #3468 is not really unassigned

Its body is one line: *"Mirror of
[compressed-tensors#1006](https://github.com/vllm-project/compressed-tensors/issues/1006)"*.
That upstream issue is **assigned to `GOavi101`**. The mirror is unassigned only
because it is a tracking stub. Ask upstream before touching it.

## Verification performed

Claims here were checked against the tree at `c21cbc8b`, not taken from the
issues. **Four of the five bugs were independently reproduced** in a Python 3.13
+ CPU-torch environment:

| Issue | Evidence |
|---|---|
| #3306 | `rank 0: IndexError: Index 127 out of range for dataset of size 100` — matches the report exactly |
| #3301 | `experts.1.gate_proj: ILP=W8A16 applied_bits=4 matching_groups=2` — the assigned 8-bit layer gets 4 bits |
| #3303 | compressed-first `int8`/107,152 B/`int-quantized` vs frozen-then-compressed **`float32`/219,064 B/`dense`** |
| #3324 | `run 0: OK` / `run 1: RuntimeError: Cannot initialize a modifier that has already been initialized` |
| #3337 | Source-level: `mappings.py:64` has `"re:.*mlp.gate$"`, `dynamic_mappings.py:120` has only a TODO. The end-to-end numbers need a Qwen3.5-MoE checkpoint. |

**The #3306 fix was also implemented and tested, then reverted** (this branch is
documentation only):

| World size | Per-rank sample counts | Sum |
|---|---|---|
| 2 | 50, 50 | 100 |
| 3 | 33, 33, 34 | 100 |
| 7 | 14, 14, 14, 15, 14, 14, 15 | 100 |

Every row used exactly once, including when the count does not divide evenly.
That run also established a trap worth knowing: **the fix must be additive.**
Moving the existing clamp instead of adding one leaves `num_samples` larger than
the rank's slice, and `RandomSampler` then draws with replacement — silently
duplicating calibration data, which is worse than the crash it replaces.

All 28 Mermaid diagrams were parsed with Mermaid's own parser.

## Findings not in any issue body

- **#3303 is the same bug class as
  [#3369](https://github.com/vllm-project/llm-compressor/issues/3369).** Both
  overwrite an *unresolved default* with a concrete value and lose the
  information that it was only a default — `format=None` → `dense` here,
  `Sentinel("static")` → `"static"` there.
- **#3301 is the same shape as
  [#3329](https://github.com/vllm-project/llm-compressor/issues/3329).** Both
  use a name as an identity when it is not unique. Fixing #3329 makes #3301
  fire *more* often, because more candidates reaching the ILP means more mixed
  assignments.
- **#3479 contains a separable regression.** `disable_lm_head` has no effect in
  the sequential pipeline, because subgraph onload restores `module.forward`
  from a `_original_forward_func` captured before the patch. The last subgraph
  materializes full logits — **9.96 GB** for a 16×2048 batch at Qwen3's 151,936
  vocab, and 1.47 GiB instead of 0.53 GiB even at batch size 1. It is a
  regression from 0.14 → 0.15 with a verified candidate fix. That should be its
  own PR and should not wait on the RFC.
- **#3479's token-loss problem is the sampler, not the collator.** At batch size
  16 the *default* shuffled sampler keeps 21 % of calibration tokens;
  length-aware sampling keeps 93.5 %. Anyone raising `batch_size` for speed
  today silently discards most of their data.
- **#3468's consolidation risk is #3301's bug at config scale.** An over-broad
  regex silently quantizes a module that should have been ignored, so the
  consolidator needs round-trip verification rather than the "95 % of cases"
  the design accepts.

## Caveat

These are plans, not endorsements. Each names what is unresolved and what to ask
before coding — notably #3303, where whether `save_pretrained` may mutate the
live model is a maintainer's call, and that answer decides whether the
deliverable is a patch or a documentation change.
