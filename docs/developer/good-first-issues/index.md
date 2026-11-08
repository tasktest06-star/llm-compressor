# Good first issues — implementation plans

Implementation plans for every issue carrying the
[`good first issue`](https://github.com/vllm-project/llm-compressor/issues?q=is%3Aopen+label%3A%22good+first+issue%22)
label on `vllm-project/llm-compressor`, captured on **2026-11-08**.

Each issue has an overview page. Where more than one viable design exists, each
one gets its own page so the trade-offs can be compared side by side rather than
buried in prose.

!!! warning "Read the overview before writing code"
    **Four of the five issues have been re-scoped, blocked, or contradicted by
    their own comment threads.** The issue body alone is misleading in each
    case — see the "status" column below. Every overview page opens with what
    actually changed.

## Summary

| Issue | Topic | Assignee | Status | Plans |
|---|---|---|---|---|
| [#3453](3453-aimer.md) | AIMER expert pruning | `rohan9446` | ⚠️ Approach superseded in comments | [A: converter](3453-approach-a-converter.md) ✅ · [B: modifier](3453-approach-b-modifier.md) |
| [#3340](3340-masking.md) | Masking for modifiers & observers | `HDCharles` (tracker) | 🔨 4 sub-tasks, all claimed | [GPTQ](3340-task-gptq-hessian.md) · [iMatrix](3340-task-imatrix-observer.md) · [REAP](3340-task-reap-saliency.md) · [activations+KV](3340-task-activation-kv-observers.md) |
| [#3161](3161-pytree.md) | Replace nested tensor logic with pytree | `aayush7511` | ⚠️ Unresolved design disagreement | [A: full pytree](3161-approach-a-full-pytree.md) · [B: targeted fix](3161-approach-b-targeted-fix.md) · [C: pytree + registration](3161-approach-c-pytree-with-registration.md) ✅ |
| [#2974](2974-rac.md) | Reasoning-Aware Compression | `HDCharles` / `AbigaleD` | ⚠️ Re-scoped; headline result not reproduced | [A: script](2974-approach-a-standalone-script.md) ✅ · [B: datasets module](2974-approach-b-datasets-module.md) ❌ declined |
| [#2480](2480-gpu-offload.md) | GPU offloading | `HDCharles` | 🚫 **Blocked** on an in-flight refactor | [A: rebase](2480-approach-a-wait-and-rebase.md) ✅ · [B: CT primitive](2480-approach-b-compressed-tensors-util.md) · [C: dist util](2480-approach-c-dist-util.md) |

✅ recommended · ❌ declined by maintainers

**20 pages: 5 overviews, 11 approach pages, 4 sub-task pages.**

## If you are picking one up

Every one of the five is **already assigned**. `CONTRIBUTING.md` asks you to
comment on the issue and wait for a maintainer to assign it before starting:

> **Wait for the "Green Light"**: A maintainer will assign the issue to you.

So check the thread before writing code. Several of these have had work
abandoned and re-claimed, and at least one has a parked branch whose author
offered to finish it.

By difficulty, if one frees up:

| Rank | Issue | Why |
|---|---|---|
| 1 | [#3453 AIMER](3453-aimer.md) | Closed-form score from weights. No calibration loop, no hooks, CPU-only, shard-at-a-time. The scoring function is the only new logic. |
| 2 | [#3340 sub-tasks](3340-masking.md) | Shared helpers already written. Each sub-task is one accumulation site. CPU-testable. |
| 3 | [#3161 pytree](3161-pytree.md) | Small surface, but needs a design decision resolved with a maintainer first. |
| 4 | [#2974 RAC](2974-rac.md) | Little code; cost is GPU time across three calibration arms and two bit-widths. |
| 5 | [#2480 GPU offload](2480-gpu-offload.md) | **Not a good first issue today.** Blocked, needs multi-GPU hardware, touches distributed collectives. Four prior volunteers. |

## Findings worth carrying across issues

Things established in comment threads and linked PRs that are **not** in any
issue body, and that cost real accuracy if missed.

### Keep BOS in any calibration mask

[PR #3365](https://github.com/vllm-project/llm-compressor/pull/3365) found that
masking the chat prefix wholesale makes things *worse*. Llama-3-8B-Instruct,
gsm8k:

| Mask | W4A16 | W3A16 |
|---|---|---|
| None | 0.738 | 0.613 |
| First 5 tokens masked | 0.691 | 0.384 |
| Same, position 0 restored | recovered | recovered |

Position 0 is the **attention sink**: its activations are enormous and GPTQ must
see them to compensate their quantization error. Affects
[#3340](3340-masking.md) and [#2974](2974-rac.md). Note this may *invert* for
activation-range observers — see
[that sub-task](3340-task-activation-kv-observers.md).

### Masking helps; on-policy traces are unproven

`AbigaleD` could not reproduce RAC's headline gain at single-GPU scale. At
matched token budgets on Qwen3-1.7B W4A16, masked *dataset* responses beat
masked *own-CoT* traces on MATH-500 (0.778 vs 0.752). The reproduced effect is
prompt masking (p = 0.009); on-policy generation was p = 0.12 *against*. Any
comparison must equalize loss-mask tokens — the original apparent win came from
CoT traces simply being longer. See [#2974](2974-rac.md).

### Offload copies must be written after the collective completes

On NCCL, a broadcast's write to its target tensor is only observable once the
work handle is waited on. Copying into an offloaded parameter before
`_wait_for_comms` is a race that passes on small models and corrupts weights
intermittently under load. See [#2480](2480-gpu-offload.md).

### Verified pytree behaviour (torch 2.15.0, transformers 5.22.2)

The [#3161](3161-pytree.md) disagreement turns on facts both parties got
partly right:

```
dict / OrderedDict / list / tuple / namedtuple   traversed, type preserved
plain dataclass                                   single leaf — NOT traversed
ModelOutput                                       registered, rebuilt as itself
```

And a bug found while checking: `tensors_to_device` matches `str` as `Iterable`,
so a text field in a batch raises `RecursionError`; `None` and `int` raise
`ValueError`. Also confirmed that today's walker downgrades a `ModelOutput` to
an `OrderedDict`, breaking attribute access.

### Stale API references in issue bodies

[#2480](2480-gpu-offload.md)'s pseudocode uses `DeviceCache` and
`update_parameter`; neither exists in compressed-tensors 0.20.2a20261103.
[#3453](3453-aimer.md)'s title asks for a modifier inheriting
`ExpertSparsityBase`, which a maintainer later advised against. Check APIs
against the installed version before planning around them.

## How these plans were produced

- Every issue body **and every comment** read, including the re-scoping comments
  that supersede four of the five bodies.
- Cited file paths, line numbers and class names checked against the tree at
  `c21cbc8b`.
- Behavioural claims verified by running them — pytree specs, the
  `ModelOutput` downgrade, the `str` recursion, the observer registry, the
  `ExpertPruner` metric interface, and the available `compressed_tensors.offload`
  exports.
- All 30 Mermaid diagrams parsed with Mermaid's own parser, not eyeballed.

These are plans, not endorsements: each one names what is unresolved and what
should be asked on the issue before coding starts.
