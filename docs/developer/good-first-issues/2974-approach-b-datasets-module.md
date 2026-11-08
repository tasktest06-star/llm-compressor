# #2974 approach B — a `datasets` module in `src/` (declined)

Parent: [#2974 overview](2974-rac.md) · Build instead:
[A — standalone script](2974-approach-a-standalone-script.md)

This is the original RFC. **It was declined on scope.** Recorded so that nobody
re-proposes it without knowing why, and so the reasoning is available if the
decision is ever revisited.

## What was proposed

A first-class library feature:

```python
from llmcompressor.datasets import generate_calibration_traces

traces = generate_calibration_traces(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    dataset="open-r1/OpenR1-Math-220k",
    prompt_column="problem",
    num_prompts=512,
    max_new_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    backend="auto",      # vllm if installed, else transformers
    save_to=None,
)
```

Plus:

- new `TextGenerationDataset` subclasses — `open_r1_math`, `codeforces` —
  registered like the existing `gsm8k.py`
- `examples/reasoning_aware_compression/`
- a `docs/guides/` page on calibrating reasoning models
- unit tests for the trace builder and a small end-to-end `oneshot` test
- `vllm` behind an optional extra (`pip install llmcompressor[traces]`)

Explicitly excluded by the RFC itself: no new modifier, no change to Hessian
accumulation, no prompt-vs-decode reweighting.

## Why it was declined

Three maintainer objections, all on the record.

**1. It imports nothing from the project.** `brian-dellabetta`:

> it sounds like the `llmcompressor.datasets.generate_calibration_traces`
> function you are suggesting would not really import any logic or abstractions
> from llm-compressor itself?

Correct, and decisive. The function would sample from a model and write a
dataset. Nothing in it is specific to compression, so `src/` buys only an import
path.

**2. Dependency friction.** `brian-dellabetta`:

> vllm and llm-compressor often have incompatible dependency versions, so this
> may lead to friction for users.

An optional extra does not fully solve this: a resolver still has to satisfy
both, and `backend="auto"` means behaviour silently depends on whether vLLM
happens to be installed — two users running identical code get different
sampling paths.

**3. It is data preparation.** `HDCharles`:

> This data generation is not really an llm-compressor feature, its just
> generating data to use for calibration

and, on where it belongs:

> The data generation should probably happen in vllm along the same lines as
> what we do for speculators training

## The disagreement

`PKUWZP` (the paper's author) maintains the framing is wrong:

> I would say it's a novel model pruning method rather than data
> preparation/generation ... It's the E2E sparsity pipeline.

`HDCharles` asked directly what the gap was:

> If you just generate prompt + CoT + answer traces from your desired task and
> then mask out the prompt, that would be sufficient, right?

That question was not answered with a mechanism. And the RFC's own text supports
the maintainer's reading — it says the activation matrix is a **plain
concatenation with no reweighting**, so a teacher-forced pass over
`prompt + CoT + answer` reproduces it exactly. If that holds, there is nothing
left for a modifier to do, and the method genuinely is the data.

Two caveats worth preserving:

- The paper targets **sparsity at high ratios**, where `HDCharles` notes
  sparsity support may be reduced anyway ("it looks more and more likely that
  we'll remove some of that"). The regime where RAC's advantage is largest is
  the one the project is least invested in.
- The `oneshot` path was also noted to have largely **deprecated
  sparsification**, so the RFC's `SparseGPTModifier` recipe is not the natural
  target any more.

## What survives from the RFC

Approach A keeps the parts that were never contested:

| RFC element | Fate |
|---|---|
| Trace generation utility | Kept, as a script in `examples/`, not `src/` |
| Chat-template correctness (concatenate ids) | Kept — a real correctness point |
| Token budgeting (tokens vs samples) | Kept, and reinforced by the replication result |
| Example directory | Kept |
| Docs guide | Folded into the example README |
| Registered dataset subclasses | Dropped — unnecessary once generation is a script |
| `vllm` optional extra | Dropped — no dependency at all |
| Optional correctness filtering | Dropped from v1; the paper's headline numbers do not filter |

## If this is ever revisited

Conditions that would change the calculus:

1. A demonstrated mechanism that is *not* reproducible by teacher-forcing the
   concatenated string — that would make it an algorithm, not data.
2. Reproduced gains over masked dataset responses at matched token budget. The
   current evidence ([overview](2974-rac.md)) shows p = 0.12 *against*, at small
   scale.
3. vLLM and llm-compressor converging on compatible dependency ranges.
4. Renewed project investment in sparsity at high ratios.

Until then, the script-plus-example shape delivers the same user-visible
capability at a fraction of the maintenance cost.
