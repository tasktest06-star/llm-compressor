# #3161 — Replace nested tensor logic with pytree

- **Issue:** [vllm-project/llm-compressor#3161](https://github.com/vllm-project/llm-compressor/issues/3161)
- **Labels:** `enhancement`, `good first issue`, `good follow-up issue`
- **Assignee:** `aayush7511`
- **Related PR:** [compressed-tensors#964](https://github.com/vllm-project/compressed-tensors/pull/964) (`send_tensors` → `tree_map`)
- **Approaches:** [A — full pytree migration](3161-approach-a-full-pytree.md) ·
  [B — targeted fix, no pytree](3161-approach-b-targeted-fix.md) ·
  [C — pytree plus registration](3161-approach-c-pytree-with-registration.md) (recommended)

## What is being asked

Both LLM Compressor and compressed-tensors hand-roll recursive walkers that
flatten and rebuild nested structures containing tensors.
`torch.utils._pytree` does the same thing generically. The issue asks whether
these can be replaced.

Suggested steps: find all the walkers, try pytree, add tests for edge cases.

## The walkers (survey settled on the issue)

`amacharla15` and `aayush7511` between them found all of them:

| Walker | Location | Notes |
|---|---|---|
| `IntermediatesCache` | `src/llmcompressor/pipelines/cache.py` | `_offload_value` / `_onload_value` transform and rebuild; `_pin_intermediate` / `_size_helper` only traverse |
| `tensors_to_device` | `src/llmcompressor/pytorch/utils/helpers.py:38` | Only production caller is `BasicPipeline` |
| `tensors_to_precision` | `src/llmcompressor/pytorch/utils/helpers.py:88` | **No production caller.** `kylesayrs`: "can probably be removed" |
| `send_tensors` | compressed-tensors `offload/utils.py` | Handled in compressed-tensors#964 |

So the LLM Compressor side is: `IntermediatesCache`, `tensors_to_device`, and
deleting `tensors_to_precision`.

## The disagreement to resolve first

This is the crux, and it is **not settled**. `kylesayrs` (2026-10-20):

> Unfortunately, I think the fact that `tree_map` doesn't seem to handle dicts
> and dataclasses seems to mean that we're not really saving that much code by
> doing this. What do you think?

`aayush7511` (2026-10-27) pushed back with a concrete behavioural argument: that
pytree isn't about saving lines, it's about *registered* types being rebuilt as
themselves — `transformers` registers `ModelOutput`, which the current code
rebuilds as a plain dict, breaking attribute access.

**Both are factually correct.** Verified against this tree (torch 2.15.0,
transformers 5.22.2):

```
dict           leaves=1  spec=TreeSpec(dict, ['x'], [*])
OrderedDict    leaves=1  spec=TreeSpec(OrderedDict, ['x'], [*])
list           leaves=1  spec=TreeSpec(list, None, [*])
tuple          leaves=1  spec=TreeSpec(tuple, None, [*])
dataclass      leaves=1  spec=*            <- NOT traversed
None           leaves=1  spec=*
namedtuple     leaves=2  rebuilt=NT
ModelOutput    leaves=1  rebuilt_type=BaseModelOutput   <- preserved
```

- `kylesayrs` is right that a **plain dataclass is a single leaf** — pytree does
  not look inside it, so the dataclass branch cannot be deleted. (Dicts *are*
  handled, including `OrderedDict`; that part of the comment is mistaken.)
- `aayush7511` is right about `ModelOutput`. Confirmed on the current tree:

```python
>>> tensors_to_device(BaseModelOutput(last_hidden_state=torch.ones(2,2)), "cpu")
# -> type: OrderedDict
# -> out.last_hidden_state: AttributeError
```

So today's walker silently downgrades a `ModelOutput` to an `OrderedDict`, and
any caller relying on attribute access breaks.

## A bug found while verifying

`tensors_to_device` recurses on anything matching `Iterable()`, and `str` is
iterable:

```python
>>> tensors_to_device("str", "cpu")
RecursionError
>>> tensors_to_device(None, "cpu")
ValueError
>>> tensors_to_device(5, "cpu")
ValueError
```

A string in a batch dict — a text field, a sample id — crashes with
`RecursionError` rather than a useful message. Not mentioned on the issue.
Whichever approach is taken should fix this, and it is a decent argument on its
own for touching this code.

## Approach comparison

| | A: full pytree | B: targeted fix | C: pytree + registration |
|---|---|---|---|
| Fixes `ModelOutput` | Yes | Yes | Yes |
| Fixes `str` recursion | Yes | Yes | Yes |
| Traverses dataclasses | No (stays special-cased) | Yes (existing branch) | Yes (via registration) |
| Lines removed | Few | ~0 | Moderate |
| Answers the maintainer's objection | No | Sidesteps it | Yes |
| New dependency on a private API | `torch.utils._pytree` | None | `torch.utils._pytree` |
| Risk | Medium | Low | Medium |

## Recommendation

**Approach C.** It is the only one that actually answers the objection: register
the dataclass types pytree doesn't handle, and then the dataclass branch *does*
go away, so the "we're not saving much code" argument no longer holds. It also
keeps the `ModelOutput` win.

If a maintainer still prefers minimal churn, **approach B** delivers both bug
fixes with almost no risk and leaves the pytree question open — a perfectly
respectable outcome that a reviewer is likely to merge quickly.

**Do not start with approach A**: it takes the dependency without resolving the
objection that stalled the issue.

## One caution on `torch.utils._pytree`

The leading underscore means private. It is widely used across the PyTorch
ecosystem and is unlikely to vanish, but it carries no stability guarantee.
`torch.utils.pytree` (public) exists in newer versions; check what the project's
`torch>=2.10` floor supports before committing, and say which you used and why
in the PR.

## Semantics that must be preserved

`amacharla15` flagged that the three walkers deliberately differ on unknown
leaves:

| Walker | Unknown leaf |
|---|---|
| `send_tensors` | passes through |
| `IntermediatesCache` | wraps and warns |
| `tensors_to_device` / `tensors_to_precision` | raises `ValueError` |

> I'd preserve those semantics unless there's a reason to standardize them.

`aayush7511` then asked whether `tensors_to_device` should switch to
pass-through, noting the only caller is `BasicPipeline` and that pass-through
"seems safer there anyway". **That question is unanswered.** Pass-through is
probably right — it is what fixes the `str` and `None` cases — but it is a
behaviour change to a public-ish helper and should be called out explicitly in
the PR rather than slipped in.
