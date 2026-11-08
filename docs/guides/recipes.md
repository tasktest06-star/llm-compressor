# Recipes

A **recipe** is the declarative description of *what* to do to a model. Every
entrypoint — `oneshot`, `train`, `model_free_ptq` — is ultimately handed a
recipe, and every compressed model gets a `recipe.yaml` saved beside its weights
recording how it was produced.

This guide explains how a recipe is structured, the vocabulary it uses, and the
rules that govern how it is written to disk and read back.

## The shape of a recipe

A recipe nests three levels: **stages**, **modifier groups**, and
**modifiers** with their arguments.

```yaml
quant_stage:                 # 1. a stage
  quant_modifiers:           # 2. a modifier group
    GPTQModifier:            # 3. a modifier
      scheme: W4A16          #    ...and its arguments
      targets: [Linear]
      ignore: [lm_head]
```

### Stages

A stage is one pass over the model. A top-level key is read as a stage when it
matches `<name>_stage`, optionally with a numeric suffix (`<name>_stage_0`).
The leading part is the stage name, so `quant_stage` is the stage `quant`.

Stages matter because compression is often applied in more than one pass — for
example a transform pass followed by a quantization pass, or a second pass that
quantizes only `lm_head`. Keys that don't match the stage pattern (such as
`args`) are not treated as stages.

### Modifier groups

Inside a stage, a key ending in `_modifiers` holds a group of modifiers. The
part before the suffix becomes the **group** that each modifier in it belongs
to: modifiers under `quant_modifiers` land in the group `quant`, and under
`pruning_modifiers` in the group `pruning`.

The group is bookkeeping, not behaviour — it labels where a modifier came from
and determines where it is written back out. Because the group is already
encoded in the group key, it is *stripped* from a modifier's arguments when a
recipe is serialized, and inferred again from the key when it is read back.

You can also set `group` explicitly on a modifier, in which case your value wins
over the one inferred from the key:

```yaml
test_stage:
  smoothquant_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.75
      group: custom_group      # overrides the inferred "smoothquant"
```

### Modifiers and their arguments

The innermost key is the modifier's class name, resolved through
`ModifierFactory` against the registered, main, and experimental modifier
registries. Its value is a mapping of argument names to values, matching the
modifier's own fields.

A modifier that needs no arguments is written with an empty value, which is
simply `None` in YAML. It is built entirely from its defaults:

```yaml
test_stage:
  smoothquant_modifiers:
    SmoothQuantModifier:
```

Anything that is neither a mapping nor empty (`SmoothQuantModifier: 5`) is
rejected, since there is no way to turn it into arguments.

## Ways to express a recipe

`Recipe.create_instance` accepts all of the following, so "the recipe" can be
whichever form is most convenient:

| Input | Notes |
|---|---|
| YAML string | The form shown throughout this guide. |
| JSON string | Tried first; YAML parsing is the fallback. |
| Path to a `.yaml`/`.json` file | Including a `recipe.yaml` from a saved model. |
| Path to a `.md` recipe card | The YAML front matter between `---` delimiters is extracted. |
| A `Modifier`, or a list of them | Wrapped into a single stage via `Recipe.from_modifiers`. |
| An existing `Recipe` | Returned as-is. |

Passing modifier objects directly is usually the clearest option in Python:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

oneshot(
    model="meta-llama/Llama-3.2-1B-Instruct",
    recipe=GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]),
    dataset="open_platypus",
)
```

`target_stage` can be passed to load only the stages whose names match, which is
how a single multi-stage recipe file can be reused for one pass at a time.

## Serialization and the round trip

`Recipe.yaml()` renders a recipe back to YAML, and `oneshot` writes that next to
the model as `recipe.yaml`. This is not only a record: a saved recipe is read
back in two places that matter.

- Passing a saved `recipe.yaml` as `recipe=` to another run.
- Saving a model that already carries a recipe. `update_and_save_recipe` loads
  the existing `recipe.yaml` and appends the new stage to it, so a second pass
  over a previously compressed directory *must* be able to parse the first
  recipe. When it can't, saving fails after calibration has already run.

Round-trip fidelity is therefore a correctness property, not a nicety. Four
rules keep it:

**Only plain YAML.** Recipes are read with `yaml.safe_load`, which refuses
Python object tags, so they are written with `yaml.safe_dump` from a JSON-mode
model dump. Every value is a plain scalar, list, or mapping. Without this, values
that are ordinary Python objects internally — an enum such as
`QuantizationType.INT`, a `torch.device`, a tuple — serialize as
`!!python/object/apply:...` and the recipe cannot be loaded again by
llm-compressor itself.

**The group is dropped.** As described above, it is recovered from the group key.

**An unset `None` is dropped; an explicit `None` is kept.** An argument that is
`None` only because nobody set it carries no information. An argument the user
*set* to `None` does: for GPTQ `actorder`, `None` is how activation ordering is
switched off. Writing `actorder: null` is what keeps that choice across a save
and load.

**Unresolved defaults are omitted.** Some arguments default to a `Sentinel`
standing for "not provided", which is resolved later from other settings.
`GPTQModifier.actorder` defaults to `Sentinel("static")`, and is reconciled with
any `actorder` in `config_groups` when the quantization config is resolved.
A `Sentinel` serializes to its bare name, so writing it out would turn
"nobody chose" into "the user chose `static`" — and then conflict with a
`config_groups` value that differs. Omitting it lets the default resolve again
on load.

### Multi-stage recipes

When a stage is appended to a recipe that already has a stage of the same name,
both are given numeric suffixes, starting at zero:

```yaml
quant_stage_0:    # from the first pass
  ...
quant_stage_1:    # appended by the second
  ...
```

This is why the stage pattern allows a `_<n>` suffix: a recipe that has been
through two passes still parses as two stages with the same name.

## Common errors

| Symptom | Cause |
|---|---|
| `Could not parse recipe from string ...` | The file isn't valid JSON or YAML. If you meant to load a file, check the path exists — a non-existent path is treated as recipe *text*. |
| `No modifier of type 'X' found.` | The modifier name doesn't resolve. Check spelling and that the modifier still exists in your installed version. |
| `Recipe includes <X> with no subsequent quantization modifier` | A transform modifier such as AWQ or SpinQuant has to be followed by a quantization modifier in the same recipe. |
| A `recipe.yaml` your tooling can't read | A recipe produced by an older release may carry Python object tags; re-saving it with a current version writes plain YAML. |

## See also

- [Compression Schemes](compression_schemes.md) — what to put in `scheme` and
  `config_groups`.
- [Observers](observers.md) — calibration statistics, and the `actorder`
  semantics referenced above.
- [Saving a Compressed Model](saving_a_model.md) — what else is written
  alongside `recipe.yaml`.
- [Adding a New Modifier](../developer-tutorials/add-modifier.md) — defining the
  arguments a recipe can set.
