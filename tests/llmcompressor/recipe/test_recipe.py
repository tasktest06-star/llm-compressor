import tempfile
import warnings

import pytest
import yaml

# Suppress deprecation warning for AWQModifier import as this test
# explicitly tests backward compatibility of the deprecated API
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from llmcompressor.modifiers.awq import AWQModifier

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier as AWQTransformModifier
from llmcompressor.recipe import Recipe
from llmcompressor.recipe.utils import append_recipe_dict
from llmcompressor.sentinel import Sentinel
from tests.llmcompressor.helpers import valid_recipe_strings


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_recipe_create_instance_accepts_valid_recipe_string(recipe_str):
    recipe = Recipe.create_instance(recipe_str)
    assert recipe is not None, "Recipe could not be created from string"


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_recipe_create_instance_accepts_valid_recipe_file(recipe_str):
    content = yaml.safe_load(recipe_str)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(content, f)
        recipe = Recipe.create_instance(f.name)
        assert recipe is not None, "Recipe could not be created from file"


@pytest.mark.parametrize("recipe_str", valid_recipe_strings())
def test_serialization(recipe_str):
    recipe_instance = Recipe.create_instance(recipe_str)
    serialized_recipe = recipe_instance.yaml()
    recipe_from_serialized = Recipe.create_instance(serialized_recipe)

    expected_dict = recipe_instance.to_dict()
    actual_dict = recipe_from_serialized.to_dict()

    assert expected_dict == actual_dict


def test_recipe_creates_correct_modifier():
    start = 1
    end = 10
    targets = "Linear"

    yaml_str = f"""
        test_stage:
            quantization_modifiers:
                GPTQModifier:
                    start: {start}
                    end: {end}
                    targets: {targets}
        """

    recipe_instance = Recipe.create_instance(yaml_str)

    stage_modifiers = recipe_instance.modifiers
    assert len(modifiers := stage_modifiers) == 1
    from llmcompressor.modifiers.gptq import GPTQModifier

    assert isinstance(modifier := modifiers[0], GPTQModifier)
    assert modifier.start == start
    assert modifier.end == end


@pytest.mark.parametrize("stage_name", ["default", "custom", "custom_stage"])
def test_recipe_from_dict_loads_numbered_stages(stage_name):
    first = Recipe.from_modifiers(
        QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC"),
        modifier_group_name=stage_name,
    )
    second = Recipe.from_modifiers(
        QuantizationModifier(targets="Linear", scheme="W8A8"),
        modifier_group_name=stage_name,
    )
    merged = append_recipe_dict(first.to_dict(), second.to_dict())
    assert list(merged) == [f"{stage_name}_stage_0", f"{stage_name}_stage_1"]

    recipe = Recipe.from_dict(merged)

    assert [m.scheme for m in recipe.modifiers] == ["FP8_DYNAMIC", "W8A8"]
    assert recipe.stage == stage_name


def test_recipe_can_be_created_from_modifier_instances():
    modifier = QuantizationModifier(
        scheme="FP8",
        group="quantization",
    )
    group_name = "dummy"

    # for pep8 compliance
    recipe_str = (
        f"{group_name}_stage:\n"
        "   quantization_modifiers:\n"
        "       QuantizationModifier:\n"
        "           scheme: FP8\n"
    )

    expected_recipe_instance = Recipe.create_instance(recipe_str)
    expected_modifiers = expected_recipe_instance.modifiers

    actual_recipe_instance = Recipe.create_instance(
        [modifier], modifier_group_name=group_name
    )
    actual_modifiers = actual_recipe_instance.modifiers

    # assert num stages is the same
    assert len(actual_modifiers) == len(expected_modifiers)

    # assert num modifiers in each stage is the same
    assert len(actual_modifiers) == len(expected_modifiers)

    # assert modifiers in each stage are the same type
    # and have the same parameters
    for actual_modifier, expected_modifier in zip(actual_modifiers, expected_modifiers):
        assert isinstance(actual_modifier, type(expected_modifier))
        assert actual_modifier.model_dump() == expected_modifier.model_dump()


@pytest.mark.parametrize(
    "modifiers,is_valid",
    [
        (
            [
                AWQModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                    duo_scaling="both",
                )
            ],
            True,
        ),
        (
            AWQModifier(
                ignore=["lm_head"],
                scheme="W4A16_ASYM",
                targets=["Linear"],
                duo_scaling="both",
            ),
            True,
        ),
        (
            [
                AWQTransformModifier(
                    duo_scaling="both",
                )
            ],
            False,
        ),
        (
            AWQTransformModifier(
                duo_scaling="both",
            ),
            False,
        ),
        (
            [
                AWQTransformModifier(
                    duo_scaling="both",
                ),
                QuantizationModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                ),
            ],
            True,
        ),
        (
            [
                QuantizationModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                ),
                AWQTransformModifier(
                    duo_scaling="both",
                ),
            ],
            False,
        ),
        (
            [
                AWQTransformModifier(
                    duo_scaling="both",
                ),
                GPTQModifier(
                    ignore=["lm_head"],
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                ),
            ],
            True,
        ),
    ],
)
def test_recipe_validate_after(modifiers, is_valid):
    if is_valid:
        Recipe.create_instance(modifiers)
    else:
        with pytest.raises(ValueError):
            Recipe.create_instance(modifiers)


def _round_trip(modifier):
    """Serialize a modifier to a recipe and load it back."""
    return Recipe.create_instance(Recipe.from_modifiers(modifier).yaml()).modifiers[0]


def _resolved_actorder(modifier):
    config = modifier.resolve_quantization_config()
    return config.config_groups["group_0"].weights.actorder


def test_gptq_explicit_actorder_none_survives_round_trip():
    # `None` is how the observers guide documents turning activation ordering
    # off. A `None` the user set was dropped on the way out, so the reloaded
    # recipe fell back to the `Sentinel("static")` default and quietly turned
    # activation ordering back on.
    modifier = GPTQModifier(
        targets="Linear", scheme="W4A16", ignore=["lm_head"], actorder=None
    )

    assert _resolved_actorder(modifier) is None
    assert _resolved_actorder(_round_trip(modifier)) is None


def test_gptq_unset_actorder_is_not_serialized():
    # The default is a `Sentinel`, which serializes to its name. Writing
    # `actorder: static` made a later load treat the default as a value the
    # user set on the modifier.
    modifier = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

    assert (
        "actorder"
        not in yaml.safe_load(Recipe.from_modifiers(modifier).yaml())["default_stage"][
            "default_modifiers"
        ]["GPTQModifier"]
    )
    # The reloaded modifier still holds an unresolved default.
    assert isinstance(_round_trip(modifier).actorder, Sentinel)


def test_gptq_config_groups_actorder_survives_round_trip():
    # With `actorder: static` written out as a plain string, a differing
    # `actorder` in `config_groups` made the reloaded recipe raise
    # "both are provided and differ" when the config was resolved.
    weights = {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "actorder": "weight",
    }
    modifier = GPTQModifier(
        ignore=["lm_head"],
        config_groups={"group_0": {"targets": ["Linear"], "weights": weights}},
    )

    assert _resolved_actorder(modifier) == "weight"
    assert _resolved_actorder(_round_trip(modifier)) == "weight"
