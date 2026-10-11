import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from stepbook.runtime.loop import STATE_OUTPUT_SCHEMA, Skill
from stepbook.state.patch_schema import patch_shape, state_output_schema
from stepbook.state.skill import render_schema_section
from stepbook.state.validator import PatchError, Validator


def test_partial_objects_null_deletions_and_array_replacement():
    schema = {
        "type": "object",
        "required": ["settings"],
        "additionalProperties": False,
        "properties": {
            "settings": {
                "type": "object",
                "required": ["phase", "count"],
                "additionalProperties": False,
                "properties": {
                    "phase": {"type": "string", "enum": ["run", "decide"]},
                    "count": {"type": "integer"},
                    "optional": {"type": "string"},
                },
            },
            "items": {"type": "array", "items": {"type": "object", "required": ["id"]}},
        },
    }
    state = {"settings": {"phase": "run", "count": 1, "optional": "old"}, "items": [{"id": 1}]}
    original = copy.deepcopy(schema)
    shape = Draft202012Validator(patch_shape(schema, root=True))
    host = Validator(schema)
    settings = copy.deepcopy(state["settings"])
    for patch, expected, changed in (
        ({}, state, []),
        (
            {"settings": {"phase": "decide"}},
            {"settings": {**settings, "phase": "decide"}, "items": [{"id": 1}]},
            ["settings.phase"],
        ),
        (
            {"settings": {"optional": None}},
            {"settings": {"phase": "run", "count": 1}, "items": [{"id": 1}]},
            ["settings.optional"],
        ),
        ({"items": []}, {"settings": settings, "items": []}, ["items"]),
        ({"items": None}, {"settings": settings}, ["items"]),
        ({"absent": None}, state, []),
        ({"settings": {"absent": None}}, state, []),
    ):
        assert host.apply(state, patch) == (expected, changed)
        shape.validate(patch)
    assert not shape.is_valid({"settings": {"phase": "research/select"}})
    assert not shape.is_valid({"settings": "{}"})
    assert not shape.is_valid(None)
    # Shape success is NOT execution permission: required fields and full array
    # items, protected fields, budgets and semantic checks still belong to the host.
    for patch, message in (
        ({"settings": None}, "schema violation at '<root>': 'settings' is a required property"),
        ({"items": [{}]}, "schema violation at 'items.0': 'id' is a required property"),
    ):
        shape.validate(patch)
        with pytest.raises(PatchError) as rejected:
            host.apply(state, patch)
        assert str(rejected.value) == message
    assert schema == original
    assert state == {"settings": {"phase": "run", "count": 1, "optional": "old"}, "items": [{"id": 1}]}


@pytest.mark.parametrize(
    "complex_node",
    [
        {"$ref": "#/$defs/value"},
        {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        {"type": "object", "patternProperties": {"^x": {"type": "number"}}, "additionalProperties": False},
        {"properties": {"x": {"type": "integer"}}},
    ],
)
def test_complex_nodes_remain_permissive_not_broken_references(complex_node):
    assert patch_shape(complex_node) == {}


def test_all_shipped_skill_shapes_are_valid_and_do_not_change_base_schema():
    root = Path(__file__).resolve().parents[1]
    original = copy.deepcopy(STATE_OUTPUT_SCHEMA)
    shipped = sorted(path.parent.name for path in (root / "stepbook/templates").glob("*/schema.json"))
    # The three public skills must all be present; PRIVATE-marked skills are checkout-specific.
    assert {"eval-loop-adapter", "eval-loop", "checklist"} <= set(shipped)
    for path in [
        *(root / "stepbook/templates").glob("*/schema.json"),
    ]:
        schema = json.loads(path.read_text())
        out = state_output_schema(STATE_OUTPUT_SCHEMA, schema)
        Draft202012Validator.check_schema(out)
        checker = Draft202012Validator(out)
        checker.validate({"reasoning": "No change", "state_patch": {}, "action": "stop"})
        assert not checker.is_valid({"reasoning": "<parameter name=state_patch>{}", "action": "stop"})
        assert Skill.load(path.parent).initial_state == json.loads((path.parent / "initial_state.json").read_text())
    assert STATE_OUTPUT_SCHEMA == original


def test_eval_loop_patch_fields_exposed_but_host_protection_remains():
    skill = Skill.load(Path(__file__).resolve().parents[1] / "stepbook/templates/eval-loop-adapter")
    output = state_output_schema(STATE_OUTPUT_SCHEMA, skill.schema)
    patch = output["properties"]["state_patch"]
    assert "current" in patch["properties"]
    assert "run" in patch["properties"]["current"]["properties"]["phase"]["enum"]
    assert patch["properties"]["current"]["properties"]["phase"]["enum"] == [
        "research",
        "select",
        "apply",
        "run",
        "decide",
        "done",
        None,
    ]
    checker = Draft202012Validator(patch)
    candidate = {"protocol": {"baseline_runs": 99}}
    checker.validate(candidate)
    with pytest.raises(PatchError) as rejected:
        Validator(skill.schema, skill.protected_paths).apply(skill.initial_state, candidate)
    assert str(rejected.value) == "'protocol.baseline_runs' is protected and cannot be changed by the model"


def test_true_root_schema_renders_state_any_instead_of_crashing():
    """A true root schema must render as any state and remain a valid JSON schema."""
    section = render_schema_section(True, ())
    assert section.splitlines() == [
        "State schema (your state_patch must keep the state valid against this; unknown keys are rejected):",
        "<state>: any",
        "Patch rules: merge patch, only changed keys; null deletes a key (a missing key means empty); "
        "never rewrite whole objects.",
    ]
    assert patch_shape(True, root=True) == {"type": "object"}
    Draft202012Validator.check_schema(True)


def test_false_root_schema_renders_state_any_instead_of_crashing():
    """A false root schema must render without keyword descriptions and remain a valid JSON schema."""
    section = render_schema_section(False, ())
    assert section.splitlines() == [
        "State schema (your state_patch must keep the state valid against this; unknown keys are rejected):",
        "<state>: any",
        "Patch rules: merge patch, only changed keys; null deletes a key (a missing key means empty); "
        "never rewrite whole objects.",
    ]
    assert patch_shape(False, root=True) == {"type": "object"}
    Draft202012Validator.check_schema(False)
