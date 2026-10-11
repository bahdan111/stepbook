import pytest

from stepbook.state.validator import PatchError, Validator

SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "inventory": {"type": "object", "additionalProperties": {"type": "string"}},
        "count": {"type": "integer"},
    },
    "required": ["goal", "inventory"],
    "additionalProperties": False,
}


@pytest.fixture
def v():
    return Validator(SCHEMA, protected_paths=("goal",), max_deletions=2)


def test_valid_patch_applies(v):
    state = {"goal": "g", "inventory": {"s1": "i1"}}
    new, changed = v.apply(state, {"inventory": {"s2": "i2", "s1": None}})
    assert new == {"goal": "g", "inventory": {"s2": "i2"}}
    assert changed == ["inventory.s1", "inventory.s2"]


def test_rejects_non_object(v):
    state = {"goal": "g", "inventory": {}}
    with pytest.raises(PatchError) as error:
        v.apply(state, ["nope"])
    assert str(error.value) == "state_patch must be a JSON object, got list"
    for patch, name in (("nope", "str"), (3, "int"), (True, "bool"), (None, "NoneType")):
        with pytest.raises(PatchError) as other:
            v.apply(state, patch)
        assert str(other.value) == f"state_patch must be a JSON object, got {name}"
    assert state == {"goal": "g", "inventory": {}}


@pytest.mark.parametrize("value", ["...", "…", "  ...  "])
def test_rejects_placeholders(v, value):
    with pytest.raises(PatchError, match="placeholder"):
        v.apply({"goal": "g", "inventory": {}}, {"inventory": {"s1": value}})


@pytest.mark.parametrize("value", ["unchanged", "same", "same as before", "no change", "[previous content preserved]"])
def test_accepts_words_that_can_be_real_values(v, value):
    """A status or a summary is a value, not a placeholder; only "..." and "…" stand for data left out."""
    new, changed = v.apply({"goal": "g", "inventory": {}}, {"inventory": {"s1": value}})
    assert new["inventory"] == {"s1": value} and changed == ["inventory.s1"]


def test_rejects_mass_deletion(v):
    with pytest.raises(PatchError, match="deletions"):
        v.apply(
            {"goal": "g", "inventory": {"a": "1", "b": "2", "c": "3"}}, {"inventory": {"a": None, "b": None, "c": None}}
        )


def test_rejects_protected_change_but_allows_noop(v):
    state = {"goal": "g", "inventory": {}}
    with pytest.raises(PatchError, match="protected"):
        v.apply(state, {"goal": "other"})
    new, changed = v.apply(state, {"goal": "g"})  # same value -> no change -> fine
    assert changed == []


def test_rejects_schema_violation_of_result(v):
    state = {"goal": "g", "inventory": {}}
    with pytest.raises(PatchError, match="schema violation"):
        v.apply(state, {"count": "twelve"})
    with pytest.raises(PatchError, match="schema violation"):
        v.apply(state, {"inventory": None})  # would drop a required key


def test_error_message_is_feedback_for_the_model(v):
    with pytest.raises(PatchError) as error:
        v.apply({"goal": "g", "inventory": {}}, {"extra": 1})
    e = error.value
    text = e.as_observation()
    assert "REJECTED" in text and "extra" in text
    assert text == (
        "STATE PATCH REJECTED. The state was NOT changed and no action was executed."
        " Fix the patch and answer again for the same observation.\n"
        "- schema violation at '<root>': Additional properties are not allowed ('extra' was unexpected)"
    )


def test_schema_section_lists_fields_and_protected():
    from pathlib import Path

    from stepbook.runtime.loop import Skill

    sk = Skill.load(Path(__file__).resolve().parent.parent / "stepbook/templates" / "eval-loop-adapter")
    assert "State schema" in sk.instructions
    assert "hypotheses: object" in sk.instructions
    assert "enum" not in sk.schema["properties"]["hypotheses"]["additionalProperties"]["properties"]["kind"]
    assert "goal: string (required. PROTECTED" in sk.instructions or "PROTECTED" in sk.instructions
    protected = [line for line in sk.instructions.splitlines() if "PROTECTED" in line]
    assert protected == [
        "  goal: string (PROTECTED, never change; required.)",
        "  protocol: object (PROTECTED, never change; required.)",
        "    runs_max: ['integer', 'null'] (PROTECTED, never change)",
        "  user_requirements: array (PROTECTED, never change)",
    ]
    assert set(sk.protected_paths) == {"goal", "protocol", "user_requirements", "budget.runs_max"}
    assert "  hypotheses: object (required.)" in sk.instructions.splitlines()
    plain = Skill.load(
        Path(__file__).resolve().parent.parent / "stepbook/templates" / "eval-loop-adapter", include_schema=False
    )
    assert "State schema" not in plain.instructions


@pytest.mark.parametrize("character", ["x", "😀"])
def test_length_feedback_is_numeric_without_echoing_or_trimming_value(character):
    from copy import deepcopy

    schema = {
        "type": "object",
        "properties": {"finding": {"type": "string", "maxLength": 600}},
        "required": ["finding"],
    }
    validator = Validator(schema)
    state, patch = {"finding": "supported evidence"}, {"finding": character * 613}
    original_state, original_patch = deepcopy(state), deepcopy(patch)
    with pytest.raises(PatchError) as error:
        validator.apply(state, patch)
    text = str(error.value)
    assert "schema violation at 'finding'" in text
    assert "613 characters (Unicode code points)" in text
    assert "maxLength is 600, excess 13" in text
    assert patch["finding"] not in text
    assert state == original_state and patch == original_patch
    assert validator.apply(state, {"finding": character * 600})[0]["finding"] == character * 600
