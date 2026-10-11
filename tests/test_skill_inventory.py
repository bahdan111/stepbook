"""The shipped skill folders and their public validation feedback."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from stepbook import Skill
from stepbook.cli import main
from stepbook.runtime.loop import render_state_prompt
from stepbook.state.validator import PatchError, Validator

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SKILLS = ["eval-loop-adapter", "eval-loop", "checklist"]
SKILLS = PUBLIC_SKILLS


def skill_folder(name):
    return ROOT / "stepbook/templates" / name


def test_public_skill_inventory():
    public = {p.name for p in (ROOT / "stepbook/templates").iterdir() if p.is_dir() and not (p / "PRIVATE").exists()}
    assert public == set(PUBLIC_SKILLS)


def test_eval_loop_target_has_no_value_until_explicitly_set():
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop-adapter")
    assert skill.initial_state["target"] == {"metric": "primary"}
    prompt = render_state_prompt(skill.initial_state, "ready", [])
    rendered = json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert rendered["target"] == {"metric": "primary"}
    state, _ = Validator(skill.schema).apply(skill.initial_state, {"target": {"value": 0.8}})
    assert state["target"] == {"metric": "primary", "value": 0.8}


@pytest.mark.parametrize("name", SKILLS)
def test_shipped_skill_required_files_load_and_validate(name, capsys):
    """Bug: a named shipped skill disappears or ships an invalid initial state."""
    folder = skill_folder(name)
    for filename in ("SKILL.md", "schema.json", "initial_state.json"):
        assert (folder / filename).is_file(), f"{name}/{filename} missing"
    skill = Skill.load(folder)
    assert skill.name == name and skill.instructions.strip()
    assert skill.schema == json.loads((folder / "schema.json").read_text())
    assert skill.initial_state == json.loads((folder / "initial_state.json").read_text())
    assert skill.schema["additionalProperties"] is True
    assert Validator(skill.schema, skill.protected_paths, skill.max_deletions, skill.write_once_paths).apply(
        skill.initial_state, {}
    ) == (skill.initial_state, [])
    assert main(["validate", str(folder)]) == 0
    output = capsys.readouterr()
    assert output.out.startswith(f"ok: skill {name!r},") and output.err == ""


def test_unknown_property_feedback_names_rejected_field():
    """Bug: a closed schema silently accepts an unknown state field."""
    skill = Skill("closed-example", "Synthetic closed schema", {"type": "object", "additionalProperties": False}, {})
    with pytest.raises(PatchError) as error:
        Validator(skill.schema).apply(skill.initial_state, {"unexpected_sentinel": 1})
    assert "schema violation at '<root>'" in str(error.value)
    assert "unexpected_sentinel" in error.value.as_observation()
    assert error.value.as_observation().startswith("STATE PATCH REJECTED.")


@pytest.mark.parametrize("name", PUBLIC_SKILLS)
def test_open_shipped_schema_preserves_unknown_field(name):
    """Bug: cleanup changes an open shipped schema to reject previously accepted state fields."""
    skill = Skill.load(skill_folder(name))
    state, changed = Validator(skill.schema).apply(skill.initial_state, {"unexpected_sentinel": 1})
    assert state == {**skill.initial_state, "unexpected_sentinel": 1}
    assert changed == ["unexpected_sentinel"]


@pytest.mark.parametrize(
    "name,field",
    [
        ("checklist", "goal"),
        ("checklist", "constraints"),
        ("checklist", "checks"),
        ("checklist", "user_requirements"),
        ("eval-loop-adapter", "goal"),
        ("eval-loop-adapter", "protocol"),
        ("eval-loop-adapter", "user_requirements"),
        ("eval-loop-adapter", "budget.runs_max"),
        ("eval-loop", "goal"),
        ("eval-loop", "user_requirements"),
        ("eval-loop", "budget.runs_max"),
    ],
)
def test_protected_feedback_names_the_changed_field(name, field):
    """Bug: protected-field rejection no longer identifies the field the model tried to change."""
    skill = Skill.load(skill_folder(name))
    state = deepcopy(skill.initial_state)
    patch = {field: None} if "." not in field else {"budget": {"runs_max": 999}}
    # Missing/empty defaults still need an actual change, not a deletion no-op.
    if field not in state and "." not in field:
        patch = {field: "changed"}
    with pytest.raises(PatchError) as error:
        Validator(skill.schema, skill.protected_paths).apply(state, patch)
    assert f"'{field}' is protected and cannot be changed by the model" in error.value.as_observation()
    assert state == skill.initial_state


def test_eval_loop_shell_rendered_state_has_no_invented_target():
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop")
    prompt = render_state_prompt(skill.initial_state, "ready", [])
    rendered = json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert rendered == skill.initial_state
    assert "target" not in rendered and rendered["facts"] == {}
    assert rendered["budget"] == {"runs_used": 0, "runs_max": None}
    assert rendered["protocol"] == {"baseline_runs": 2, "runs_per_hypothesis": 2, "noise_cases": None}
    assert set(skill.protected_paths) == {"goal", "user_requirements", "budget.runs_max"}
    state, _ = Validator(skill.schema, skill.protected_paths).apply(
        skill.initial_state, {"facts": {"target": "User requested primary >= 0.8."}}
    )
    assert state["facts"]["target"] == "User requested primary >= 0.8."


@pytest.mark.parametrize("name", ["eval-loop", "checklist"])
@pytest.mark.parametrize("value", ["reader path", 12, 0.5, True, ["cases.json", None], {"count": 12}, None])
def test_facts_accept_any_json_value(name, value):
    skill = Skill.load(skill_folder(name))
    state = deepcopy(skill.initial_state)
    # Validate an existing null: null in a merge patch deletes the key instead.
    state["facts"]["known"] = value
    validated, changed = Validator(skill.schema, skill.protected_paths).apply(state, {})
    assert validated["facts"] == {"known": value} and changed == []


@pytest.mark.parametrize("done_when", [None, "out.txt contains ready"])
def test_checklist_items_accept_optional_done_when(done_when):
    skill = Skill.load(ROOT / "stepbook/templates/checklist")
    assert set(skill.protected_paths) == {"goal", "constraints", "checks", "user_requirements"}
    item = {"text": "Create out.txt", "status": "todo"}
    if done_when is not None:
        item["done_when"] = done_when
    validator = Validator(skill.schema, skill.protected_paths)
    state, _ = validator.apply(skill.initial_state, {"checklist": {"create": item}})
    assert state["checklist"] == {"create": item}
    for field in ("text", "status"):
        with pytest.raises(PatchError, match=f"'{field}' is a required property"):
            validator.apply(state, {"checklist": {"create": {field: None}}})
    with pytest.raises(PatchError, match="unexpected"):
        validator.apply(state, {"checklist": {"create": {"unexpected": True}}})


@pytest.mark.parametrize("value", [1, False, [], {}])
def test_checklist_done_when_remains_a_string(value):
    skill = Skill.load(ROOT / "stepbook/templates/checklist")
    with pytest.raises(PatchError, match="not of type"):
        Validator(skill.schema).apply(
            skill.initial_state,
            {"checklist": {"create": {"text": "Create out.txt", "status": "todo", "done_when": value}}},
        )


@pytest.mark.parametrize("config_only", [False, True])
def test_generated_skill_validates_after_relocation(tmp_path, config_only, capsys):
    """Bug: init creates a skill that depends on its original filesystem location."""
    source = tmp_path / "generated"
    assert main(["init", str(source), *(["--config-only"] if config_only else [])]) == 0
    target = tmp_path / "relocated"
    shutil.move(str(source), target)
    assert main(["validate", str(target / "skill")]) == 0
    assert "ok: skill 'skill'," in capsys.readouterr().out
    loaded = Skill.load(target / "skill")
    assert loaded.initial_state["target"] == {"metric": "primary"}
    assert loaded.schema == Skill.load(ROOT / "stepbook/templates/eval-loop-adapter").schema
    assert Validator(loaded.schema).apply(loaded.initial_state, {}) == (loaded.initial_state, [])


def test_write_once_feedback_names_field_and_preserves_original():
    """Bug: a second write to a write-once field is accepted or loses its field-specific feedback."""
    validator = Validator(
        {"type": "object", "properties": {"baseline": {"type": "string"}}}, write_once_paths=("baseline",)
    )
    state, _ = validator.apply({}, {"baseline": "verified"})
    with pytest.raises(PatchError) as error:
        validator.apply(state, {"baseline": "replacement"})
    assert error.value.problems == ["'baseline' is write-once and already set; it cannot be changed"]
    assert state == {"baseline": "verified"}


def test_merge_deletion_feedback_names_required_field():
    """Bug: deleting a required field stops producing field-specific schema feedback."""
    validator = Validator({"type": "object", "properties": {"evidence": {"type": "string"}}, "required": ["evidence"]})
    with pytest.raises(PatchError) as error:
        validator.apply({"evidence": "verified"}, {"evidence": None})
    assert error.value.problems == ["schema violation at '<root>': 'evidence' is a required property"]
