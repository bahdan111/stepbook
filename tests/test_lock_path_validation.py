"""Offline probes for ancestor removal and unaddressable dotted lock paths."""

import json
import shutil
import subprocess
import sys
from copy import deepcopy

import pytest

from stepbook.state.skill import LockPathError, Skill
from stepbook.state.validator import PatchError, Validator


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize("value", [3, 0, False])
@pytest.mark.parametrize("ancestor", ["budget", "outer"])
def test_optional_ancestor_removal(mode, value, ancestor):
    state = {"goal": "g", "outer": {"budget": {"runs_max": value}}}
    patch = {"outer": None} if ancestor == "outer" else {"outer": {"budget": None}}
    original = deepcopy(state)
    validator = Validator({"type": "object"}, **{f"{mode}_paths": ("outer.budget.runs_max",)})
    message = (
        "is protected and cannot be changed by the model"
        if mode == "protected"
        else "is write-once and already set; it cannot be changed"
    )
    with pytest.raises(PatchError) as caught:
        validator.apply(state, patch)
    assert caught.value.problems == [f"'outer.budget.runs_max' {message}"]
    assert state == original


@pytest.mark.parametrize("empty", [None, {}, [], ""])
def test_uninitialized_write_once_can_be_removed_or_initialized(empty):
    validator = Validator({"type": "object"}, write_once_paths=("budget.runs_max",))
    state = {"budget": {"runs_max": empty}}
    assert validator.apply(state, {"budget": None}) == ({}, ["budget"])
    assert validator.apply(state, {"budget": {"runs_max": 3}})[0] == {"budget": {"runs_max": 3}}
    assert validator.apply({}, {"budget": {"runs_max": 3}})[0] == {"budget": {"runs_max": 3}}


@pytest.mark.parametrize("mode", ["protected", "write_once"])
def test_list_position_lock_and_noop_regressions(mode):
    validator = Validator({"type": "object"}, **{f"{mode}_paths": ("budget.0.runs_max",)})
    state = {"budget": [{"runs_max": 3}, {"runs_max": 9}]}
    assert validator.apply(state, deepcopy(state)) == (state, [])
    for replacement in (None, [], [{"runs_max": 9}, {"runs_max": 3}]):
        with pytest.raises(PatchError, match="budget.0.runs_max"):
            validator.apply(state, {"budget": replacement})
    assert validator.apply(state, {"budget": [{"runs_max": 3, "used": 1}]})[0] == {
        "budget": [{"runs_max": 3, "used": 1}]
    }


def _skill_folder(tmp_path, schema, state, mode, path):
    schema = deepcopy(schema)
    schema["x-stepbook"] = {mode: [path]}
    folder = tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text("Inspect the synthetic lock-path fixture.\n")
    (folder / "schema.json").write_text(json.dumps(schema))
    (folder / "initial_state.json").write_text(json.dumps(state))
    return folder


@pytest.mark.parametrize("command", ["validate", "run"])
@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize("source", ["state", "schema"])
@pytest.mark.parametrize(
    "state,path",
    [
        ({"budget.runs_max": 3}, "budget.runs_max"),
        ({"outer.key": {"limit": 3}}, "outer.key.limit"),
        ({"outer": {"key.limit": 3}}, "outer.key.limit"),
    ],
    ids=["literal-lock", "literal-prefix", "nested-literal"],
)
def test_claims_report_dotted_probes_refused_at_load_and_cli(tmp_path, command, mode, source, state, path):
    def declaration(value):
        if isinstance(value, dict):
            return {"type": "object", "properties": {key: declaration(child) for key, child in value.items()}}
        return {"type": "integer"}

    schema = declaration(state) if source == "schema" else {"type": "object"}
    folder = _skill_folder(tmp_path, schema, state if source == "state" else {}, mode, path)
    message = f"lock path '{path}': dotted keys cannot be locked; paths are dot-separated"
    with pytest.raises(LockPathError) as caught:
        Skill.load(folder, include_schema=False)
    assert str(caught.value) == message
    # No adapter or executor is configured; loading must refuse before either is needed.
    result = subprocess.run(
        [sys.executable, "-m", "stepbook.cli", command, str(folder)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert result.stderr == message + "\n"
    assert "Traceback" not in result.stderr
    assert not (folder / "events.jsonl").exists()


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize(
    "subtree",
    [
        {"type": "object", "additionalProperties": {"type": "number"}},
        {"type": "object", "patternProperties": {".*": {"type": "number"}}, "additionalProperties": False},
        {"type": "object"},
        {"type": "object", "properties": {"a.b": {"type": "number"}}, "additionalProperties": False},
        {"type": "array", "items": {"type": "object"}},
    ],
    ids=["additional", "pattern", "implicit-additional", "literal-descendant", "array-descendant"],
)
def test_locked_subtree_may_admit_dotted_keys(tmp_path, mode, subtree):
    schema = {"type": "object", "properties": {"locked": subtree}}
    folder = _skill_folder(tmp_path, schema, {}, mode, "locked")
    assert Skill.load(folder).initial_state == {}


@pytest.mark.parametrize("mode", ["protected", "write_once"])
def test_dotted_keys_outside_locks_remain_valid(tmp_path, mode):
    schema = {"type": "object", "properties": {"locked": {"type": "object"}, "locked.limits": {"type": "integer"}}}
    state = {"locked": {"limit": 3}, "locked.limits": 4, "free.form": {"also.free": 3}}
    folder = _skill_folder(tmp_path, schema, state, mode, "locked.limit")
    assert Skill.load(folder).initial_state == state


def test_protected_descendant_refuses_ancestor_removal():
    validator = Validator({"type": "object"}, protected_paths=("budget.runs_max",))
    with pytest.raises(PatchError) as caught:
        validator.apply({"goal": "g", "budget": {"runs_max": 3}}, {"budget": None})
    assert caught.value.problems == ["'budget.runs_max' is protected and cannot be changed by the model"]


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize(
    "subtree,state,path",
    [
        (
            {"type": "object", "properties": {"limit": {"type": "integer"}}, "additionalProperties": False},
            {},
            "locked.limit",
        ),
        ({"type": "array", "items": {"type": "integer"}}, {"locked": [3]}, "locked.0"),
        ({"type": "object", "properties": {"0": {"type": "integer"}}}, {"locked": {"0": 3}}, "locked.0"),
        (
            {"type": "object", "propertyNames": {"pattern": "^[^.]+$"}, "additionalProperties": {"type": "number"}},
            {"locked": {"metric": 3}},
            "locked",
        ),
    ],
    ids=["optional-scalar", "list-position", "numeric-key", "dot-free-map"],
)
def test_addressable_dot_free_locks_still_load(tmp_path, mode, subtree, state, path):
    folder = _skill_folder(tmp_path, {"type": "object", "properties": {"locked": subtree}}, state, mode, path)
    assert Skill.load(folder).initial_state == state


@pytest.mark.parametrize("patch", [{"checks": {"a.b": "changed"}}, {"checks": {"a.b": None}}, {"checks": None}])
def test_checklist_protected_checks_accept_dotted_key_but_refuse_changes(tmp_path, patch):
    folder = tmp_path / "checklist"
    shutil.copytree("stepbook/templates/checklist", folder)
    state_path = folder / "initial_state.json"
    state = json.loads(state_path.read_text())
    state["checks"] = {"a.b": "original"}
    state_path.write_text(json.dumps(state))
    skill = Skill.load(folder)
    validator = Validator(skill.schema, protected_paths=skill.protected_paths)
    assert validator.apply(skill.initial_state, {}) == (state, [])
    with pytest.raises(PatchError, match="checks.*is protected"):
        validator.apply(skill.initial_state, patch)
    assert skill.initial_state == state


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "additionalProperties": False},
        {"type": "object", "properties": {"a": True}},
    ],
    ids=["absent-route", "boolean-route"],
)
def test_no_dotted_declaration_does_not_require_schema_analysis(tmp_path, schema):
    folder = _skill_folder(tmp_path, schema, {}, "protected", "a.b")
    assert Skill.load(folder).initial_state == {}


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize("array_kind,index", [("items", 0), ("prefixItems", 0), ("prefixItems", 1), ("tail", 1)])
def test_declared_dotted_lock_routes_through_arrays_are_refused(tmp_path, mode, array_kind, index):
    item = {"type": "object", "properties": {"key.limit": {"type": "integer"}}}
    array = {"type": "array", "items": item}
    if array_kind == "prefixItems":
        array = {"type": "array", "prefixItems": [{}] * index + [item], "items": False}
    elif array_kind == "tail":
        array["prefixItems"] = [{}]
    schema = {"type": "object", "properties": {"rows": array}}
    folder = _skill_folder(tmp_path, schema, {"rows": []}, mode, f"rows.{index}.key.limit")
    with pytest.raises(ValueError, match="dotted keys cannot be locked"):
        Skill.load(folder)


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize("segment", ["00", "+1", " 1", "01", "-0", "1_0", "١"])
def test_noncanonical_numeric_lock_segments_are_refused(tmp_path, mode, segment):
    folder = _skill_folder(
        tmp_path, {"type": "object"}, {"rows": [{"key.limit": 3}]}, mode, f"rows.{segment}.key.limit"
    )
    with pytest.raises(ValueError, match="numeric segments must be canonical"):
        Skill.load(folder)


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf", "$ref"])
@pytest.mark.parametrize("route", ["root", "object", "items", "prefixItems", "leaf"])
def test_composed_or_referenced_lock_routes_are_refused(tmp_path, mode, keyword, route):
    node = {keyword: "#/$defs/value" if keyword == "$ref" else [{}]}
    schema, path = node, "limit"
    if route in ("object", "leaf"):
        schema = {"properties": {"outer": node}}
        path = "outer.limit" if route == "object" else "outer"
    elif route in ("items", "prefixItems"):
        schema = {"properties": {"rows": {route: [node] if route == "prefixItems" else node}}}
        path = "rows.0.limit"
    folder = _skill_folder(tmp_path, schema, {}, mode, path)
    with pytest.raises(ValueError, match="unsupported for locks") as caught:
        Skill.load(folder)
    assert keyword in str(caught.value)


@pytest.mark.parametrize("mode", ["protected", "write_once"])
def test_array_prefix_takes_precedence_and_unrelated_composition_is_allowed(tmp_path, mode):
    schema = {
        "properties": {
            "rows": {
                "type": "array",
                "prefixItems": [{"properties": {"key": {"properties": {"limit": {"type": "integer"}}}}}],
                "items": {"allOf": [{"properties": {"key.limit": {"type": "integer"}}}]},
            },
            "unlocked": {"$ref": "#/$defs/value"},
        }
    }
    folder = _skill_folder(tmp_path, schema, {"rows": [{"key": {"limit": 3}}]}, mode, "rows.0.key.limit")
    assert Skill.load(folder).initial_state == {"rows": [{"key": {"limit": 3}}]}


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize(
    "path,state",
    [
        (".limit", {"": {"limit": 3}}),
        ("outer.", {"outer": {"": 3}}),
        ("outer..limit", {"outer": {"": {"limit": 3}}}),
        ("", {"": 3}),
    ],
)
def test_empty_lock_segments_are_refused(tmp_path, mode, path, state):
    folder = _skill_folder(tmp_path, {"type": "object"}, state, mode, path)
    with pytest.raises(LockPathError, match="empty segments are unsupported"):
        Skill.load(folder)


@pytest.mark.parametrize("command", ["validate", "run"])
@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize(
    "schema,path",
    [({"type": "object"}, ".limit"), ({"type": "object"}, "rows.00.limit"), ({"allOf": [{}]}, "limit")],
    ids=["empty", "numeric-alias", "unsupported-schema"],
)
def test_lock_policy_cli_refusals_are_one_line(tmp_path, command, mode, schema, path):
    folder = _skill_folder(tmp_path, schema, {}, mode, path)
    with pytest.raises(LockPathError) as caught:
        Skill.load(folder)
    result = subprocess.run(
        [sys.executable, "-m", "stepbook.cli", command, str(folder)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 2
    assert result.stderr == str(caught.value) + "\n"
    assert "Traceback" not in result.stderr
