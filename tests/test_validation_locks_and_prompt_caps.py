import json
import re

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv
from stepbook.runtime.rendering import render_state_prompt
from stepbook.state.aging import abbreviate_state
from stepbook.state.validator import PatchError, Validator


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize("replacement", [None, "replaced", [], {"runs_max": 4}])
def test_ancestor_cannot_erase_or_replace_locked_leaf(mode, replacement):
    validator = Validator({"type": "object"}, **{f"{mode}_paths": ("budget.runs_max",)})
    state = {"goal": "g", "budget": {"runs_max": 3, "used": 0}}
    with pytest.raises(PatchError, match="protected|write-once"):
        validator.apply(state, {"budget": replacement})
    assert state["budget"] == {"runs_max": 3, "used": 0}
    updated, changed = validator.apply(state, {"budget": {"used": 1}})
    assert updated["budget"] == {"runs_max": 3, "used": 1}
    assert changed == ["budget.used"]


def test_protected_missing_and_null_are_distinct():
    validator = Validator({"type": "object"}, protected_paths=("budget.runs_max",))
    with pytest.raises(PatchError, match="protected"):
        validator.apply({"budget": {"runs_max": None}}, {"budget": None})
    assert validator.apply({}, {"budget": {"used": 1}})[0] == {"budget": {"used": 1}}


@pytest.mark.parametrize("empty", [None, {}, [], ""])
def test_write_once_can_be_initialized(empty):
    validator = Validator({"type": "object"}, write_once_paths=("budget.runs_max",))
    assert validator.apply({"budget": {"runs_max": empty}}, {"budget": {"runs_max": 3}})[0] == {
        "budget": {"runs_max": 3}
    }


@pytest.mark.parametrize("value", ["x" * 100000, '\n"\\é😀' * 20000], ids=["ascii", "escaped"])
def test_scalar_cap_counts_rendered_escapes_and_marks_exact_cut(value):
    state = {"goal": "g", "notes": value}
    shown, notes = abbreviate_state(state, 1000)
    encoded = json.dumps(shown, separators=(",", ":"), sort_keys=True)
    assert 980 <= len(encoded) <= 1000
    prefix, cut = re.fullmatch(r"(.*)\[(\d+) characters cut; full value in state.json\]", shown["notes"], re.S).groups()
    assert prefix == value[: len(prefix)]
    assert int(cut) == len(value) - len(prefix)
    assert notes and state == {"goal": "g", "notes": value}
    assert render_state_prompt(state, "obs", state_max_chars=1000).startswith(
        "Note: the state below is abbreviated for size ("
    )


def test_multiple_scalars_and_tiny_cap_are_best_effort():
    state = {"a": "a" * 3000, "b": "b" * 2000, "count": 3}
    shown, notes = abbreviate_state(state, 500)
    assert len(json.dumps(shown, separators=(",", ":"))) <= 500
    assert all(shown[k].endswith("full value in state.json]") for k in ("a", "b"))
    tiny, notes = abbreviate_state({"a": "x" * 3000}, 1)
    assert tiny["a"] == "[3000 characters cut; full value in state.json]"
    assert notes


def test_under_cap_is_identical():
    state = {"notes": "é\ntext", "nested": {"a": [1, "b"]}}
    assert render_state_prompt(state, "obs", state_max_chars=1000) == render_state_prompt(state, "obs")


def test_scalar_cap_only_changes_prompt_not_saved_state(tmp_path):
    state = {"goal": "g", "notes": "x" * 100000}
    skill = Skill("cap", "work", {"type": "object"}, state)
    prompts = []
    executor = ScriptedExecutor(lambda system, prompt: prompts.append(prompt) or Envelope("done", {}, "done"))
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    result = Runtime(skill, executor, env, log_dir=tmp_path / "run", state_max_chars=1000).run(1)
    assert result.final_state == state
    assert json.loads((tmp_path / "run" / "state.json").read_text()) == state
    assert "characters cut; full value in state.json]" in prompts[0]
    assert state["notes"] not in prompts[0]


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize(
    "replacement",
    [[{"runs_max": 4}], None, [], [{"runs_max": 9}, {"runs_max": 3}]],
    ids=["replace", "delete-parent", "delete-position", "reorder"],
)
def test_locked_list_position_cannot_change(mode, replacement):
    validator = Validator({"type": "object"}, **{f"{mode}_paths": ("budget.0.runs_max",)})
    state = {"budget": [{"runs_max": 3}, {"runs_max": 9}]}
    with pytest.raises(PatchError, match=r"budget\.0\.runs_max.*(protected|write-once)"):
        validator.apply(state, {"budget": replacement})
    assert state == {"budget": [{"runs_max": 3}, {"runs_max": 9}]}
    updated, _ = validator.apply(state, {"budget": [{"runs_max": 3, "used": 1}]})
    assert updated == {"budget": [{"runs_max": 3, "used": 1}]}


def test_absent_write_once_list_position_initializes_then_locks():
    validator = Validator({"type": "object"}, write_once_paths=("budget.1.runs_max",))
    state, _ = validator.apply({"budget": [{}]}, {"budget": [{}, {"runs_max": 3}]})
    assert state == {"budget": [{}, {"runs_max": 3}]}
    with pytest.raises(PatchError, match="write-once"):
        validator.apply(state, {"budget": [{}]})


@pytest.mark.parametrize("mode", ["protected", "write_once"])
def test_numeric_dictionary_key_remains_a_key(mode):
    validator = Validator({"type": "object"}, **{f"{mode}_paths": ("budget.0.runs_max",)})
    with pytest.raises(PatchError, match="protected|write-once"):
        validator.apply({"budget": {"0": {"runs_max": 3}}}, {"budget": {"0": {"runs_max": 4}}})


def test_absent_protected_parent_initializes_then_locks():
    validator = Validator({"type": "object"}, protected_paths=("budget.runs_max",))
    state, _ = validator.apply({}, {"budget": {"runs_max": 3}})
    assert state == {"budget": {"runs_max": 3}}
    for patch in ({"budget": None}, {"budget": []}, {"budget": {"runs_max": 4}}):
        with pytest.raises(PatchError, match="protected"):
            validator.apply(state, patch)


@pytest.mark.parametrize("patch", [{"budget": None}, {"budget": {"runs_max": 3}}])
def test_explicit_protected_null_is_already_initialized(patch):
    validator = Validator({"type": "object"}, protected_paths=("budget.runs_max",))
    with pytest.raises(PatchError, match="protected"):
        validator.apply({"budget": {"runs_max": None}}, patch)


@pytest.mark.parametrize("mode", ["protected", "write_once", "mixed"])
def test_locked_diagnostics_follow_sorted_changed_paths(mode):
    paths = (
        {f"{mode}_paths": ("b", "a")} if mode != "mixed" else {"protected_paths": ("b",), "write_once_paths": ("a",)}
    )
    validator = Validator({"type": "object"}, **paths)
    with pytest.raises(PatchError) as caught:
        validator.apply({"a": 1, "b": 2}, {"b": 3, "a": 2})
    assert [problem.split("'")[1] for problem in caught.value.problems] == ["a", "b"]


@pytest.mark.parametrize("replacement", [4, None], ids=["update", "delete"])
def test_literal_dotted_protected_key_cannot_change(replacement):
    validator = Validator({"type": "object"}, protected_paths=("budget.runs_max",))
    with pytest.raises(PatchError) as caught:
        validator.apply({"budget.runs_max": 3}, {"budget.runs_max": replacement})
    assert caught.value.problems == ["'budget.runs_max' is protected and cannot be changed by the model"]


@pytest.mark.parametrize(
    "state,patch,path",
    [({"budget": {}}, {"budget": {"runs_max": 3}}, "budget.runs_max"), ({}, {"budget": 3}, "budget")],
    ids=["existing-parent", "top-level"],
)
def test_direct_protected_creation_remains_rejected(state, patch, path):
    validator = Validator({"type": "object"}, protected_paths=(path,))
    with pytest.raises(PatchError) as caught:
        validator.apply(state, patch)
    assert caught.value.problems == [f"{path!r} is protected and cannot be changed by the model"]


def test_direct_locked_diagnostics_preserve_order_and_multiplicity():
    validator = Validator(
        {"type": "object"},
        protected_paths=("a", "a.b", "a.b"),
        write_once_paths=("a.b",),
    )
    with pytest.raises(PatchError) as caught:
        validator.apply({"a": {"b": 3}, "a.b": 9}, {"a": {"b": 4}, "a.b": 10})
    assert (
        caught.value.problems
        == [
            "'a.b' is protected and cannot be changed by the model",
            "'a.b' is protected and cannot be changed by the model",
            "'a.b' is protected and cannot be changed by the model",
            "'a.b' is write-once and already set; it cannot be changed",
        ]
        * 2
    )


def test_literal_dotted_change_retains_initialized_write_once_diagnostic():
    validator = Validator({"type": "object"}, write_once_paths=("budget.runs_max",))
    with pytest.raises(PatchError) as caught:
        validator.apply({"budget.runs_max": 3, "budget": {"runs_max": 9}}, {"budget.runs_max": 4})
    assert caught.value.problems == ["'budget.runs_max' is write-once and already set; it cannot be changed"]
