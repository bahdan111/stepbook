"""State rendering under --state-max-chars (abbreviate_state / render_state_prompt)."""

import copy
import json
import signal

import pytest

from stepbook import Runtime
from stepbook.runtime.loop import abbreviate_state, render_state_prompt

from ._helpers import ROOT, Script, checklist, cmd, shell_env


def _facts(n, width):
    return {f"fact_{i:04d}": f"{i}: " + "x" * width for i in range(n)}


def test_rendering_five_thousand_entries_stays_under_two_seconds():
    state = {"goal": "g", "facts": _facts(5000, 120), "current": "fact_4999"}

    def alarm(signum, frame):
        raise TimeoutError("abbreviate_state exceeded 2 s")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        shown, notes = abbreviate_state(state, 12000)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert len(json.dumps(shown, separators=(",", ":"))) <= 12600 and notes


@pytest.mark.parametrize("field", ["facts_dict", "lessons_list"])
def test_abbreviation_never_mutates_the_state(field):
    if field == "facts_dict":
        state = {"goal": "g", "facts": {f"f{i:03d}": f"{i}: " + "x" * 60 for i in range(300)}, "current": "f299"}
    else:
        state = {"goal": "g", "lessons": [f"{i}: " + "x" * 60 for i in range(300)]}
    before = copy.deepcopy(state)
    abbreviate_state(state, 5000)
    assert state == before


def test_runtime_state_survives_capped_rendering_of_short_facts(tmp_path):
    skill = checklist()
    skill.initial_state = {**skill.initial_state, "facts": {f"f{i:03d}": f"{i}: " + "x" * 60 for i in range(200)}}
    expected = copy.deepcopy(skill.initial_state["facts"])  # the runtime only shallow-copies the initial state
    env = shell_env(tmp_path)
    script = Script(cmd("echo a"), cmd("echo b"), cmd("done"))
    res = Runtime(skill, script.executor(), env, log_dir=tmp_path / "log", state_max_chars=3000).run(5)
    assert len(res.final_state["facts"]) == 200, (
        f"{200 - len(res.final_state['facts'])} facts vanished without any patch deleting them"
    )
    assert res.final_state["facts"] == expected, "capped rendering rewrote the durable values"
    on_disk = json.loads((tmp_path / "log" / "state.json").read_text())
    assert len(on_disk["facts"]) == 200
    assert on_disk["facts"] == expected
    assert "_omitted" not in on_disk["facts"] and "abbreviated" not in json.dumps(on_disk)


def test_huge_top_level_string_is_capped_with_marker_and_note_without_changing_state_json(tmp_path):
    state = {"goal": "x" * 50000, "facts": {}}
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state))
    before = state_file.read_bytes()
    shown, notes = abbreviate_state(state, 1000)
    assert len(json.dumps(shown, separators=(",", ":"))) == 1000
    prefix, marker = shown["goal"].split("[", 1)
    assert prefix == state["goal"][: len(prefix)]
    assert marker == f"{len(state['goal']) - len(prefix)} characters cut; full value in state.json]"
    assert notes == ["goal: string shortened (full value in state.json)"]
    prompt = render_state_prompt(state, "o", state_max_chars=1000, state_path=str(state_file))
    assert prompt.startswith("Note: the state below is abbreviated for size (goal: string shortened")
    assert json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0]) == shown
    assert state_file.read_bytes() == before
    assert state == json.loads(before)


def test_lists_of_dicts_nested_dicts_and_unicode_render_without_error():
    state = {
        "goal": "g",
        "runs": [{"id": f"r{i}", "meta": {"deep": {"deeper": {"deepest": "é" * 200}}}} for i in range(20)],
        "tree": {"a": {"b": {"c": {"d": {"e": "y" * 5000}}}}},
        "notes": {f"n{i}": "日本語のメモ " * 40 for i in range(15)},
        "current": "r19",
    }
    before = copy.deepcopy(state)
    shown, notes = abbreviate_state(state, 2000)
    json.dumps(shown)
    assert state == before
    assert all(isinstance(x, str) and len(x) <= 96 for x in shown["runs"][:-6] if not isinstance(x, dict))
    assert shown["tree"] == state["tree"], "a single deeply nested entry is below keep_recent and stays whole"
    assert any(n.startswith("runs:") or n.startswith("notes:") for n in notes)
    assert (
        shown["runs"] == ["14 older entries omitted from this rendering; full text in state.json"] + state["runs"][-6:]
    ), "the six newest runs stay whole and the older ones collapse into one marker"
    assert list(shown["notes"]) == ["_omitted", "n9", "n10", "n11", "n12", "n13", "n14"]
    assert all(shown["notes"][k] == state["notes"][k] for k in list(shown["notes"])[1:])
    assert notes == [
        "runs: oldest 14 entries shown as one line",
        "notes: oldest 9 entries shown as one line",
        "notes: 9 older entries omitted (full text in state.json)",
        "runs: 14 older entries omitted (full text in state.json)",
    ]


def test_protected_references_to_missing_keys_and_a_cap_below_the_protected_set_terminate():
    state = {
        "goal": "g",
        "current": "r_missing",
        "baseline": {"runs": ["also_missing"]},
        "runs": {f"r{i:02d}": {"notes": "n" * 300} for i in range(12)},
    }
    before = copy.deepcopy(state)
    shown, notes = abbreviate_state(state, 10)
    assert state == before
    assert set(shown["runs"]) >= {f"r{i:02d}" for i in range(6, 12)}, "the last six entries always stay"
    assert all(shown["runs"][k] == state["runs"][k] for k in list(state["runs"])[-6:])


def test_cap_zero_disables_and_a_negative_cap_ages_everything_it_can():
    state = {"goal": "g", "facts": _facts(30, 200), "current": "fact_0029"}
    assert "abbreviated" not in render_state_prompt(state, "o", state_max_chars=0)
    assert render_state_prompt(state, "o", state_max_chars=0) == render_state_prompt(state, "o"), "0 means no cap"
    before = copy.deepcopy(state)
    shown, notes = abbreviate_state(state, -1)
    assert state == before and notes and "_omitted" in shown["facts"]
    assert list(shown["facts"]) == ["_omitted", *list(state["facts"])[-6:]]
    assert shown["facts"]["_omitted"] == "24 older entries omitted from this rendering; full text in state.json"
    assert all(shown["facts"][k] == state["facts"][k] for k in list(state["facts"])[-6:]), "the last six stay whole"
    assert notes == [
        "facts: oldest 24 entries shown as one line",
        "facts: 24 older entries omitted (full text in state.json)",
    ]
    assert (shown["goal"], shown["current"]) == ("g", "fact_0029"), "scalars are never aged"


def test_note_names_every_aged_field_and_points_at_the_state_file():
    state = {"goal": "g", "facts": _facts(40, 300), "lessons": ["l" * 300] * 20, "current": "fact_0039"}
    text = render_state_prompt(state, "obs", state_max_chars=3000, state_path="/r/state.json")
    head = text.split("\n", 1)[0]
    assert (
        head.startswith("Note: the state below is abbreviated for size (") and "facts:" in head and "lessons:" in head
    )
    assert head.endswith("readable at /r/state.json.")


def test_cli_rejects_state_max_chars_zero(tmp_path, capsys):
    from stepbook.cli import main

    run_dir = tmp_path / "run"  # the default log dir is cwd-relative, so name one we can watch
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                str(ROOT / "stepbook/templates" / "checklist"),
                "--workdir",
                str(tmp_path),
                "--log-dir",
                str(run_dir),
                "--goal",
                "g",
                "--state-max-chars",
                "0",
                "--steps",
                "1",
                "--quiet",
                "--no-checkpoint",
                "--unsandboxed-i-accept-the-risk",
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.splitlines()[-1].endswith("error: argument --state-max-chars: must be positive"), err
    assert not run_dir.exists() and list(tmp_path.iterdir()) == [], "a rejected --state-max-chars must not start a run"
