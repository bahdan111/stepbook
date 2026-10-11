"""Render-time state aging.
Off by default; with a cap, the oldest entries of the largest fields are rendered as one line and the prompt points at
the full state on disk. The state itself never changes."""

import json
from pathlib import Path

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv
from stepbook.runtime.loop import abbreviate_state, render_state_prompt

ROOT = Path(__file__).resolve().parent.parent


def _big_state():
    runs = {f"r{i:02d}": {"run_dir": f"/tmp/run{i}", "pass": 40 + i % 7, "notes": "x" * 300} for i in range(20)}
    lessons = [f"lesson {i}: " + "y" * 200 for i in range(10)]
    return {"goal": "g", "current": "r19", "runs": runs, "lessons": lessons, "facts": {"a": "b"}}


def test_no_cap_renders_the_whole_state():
    state = _big_state()
    text = render_state_prompt(state, "obs")
    assert json.dumps(state, separators=(",", ":"), sort_keys=True) in text and "abbreviated" not in text


def test_cap_ages_oldest_entries_of_the_largest_field_and_keeps_recent_ones_whole():
    state = _big_state()
    before = json.dumps(state, sort_keys=True)
    shown, aged = abbreviate_state(state, max_chars=3000)
    assert json.dumps(state, sort_keys=True) == before, "the state itself must not change"
    after = len(json.dumps(shown, separators=(",", ":"), sort_keys=True))
    assert after < len(before) * 0.7, "aging must shrink the rendering substantially"
    # the last 6 entries of every field stay whole even when the cap cannot be met: recency beats the cap
    assert aged[:2] == ["runs: oldest 14 entries shown as one line", "lessons: oldest 4 entries shown as one line"]
    # If still over the cap after one-lining, the oldest entries are omitted behind a marker.
    assert any(n.startswith("runs: ") and "omitted" in n for n in aged)
    assert "r00" not in shown["runs"] and shown["runs"]["_omitted"].endswith("full text in state.json")
    assert shown["runs"]["r19"] == state["runs"]["r19"], "the newest entries stay whole"
    kept = [k for k in shown["runs"] if k != "_omitted"]
    assert kept == [k for k in state["runs"] if k in shown["runs"]] and kept[-6:] == list(state["runs"])[-6:], (
        "order and recency preserved"
    )
    assert all(len(v) <= 96 for k, v in shown["runs"].items() if isinstance(v, str) and k != "_omitted")


def test_rendered_prompt_carries_the_aging_note_and_the_state_path():
    text = render_state_prompt(_big_state(), "obs", state_max_chars=3000, state_path="/runs/x/state.json")
    assert text.startswith("Note: the state below is abbreviated for size (") and "/runs/x/state.json" in text
    assert "Skill Execution State:" in text


def test_runtime_knob_is_off_by_default_and_only_changes_the_prompt(tmp_path):
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    skill.initial_state = {**skill.initial_state, "facts": {f"f{i:03d}": f"{i}: " + "x" * 60 for i in range(200)}}
    seen = []
    steps = iter([Envelope("a", {"current": "one"}, "echo ok"), Envelope("b", {}, "done")])
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    rt = Runtime(skill, ScriptedExecutor(lambda s, u: seen.append(u) or next(steps)), env, log_dir=tmp_path / "log")
    assert rt.state_max_chars is None
    uncapped = rt.run(5)
    assert seen and "abbreviated" not in seen[0]
    seen2 = []
    steps2 = iter([Envelope("a", {"current": "one"}, "echo ok"), Envelope("b", {}, "done")])
    rt2 = Runtime(
        skill,
        ScriptedExecutor(lambda s, u: seen2.append(u) or next(steps2)),
        ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False),
        log_dir=tmp_path / "log2",
        state_max_chars=50,
    )
    assert rt2.state_max_chars == 50
    capped = rt2.run(5)
    assert seen2[0].startswith("Note: the state below is abbreviated for size (") and "facts:" in seen2[0]
    assert "_omitted" in seen2[0] and len(seen2[0]) < len(seen[0]) / 2, "the capped prompt is the shorter one"
    assert capped.final_state == uncapped.final_state, "the knob changes the prompt, never the state"
    assert json.loads((tmp_path / "log2" / "state.json").read_text()) == json.loads(
        (tmp_path / "log" / "state.json").read_text()
    )
    assert len(capped.final_state["facts"]) == 200 and "_omitted" not in capped.final_state["facts"]


def test_entries_named_in_baseline_or_kept_stack_stay_whole():
    state = _big_state()
    state["baseline"] = {"runs": ["r00", "r01"], "pass_mean": 41.5}
    state["kept_stack"] = ["r05"]
    shown, aged = abbreviate_state(state, max_chars=3000)
    for ref in ("r00", "r01", "r05"):
        assert shown["runs"][ref] == state["runs"][ref], f"reference run {ref} was abbreviated"
    assert "r02" not in shown["runs"] or isinstance(shown["runs"]["r02"], str), (
        "non-reference old runs are aged or omitted"
    )
    assert aged and aged[0].startswith("runs:")


def test_short_reference_ids_are_protected_and_words_inside_a_sentence_are_not():
    """A reference is an exact recorded value: a two-character id counts, a word in prose does not."""
    state = {
        "goal": "g",
        "current": "the kept idea is h1; lessons were learned",
        "kept_stack": ["h1"],
        "hypotheses": {f"h{i}": {"evidence": "e" * 300} for i in range(1, 12)},
        "lessons": {"learned": "l" * 300, **{f"note{i}": "n" * 300 for i in range(10)}},
    }
    shown, aged = abbreviate_state(state, max_chars=1200)
    assert shown["hypotheses"]["h1"] == state["hypotheses"]["h1"], "the referenced short id stays whole"
    assert any(not isinstance(value, dict) for key, value in shown["hypotheses"].items() if key != "h1"), (
        "unreferenced entries are still aged"
    )
    assert shown["lessons"].get("learned") != state["lessons"]["learned"], (
        "a word inside the sentence is not a reference, so that entry is aged like any other"
    )
    assert aged


def test_aging_shortens_long_string_facts_too():
    """Long string-valued facts must also be abbreviated."""
    import json

    from stepbook.runtime.loop import abbreviate_state

    state = {"goal": "g", "facts": {f"fact_{i:03d}": f"{i}: " + "x" * 400 for i in range(200)}, "current": "fact_199"}
    rendered, notes = abbreviate_state(state, 12000)
    assert len(json.dumps(rendered, separators=(",", ":"), sort_keys=True)) <= 12000 + 600
    assert "fact_000" not in rendered["facts"] and rendered["facts"]["_omitted"].endswith("state.json")
    assert rendered["facts"]["fact_199"] == state["facts"]["fact_199"]  # referenced by `current`, kept whole
    assert rendered["facts"]["fact_198"] == state["facts"]["fact_198"]  # recent entries stay whole
    assert any(n.startswith("facts: oldest") for n in notes) and any("omitted" in n for n in notes)
