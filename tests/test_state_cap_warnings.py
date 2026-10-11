"""Operator-only state-cap warnings; prompts and journal contracts stay unchanged."""

import json
from unittest.mock import Mock, patch

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill, StepResult
from stepbook.runtime.rendering import render_state_prompt
from stepbook.state.aging import abbreviate_state


class FixedEnv:
    def reset(self):
        return "ready"

    def step(self, action):
        return StepResult("ready", info={"ok": True})

    def score(self):
        return 0.0


def _runtime(state, cap, **options):
    seen = []
    executor = ScriptedExecutor(lambda system, user: seen.append(user) or Envelope("continue", {}, "next"))
    runtime = Runtime(
        Skill("probe", "instructions", {"type": "object", "properties": {"facts": {"type": "object"}}}, state),
        executor,
        FixedEnv(),
        state_max_chars=cap,
        **options,
    )
    return runtime, seen


def _state_json(prompt):
    return prompt.split("Skill Execution State:\n```json\n", 1)[1].split("\n```", 1)[0]


def test_scalar_truncation_preserves_existing_rendering(capsys):
    # State-cap warnings preserve the existing truncation of top-level strings.
    state = {"status": "ok", "notes": "x" * 100_000}
    shown, notes = abbreviate_state(state, 1000)
    assert len(json.dumps(state, separators=(",", ":"))) == 100_026
    assert len(json.dumps(shown, separators=(",", ":"))) == 1000
    assert notes == ["notes: string shortened (full value in state.json)"]
    runtime, seen = _runtime(state, 1000)
    result = runtime.run(1)
    assert seen == [render_state_prompt(state, "ready", state_max_chars=1000, step=0, max_steps=1)]
    assert len(_state_json(seen[0])) == 1000
    assert result.final_state == state
    assert capsys.readouterr().err == ""


def test_retained_scalar_overrun_warns_once_even_without_verbose_or_journal(capsys):
    state = {"facts": {"notes": "x" * 100_000}}
    runtime, seen = _runtime(state, 1000, verbose=False)
    result = runtime.run(3)
    assert result.steps == 3 and result.final_state == state
    assert len(_state_json(seen[0])) == 100_022
    expected = render_state_prompt(state, "ready", state_max_chars=1000, step=0, max_steps=3)
    assert seen[0].encode() == expected.encode()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Warning: step 0 rendered state is 100022 characters, exceeding --state-max-chars 1000; "
        "state aging is best-effort and retained content can exceed the cap. "
        "Further state-cap warnings are suppressed for this run.\n"
    )


@pytest.mark.parametrize("cap", [None, 0, 100_022, 100_023])
def test_disabled_or_met_cap_is_silent_even_with_large_observation(cap, capsys):
    runtime, _ = _runtime({"facts": {"notes": "x" * 100_000}}, cap)
    runtime.env.reset = lambda: "observation" * 20_000
    runtime.run(1)
    assert capsys.readouterr().err == ""


def test_warning_can_first_appear_on_later_step_and_resets_for_next_run(capsys):
    runtime, seen = _runtime({}, 1000)
    runtime.executor = ScriptedExecutor(
        lambda system, user: seen.append(user) or Envelope("grow", {"facts": {"notes": "x" * 100_000}}, "next")
    )
    for _ in range(2):
        runtime.run(3)
        err = capsys.readouterr().err
        assert err.count("Warning:") == 1
        assert "step 1 rendered state is 100022 characters" in err


def test_warning_measures_aged_json_including_omission_markers_and_escaping(tmp_path, capsys):
    state = {"facts": {f"f{i}": "é\n" * 500 for i in range(20)}}
    runtime, seen = _runtime(state, 1000, log_dir=tmp_path / "run")
    runtime.run(2)
    shown = _state_json(seen[0])
    assert "_omitted" in shown and "\\u00e9" in shown
    assert len(shown) < len(json.dumps(state, separators=(",", ":")))
    err = capsys.readouterr().err
    assert err.count("Warning:") == 1
    assert f"rendered state is {len(shown)} characters, exceeding --state-max-chars 1000" in err
    assert json.loads((tmp_path / "run" / "state.json").read_text()) == state


def test_repair_attempts_share_the_warning_and_resume_gets_one_new_warning(tmp_path, capsys):
    state = {"facts": {"notes": "x" * 100_000}}
    runtime, seen = _runtime(state, 1000, log_dir=tmp_path / "run")
    responses = iter([Envelope("invalid", {"facts": 42}, "next"), Envelope("repaired", {}, "next")])
    runtime.executor = ScriptedExecutor(lambda system, user: seen.append(user) or next(responses))
    result = runtime.run(1)
    assert result.rejected_patches == 1 and len(seen) == 2
    assert capsys.readouterr().err.count("Warning:") == 1
    runtime.executor = ScriptedExecutor(lambda system, user: Envelope("continue", {}, "next"))
    runtime.run(3, resume=True, replay_env=False)
    err = capsys.readouterr().err
    assert err.count("Warning:") == 1
    assert "step 1 rendered state is 100022 characters" in err


@pytest.mark.parametrize("cap", [None, 1000, 100_000])
def test_each_prompt_ages_and_serializes_state_only_once(cap, monkeypatch, capsys):
    state = {"facts": {"notes": "é\n" * 1000}}
    shown = {"facts": {"notes": "é\n" * 500, "_omitted": 1}}
    age = Mock(return_value=(shown, ["probe omission"]))
    monkeypatch.setattr("stepbook.runtime.rendering.abbreviate_state", age)
    monkeypatch.setattr("stepbook.runtime.loop.abbreviate_state", age)
    runtime, _ = _runtime(state, cap)
    for step in range(3):
        age.reset_mock()
        with patch("stepbook.runtime.rendering.json.dumps", wraps=json.dumps) as dumps:
            prompt = runtime._render(state, "ready", [], step)
        assert isinstance(prompt, str)
        assert age.call_count == (1 if cap else 0)
        assert dumps.call_count == 1
        assert _state_json(prompt) == json.dumps(shown if cap else state, separators=(",", ":"), sort_keys=True)
    err = capsys.readouterr().err
    if cap == 1000:
        assert err.count("Warning:") == 1
        assert f"rendered state is {len(_state_json(prompt))} characters" in err
    else:
        assert err == ""
