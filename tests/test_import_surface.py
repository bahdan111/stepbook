"""Public constructors and result fields used by external consumers."""

import dataclasses
import inspect
from pathlib import Path

import pytest

import stepbook
from stepbook.runtime.settings import RuntimeSettings

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = {
    "Runtime": "external API consumers",
    "Skill": "external API consumers",
    "StepResult": "external API consumers",
    "Envelope": "external API consumers",
    "ScriptedExecutor": "tests/test_journal_replay.py and scripted external consumers",
    "RunResult": "external API consumers",
}


def _public(name):
    value = getattr(stepbook, name, None)
    assert callable(value), f"{CONSUMERS[name]} requires callable 'from stepbook import {name}'"
    return value


@pytest.mark.parametrize("name", CONSUMERS)
def test_public_import_exists(name):
    _public(name)


def test_envelope_positional_replace_asdict_and_mutable_tokens():
    Envelope = _public("Envelope")
    original = Envelope("Decision", {"phase": "inspected"}, "done")
    assert (original.reasoning, original.state_patch, original.action) == ("Decision", {"phase": "inspected"}, "done")
    original.input_tokens = 19
    original.output_tokens = 7
    copied = dataclasses.replace(original, action="check", cost_usd=0.125)
    assert copied is not original and copied.action == "check" and original.action == "done"
    assert (copied.input_tokens, copied.output_tokens, copied.cost_usd) == (19, 7, 0.125)
    assert dataclasses.asdict(copied) == {
        "reasoning": "Decision",
        "state_patch": {"phase": "inspected"},
        "action": "check",
        "input_tokens": 19,
        "output_tokens": 7,
        "cost_usd": 0.125,
        "duration_s": 0.0,
        "api_calls": 1,
        "usage": {},
        "tool_calls": 0,
        "tool_log": [],
        "extra": {},
        "raw": {},
    }, "external API consumers"


def test_skill_step_result_and_executor_callable_shapes():
    Skill, StepResult = _public("Skill"), _public("StepResult")
    Envelope, ScriptedExecutor = _public("Envelope"), _public("ScriptedExecutor")
    skill = Skill("surface", "Inspect", {"type": "object"}, {})
    assert (skill.name, skill.instructions, skill.schema, skill.initial_state) == (
        "surface",
        "Inspect",
        {"type": "object"},
        {},
    )
    loaded = Skill.load(ROOT / "stepbook/templates" / "checklist", include_schema=False)
    assert loaded.name == "checklist" and loaded.instructions
    result = StepResult("Observed", True, {"ok": True})
    assert (result.observation, result.done, result.info) == ("Observed", True, {"ok": True})
    assert StepResult("Observed").done is False and StepResult("Observed").info == {}
    calls = []

    def respond(system, user):
        calls.append((system, user))
        return Envelope("Decision", {}, "done")

    executor = ScriptedExecutor(respond)
    assert executor.step("system", "user", {"type": "object"}).action == "done"
    assert executor.step(system_prompt="system2", user_prompt="user2", output_schema={}).action == "done"
    assert calls == [("system", "user"), ("system2", "user2")]


@pytest.mark.parametrize("name", ["Runtime"])
def test_runtime_constructor_and_run_shapes(name, tmp_path):
    Skill, StepResult = _public("Skill"), _public("StepResult")
    Envelope, ScriptedExecutor = _public("Envelope"), _public("ScriptedExecutor")
    runtime_type = _public(name)
    skill = Skill("surface", "Finish", {"type": "object"}, {})

    class Env:
        def reset(self):
            return "Ready"

        def step(self, action):
            assert action == "done"
            return StepResult("Finished", done=True, info={"ok": True})

        def score(self):
            return 1.0

    executor = ScriptedExecutor(lambda s, u: Envelope("Finish", {}, "done"))
    # Both positional and named leading arguments remain valid for external callers.
    inspect.signature(runtime_type).bind(skill=skill, executor=executor, env=Env(), log_dir=tmp_path)
    # The signature accepts any keyword through **overrides, so binding proves nothing about
    # the settings an external caller passes; the declared fields are what the constructor really accepts.
    assert {"log_dir", "verbose"} <= set(RuntimeSettings.model_fields)
    assert (
        not {"measurement_repeats", "repeat_limit", "on_exhausted", "skip_action"} & RuntimeSettings.model_fields.keys()
    )
    assert "max_total_usd" not in RuntimeSettings.model_fields
    assert "max_total_usd" not in inspect.signature(runtime_type).parameters
    runtime = runtime_type(skill, executor, Env(), log_dir=tmp_path, verbose=False)
    result = runtime.run(1)
    assert result.steps == 1 and result.score == 1.0 and not result.aborted, CONSUMERS[name]


def test_summary_consumer_keys_and_rounded_values():
    consumer = "public RunResult API"
    RunResult = _public("RunResult")
    result = RunResult(
        "stepbook",
        3,
        3,
        1,
        0.123456,
        [10, 20, 20],
        [2, 3, 4],
        0.987654,
        12.3456,
        final_state={"private": "not in summary"},
        aborted=True,
        assisted_steps=1,
        first_attempt_score=0.112233,
        rendered_prompt_chars=[10, 20, 20],
        model_ids=["scripted"],
    )
    summary = result.summary()
    expected = {
        "runtime": "stepbook",
        "steps": 3,
        "llm_calls": 3,
        "rejected_patches": 1,
        "score": 0.1235,
        "avg_prompt_tokens": 16.7,
        "max_prompt_tokens": 20,
        "total_tokens": 59,
        "cost_usd": 0.9877,
        "cost_complete": False,
        "wall_s": 12.3,
        "aborted": True,
        "assisted_steps": 1,
        # first_attempt_score is intentionally NOT rounded by the existing API.
        "first_attempt_score": 0.112233,
        "rendered_prompt_tokens_est": [3, 6, 6],
        "rendered_prompt_tokens_est_total": 14,
        "model_ids": ["scripted"],
    }
    assert summary == expected, f"{consumer}: summary values/rounding changed"
    for key in ("score", "avg_prompt_tokens", "cost_usd", "wall_s", "first_attempt_score"):
        assert type(summary[key]) is float, f"{consumer}: {key} must remain numeric"
    assert result.prompt_tokens == [10, 20, 20], "external API consumers"


def test_conversational_runtime_is_only_local():
    from stepbook.runtime import loop as runtime

    assert not hasattr(stepbook, "ConversationalRuntime")
    assert not hasattr(runtime, "ConversationalRuntime")
