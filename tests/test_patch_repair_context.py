"""A rejected state patch must not throw away its paid investigation on repair."""

from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.reporting.usage import usage_summary
from stepbook.runtime.loop import StepResult, render_patch_repair
from stepbook.state.validator import PatchError
from stepbook.storage.events import EventLog


class Environment:
    def __init__(self):
        self.actions = []

    def reset(self):
        return "Investigate the source, then choose an experiment."

    def step(self, action):
        self.actions.append(action)
        return StepResult("accepted", done=True, info={"ok": True})

    def score(self):
        return 0


def test_repair_retains_findings_and_action_without_committing_invalid_phase(tmp_path):
    skill = Skill.load(Path(__file__).resolve().parents[1] / "stepbook/templates/eval-loop-adapter")
    env, prompts = Environment(), []

    def model(system, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return Envelope(
                "Inspected source.pdf page 2: context was truncated, not missing from the document.",
                {"current": {"phase": "research/select"}},
                "propose candidate-7",
                cost_usd=0.08,
                input_tokens=100,
                output_tokens=30,
                tool_calls=1,
                tool_log=[{"tool": "Read", "input": {"file_path": "source.pdf", "pages": "2"}}],
            )
        assert "context was truncated" in prompt and "propose candidate-7" in prompt
        assert "NOT COMMITTED, ACTION NOT EXECUTED" in prompt
        assert not env.actions
        return Envelope(
            "Repair only the invalid phase; no repeated source inspection.",
            {"current": {"phase": "research"}},
            "propose candidate-7",
            cost_usd=0.01,
            input_tokens=50,
            output_tokens=10,
        )

    result = Runtime(skill, ScriptedExecutor(model), env, log_dir=tmp_path).run(1)
    assert result.rejected_patches == 1 and result.final_state["current"]["phase"] == "research"
    assert env.actions == ["propose candidate-7"] and len(prompts) == 2
    events = EventLog(tmp_path / "events.jsonl").read()
    rejected = next(e for e in events if e["kind"] == "patch_rejected")
    assert rejected["action"] == "propose candidate-7"
    calls = [e for e in events if e["kind"] == "model_call"]
    assert calls[0]["tool_log"][0]["input"]["file_path"] == "source.pdf"
    assert calls[1]["tool_log"] == []
    assert round(usage_summary(events)["known_total_usd"], 2) == 0.09  # synthetic test charges, not a live saving


def test_repair_context_is_bounded_but_full_rejected_response_is_retained(tmp_path):
    skill = Skill("small", "work", {"type": "object", "additionalProperties": False}, {})
    seen = []
    # Distinguishable head, middle and tail: a same-length corruption must not pass as "retained".
    reasoning = "HEAD-FINDING " + "x" * 19000 + " MIDDLE-FINDING " + "y" * 20958 + " TAIL-FINDING"
    assert len(reasoning) == 40000

    def model(system, prompt):
        seen.append(prompt)
        return Envelope(reasoning, {"bad": True}, "work") if len(seen) == 1 else Envelope("fixed", {}, "work")

    Runtime(skill, ScriptedExecutor(model), Environment(), log_dir=tmp_path).run(1)
    assert len(seen[1]) < 19000 and "truncated" in seen[1]
    assert "HEAD-FINDING" in seen[1] and "TAIL-FINDING" not in seen[1], "the repair prompt keeps the head"
    rejected = next(e for e in EventLog(tmp_path / "events.jsonl").read() if e["kind"] == "patch_rejected")
    assert len(rejected["reasoning"]) == 40000 and rejected["patch"] == {"bad": True}
    assert rejected["reasoning"] == reasoning, "the journal keeps the paid investigation verbatim, not its length"
    assert rejected["action"] == "work"


@pytest.mark.parametrize("include_response", [False, True])
def test_repair_prompt_explicit_response_retention(include_response):
    error = PatchError(["invalid phase"])
    response = Envelope("UNCOMMITTED_FINDING", {"phase": "bad"}, "propose pending")
    prompt = render_patch_repair("source observation", error, response, include_response=include_response)
    base = "source observation\n\n" + error.as_observation()
    assert prompt.startswith(base)
    if include_response:
        assert "UNCOMMITTED_FINDING" in prompt and "propose pending" in prompt
        assert "NOT COMMITTED, ACTION NOT EXECUTED" in prompt
    else:
        assert prompt == base
