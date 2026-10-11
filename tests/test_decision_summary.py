from types import SimpleNamespace

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill, StepResult
from stepbook.runtime.loop import (
    DECISION_SUMMARY,
    STATE_OUTPUT_SCHEMA,
)


def test_same_concise_summary_contract_without_evidence_truncation():
    for schema in (STATE_OUTPUT_SCHEMA,):
        assert schema["properties"]["reasoning"]["description"] == DECISION_SUMMARY
        assert "maxLength" not in schema["properties"]["reasoning"]
    seen = []

    def respond(system_prompt, user_prompt):
        seen.append(user_prompt)
        return Envelope("retain evidence", {}, "stop")

    skill = Skill("decision-summary", "retain evidence", {"type": "object"}, {})
    for runtime_class in (Runtime,):
        env = SimpleNamespace(
            reset=lambda: "evidence",
            step=lambda action: StepResult("stopped", done=True),
            score=lambda: 0.0,
        )
        runtime_class(skill, ScriptedExecutor(respond), env).run(1)
    assert len(seen) == 1
    for prompt in seen:
        assert DECISION_SUMMARY in prompt
        assert "evidence" in prompt
        assert "Step-by-step reasoning" not in prompt
