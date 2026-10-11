"""Resume preserves operator notes and full state even without user_requirements."""

import json

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill, StepResult


class Observations:
    def __init__(self, initial):
        self.initial = initial
        self.actions = []

    def reset(self):
        return self.initial

    def step(self, action):
        self.actions.append(action)
        return StepResult(f"result:{action}", info={"ok": True})

    def score(self):
        return 0.0


def test_resume_keeps_notes_exact_state_and_latest_observation(tmp_path):
    """Bug: resuming a capped state without user_requirements loses durable facts or operator notes."""
    facts = {f"f{i:03}": f"evidence-{i}: " + "雪" * 100 for i in range(100)}
    skill = Skill(
        "notes",
        "preserve evidence",
        {"type": "object", "properties": {"facts": {"type": "object"}}, "additionalProperties": False},
        {"facts": {}},
    )
    log = tmp_path / "log"
    log.mkdir()
    (log / "inbox.txt").write_text("keep café evidence\n")
    Runtime(
        skill,
        ScriptedExecutor(lambda s, u: Envelope("record", {"facts": facts}, "old")),
        Observations("initial-old"),
        log_dir=log,
        state_max_chars=2000,
    ).run(1)
    env, seen = Observations("fresh-reset-must-not-replace-journal"), []
    actions = iter(["new", "next"])

    def respond(system, user):
        seen.append(user)
        return Envelope("continue", {}, next(actions))

    result = Runtime(skill, ScriptedExecutor(respond), env, log_dir=log, state_max_chars=2000).run(
        3, resume=True, replay_env=False
    )
    assert env.actions == ["new", "next"]
    assert result.final_state == {"facts": facts}
    assert json.loads((log / "state.json").read_text()) == {"facts": facts}
    for prompt, observation in zip(seen, ["result:old", "result:new"], strict=True):
        assert (
            "STANDING OPERATOR NOTES (delivered earlier in this run; still in force):\n  - keep café evidence\n"
            in prompt
        )
        assert "abbreviated for size" in prompt
        assert prompt.split("<<<OBSERVATION\n", 1)[1].split("\nOBSERVATION>>>", 1)[0] == observation
        assert "fresh-reset-must-not-replace-journal" not in prompt
    assert "user_requirements" not in result.final_state
