from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv
from stepbook.storage.events import EventLog


@pytest.mark.parametrize("retry", [False, True])
def test_journal_measures_sent_prompt_with_state_cap_and_standing_note(tmp_path, retry):
    skill = Skill.load(Path(__file__).resolve().parents[1] / "stepbook/templates" / "checklist")
    skill.initial_state["facts"] = {f"f{i:03d}": f"{i}: " + "x" * 100 for i in range(100)}
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "inbox.txt").write_text("Preserve the operator's evidence.\n")
    prompts = []

    def executor(system_prompt, user_prompt):
        prompts.append((system_prompt, user_prompt))
        patch = {"facts": 42} if retry and len(prompts) == 1 else {"facts": {"new": "accepted"}}
        return Envelope("record evidence", patch, "done")

    env = ShellEnv(workdir=tmp_path, goal="g", require_sandbox=False, checkpoint=False, checks={})
    result = Runtime(skill, ScriptedExecutor(executor), env, log_dir=log_dir, state_max_chars=1500).run(1)

    assert result.steps == 1
    assert len(prompts) == (2 if retry else 1)
    system_prompt, user_prompt = prompts[-1]
    assert "state below is abbreviated" in user_prompt
    assert "STANDING OPERATOR NOTES" in user_prompt
    assert "Preserve the operator's evidence." in user_prompt
    assert result.final_state["facts"]["new"] == "accepted"
    steps = [event for event in EventLog(log_dir / "events.jsonl").read() if event["kind"] == "step"]
    assert steps[0]["rendered_prompt_chars"] == len(system_prompt) + len(user_prompt)
