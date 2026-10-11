"""Bad controller output (invalid JSON, protected/typed patch violations, odd action strings)."""

import json
import sys

import pytest

from stepbook import CommandExecutor, Envelope, Runtime
from stepbook.contracts.errors import ExecutorError
from stepbook.run_pointer import classify_run

from ._helpers import Script, checklist, cmd, events, kinds, shell_env, steps


def test_command_executor_invalid_json_retries_then_raises_and_leaves_a_resumable_journal(tmp_path, monkeypatch):
    monkeypatch.setattr("stepbook.runtime.loop.time.sleep", lambda s: None)
    agent = tmp_path / "agent.py"
    agent.write_text('import sys\nsys.stdin.read()\nprint(\'{"reasoning": "oops", "state_patch": \')\n')
    env = shell_env(tmp_path)
    with pytest.raises(ExecutorError, match="failed 3 times"):
        Runtime(checklist(), CommandExecutor(command=f"{sys.executable} {agent}"), env, log_dir=tmp_path / "log").run(3)
    k = kinds(tmp_path / "log")
    assert k.count("executor_error") == 3 and k.count("failed_model_call") == 3 and "step" not in k
    info = classify_run(tmp_path / "log")
    assert info["status"] == "interrupted" and info["pending_actions"] == 0


def test_patch_removing_a_protected_field_is_rejected_every_time_and_the_run_aborts(tmp_path):
    bad = Envelope("r", {"goal": None}, "echo never >> marks.txt")
    env = shell_env(tmp_path)
    res = Runtime(checklist(), Script(exhausted=bad).executor(), env, log_dir=tmp_path / "log").run(3)
    assert res.aborted and res.rejected_patches == 4 and res.steps == 0 and not (tmp_path / "marks.txt").exists()
    rejections = [e["problems"] for e in events(tmp_path / "log") if e.get("kind") == "patch_rejected"]
    assert len(rejections) == 4 and all(any("'goal' is protected" in p for p in ps) for ps in rejections)
    assert kinds(tmp_path / "log")[-1] == "abort"


def test_patch_setting_an_object_field_to_a_string_or_none_patch_is_rejected(tmp_path):
    env = shell_env(tmp_path)
    script = Script(Envelope("r", {"checklist": "all done"}, "echo x"), Envelope("r", None, "echo x"), cmd("done"))
    res = Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log").run(3)
    rej = [e for e in events(tmp_path / "log") if e.get("kind") == "patch_rejected"]
    assert len(rej) == 2 and "schema violation at 'checklist'" in rej[0]["problems"][0]
    assert "must be a JSON object, got NoneType" in rej[1]["problems"][0]
    assert res.steps == 1 and res.assisted_steps == 1


@pytest.mark.parametrize("action", ["", "   \n"])
def test_empty_action_is_rejected_not_executed(tmp_path, action):
    env = shell_env(tmp_path)
    script = Script(Envelope("r", {}, action), cmd("done"))
    res = Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log").run(3)
    executed = [e["action"] for e in steps(tmp_path / "log")]
    assert all(a.strip() for a in executed), f"an empty action was journaled as executed: {executed!r}"
    assert executed == ["done"], "the empty action is refused and the retry's action is the one that runs"
    rejected = [e for e in events(tmp_path / "log") if e.get("kind") == "patch_rejected"]
    assert [(e["step"], e["attempt"], e["action"], e["problems"]) for e in rejected] == [
        (0, 0, action, ["action must be a nonempty command string"])
    ]
    assert kinds(tmp_path / "log") == [
        "run_start",
        "model_call",
        "patch_rejected",
        "model_call",
        "action_started",
        "action_result",
        "step",
    ]
    assert script.calls == 2 and res.steps == 1 and res.assisted_steps == 1 and res.rejected_patches == 1


@pytest.mark.parametrize("closing", ["DONE", "\n  done \n"])
def test_done_variants_complete_and_close_the_journal(tmp_path, closing):
    env = shell_env(tmp_path)
    res = Runtime(checklist(), Script(cmd(closing)).executor(), env, log_dir=tmp_path / "log").run(3)
    assert res.score == 1.0 and steps(tmp_path / "log")[-1]["done"]
    again = Script(exhausted=None)
    Runtime(checklist(), again.executor(), shell_env(tmp_path), log_dir=tmp_path / "log").run(
        3, resume=True, replay_env=False
    )
    assert again.calls == 0


def test_state_growing_past_the_cap_every_step_keeps_the_prompt_bounded_and_the_state_intact(tmp_path):
    env = shell_env(tmp_path)
    grow = [Envelope("r", {"facts": {f"fact_{i:02d}": f"{i}: " + "x" * 500}}, "echo x") for i in range(10)]
    script = Script(*grow, cmd("done"))
    res = Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log", state_max_chars=3000).run(20)
    assert len(res.final_state["facts"]) == 10
    assert res.final_state["facts"] == {f"fact_{i:02d}": f"{i}: " + "x" * 500 for i in range(10)}, (
        "the durable state keeps every value in full while the rendering is capped"
    )
    blocks = [p.split("```json\n", 1)[1].split("\n```", 1)[0] for p in script.prompts]
    assert max(len(b) for b in blocks) < 3000 + 900, "rendering must stay near the cap while the state keeps growing"
    assert all(len(json.loads(b)["facts"]) >= 6 for b in blocks[6:])
    assert [list(json.loads(b)["facts"]) for b in blocks[-2:]] == [
        ["_omitted", "fact_03", "fact_04", "fact_05", "fact_06", "fact_07", "fact_08"],
        ["_omitted", "fact_04", "fact_05", "fact_06", "fact_07", "fact_08", "fact_09"],
    ], "once over the cap the window slides: one marker plus the six newest facts"
    last = json.loads(blocks[-1])["facts"]
    assert last["fact_09"] == res.final_state["facts"]["fact_09"], "the newest entry is rendered whole"
    assert last["_omitted"] == "4 older entries omitted from this rendering; full text in state.json"
