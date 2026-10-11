"""The operator inbox (pause lines, messages, timing relative to the step)."""

import os
import stat
import threading
import time

import pytest

from stepbook import Envelope, Runtime
from stepbook.run_pointer import classify_run

from ._helpers import Script, checklist, cmd, events, kinds, shell_env, steps


def _rt(tmp_path, script, log="log", **env_kw):
    env = shell_env(tmp_path, log_dir=tmp_path / log, **env_kw)
    return Runtime(checklist(), script.executor(), env, log_dir=tmp_path / log), env


def _inbox(tmp_path, text, log="log"):
    (tmp_path / log).mkdir(exist_ok=True)
    (tmp_path / log / "inbox.txt").write_text(text)


def test_pause_before_the_first_step_pauses_without_a_model_call_and_resumes(tmp_path):
    _inbox(tmp_path, "pause\n")
    script = Script(exhausted=None)
    rt, _ = _rt(tmp_path, script)
    res = rt.run(5)
    assert res.aborted and script.calls == 0 and res.steps == 0
    paused = [e for e in events(tmp_path / "log") if e.get("kind") == "paused"]
    assert paused and paused[-1]["step"] == 0
    assert classify_run(tmp_path / "log")["status"] == "paused"
    again = Script(cmd("echo a"), cmd("done"))
    rt2, _ = _rt(tmp_path, again)
    res2 = rt2.run(5, resume=True, replay_env=False)
    assert again.calls == 2 and res2.steps == 2 and not res2.aborted


def test_pause_plus_messages_land_in_requirements_and_survive_resume(tmp_path):
    _inbox(tmp_path, "pause\nkeep the README untouched\n")
    rt, _ = _rt(tmp_path, Script(exhausted=None))
    rt.run(5)
    ops = [e for e in events(tmp_path / "log") if e.get("kind") == "operator_messages"]
    assert ops and ops[0]["messages"] == ["pause", "keep the README untouched"]
    again = Script(cmd("done"))
    rt2, _ = _rt(tmp_path, again)
    res = rt2.run(5, resume=True, replay_env=False)
    assert [r["text"] for r in res.final_state["user_requirements"]] == ["keep the README untouched"]
    assert "keep the README untouched" in again.prompts[0]


def test_whitespace_only_inbox_is_consumed_silently(tmp_path):
    _inbox(tmp_path, "  \n\n\t \n")
    rt, _ = _rt(tmp_path, Script(cmd("done")))
    res = rt.run(3)
    assert res.steps == 1 and "operator_messages" not in kinds(tmp_path / "log")
    assert not (tmp_path / "log" / "inbox.txt").exists(), "an empty inbox file is still consumed (renamed)"


def test_pause_line_case_and_surrounding_whitespace_still_pause(tmp_path):
    _inbox(tmp_path, "   PAUSE  \n")
    script = Script(exhausted=None)
    rt, _ = _rt(tmp_path, script)
    assert rt.run(3).aborted and script.calls == 0 and "paused" in kinds(tmp_path / "log")


def test_pause_arriving_during_the_controller_call_withholds_the_proposed_action(tmp_path):
    def answer(prompt):
        _inbox(tmp_path, "pause\n")
        return cmd("echo ran >> marks.txt")

    script = Script(answer)
    rt, _ = _rt(tmp_path, script)
    res = rt.run(5)
    assert res.aborted and script.calls == 1 and not (tmp_path / "marks.txt").exists()
    k = kinds(tmp_path / "log")
    assert "action_withheld" in k and "paused" in k and "step" not in k
    again = Script(cmd("done"))
    rt2, _ = _rt(tmp_path, again)
    rt2.run(5, resume=True, replay_env=False)
    assert "HOST RECOVERY FACT" in again.prompts[0] and "echo ran >> marks.txt" in again.prompts[0]


def test_pause_arriving_during_the_command_pauses_after_the_step_commits(tmp_path):
    script = Script(cmd("printf 'pause\\n' > log/inbox.txt; echo did-it"), exhausted=None)
    rt, _ = _rt(tmp_path, script)
    res = rt.run(5)
    assert res.aborted and script.calls == 1 and res.steps == 1
    assert kinds(tmp_path / "log")[-1] == "paused"
    again = Script(cmd("done"))
    rt2, _ = _rt(tmp_path, again)
    rt2.run(5, resume=True, replay_env=False)
    assert "did-it" in again.prompts[0], (
        "resume must re-feed the observation of the step that committed before the pause"
    )


def test_operator_messages_are_inside_the_observation_frame_before_the_tool_output(tmp_path):
    _inbox(tmp_path, "note one\nnote two\n")
    script = Script(cmd("echo tool-output"), cmd("done"))
    rt, _ = _rt(tmp_path, script)
    res = rt.run(5)
    first = script.prompts[0]
    frame = first.index("<<<OBSERVATION")
    body = first[frame:]  # the note text also appears in the standing-notes block above the state
    assert (
        body.index("OPERATOR MESSAGES")
        < body.index("- note one")
        < body.index("- note two")
        < body.index("Working directory:")
    )
    assert [r["text"] for r in res.final_state["user_requirements"]] == ["note one", "note two"]
    assert "OPERATOR MESSAGES" not in script.prompts[1], "delivered once in the frame; afterwards it is a standing note"
    assert "STANDING OPERATOR NOTES" in script.prompts[1] and "note one" in script.prompts[1]


def test_pause_during_a_wait_action_ends_the_wait_and_pauses_before_the_next_call(tmp_path):
    script = Script(cmd("wait 20s until false every 1s"), exhausted=None)
    rt, env = _rt(tmp_path, script)
    threading.Timer(0.3, lambda: _inbox(tmp_path, "pause\n")).start()
    t0 = time.monotonic()
    res = rt.run(5)
    assert time.monotonic() - t0 < 2.5
    assert res.aborted and res.steps == 1 and script.calls == 1
    assert steps(tmp_path / "log")[-1]["info"]["wait"] == "operator message waiting"
    assert kinds(tmp_path / "log")[-1] == "paused"


def test_pause_on_resume_pauses_again_without_a_call(tmp_path):
    _inbox(tmp_path, "pause\n")
    rt, _ = _rt(tmp_path, Script(exhausted=None))
    rt.run(5)
    _inbox(tmp_path, "pause\n")
    again = Script(exhausted=None)
    rt2, _ = _rt(tmp_path, again)
    assert rt2.run(5, resume=True, replay_env=False).aborted and again.calls == 0
    assert kinds(tmp_path / "log").count("paused") == 2


def test_undecodable_inbox_bytes_do_not_crash_the_run(tmp_path):
    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "inbox.txt").write_bytes(b"pause\nkeep the caf\xe9 file\n")  # latin-1 paste
    script = Script(cmd("done"))
    rt, _ = _rt(tmp_path, script)
    res = rt.run(3)  # Invalid text must still be consumed without a decoding exception.
    assert res.aborted and "paused" in kinds(tmp_path / "log")
    assert any(e.get("kind") == "operator_messages" for e in events(tmp_path / "log"))
    ops = [e for e in events(tmp_path / "log") if e.get("kind") == "operator_messages"]
    assert [e["messages"] for e in ops] == [["pause", "keep the caf� file"]], (
        "the undecodable byte becomes U+FFFD; the rest of the message survives"
    )
    assert script.calls == 0 and res.steps == 0 and "step" not in kinds(tmp_path / "log")


def test_read_only_run_dir_fails_loudly_and_keeps_the_inbox_unconsumed(tmp_path):
    """Behaviour record: when the run dir cannot be written the inbox rename raises PermissionError out of run();
    the message stays in inbox.txt and the journal has no pending action, so the run is resumable once fixed."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    script = Script(cmd("echo first"), cmd("done"))
    rt, _ = _rt(tmp_path, script)
    log = tmp_path / "log"

    trace = tmp_path / "second-ran.txt"  # dispatching the withheld action would create this file

    def answer(prompt):
        _inbox(tmp_path, "pause\n")
        os.chmod(log, stat.S_IRUSR | stat.S_IXUSR)
        return cmd(f"touch {trace}")

    script.queue.insert(0, answer)
    try:
        with pytest.raises(PermissionError):
            rt.run(5)
    finally:
        os.chmod(log, 0o755)
    assert (log / "inbox.txt").read_text() == "pause\n"
    info = classify_run(log)
    assert info["status"] == "interrupted" and info["pending_actions"] == 0
    assert script.calls == 1, "the failure happened after the paid response, before any further query"
    assert "step" not in kinds(log) and not any(e.get("kind") == "operator_messages" for e in events(log)), (
        "an inbox that could not be renamed was not consumed, so its message is not journalled either"
    )
    withheld = [e for e in events(log) if e.get("kind") == "action_withheld"]
    assert [e["action"] for e in withheld] == [f"touch {trace}"], "the proposed action is recorded, never dispatched"
    assert not trace.exists() and not list(tmp_path.glob("second*"))


def test_operator_notes_stay_in_every_later_prompt_even_without_a_state_field(tmp_path):
    """Standing operator notes must remain in every later prompt independently of the state schema,
    and must be rebuilt from the journal on resume."""
    from stepbook import Runtime, ScriptedExecutor, Skill
    from stepbook.envs.shell import ShellEnv

    skill = Skill("notes-example", "Run the requested shell commands.", {"type": "object"}, {})
    assert "user_requirements" not in skill.schema.get("properties", {}), "fixture assumes this skill has no such field"
    log = tmp_path / "log"
    log.mkdir()
    (log / "inbox.txt").write_text("operator: the example task is blocked by infrastructure; do not retry it\n")
    seen = []
    steps = iter([Envelope("a", {}, "echo one"), Envelope("b", {}, "echo two"), Envelope("c", {}, "done")])
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    Runtime(skill, ScriptedExecutor(lambda s, u: seen.append(u) or next(steps)), env, log_dir=log).run(4)
    assert len(seen) >= 3
    assert all("blocked by infrastructure" in u for u in seen[1:]), (
        "the note must outlive the observation it arrived in"
    )
    assert "STANDING OPERATOR NOTES" in seen[1]
