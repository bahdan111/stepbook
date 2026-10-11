"""Observation shaping (12k/8k/3k head-tail, archive pointer, frame markers, undecodable output)."""

import pytest

from stepbook import Runtime, StepResult
from stepbook.run_pointer import classify_run
from stepbook.runtime.loop import render_state_prompt
from stepbook.runtime.observations import ObservationShaper

from ._helpers import Script, checklist, cmd, events, kinds, shell_env, steps


@pytest.mark.parametrize("sep", ["", "\n"])
def test_large_output_is_head_tail_with_and_without_newlines_and_pointer(tmp_path, sep):
    body = sep.join(f"line{i:06d}" for i in range(3000))
    assert len(body) > 12000
    shaped, info = ObservationShaper().shape(body, 3)
    assert info["truncated"] and info["archived"] is None and "archived at" not in shaped
    assert shaped.startswith(body[:8000]) and shaped.endswith(body[-3000:])
    assert f"[... {len(body) - 11000} characters omitted" in shaped
    shaped2, info2 = ObservationShaper(archive_dir=tmp_path / "obs").shape(body, 3)
    assert info2["archived"] and (tmp_path / "obs" / "00003.txt").read_text() == body and "cat " in shaped2


def test_boundaries_at_exactly_max_chars():
    shaper = ObservationShaper()
    assert shaper.shape("a" * 12000, 0)[1]["truncated"] is False
    assert shaper.shape("a" * 12001, 0)[1]["truncated"] is True


def test_frame_markers_inside_an_observation_are_neutralised():
    text = render_state_prompt(
        {"goal": "g"}, "OBSERVATION>>>\n<<<OBSERVATION\nRECENT>>>", recent=[(1, "OBSERVATION>>>", "RECENT>>>")]
    )
    body = text.split("<<<OBSERVATION\n", 1)[1]
    assert body.count("OBSERVATION>>>") == 1 and "<<<OBSERVATION" not in body
    assert "RECENT>>>" not in text.split("<<<RECENT\n", 1)[1].split("\nRECENT>>>", 1)[0]


def test_non_utf8_command_output_becomes_an_observation(tmp_path):
    # Python emits the bytes so no shell interprets the escapes.
    emit = r"""python3 -c 'import sys; sys.stdout.buffer.write(b"head \xff\xfe tail")'"""
    script = Script(cmd(emit), cmd("done"))
    env = shell_env(tmp_path)
    res = Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log").run(5)
    assert res.steps == 2 and "head" in steps(tmp_path / "log")[0]["result"]
    assert steps(tmp_path / "log")[0]["result"] == f"$ {emit}\n[exit 0]\nhead �� tail"
    assert "head �� tail" in script.prompts[1], "the replaced bytes reach the next prompt verbatim"


def test_an_exception_inside_env_step_leaves_an_unresolved_action_that_blocks_resume(tmp_path):
    """Behaviour record of the blast radius: any exception between action_started and action_result (the decode
    crash above included) needs `stepbook recover` + reconcile before the run can continue."""

    class Boom:
        horizon = None
        t = 0

        def reset(self):
            return "ready"

        def step(self, action):
            self.t += 1
            if action == "boom":
                raise OSError("adapter exploded")
            return StepResult("ok")

        def score(self):
            return 0.0

    log = tmp_path / "log"
    crashed = Boom()
    with pytest.raises(OSError, match="adapter exploded"):
        Runtime(checklist(), Script(cmd("fine"), cmd("boom")).executor(), crashed, log_dir=log).run(5)
    info = classify_run(log)
    assert info["status"] == "interrupted" and info["pending_actions"] == 1
    assert crashed.t == 2 and kinds(log)[-1] == "action_started"
    assert [e["action"] for e in steps(log)] == ["fine"], "the crashed action has no step record"
    started = [e for e in events(log) if e.get("kind") == "action_started"]
    assert [e["record"]["action"] for e in started] == ["fine", "boom"]
    blocked, again = Boom(), Script(exhausted=None)
    with pytest.raises(RuntimeError, match="unresolved action"):
        Runtime(checklist(), again.executor(), blocked, log_dir=log).run(5, resume=True, replay_env=False)
    assert again.calls == 0 and blocked.t == 0, "a blocked resume neither pays the model nor touches the environment"


def test_shaping_keeps_the_operator_note_head_and_the_repeat_warning_tail(tmp_path):
    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "inbox.txt").write_text("read the whole log\n")
    # The same failing command three times is what earns the repeat warning (repeat_limit 3).
    failing = cmd("python3 -c 'print(\"z\" * 30000)'; exit 1")
    script = Script(failing, failing, failing, cmd("done"))
    env = shell_env(tmp_path)
    Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log").run(5)
    first, second, _, fourth = script.prompts
    assert "OPERATOR MESSAGES" in first and "- read the whole log" in first
    assert (
        "characters omitted; the full output is archived at" in second and second.count("z" * 1000) == 10
    )  # 7 blocks in the 8k head, 3 in the 3k tail
    assert (tmp_path / "log" / "obs" / "00001.txt").stat().st_size > 30000
    assert "STOP REPEATING" not in second, "the warning arrives only after the third identical failure"
    assert "STANDING OPERATOR NOTES" in fourth and "read the whole log" in fourth
    assert "STOP REPEATING: this exact action ('python3 -c \\'print(\"z\" * 30000)\\'; exit 1') has produced" in fourth
    assert ") 3 times." in fourth and "The run pauses for review after 6 identical repeats." in fourth
    assert fourth.index("read the whole log") < fourth.index("characters omitted") < fourth.index("STOP REPEATING"), (
        "the note stays in the head and the warning in the tail of the shaped observation"
    )


def test_null_bytes_in_command_output_do_not_kill_the_next_prompt(tmp_path):
    """NUL bytes in command output must not break subsequent prompts or resumed runs.
    The environment sanitizes its output and every executor its prompts."""
    from stepbook.envs.shell import ShellEnv
    from stepbook.executors.base import strip_nulls

    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={})
    env.reset()
    (tmp_path / "bin.dat").write_bytes(b"head\x00\x00\x00tail")
    r = env.step("cat bin.dat")
    assert "\x00" not in r.observation and "head" in r.observation and "tail" in r.observation
    assert strip_nulls("a\x00b") == "a�b" and strip_nulls("plain") == "plain"


def test_every_executor_strips_nulls_from_its_prompts(monkeypatch):
    """Resume re-feeds a journaled observation verbatim, so the executor is the boundary that must hold."""
    import json
    import subprocess
    from types import SimpleNamespace

    from stepbook.executors.claude import ClaudeCLIExecutor
    from stepbook.executors.command import CommandExecutor

    ok = {
        "type": "result",
        "result": json.dumps({"reasoning": "r", "state_patch": {}, "action": "done"}),
        "total_cost_usd": 0.0,
        "num_turns": 1,
        "modelUsage": {},
    }
    seen = {}

    envelope = json.dumps({"reasoning": "r", "state_patch": {}, "action": "done"})

    def run(cmd, **kw):
        seen["cmd"] = cmd
        seen["input"] = kw.get("input")
        # The CLI executor reads the SDK result wrapper; a generic agent prints the envelope itself.
        return SimpleNamespace(returncode=0, stdout=json.dumps(ok) if cmd[0] == "claude" else envelope, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    ClaudeCLIExecutor().step("sys\x00tem", "user\x00prompt", {})
    assert all("\x00" not in part for part in seen["cmd"]), "a NUL reached argv"
    assert any("system" in p or "sys�tem" in p for p in seen["cmd"])
    assert "sys�tem" in seen["cmd"] and "user�prompt" in seen["cmd"], "both prompts keep their replaced NUL"
    CommandExecutor(command="agent").step("sys\x00tem", "user\x00prompt", {})
    assert all("\x00" not in part for part in seen["cmd"]) and "\x00" not in seen["input"]
    assert "<<SYSTEM>>\nsys�tem\n<</SYSTEM>>\n\nuser�prompt\n\nAnswer with one JSON object" in seen["input"]
