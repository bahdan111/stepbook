"""Completion, reopen and resume preserve the distinction between completion and a horizon stop."""

import pytest

from stepbook import Runtime
from stepbook.run_pointer import RESUMABLE, classify_run, clear_active, describe, write_launch_pointers

from ._helpers import ROOT, Script, checklist, cmd, kinds, shell_env, steps


def _first(tmp_path, script, n, **env_kw):
    env = shell_env(tmp_path, **env_kw)
    return Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log").run(n), env


def _resume(tmp_path, script, n, reopen=None, **env_kw):
    env = shell_env(tmp_path, **env_kw)
    res = Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log").run(
        n, resume=True, replay_env=False, reopen=reopen
    )
    return res, env


def _marks(tmp_path):
    p = tmp_path / "marks.txt"
    return p.read_text().count("run") if p.exists() else 0


def test_reopen_twice_continues_each_time_and_the_second_done_closes_again(tmp_path):
    _first(tmp_path, Script(cmd("echo run >> marks.txt"), cmd("done")), 10)
    _resume(tmp_path, Script(cmd("echo run >> marks.txt"), cmd("done")), 10, reopen="first reopen")
    _resume(tmp_path, Script(cmd("echo run >> marks.txt"), cmd("done")), 10, reopen="second reopen")
    assert _marks(tmp_path) == 3 and kinds(tmp_path / "log").count("reopened") == 2
    plain = Script(exhausted=None)
    _resume(tmp_path, plain, 10)
    assert plain.calls == 0, "after the second reopen ended with a fresh done, a plain resume must run nothing"


@pytest.mark.parametrize("closing", ["done", "stop nothing left to do"])
def test_agent_done_on_the_horizon_step_is_completion_not_a_horizon_stop(tmp_path, closing):
    """recovery.execute_action marks EVERY done on t >= horizon as the guard's stop, including the agent's own."""
    _first(tmp_path, Script(cmd("echo run >> marks.txt"), cmd(closing)), 2, horizon=2)
    last = steps(tmp_path / "log")[-1]
    assert last["done"] and last["action"] == closing
    assert not last.get("horizon_stop"), "the agent closed the run; the step guard did not"
    again = Script(cmd("echo run >> marks.txt"))
    _resume(tmp_path, again, 10, horizon=10)
    assert again.calls == 0 and _marks(tmp_path) == 1, "a completed journal was resumed and paid for another model call"


def test_horizon_stop_is_classified_as_resumable(tmp_path, monkeypatch, capsys):
    """The pointer classifier and CLI must treat a horizon stop as resumable."""
    _first(tmp_path, Script(cmd("echo run >> marks.txt"), cmd("echo run >> marks.txt")), 2, horizon=2)
    log = tmp_path / "log"
    assert steps(log)[-1].get("horizon_stop") is True
    info = classify_run(log)
    assert info["status"] in RESUMABLE, f"{info['status']!r}: {describe(info)}"
    # and the CLI's pointer-based resume must not refuse it either
    from stepbook.cli import main

    write_launch_pointers(log, cwd=tmp_path)
    clear_active(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("stepbook.cli.run.ClaudeCLIExecutor", lambda **_: Script(cmd("done")).executor())
    rc = main(
        [
            "run",
            str(ROOT / "stepbook/templates" / "checklist"),
            "--workdir",
            str(tmp_path),
            "--goal",
            "g",
            "--executor",
            "cli",
            "--steps",
            "10",
            "--resume",
            "--quiet",
            "--no-checkpoint",
            "--unsandboxed-i-accept-the-risk",
        ]
    )
    assert rc == 0, capsys.readouterr().err


def test_done_refused_by_failing_checks_closes_the_journal_until_reopen(tmp_path):
    checks = {"f": "test -f never"}
    res, _ = _first(tmp_path, Script(cmd("done")), 5, checks=checks)
    assert res.score == 0.0 and steps(tmp_path / "log")[-1]["info"]["ok"] is False
    plain = Script(exhausted=None)
    _resume(tmp_path, plain, 5, checks=checks)
    assert plain.calls == 0, "a refused done still ends the episode; plain resume must not continue it"
    res2, env2 = _resume(
        tmp_path, Script(cmd("touch never"), cmd("done")), 5, reopen="fix it and finish", checks=checks
    )
    assert res2.score == 1.0 and env2.last_checks == {"f": "pass"}


def test_stop_with_reason_closes_the_journal_and_reopen_continues(tmp_path):
    _first(tmp_path, Script(cmd("stop waiting for the operator")), 5)
    plain = Script(exhausted=None)
    _resume(tmp_path, plain, 5)
    assert plain.calls == 0
    again = Script(cmd("echo run >> marks.txt"), cmd("done"))
    _resume(tmp_path, again, 5, reopen="operator is back, continue")
    assert again.calls == 2 and _marks(tmp_path) == 1 and "OPERATOR MESSAGE (run reopened" in again.prompts[0]


def test_reopen_with_a_blank_message_is_ignored_and_runs_nothing(tmp_path):
    _first(tmp_path, Script(cmd("done")), 5)
    again = Script(exhausted=None)
    _resume(tmp_path, again, 5, reopen="   ")
    assert again.calls == 0 and "reopened" not in kinds(tmp_path / "log")


def test_horizon_stop_logs_max_steps_at_once_and_an_equal_resume_appends_it_again(tmp_path):
    """The step guard's done is the horizon stop, not a completion, so the first pass writes the
    terminal record itself and classifies without a resume; an equal --steps on resume still runs nothing and
    appends the record again, which is what keeps the resumed journal classified as stopped (the classifier reads
    the last event, so a trailing resumed would read as a crash)."""
    _first(tmp_path, Script(cmd("echo run >> marks.txt"), cmd("echo run >> marks.txt")), 2, horizon=2)
    assert kinds(tmp_path / "log")[-1] == "max_steps_reached"
    assert classify_run(tmp_path / "log", active=False)["status"] == "stopped"
    again = Script(exhausted=None)
    res, _ = _resume(tmp_path, again, 2, horizon=2)
    assert again.calls == 0 and res.steps == 2 and kinds(tmp_path / "log").count("max_steps_reached") == 2
    assert kinds(tmp_path / "log")[-1] == "max_steps_reached"
    assert classify_run(tmp_path / "log", active=False)["status"] == "stopped"
