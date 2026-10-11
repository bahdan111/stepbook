"""Kills at every journal boundary, and what resume rebuilds from the journal."""

import uuid

import pytest

from stepbook import Runtime
from stepbook.run_pointer import classify_run
from stepbook.storage.events import EventLog

from ._helpers import Script, checklist, cmd, events, git, git_repo, rewrite_events, shell_env, steps


def _run(workdir, script, n, log, resume=False, **env_kw):
    env = shell_env(workdir, **env_kw)
    res = Runtime(checklist(), script.executor(), env, log_dir=log).run(n, resume=resume, replay_env=False)
    return res, env


def test_action_started_without_a_receipt_refuses_resume_and_is_visible_in_status(tmp_path):
    log = tmp_path / "log"
    _run(tmp_path, Script(cmd("echo one >> marks.txt"), cmd("echo two >> marks.txt")), 2, log)
    EventLog(log / "events.jsonl").append(
        "action_started", action_id=uuid.uuid4().hex, record={"step": 2, "action": "echo three >> marks.txt"}
    )
    info = classify_run(log)
    assert info["status"] == "interrupted" and info["pending_actions"] == 1
    again = Script(exhausted=None)
    with pytest.raises(RuntimeError, match="unresolved action"):
        _run(tmp_path, again, 10, log, resume=True)
    assert again.calls == 0 and (tmp_path / "marks.txt").read_text().count("\n") == 2


def test_receipt_without_step_on_the_horizon_step_still_resumes_with_a_higher_horizon(tmp_path):
    log = tmp_path / "log"
    _run(tmp_path, Script(cmd("echo run >> marks.txt"), cmd("echo run >> marks.txt")), 2, log, horizon=2)
    last = steps(log)[-1]
    assert last.get("horizon_stop") is True
    # the process died after the receipt was journaled but before the step record (the last line)
    rewrite_events(log, lambda e: not (e.get("kind") == "step" and e.get("step") == 1))
    again = Script(cmd("echo run >> marks.txt"), cmd("done"))
    _run(tmp_path, again, 10, log, resume=True, horizon=10)
    recovered = steps(log)[-1] if again.calls == 0 else steps(log)[1]
    assert recovered.get("recovered_from_receipt") is True
    assert again.calls == 2 and (tmp_path / "marks.txt").read_text().count("run") == 3


def test_model_call_without_a_step_is_paid_once_and_the_step_number_is_reused(tmp_path):
    log = tmp_path / "log"
    _run(tmp_path, Script(cmd("echo one")), 1, log)
    EventLog(log / "events.jsonl").append(
        "model_call",
        step=1,
        attempt=0,
        input_tokens=500,
        output_tokens=50,
        cost_usd=0.01,
        api_calls=1,
        usage_unknown=False,
    )
    again = Script(cmd("echo two"), cmd("done"))
    res, env = _run(tmp_path, again, 10, log, resume=True)
    numbers = [e["step"] for e in steps(log)]
    assert numbers == [0, 1, 2] and len(set(numbers)) == 3
    assert res.llm_calls == 4 and res.cost_usd == pytest.approx(0.01), "the orphaned paid call is counted exactly once"
    assert env.t == 3


def test_duplicated_step_number_in_the_journal_shifts_the_counter_but_does_not_crash(tmp_path):
    """Behaviour record: restore_from_events counts records, not distinct step numbers."""
    log = tmp_path / "log"
    _run(tmp_path, Script(cmd("echo one")), 1, log)
    path = log / "events.jsonl"
    step_line = next(line for line in path.read_text().splitlines() if '"kind": "step"' in line)
    path.write_text(path.read_text() + step_line + "\n")
    again = Script(cmd("done"))
    res, env = _run(tmp_path, again, 10, log, resume=True)
    assert env.t == 3 and [e["step"] for e in steps(log)] == [0, 0, 2]


def test_resume_after_the_checkpoint_branch_was_deleted_does_not_commit_to_the_base_branch(tmp_path):
    repo = git_repo(tmp_path)
    log = tmp_path / "log"
    base = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _run(repo, Script(cmd("echo run >> marks.txt")), 1, log, checkpoint=True)
    ss_branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert ss_branch.startswith("stepbook/")
    git(repo, "checkout", "-q", base)
    git(repo, "branch", "-D", ss_branch)
    base_head = git(repo, "rev-parse", "HEAD")
    again = Script(cmd("echo run >> marks.txt"), cmd("done"))
    _, env = _run(repo, again, 10, log, resume=True, checkpoint=True)
    assert git(repo, "rev-parse", base) == base_head, f"resume committed onto {base}"
    assert env.branch and env.branch.startswith("stepbook/"), f"resumed on {env.branch!r}"


def test_resume_restores_step_counter_command_history_and_last_checks(tmp_path):
    log = tmp_path / "log"
    checks = {"f": "test -f flag"}
    _run(tmp_path, Script(cmd("echo one"), cmd("check"), cmd("touch flag")), 3, log, checks=checks)
    # Before the resumed `done` reruns the checks, the restored value must be the journalled failure.
    restored = shell_env(tmp_path, checks=checks)
    restored.restore_from_events(events(log))
    assert restored.t == 3 and restored.last_checks == {"f": "fail"}
    assert [(c["cmd"], c["rc"]) for c in restored.commands] == [("echo one", 0), ("touch flag", 0)]
    again = Script(cmd("done"))
    res, env = _run(tmp_path, again, 10, log, resume=True, checks=checks)
    assert env.t == 4 and [c["cmd"] for c in env.commands][:2] == ["echo one", "touch flag"]
    assert env.last_checks["f"] == "pass" and res.score == 1.0


def test_torn_final_journal_line_is_preserved_and_resume_continues(tmp_path):
    log = tmp_path / "log"
    _run(tmp_path, Script(cmd("echo one")), 1, log)
    torn_line = b'{"ts": 1, "kind": "step", "step": 1, "action": "echo tw'
    with (log / "events.jsonl").open("ab") as fh:
        fh.write(torn_line)
    again = Script(cmd("echo two"), cmd("done"))
    res, env = _run(tmp_path, again, 10, log, resume=True)
    assert res.steps == 3 and env.t == 3
    saved = list(log.glob("events.jsonl.torn-*"))
    assert saved, "the torn tail must be kept for repair, not dropped"
    assert [path.read_bytes() for path in saved] == [torn_line], "the exact torn bytes, nothing added or lost"
    assert torn_line not in (log / "events.jsonl").read_bytes(), "the unreadable tail is cut from the journal"
    assert [e["step"] for e in steps(log)] == [0, 1, 2] and [e["action"] for e in steps(log)][1:] == [
        "echo two",
        "done",
    ]
