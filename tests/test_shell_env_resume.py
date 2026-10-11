"""Resuming a shell run must not re-execute journaled commands."""

import subprocess
from pathlib import Path

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv

ROOT = Path(__file__).resolve().parent.parent


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    (repo / "README").write_text("x\n")
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], cwd=repo, check=True
    )
    return repo


def _branch(repo):
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def test_resume_without_replay_does_not_rerun_commands_and_keeps_the_branch(tmp_path):
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    repo = _git_repo(tmp_path)
    log = tmp_path / "log"
    # first run: two commands that leave a visible trace each time they execute, then the process "dies" (horizon 2)
    first = iter(
        [
            Envelope("a", {"current": "one"}, "echo run >> marks.txt"),
            Envelope("b", {"current": "two"}, "echo run >> marks.txt"),
        ]
    )
    env1 = ShellEnv(require_sandbox=False, workdir=repo, goal="g", checks={})
    Runtime(skill, ScriptedExecutor(lambda s, u: next(first)), env1, log_dir=log).run(2)
    branch_after_first = _branch(repo)
    assert branch_after_first.startswith("stepbook/") and (repo / "marks.txt").read_text().count("run") == 2
    # resume: one more command, then done. Nothing from before may run again, and the branch must be the same one.
    second = iter([Envelope("c", {"current": "three"}, "echo run >> marks.txt"), Envelope("d", {}, "done")])
    env2 = ShellEnv(require_sandbox=False, workdir=repo, goal="g", checks={})
    res = Runtime(skill, ScriptedExecutor(lambda s, u: next(second)), env2, log_dir=log).run(
        10, resume=True, replay_env=False
    )
    assert (repo / "marks.txt").read_text().count("run") == 3, "resume re-executed journaled commands"
    assert _branch(repo) == branch_after_first, "resume created a new checkpoint branch instead of continuing the run's"
    assert env2.t >= 3 and len(env2.commands) >= 3, "step counter and command history must be restored from the journal"
    kinds = [
        e.get("kind")
        for e in __import__("stepbook.storage.events", fromlist=["EventLog"]).EventLog(log / "events.jsonl").read()
    ]
    assert "resumed" in kinds and res.final_state["current"] == "three"


def test_replay_env_true_would_rerun_which_is_why_shell_runs_disable_it(tmp_path):
    """Documents the contrast: with replay_env=True the journaled commands execute again."""
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    repo = _git_repo(tmp_path)
    log = tmp_path / "log"
    first = iter([Envelope("a", {"current": "one"}, "echo run >> marks.txt")])
    Runtime(
        skill,
        ScriptedExecutor(lambda s, u: next(first)),
        ShellEnv(require_sandbox=False, workdir=repo, goal="g", checks={}, checkpoint=False),
        log_dir=log,
    ).run(1)
    second = iter([Envelope("d", {}, "done")])
    Runtime(
        skill,
        ScriptedExecutor(lambda s, u: next(second)),
        ShellEnv(require_sandbox=False, workdir=repo, goal="g", checks={}, checkpoint=False),
        log_dir=log,
    ).run(5, resume=True, replay_env=True)
    assert (repo / "marks.txt").read_text().count("run") == 2
