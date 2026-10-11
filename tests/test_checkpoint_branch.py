"""Checkpoint contracts exercised only in pytest's disposable git repositories."""

import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=1
    ).stdout.strip()


def frozen_clock(hour):
    strftime = time.strftime
    instant = (2026, 1, 2, hour, 4, 5, 4, 2, -1)
    return patch("time.strftime", side_effect=lambda fmt: strftime(fmt, instant))


def test_checkpoint_branch_has_exact_local_timestamp(tmp_path):
    """Bug: checkpoint creation changes the externally consumed stepbook/YYYYmmdd-HHMMSS branch name."""
    git(tmp_path, "init", "-q", "-b", "operator")
    env = ShellEnv(tmp_path, "g", require_sandbox=False)
    with frozen_clock(3):
        env.reset()
    assert git(tmp_path, "branch", "--show-current") == "stepbook/20260102-030405"
    assert git(tmp_path, "log", "-1", "--format=%s") == "stepbook: start"


def test_resume_with_deleted_branch_checkpoints_off_operator_branch(tmp_path):
    """Bug: resuming after branch deletion commits new work onto the operator's branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "operator")
    git(repo, "-c", "user.name=test", "-c", "user.email=test@local", "commit", "-qm", "operator", "--allow-empty")
    operator = git(repo, "rev-parse", "HEAD")
    skill = Skill.load(ROOT / "stepbook/templates/checklist")
    log = tmp_path / "log"
    with frozen_clock(3):
        Runtime(
            skill,
            ScriptedExecutor(lambda s, u: Envelope("first", {}, "printf first > first.txt")),
            ShellEnv(repo, "g", require_sandbox=False),
            log_dir=log,
        ).run(1)
    git(repo, "switch", "operator")
    git(repo, "branch", "-D", "stepbook/20260102-030405")
    with frozen_clock(6):
        result = Runtime(
            skill,
            ScriptedExecutor(lambda s, u: Envelope("next", {}, "printf next > next.txt")),
            ShellEnv(repo, "g", require_sandbox=False),
            log_dir=log,
        ).run(2, resume=True, replay_env=False)
    assert result.steps == 2
    assert git(repo, "branch", "--show-current") == "stepbook/20260102-060405"
    assert git(repo, "rev-parse", "operator") == operator
    assert git(repo, "show", "HEAD:next.txt") == "next"
    assert not (repo / "first.txt").exists(), "resume must not replay the deleted branch's action"
