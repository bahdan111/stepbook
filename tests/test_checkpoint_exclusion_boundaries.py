"""Repository-wide checkpoints exclude only the two explicitly owned directories."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook.envs.shell import ShellEnv
from tests.test_shell_env_resume import _git_repo

ROOT = Path(__file__).resolve().parent.parent


def git(repo, *args):
    assert repo.resolve() != ROOT
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def commit(repo):
    git(repo, "add", "-A")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "fixture")


@pytest.mark.parametrize("layout", ["file", "symlink", "root"])
@pytest.mark.parametrize("operation", ["preflight", "checkpoint"])
def test_user_launch_json_never_hides_source(tmp_path, monkeypatch, layout, operation):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    folder = repo if layout == "root" else repo / "examples"
    folder.mkdir(exist_ok=True)
    record = dict(argv=["run"], program="user-tool", cwd=".", started_at=0, resume_command="user-tool run")
    target = tmp_path / "user-launch.json" if layout == "symlink" else folder / "launch.json"
    target.write_text(json.dumps(record))
    if layout == "symlink":
        (folder / "launch.json").symlink_to(target)
    source = folder / "source.py"
    source.write_text("old\n")
    commit(repo)
    env = ShellEnv(repo, "g", log_dir=tmp_path / "run", require_sandbox=False)
    env.reset()
    source.write_text("new\n")
    if operation == "preflight":
        with pytest.raises(ValueError, match="source.py.*commit or stash"):
            ShellEnv(repo, "g", log_dir=tmp_path / "run", require_sandbox=False).preflight()
    else:
        assert env._checkpoint("edit")
        assert git(repo, "show", f"HEAD:{source.relative_to(repo)}") == "new"


def test_exclusions_do_not_read_files_or_last_run(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "launch.json").write_text("{}")
    (repo / ".stepbook").mkdir()
    (repo / ".stepbook/LAST_RUN").write_text(str(repo / "user"))
    (repo / "user").mkdir()
    (repo / "user/source.py").write_text("uncommitted")

    def no_read(path, *args, **kwargs):
        raise AssertionError(f"ownership opened {path}")

    monkeypatch.setattr(Path, "open", no_read)
    with pytest.raises(ValueError, match="user/source.py"):
        ShellEnv(repo, "g", require_sandbox=False).preflight()


def test_fifo_launch_symlink_does_not_block(tmp_path):
    repo = _git_repo(tmp_path)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    (repo / "launch.json").symlink_to(fifo)
    commit(repo)
    child = (
        "from pathlib import Path; from stepbook.envs.shell import ShellEnv; import sys; "
        "env = ShellEnv(Path(sys.argv[1]), 'g', require_sandbox=False); env.reset(); "
        "assert env.branch; print('returned')"
    )
    result = subprocess.run(
        [sys.executable, "-c", child, str(repo)], cwd=ROOT, capture_output=True, text=True, timeout=3
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "returned"


@pytest.mark.parametrize("operation", ["preflight", "checkpoint"])
def test_subdir_uses_repository_scope_and_sibling_exclusions(tmp_path, monkeypatch, operation):
    repo = _git_repo(tmp_path)
    work = repo / "work"
    work.mkdir()
    (work / "child.txt").write_text("old")
    commit(repo)
    monkeypatch.chdir(repo)
    run = repo / "logs[1]/run"
    run.mkdir(parents=True)
    (run / "events.jsonl").write_text("artifact")
    (repo / ".stepbook").mkdir()
    (repo / ".stepbook/LAST_RUN").write_text(str(run))
    env = ShellEnv(work, "g", log_dir=run, require_sandbox=False)
    env.reset()
    (repo / "README").write_text("new")
    if operation == "preflight":
        with pytest.raises(ValueError, match="README.*commit or stash"):
            ShellEnv(work, "g", log_dir=run, require_sandbox=False).preflight()
    else:
        (work / "child.txt").write_text("new")
        assert env._checkpoint("edit")
        assert git(repo, "show", "HEAD:README") == "new"
        assert git(repo, "show", "HEAD:work/child.txt") == "new"
        assert git(repo, "ls-files").splitlines() == ["README", "work/child.txt"]


@pytest.mark.parametrize("initial", ["tracked", "untracked", "unborn"])
def test_staged_artifacts_stay_out_of_start_and_step_commits(tmp_path, monkeypatch, initial):
    repo = _git_repo(tmp_path) if initial != "unborn" else tmp_path
    if initial == "unborn":
        git(repo, "init", "-q", ".")
    monkeypatch.chdir(repo)
    artifacts = [repo / ".stepbook/LAST_RUN", repo / "logs/run/events.jsonl"]
    for artifact in artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("old\n")
    if initial == "tracked":
        commit(repo)
    for artifact in artifacts:
        artifact.write_text("staged harness content\n")
    git(repo, "add", "-A")
    env = ShellEnv(repo, "g", log_dir=artifacts[1].parent, require_sandbox=False)
    env.reset()
    assert env.branch
    for phase in ("start", "step"):
        if phase == "step":
            (repo / "README").write_text("new\n")
            assert env._checkpoint("edit")
            assert git(repo, "show", "HEAD:README") == "new"
        files = git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        for artifact in artifacts:
            relative = artifact.relative_to(repo).as_posix()
            if initial == "tracked":
                assert git(repo, "show", f"HEAD:{relative}") == "old"
            else:
                assert relative not in files
            assert git(repo, "show", f":{relative}") == "staged harness content"


def test_dirty_refusal_names_three_paths_and_remainder(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    for name in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"):
        (repo / name).write_text("user change")
    with pytest.raises(ValueError) as error:
        ShellEnv(repo, "g", require_sandbox=False).preflight()
    assert str(error.value) == (
        "Working folder is dirty: a.txt, b.txt, c.txt and 2 more; commit or stash first, or pass --allow-dirty."
    )
