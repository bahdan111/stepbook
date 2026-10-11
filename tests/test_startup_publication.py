"""Startup refusals must precede launch publication in both CLIs; no model calls."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook import Envelope, ScriptedExecutor
from stepbook.envs.shell import ShellEnv
from tests.probe_workflows._helpers import git, git_repo
from tests.test_launch_recovery import launch_args as product_launch_args

ROOT = Path(__file__).resolve().parents[1]


def launch_args(tmp_path, work, cli):
    """The product implies shell mode; the research harness still names its environment."""
    args = product_launch_args(tmp_path, work)
    return args + ["--env", "shell"] if cli.__name__.startswith("bench.") else args


@pytest.fixture(params=["stepbook"])
def cli_modules(request, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tuple(
        importlib.import_module(
            f"{request.param}.{name.replace('cli_', 'cli.').replace('cli.inspect', 'cli.inspection') if request.param == 'stepbook' else name}"
        )
        for name in ("cli", "cli_run")
    )


@pytest.mark.parametrize("explicit_log", [False, True])
@pytest.mark.parametrize("previous_pointers", [False, True])
def test_dirty_refusal_leaves_no_launch(cli_modules, tmp_path, monkeypatch, capsys, explicit_log, previous_pointers):
    cli, runner = cli_modules
    repo = git_repo(tmp_path)
    (repo / "README").write_text("operator edit\n")
    before = (git(repo, "rev-parse", "HEAD"), git(repo, "branch", "--show-current"), git(repo, "status", "--porcelain"))
    args = launch_args(tmp_path, repo, cli)
    if not explicit_log:
        index = args.index("--log-dir")
        del args[index : index + 2]
    pointers = tmp_path / ".stepbook"
    if previous_pointers:
        pointers.mkdir()
        for name in ("ACTIVE", "LAST_RUN"):
            (pointers / name).write_text(str(tmp_path / "previous"))
    saved = {p.name: p.read_bytes() for p in pointers.glob("*")}

    def forbidden(*args, **kwargs):
        pytest.fail("dirty refusal must precede executor selection and launch publication")

    monkeypatch.setattr(runner, "_select_run_executor", forbidden)
    monkeypatch.setattr(runner, "write_launch_pointers", forbidden)
    monkeypatch.setattr(runner, "write_launch_record", forbidden)
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Working folder is dirty: README; commit or stash first, or pass --allow-dirty.\n"
    assert not (tmp_path / "run").exists()
    assert {p.name: p.read_bytes() for p in pointers.glob("*")} == saved
    assert before == (
        git(repo, "rev-parse", "HEAD"),
        git(repo, "branch", "--show-current"),
        git(repo, "status", "--porcelain"),
    )
    assert cli.main(["status", str(tmp_path / "run")]) == 2
    assert capsys.readouterr().err == f"no run at {tmp_path / 'run'}\n"
    if not previous_pointers:
        assert cli.main(["status"]) == 2
        assert capsys.readouterr().err == "no run found; start a run or supply --run-dir\n"
        assert not pointers.exists()


@pytest.mark.parametrize(
    "case,detail",
    [
        ("workdir", "working directory does not exist"),
        ("sandbox", "ShellEnv refuses to run outside a container or VM"),
        ("executor", "was not found"),
        ("empty-command", "--command needs an executable"),
        ("bad-command", "No closing quotation"),
        ("reopen", "--reopen needs --resume"),
        ("check", "--check must be"),
        ("skill", "missing-skill"),
    ],
)
def test_other_startup_refusals_publish_nothing(cli_modules, tmp_path, monkeypatch, capsys, case, detail):
    cli, runner = cli_modules
    work = tmp_path / "work"
    work.mkdir()
    args = launch_args(tmp_path, work, cli) + ["--no-checkpoint"]
    if case == "workdir":
        work.rmdir()
    elif case == "sandbox":
        args.remove("--unsandboxed-i-accept-the-risk")
        monkeypatch.setattr(ShellEnv, "in_sandbox", staticmethod(lambda: False))
    elif case in ("executor", "empty-command", "bad-command"):
        args[args.index("--command") + 1] = {
            "executor": "./missing-agent",
            "empty-command": " ",
            "bad-command": "'",
        }[case]
    elif case == "reopen":
        if cli.__name__ == "stepbook.cli":
            pytest.skip("the product CLI infers --resume from --reopen; see tests/test_launch_recovery.py")
        args += ["--reopen", "continue"]
    elif case == "check":
        args += ["--check", "invalid"]
    elif case == "skill":
        args[1] = str(tmp_path / "missing-skill")

    def forbidden(*args, **kwargs):
        pytest.fail("startup refusal must precede launch publication and runtime construction")

    monkeypatch.setattr(runner, "write_launch_pointers", forbidden)
    monkeypatch.setattr(runner, "write_launch_record", forbidden)
    monkeypatch.setattr(runner, "_build_runtime", forbidden)
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert detail in captured.err
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / ".stepbook").exists()
    assert cli.main(["status"]) == 2
    assert capsys.readouterr().err == "no run found; start a run or supply --run-dir\n"


def test_existing_journal_refusal_preserves_artifacts(cli_modules, tmp_path, monkeypatch, capsys):
    cli, runner = cli_modules
    work = tmp_path / "work"
    work.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    journal = run / "events.jsonl"
    journal.write_text('{"kind": "paused", "step": 0}\n')
    before = journal.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("journal refusal must precede publication")

    monkeypatch.setattr(runner, "write_launch_pointers", forbidden)
    monkeypatch.setattr(runner, "write_launch_record", forbidden)
    assert cli.main(launch_args(tmp_path, work, cli) + ["--no-checkpoint"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and len(captured.err.splitlines()) == 1
    assert "already has events" in captured.err
    assert journal.read_bytes() == before
    assert sorted(p.name for p in run.iterdir()) == ["events.jsonl"]
    assert not (tmp_path / ".stepbook").exists()


def test_dirty_refusal_process_exit(cli_modules, tmp_path):
    cli, _ = cli_modules
    repo = git_repo(tmp_path)
    (repo / "dirty.txt").write_text("untracked operator work\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", cli.__name__, *launch_args(tmp_path, repo, cli)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "Working folder is dirty: dirty.txt; commit or stash first, or pass --allow-dirty.\n"
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / ".stepbook").exists()


def test_preflight_allows_dirty_resume_without_recheckpointing(cli_modules, tmp_path, monkeypatch, capsys):
    cli, runner = cli_modules
    repo = git_repo(tmp_path)
    args = launch_args(tmp_path, repo, cli)
    calls = []

    def respond(*args):
        calls.append(1)
        return Envelope("offline", {}, "echo ready")

    monkeypatch.setattr(runner, "_select_run_executor", lambda *args: (ScriptedExecutor(respond), "command"))
    assert cli.main(args) == 3
    capsys.readouterr()
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    assert branch.startswith("stepbook/")
    (repo / "README").write_text("pending run edit\n")
    assert cli.main([*args, "--resume"]) == 3
    assert "Working folder is dirty" not in capsys.readouterr().err
    assert len(calls) == 1  # An exhausted horizon must not dispatch or replay.
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "branch", "--show-current") == branch
    assert (repo / "README").read_text() == "pending run edit\n"
