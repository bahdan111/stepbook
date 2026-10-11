"""Offline launch and recovery regressions; Git mutations are confined to fixture repos."""

import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from stepbook.cli import main
from stepbook.cli.run import _preflight_executor
from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.envs.shell import ShellEnv
from stepbook.executors.base import Envelope
from stepbook.executors.command import CommandExecutor, ScriptedExecutor
from stepbook.run_pointer import classify_run
from stepbook.runtime.controls import read_inbox
from stepbook.state.skill import skill_content_hash
from stepbook.storage.events import EventLog
from tests.test_shell_env_resume import _git_repo

ROOT = Path(__file__).resolve().parent.parent


def launch_args(tmp_path, work):
    return [
        "run",
        str(ROOT / "stepbook/templates/checklist"),
        "--workdir",
        str(work),
        "--goal",
        "test",
        "--steps",
        "1",
        "--executor",
        "command",
        "--command",
        f"{sys.executable} -c 'print(\"unused\")'",
        "--log-dir",
        str(tmp_path / "run"),
        "--unsandboxed-i-accept-the-risk",
        "--quiet",
    ]


@pytest.mark.parametrize("after_write", [False, True])
def test_claim_survives_crash_without_duplicate_delivery(tmp_path, monkeypatch, after_write):
    log = EventLog(tmp_path / "events.jsonl")
    (tmp_path / "inbox.txt").write_text("keep this\n")
    append = log.append

    def crash(kind, **data):
        assert not (tmp_path / "inbox.txt").exists()
        if after_write:
            append(kind, **data)
        raise RuntimeError("crash after claim")

    monkeypatch.setattr(log, "append", crash)
    with pytest.raises(RuntimeError, match="crash after claim"):
        read_inbox(log, 0)
    resumed = EventLog(log.path)
    assert read_inbox(resumed, 0) == ([] if after_write else ["keep this"])
    assert read_inbox(resumed, 1) == []
    assert [e["messages"] for e in resumed.read()] == [["keep this"]]


def test_dirty_shell_refuses_before_checkpoint(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "README").write_text("operator edit\n")
    before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
    with pytest.raises(ValueError, match="commit or stash"):
        ShellEnv(repo, "test", require_sandbox=False).reset()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo) == before
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip()


def test_missing_command_does_not_create_phantom_run(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = launch_args(tmp_path, repo)
    args[args.index("--command") + 1] = "./missing-agent"
    before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
    assert main(args) == 2
    assert "missing-agent" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / ".stepbook/LAST_RUN").exists()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo) == before


def test_failure_after_creation_is_not_resumable(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fail(self):
        raise OSError("injected reset failure")

    monkeypatch.setattr(ShellEnv, "reset", fail)
    assert main(launch_args(tmp_path, repo)) == 2
    assert "injected reset failure" in capsys.readouterr().err
    info = classify_run(tmp_path / "run")
    assert info["status"] == "failed_to_start"
    assert info["resume_command"] is None
    assert not (tmp_path / ".stepbook/ACTIVE").exists()
    assert not (tmp_path / ".stepbook/LAST_RUN").exists()


def test_changed_skill_resume_warns_and_records_hash(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "skill"
    shutil.copytree(ROOT / "stepbook/templates/checklist", skill)
    args = launch_args(tmp_path, repo)
    args[1] = str(skill)
    run = tmp_path / "run"
    run.mkdir()
    (run / "inbox.txt").write_text("pause\n")
    assert main(args) == 3
    launch = json.loads((run / "launch.json").read_text())
    original_hash = launch.get("skill_hash")
    capsys.readouterr()
    with (skill / "SKILL.md").open("a") as fh:
        fh.write("\nUse the changed instructions.\n")
    prompts = []

    def current_skill(system, user):
        prompts.append(system)
        return Envelope("finish", {}, "done")

    monkeypatch.setattr(
        "stepbook.cli.run._select_run_executor",
        lambda *args: (ScriptedExecutor(current_skill), "command"),
    )
    assert main(args + ["--resume"]) == 0
    stderr = capsys.readouterr().err
    assert len(stderr.splitlines()) == 1 and "skill changed" in stderr.lower()
    assert original_hash
    assert "Use the changed instructions." in prompts[0]
    record = json.loads((run / "launch.json").read_text())
    assert record["skill_hash"] == original_hash
    assert record["resumes"][-1]["skill_hash_changed"] == {
        "from": original_hash,
        "to": record["resumes"][-1]["skill_hash"],
    }


@pytest.mark.parametrize("command", ["say", "pause"])
def test_control_commands_queue_existing_inbox(tmp_path, monkeypatch, capsys, command):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    EventLog(run / "events.jsonl").append("paused", step=0)
    (run / "inbox.txt").write_text("earlier\n")
    argv = [command, *(["keep the tests"] if command == "say" else []), "--log-dir", str(run)]
    assert main(argv) == 0
    message = "keep the tests" if command == "say" else "pause"
    assert (run / "inbox.txt").read_text() == f"earlier\n{message}\n"
    assert str(run) in capsys.readouterr().out
    assert read_inbox(EventLog(run / "events.jsonl"), 0) == ["earlier", message]


def test_dirty_cli_opt_in_and_resume(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (repo / "README").write_text("operator edit\n")
    args = launch_args(tmp_path, repo)
    assert main(args) == 2
    assert "commit or stash" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    run = tmp_path / "run"
    run.mkdir()
    (run / "inbox.txt").write_text("pause\n")
    assert main(args + ["--allow-dirty"]) == 3
    assert subprocess.check_output(["git", "show", "HEAD:README"], cwd=repo) == b"operator edit\n"
    (repo / "README").write_text("run edit\n")
    (run / "inbox.txt").write_text("pause\n")
    assert main(args + ["--resume"]) == 3
    assert (repo / "README").read_text() == "run edit\n"


@pytest.mark.parametrize("pointer", ["ACTIVE", "LAST_RUN", None])
def test_controls_resolve_pointers_or_refuse(tmp_path, monkeypatch, capsys, pointer):
    monkeypatch.chdir(tmp_path)
    if pointer:
        run = tmp_path / "run"
        EventLog(run / "events.jsonl").append("paused", step=0)
        base = tmp_path / ".stepbook"
        base.mkdir()
        (base / pointer).write_text(str(run))
        assert main(["pause"]) == 0
        assert (run / "inbox.txt").read_text() == "pause\n"
    else:
        assert main(["pause"]) == 2
        assert len(capsys.readouterr().err.splitlines()) == 1
        assert not (tmp_path / ".stepbook").exists()


@pytest.mark.parametrize("name", ["SKILL.md", "schema.json", "initial_state.json"])
def test_hash_tracks_loaded_files_only(tmp_path, name):
    skill = tmp_path / "skill"
    shutil.copytree(ROOT / "stepbook/templates/checklist", skill)
    before = skill_content_hash(skill)
    (skill / "unused.txt").write_text("irrelevant")
    assert skill_content_hash(skill) == before
    path = skill / name
    path.write_text(path.read_text() + "\n")
    assert skill_content_hash(skill) != before
    if name == "initial_state.json":
        path.unlink()
        missing = skill_content_hash(skill)
        path.write_text("")
        assert skill_content_hash(skill) != missing


def test_legacy_resume_has_no_hash_warning(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = launch_args(tmp_path, repo)
    run = tmp_path / "run"
    run.mkdir()
    (run / "inbox.txt").write_text("pause\n")
    assert main(args) == 3
    path = run / "launch.json"
    record = json.loads(path.read_text())
    del record["skill_hash"]
    path.write_text(json.dumps(record))
    (run / "inbox.txt").write_text("pause\n")
    capsys.readouterr()
    assert main(args + ["--resume"]) == 3
    assert capsys.readouterr().err == ""
    record = json.loads(path.read_text())
    assert "skill_hash" not in record
    assert set(record["resumes"][-1]) == {"resumed_at", "argv"}


def test_legacy_consumed_inbox_is_not_redelivered(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    (tmp_path / "inbox.legacy.consumed").write_text("old message\npause\n")
    (tmp_path / "inbox.txt").write_text("new message\n")
    assert read_inbox(log, 0) == ["new message"]
    assert read_inbox(log, 1) == []


def test_missing_cli_binary_preflight_preserves_prior_pointer(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = launch_args(tmp_path, repo)
    index = args.index("--executor")
    args[index + 1] = "cli"
    del args[index + 2 : index + 4]
    base = tmp_path / ".stepbook"
    base.mkdir()
    (base / "LAST_RUN").write_text("previous")
    monkeypatch.setattr("stepbook.cli.run.shutil.which", lambda *a, **kw: None)
    assert main(args) == 2
    assert "claude" in capsys.readouterr().err
    assert (base / "LAST_RUN").read_text() == "previous"
    assert not (tmp_path / "run").exists()


def test_bad_command_after_preflight_is_failed_to_start(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = launch_args(tmp_path, repo)
    args[args.index("--command") + 1] = shlex.join([sys.executable, "missing-agent.py"])
    monkeypatch.setattr("stepbook.runtime.retry.time.sleep", lambda _: None)
    assert main(args) == 2
    assert "missing-agent.py" in capsys.readouterr().err
    info = classify_run(tmp_path / "run")
    assert info["status"] == "failed_to_start" and info["resume_command"] is None
    assert main(args + ["--resume"]) == 2
    assert "failed to start" in capsys.readouterr().err
    assert main(["status", "--log-dir", str(tmp_path / "run")]) == 0
    assert "resume:" not in capsys.readouterr().out


def test_controls_precedence_and_unterminated_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".stepbook"
    base.mkdir()
    for name in ("ACTIVE", "LAST_RUN", "explicit"):
        run = tmp_path / name
        EventLog(run / "events.jsonl").append("paused", step=0)
        (base / name).write_text(str(run))
    (tmp_path / "ACTIVE/inbox.txt").write_text("earlier")
    assert main(["say", "new"]) == 0
    assert (tmp_path / "ACTIVE/inbox.txt").read_text() == "earlier\nnew\n"
    assert main(["pause", "--log-dir", str(tmp_path / "explicit")]) == 0
    assert (tmp_path / "explicit/inbox.txt").read_text() == "pause\n"
    assert not (tmp_path / "LAST_RUN/inbox.txt").exists()


def test_failure_with_started_action_keeps_recovery(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "stepbook.cli.run._select_run_executor",
        lambda *args: (ScriptedExecutor(lambda s, u: Envelope("act", {}, "touch marker")), "command"),
    )

    def uncertain_action(self, action):
        (self.workdir / "marker").touch()
        raise OSError("lost response")

    monkeypatch.setattr(ShellEnv, "step", uncertain_action)
    assert main(launch_args(tmp_path, repo)) == 2
    info = classify_run(tmp_path / "run")
    assert info["status"] == "interrupted" and info["pending_actions"] == 1
    assert info["resume_command"]
    assert (tmp_path / ".stepbook/LAST_RUN").exists()
    assert not (tmp_path / ".stepbook/ACTIVE").exists()


def test_claimed_note_and_pause_survive_runtime_restart(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = launch_args(tmp_path, repo)
    log = EventLog(tmp_path / "run/events.jsonl")
    (log.path.parent / "inbox.txt").write_text("retain this requirement\npause\n")

    def crash(*args, **kwargs):
        raise RuntimeError("crashed after claiming")

    monkeypatch.setattr(log, "append", crash)
    with pytest.raises(RuntimeError):
        read_inbox(log, 0)
    assert main(args) == 3
    assert classify_run(log.path.parent)["status"] == "paused"
    state = json.loads((log.path.parent / "state.json").read_text())
    assert {"text": "retain this requirement", "done": False} in state["user_requirements"]
    monkeypatch.setattr(
        "stepbook.cli.run._select_run_executor",
        lambda *args: (ScriptedExecutor(lambda s, u: Envelope("finish", {}, "done")), "command"),
    )
    assert main(args + ["--resume"]) == 0
    events = EventLog(log.path).read()
    assert sum(e["kind"] == "operator_messages" for e in events) == 1


def test_reopen_implies_resume_and_still_needs_a_run_to_continue(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = launch_args(tmp_path, repo)
    del args[args.index("--log-dir") : args.index("--log-dir") + 2]
    # --reopen continues an existing journal, so it no longer needs --resume spelled out; without a
    # directory or a pointer there is nothing to continue, and the launch is refused.
    assert main(args + ["--reopen", "continue"]) == 2
    assert "--resume needs --log-dir pointing at the interrupted run" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / ".stepbook").exists()


def scripted_done(monkeypatch):
    monkeypatch.setattr(
        "stepbook.cli.run._select_run_executor",
        lambda *args: (ScriptedExecutor(lambda s, u: Envelope("finish", {}, "done")), "command"),
    )
    monkeypatch.setattr("stepbook.cli.run.write_report_md", lambda *args: "suppressed")


def test_fresh_refusal_preserves_paused_run(tmp_path, monkeypatch, capsys):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)
    scripted_done(monkeypatch)
    args = launch_args(tmp_path, repo) + ["--no-checkpoint"]
    run = tmp_path / "run"
    run.mkdir()
    (run / "inbox.txt").write_text("pause\n")
    assert main(args) == 3
    before = {p: p.read_bytes() for p in (run / "launch.json", run / "events.jsonl", repo / ".stepbook/LAST_RUN")}
    # Even a live pointer must not be removed by a refused fresh launch.
    active = repo / ".stepbook/ACTIVE"
    active.write_text(str(run))
    before[active] = active.read_bytes()
    assert main(args) == 2
    assert "already has events" in capsys.readouterr().err
    assert classify_run(run)["status"] == "paused"
    assert all(p.read_bytes() == value for p, value in before.items())
    assert main(args + ["--resume"]) == 0
    assert classify_run(run)["status"] == "completed"


def test_failure_in_preexisting_empty_directory_is_not_marked(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    run.mkdir()

    def fail(self):
        raise OSError("reset failure")

    monkeypatch.setattr(ShellEnv, "reset", fail)
    assert main(launch_args(tmp_path, repo)) == 2
    assert "failed_to_start" not in json.loads((run / "launch.json").read_text())
    assert (tmp_path / ".stepbook/LAST_RUN").exists()


def interleaved_control(monkeypatch, inbox, argv, consume):
    """Drain after append-open; a contended real flock releases the writer to finish."""
    boundary = threading.Event()
    original_open, original_flock = Path.open, fcntl.flock
    futures = []

    def drain():
        try:
            return consume()
        finally:
            boundary.set()

    def flock(fd, operation):
        if operation == fcntl.LOCK_EX:
            try:
                return original_flock(fd, operation | fcntl.LOCK_NB)
            except BlockingIOError:
                boundary.set()
        return original_flock(fd, operation)

    with ThreadPoolExecutor(max_workers=1) as pool:

        def opened(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path == inbox and args and args[0] == "a+":
                futures.append(pool.submit(drain))
                assert boundary.wait(5), "consumer did not reach the interleaving boundary"
            return stream

        monkeypatch.setattr(fcntl, "flock", flock)
        monkeypatch.setattr(Path, "open", opened)
        assert main(argv) == 0
        return futures[0].result(timeout=5)


@pytest.mark.parametrize("command", ["say", "pause"])
@pytest.mark.parametrize("consumer", ["runtime", "adapter"])
def test_control_append_survives_concurrent_claim(tmp_path, monkeypatch, command, consumer):
    log = EventLog(tmp_path / "events.jsonl")
    inbox = tmp_path / "inbox.txt"
    inbox.write_text("earlier\n" + ("pause\n" if consumer == "adapter" else ""))
    message = "critical new instruction" if command == "say" else "pause"
    argv = [command, *([message] if command == "say" else []), "--log-dir", str(tmp_path)]
    adapter = AdapterEvalEnv(adapter="fake", log_dir=tmp_path, inbox_poll_s=0.01)
    process = Mock()
    process.communicate.side_effect = [subprocess.TimeoutExpired("fake", 0.01), ("ok", "")]
    consume = (
        (lambda: read_inbox(log, 0)) if consumer == "runtime" else lambda: adapter._communicate_with_inbox_poll(process)
    )
    interleaved_control(monkeypatch, inbox, argv, consume)
    read_inbox(log, 1)
    messages = [m for event in log.read() for m in event.get("messages", [])]
    assert messages == (["earlier"] if consumer == "adapter" and command == "pause" else ["earlier", message])
    if consumer == "adapter":
        assert adapter.pause_requested
    assert read_inbox(log, 2) == []


@pytest.mark.parametrize("layout", [".stepbook", "logs"])
def test_two_launches_exclude_only_harness_artifacts(tmp_path, monkeypatch, capsys, layout):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)
    scripted_done(monkeypatch)
    before_exclude = (repo / ".git/info/exclude").read_bytes()
    for name in ("run", "run2"):
        args = launch_args(repo / layout, repo)
        args[args.index("--log-dir") + 1] = str(repo / layout / name)
        expected = 2 if layout == "logs" and name == "run2" else 0
        assert main(args) == expected
        if expected == 2:
            error = capsys.readouterr().err
            assert "logs/run/events.jsonl" in error
            assert "logs/run/inbox.lock" in error
            assert "logs/run/launch.json" in error
            assert "and 3 more" in error  # additive launcher.json accompanies the existing run artifacts
            assert error.endswith("commit or stash first, or pass --allow-dirty.\n")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    assert tracked == ["README"]
    assert not (repo / ".gitignore").exists()
    assert (repo / ".git/info/exclude").read_bytes() == before_exclude
    (repo / "logs").mkdir(exist_ok=True)
    (repo / "logs/user.txt").write_text("operator work")
    args[args.index("--log-dir") + 1] = str(repo / layout / "run4")
    assert main(args) == 2


@pytest.mark.parametrize("tracked", [False, True])
def test_checkpoint_excludes_harness_artifacts(tmp_path, monkeypatch, tracked):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)
    run = repo / "logs/run"
    run.mkdir(parents=True)
    (repo / ".stepbook").mkdir()
    (repo / ".stepbook/LAST_RUN").write_text(str(run))
    artifact = run / "events.jsonl"
    artifact.write_text("old\n")
    if tracked:
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "old harness"], cwd=repo, check=True
        )
    artifact.write_text("new\n")
    env = ShellEnv(repo, "g", require_sandbox=False, log_dir=run)
    env.reset()
    (repo / "README").write_text("intentional change\n")
    env._checkpoint("change")
    assert subprocess.check_output(["git", "show", "HEAD:README"], cwd=repo) == b"intentional change\n"
    files = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    if tracked:
        assert subprocess.check_output(["git", "show", "HEAD:logs/run/events.jsonl"], cwd=repo) == b"old\n"
    else:
        assert files == ["README"]


@pytest.mark.parametrize("entry", ["bin", ""])
def test_preflight_resolves_path_in_executor_directory(tmp_path, monkeypatch, entry):
    work = tmp_path / "work"
    binary_dir = work / entry
    binary_dir.mkdir(parents=True)
    binary = binary_dir / "a3-offline-agent"
    binary.write_text(
        f"#!{sys.executable}\nimport json\nprint(json.dumps(dict(reasoning='ok',state_patch={{}},action='done')))\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", entry + os.pathsep + os.environ["PATH"])
    ex = CommandExecutor(binary.name, cwd=work)
    _preflight_executor(ex)
    assert ex.step("s", "u", {}).action == "done"


def test_post_step_exception_propagates_and_cleans_active(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(repo)
    count = 0

    def step(s, u):
        nonlocal count
        count += 1
        if count == 1:
            return Envelope("one", {}, "status")
        raise ValueError("injected after committed step 0")

    monkeypatch.setattr("stepbook.cli.run._select_run_executor", lambda *a: (ScriptedExecutor(step), "command"))
    args = launch_args(tmp_path, repo) + ["--no-checkpoint"]
    args[args.index("--steps") + 1] = "2"
    with pytest.raises(ValueError, match="injected after committed step 0"):
        main(args)
    run = tmp_path / "run"
    assert classify_run(run)["steps"] == 1
    assert classify_run(run)["status"] == "interrupted"
    assert "failed_to_start" not in json.loads((run / "launch.json").read_text())
    assert (repo / ".stepbook/LAST_RUN").exists()
    assert not (repo / ".stepbook/ACTIVE").exists()


def test_unborn_repo_with_untracked_files_still_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    (tmp_path / "user.txt").write_text("uncommitted")
    with pytest.raises(ValueError, match="commit or stash"):
        ShellEnv(tmp_path, "g", require_sandbox=False).preflight()


def test_whitespace_journal_allows_fresh_launch(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    scripted_done(monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text("\n")
    assert main(launch_args(tmp_path, repo)) == 0
