"""Paused-run discovery from a fresh chat: LAST_RUN pointer, status, hooks and --resume without --log-dir.

Offline: a fake agent through CommandExecutor and the shell environment; no model calls."""

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from stepbook.run_pointer import classify_run, in_sandbox, resume_argv
from tests.test_cli_end_to_end import FAKE_AGENT

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "hooks"

# On its second call (checklist item in progress) the agent also drops a pause request into the
# run's inbox, as a human would through '@loop pause' while the controller is thinking.
PAUSING_AGENT = FAKE_AGENT.replace(
    'elif state["checklist"]["c1"]["status"] == "doing":\n',
    'elif state["checklist"]["c1"]["status"] == "doing":\n'
    "    import os\n"
    '    if os.environ.get("STEPBOOK_TEST_INBOX"):\n'
    '        open(os.environ["STEPBOOK_TEST_INBOX"], "a").write("pause\\n")\n',
)
assert PAUSING_AGENT != FAKE_AGENT


def _hook(script: str, payload: dict, hooks: Path = HOOKS) -> str:
    return subprocess.run(
        [sys.executable, str(hooks / script)], input=json.dumps(payload), capture_output=True, text=True, check=True
    ).stdout


def _launch(tmp_path: Path, agent_source: str):
    agent = tmp_path / "agent.py"
    agent.write_text(agent_source)
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    log = tmp_path / ".stepbook" / "log"
    argv = [
        "run",
        str(ROOT / "stepbook/templates" / "checklist"),
        "--workdir",
        str(work),
        "--goal",
        "create out.txt containing hello",
        "--check",
        "f=grep -q hello out.txt",
        "--steps",
        "6",
        "--executor",
        "command",
        "--command",
        f"{sys.executable} {agent}",
        "--quiet",
        "--no-checkpoint",
        "--unsandboxed-i-accept-the-risk",
    ]
    return argv, log, work


def _without_age(text: str) -> str:
    """Two status calls straddling a second boundary differ only in the "Ns ago" token."""
    return re.sub(r"\b\d+s ago\b", "<age>s ago", text)


def _cli(argv, cwd: Path, env=None):
    return subprocess.run(
        [sys.executable, "-m", "stepbook.cli", *argv],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
        env={**os.environ, **(env or {})},
    )


def test_pause_leaves_a_durable_pointer_and_resume_finds_it_without_log_dir(tmp_path):
    argv, log, work = _launch(tmp_path, PAUSING_AGENT)
    log.mkdir(parents=True)
    first = _cli(argv + ["--log-dir", str(log)], tmp_path, {"STEPBOOK_TEST_INBOX": str(log / "inbox.txt")})
    assert first.returncode == 3, first.stderr[-800:]
    pointers = tmp_path / ".stepbook"
    assert not (pointers / "ACTIVE").exists()
    assert Path((pointers / "LAST_RUN").read_text()) == log.resolve()
    events = [json.loads(line) for line in (log / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "paused"
    assert (work / "out.txt").read_text().strip() == "hello"  # step 0 executed before the pause
    launch = json.loads((log / "launch.json").read_text())
    assert launch["argv"][:2] == argv[:2] and launch["resume_command"].endswith(f"--resume --log-dir {log.resolve()}")

    info = classify_run(log)
    assert info["status"] == "paused" and info["steps"] == 1 and info["pending_actions"] == 0

    status = _cli(["status"], tmp_path)
    assert status.returncode == 0, status.stdout + status.stderr
    assert "status: paused" in status.stdout and "active marker: no" in status.stdout
    # The short form: the launch recorded in that directory supplies the skill and every option. The
    # acknowledgement is added only where a resume needs it again: outside a container (CI runs in one).
    acknowledgement = "" if in_sandbox() else " --unsandboxed-i-accept-the-risk"
    assert f"run --resume {log.resolve()}{acknowledgement}\n" in status.stdout

    context = json.loads(_hook("reinject_state.py", {"cwd": str(tmp_path)}))["hookSpecificOutput"]["additionalContext"]
    assert "PAUSED" in context and "NOT running" in context and str(log.resolve()) in context
    assert f"Resume it with: stepbook run --resume {log.resolve()} " in context
    assert "--unsandboxed" not in context  # a hook hint never adds a safety bypass
    assert '"goal": "create out.txt containing hello"' in context  # recovery summary still delivered

    relay = json.loads(_hook("inbox_relay.py", {"cwd": str(tmp_path), "prompt": "@loop keep going"}))
    ctx = relay["hookSpecificOutput"]["additionalContext"]
    assert "decision" not in relay and "paused" in ctx and f"stepbook run --resume {log.resolve()}" in ctx
    assert not (log / "inbox.txt").exists()  # nothing written into an inbox nobody reads

    second = _cli(argv + ["--resume"], tmp_path)  # no --log-dir: LAST_RUN supplies it
    assert second.returncode == 0, second.stderr[-800:]
    assert f"resuming {log.resolve()} (paused" in second.stderr
    summary = json.loads(second.stdout[second.stdout.index("{") :])
    assert summary["score"] == 1.0 and Path(summary["log_dir"]).resolve() == log.resolve()
    kinds = [json.loads(line)["kind"] for line in (log / "events.jsonl").read_text().splitlines()]
    assert "resumed" in kinds and kinds.index("resumed") > kinds.index("paused")
    state = json.loads((log / "state.json").read_text())
    assert state["checklist"]["c1"]["status"] == "done"
    assert classify_run(log)["status"] == "completed"
    assert not (pointers / "ACTIVE").exists()
    assert len(json.loads((log / "launch.json").read_text())["resumes"]) == 1


def test_completed_run_is_reported_but_never_offered_for_resume(tmp_path):
    argv, log, _work = _launch(tmp_path, FAKE_AGENT)
    done = _cli(argv + ["--log-dir", str(log)], tmp_path)
    assert done.returncode == 0, done.stderr[-800:]
    status = _cli(["status"], tmp_path)
    assert status.returncode == 0 and "status: completed" in status.stdout and "resume:" not in status.stdout
    context = json.loads(_hook("reinject_state.py", {"cwd": str(tmp_path)}))["hookSpecificOutput"]["additionalContext"]
    assert "completed" in context and "Nothing to resume" in context and "--resume" not in context
    relay = json.loads(_hook("inbox_relay.py", {"cwd": str(tmp_path), "prompt": "@loop pause"}))
    assert "decision" not in relay and "no active run" in relay["hookSpecificOutput"]["additionalContext"]
    refused = _cli(argv + ["--resume"], tmp_path)
    assert refused.returncode == 2 and "is completed" in refused.stderr
    assert (tmp_path / ".stepbook" / "LAST_RUN").exists()


def test_a_run_that_ends_at_its_horizon_classifies_as_stopped_and_an_agent_done_as_completed(tmp_path):
    """A bounded run that ends exactly as asked needs no resume to be told apart from a kill."""
    from stepbook import Envelope, Runtime, ScriptedExecutor, Skill, StepResult

    class Env:
        horizon = 2

        def __init__(self):
            self.t = 0

        def reset(self):
            return "go"

        def step(self, action):
            self.t += 1
            return StepResult("ok", done=action == "done" or self.t >= self.horizon, info={"ok": True})

        def score(self):
            return 1.0

    def run(directory, actions):
        answers = iter([Envelope("r", {}, a) for a in actions])
        skill = Skill("pointer", "act", {"type": "object"}, {})
        Runtime(skill, ScriptedExecutor(lambda s, u: next(answers)), Env(), log_dir=directory).run(2)
        return classify_run(directory, active=False)

    stopped = run(tmp_path / "bounded", ["echo one", "echo two"])
    assert stopped["status"] == "stopped" and stopped["last_event"] == "max_steps_reached"
    assert run(tmp_path / "finished", ["done"])["status"] == "completed"


def test_resume_without_any_pointer_still_requires_log_dir(tmp_path):
    argv, _log, _work = _launch(tmp_path, FAKE_AGENT)
    refused = _cli(argv + ["--resume"], tmp_path)
    assert refused.returncode == 2 and "no .stepbook/LAST_RUN pointer" in refused.stderr
    assert not (tmp_path / ".stepbook" / "ACTIVE").exists()


def test_interrupted_run_names_pending_actions_and_recover(tmp_path):
    run = tmp_path / ".stepbook" / "loop-7"
    run.mkdir(parents=True)
    (tmp_path / ".stepbook" / "LAST_RUN").write_text(str(run))
    (run / "state.json").write_text(json.dumps({"goal": "tune retrieval", "current": {"phase": "measure", "h": "h2"}}))
    (run / "events.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"ts": 1.0, "kind": "run_start"},
                {"ts": 2.0, "kind": "step", "step": 0, "action": "apply h2", "action_id": "a1"},
                {"ts": 3.0, "kind": "action_started", "action_id": "a2", "record": {"step": 1, "action": "run"}},
            ]
        )
        + "\n"
    )
    info = classify_run(run)
    assert info["status"] == "interrupted" and info["steps"] == 1 and info["pending_actions"] == 1
    assert info["resume_command"] is None
    context = json.loads(_hook("reinject_state.py", {"cwd": str(tmp_path)}))["hookSpecificOutput"]["additionalContext"]
    assert "INTERRUPTED" in context and f"stepbook recover {run}" in context
    assert f"stepbook resume command for {run}" in context and "tune retrieval" in context
    status = _cli(["status"], tmp_path)
    assert status.returncode == 0 and "status: interrupted" in status.stdout and "recover" in status.stdout
    alias = _cli(["status", "--log-dir", str(run)], tmp_path)  # the chat naturally reuses run's flag name
    assert alias.returncode == 0 and _without_age(alias.stdout) == _without_age(status.stdout)


def test_active_pointer_still_wins_over_last_run(tmp_path):
    live = tmp_path / ".stepbook" / "live"
    old = tmp_path / ".stepbook" / "old"
    for run in (live, old):
        run.mkdir(parents=True)
        (run / "state.json").write_text(json.dumps({"goal": run.name}))
    (old / "events.jsonl").write_text(json.dumps({"ts": 1.0, "kind": "paused", "step": 3}) + "\n")
    (tmp_path / ".stepbook" / "ACTIVE").write_text(str(live))
    (tmp_path / ".stepbook" / "LAST_RUN").write_text(str(old))
    context = json.loads(_hook("reinject_state.py", {"cwd": str(tmp_path)}))["hookSpecificOutput"]["additionalContext"]
    assert (
        f"ACTIVE points to {live}" in context and "liveness is not verified" in context and '"goal": "live"' in context
    )
    relay = json.loads(_hook("inbox_relay.py", {"cwd": str(tmp_path), "prompt": "@loop pause"}))
    assert "decision" not in relay and "relayed" in relay["hookSpecificOutput"]["additionalContext"]
    assert (live / "inbox.txt").read_text() == "pause\n"


def test_root_hooks_load_run_pointer_from_the_package(tmp_path):
    hooks = ROOT / "hooks"
    assert (ROOT / "stepbook/run_pointer.py").exists()
    project = tmp_path / "project"
    run = project / ".stepbook" / "loop-1"
    run.mkdir(parents=True)
    (project / ".stepbook" / "LAST_RUN").write_text(".stepbook/loop-1")
    (run / "state.json").write_text(json.dumps({"goal": "bundled"}))
    (run / "events.jsonl").write_text(json.dumps({"ts": 1.0, "kind": "paused", "step": 2}) + "\n")
    emitted = json.loads(_hook("reinject_state.py", {"cwd": str(project)}, hooks))
    context = emitted["hookSpecificOutput"]["additionalContext"]
    assert "PAUSED" in context and '"goal": "bundled"' in context
    assert set(emitted) == {"hookSpecificOutput"}
    assert emitted["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    relay = json.loads(_hook("inbox_relay.py", {"cwd": str(project), "prompt": "@loop hi"}, hooks))
    assert "paused" in relay["hookSpecificOutput"]["additionalContext"] and not (run / "inbox.txt").exists()

    # The packaged pointer must load without site packages, so a new third-party
    # import in run_pointer.py cannot pass on this dependency-populated interpreter.
    bare = [sys.executable, "-I", "-S"]
    assert (
        subprocess.run(
            [*bare, "-c", "import sys; print([p for p in sys.path if 'site-packages' in p])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == "[]"
    ), "the isolation flags must actually remove site packages"
    assert subprocess.run([*bare, "-c", "import jsonschema"], capture_output=True).returncode == 1
    for script, payload in (
        ("reinject_state.py", {"cwd": str(project)}),
        ("inbox_relay.py", {"cwd": str(project), "prompt": "@loop hi"}),
    ):
        isolated = subprocess.run(
            [*bare, str(hooks / script)], input=json.dumps(payload), capture_output=True, text=True
        )
        assert isolated.returncode == 0, (script, isolated.stderr[-600:])
        assert _without_age(isolated.stdout) == _without_age(_hook(script, payload, hooks))


def test_resume_argv_replaces_earlier_resume_and_log_dir_flags(tmp_path):
    argv = [
        "run",
        "stepbook/templates/eval-loop-adapter",
        "--resume",
        "--log-dir",
        "old",
        "--log-dir=older",
        "--steps",
        "3",
    ]
    assert resume_argv(argv, tmp_path / "new") == [
        "run",
        "stepbook/templates/eval-loop-adapter",
        "--steps",
        "3",
        "--resume",
        "--log-dir",
        str((tmp_path / "new").resolve()),
    ]
