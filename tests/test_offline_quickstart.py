"""Exercise the published offline quickstart through the installed CLI."""

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook.runtime.rendering import render_state_prompt

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT / "tests/fixtures/offline"
# Loaded by path: the fixture is a standalone script, not part of the tests package
_spec = importlib.util.spec_from_file_location("scripted_agent", OFFLINE / "scripted_agent.py")
_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agent)
envelope = _agent.envelope


def test_offline_quickstart(tmp_path, monkeypatch):
    repo = tmp_path / "toy repo"
    shutil.copytree(OFFLINE / "toy-repo", repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    original_tests = {p.relative_to(repo): p.read_bytes() for p in (repo / "tests").glob("*.py")}
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    # Scope all git mutations, including the runtime's checkpoints, to this disposable repository.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)

    def run(*args, cwd=tmp_path):
        return subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=60)

    for args in (
        ("init", "-q"),
        ("config", "user.name", "Offline Quickstart"),
        ("config", "user.email", "offline@example.invalid"),
        ("add", "."),
        ("commit", "-qm", "Initial failing example"),
    ):
        result = run("git", *args, cwd=repo)
        assert result.returncode == 0, result.stderr

    before = run("python3", "-m", "unittest", "-q", cwd=repo)
    assert before.returncode == 1 and "FAILED (failures=1)" in before.stderr
    cli = str(Path(sys.executable).with_name("stepbook"))
    result = run(
        cli,
        "run",
        str(ROOT / "stepbook/templates/checklist"),
        "--workdir",
        str(repo),
        "--goal",
        "Make the tests pass without editing tests",
        "--check",
        "tests=python3 -m unittest -q",
        "--executor",
        "command",
        "--command",
        shlex.join(["python3", str(OFFLINE / "scripted_agent.py")]),
        "--steps",
        "10",
        "--unsandboxed-i-accept-the-risk",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The requested command prints step traces before its final JSON summary.
    summary = json.loads(result.stdout[result.stdout.index("{\n") :])
    assert summary["steps"] == 5 and summary["score"] == 1.0
    run_dir = Path(summary["log_dir"])
    if not run_dir.is_absolute():
        run_dir = tmp_path / run_dir
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    steps = [event for event in events if event["kind"] == "step"]
    assert steps[1]["info"]["checks"] == {"tests": "fail"}
    assert steps[-2]["info"]["checks"] == {"tests": "pass"}
    assert steps[-1]["action"] == "done" and steps[-1]["done"]
    assert steps[-1]["info"]["checks"] == {"tests": "pass"}
    assert steps[2]["patch"]["checks_status"] == {"tests": "fail"}
    final_state = json.loads((run_dir / "state.json").read_text())
    assert final_state["current"] == "finished; all checks passed"
    assert final_state["checks_status"] == {"tests": "pass"}
    assert original_tests == {p.relative_to(repo): p.read_bytes() for p in (repo / "tests").glob("*.py")}
    after = run("python3", "-m", "unittest", "-q", cwd=repo)
    assert after.returncode == 0 and "OK" in after.stderr
    status = run(cli, "status")
    assert status.returncode == 0 and "status: completed" in status.stdout, status.stdout + status.stderr
    assert "current: finished; all checks passed" in status.stdout
    assert status.stdout.index("Host-recorded checks") < status.stdout.index("Agent-maintained")
    assert 'checks_status: {"tests": "pass"}' in status.stdout
    report = run(cli, "report", str(run_dir))
    assert report.returncode == 0 and "done" in report.stdout, report.stdout + report.stderr


def test_optional_dependencies_absent():
    """Product tests run without the optional SDK installed."""
    code = """
import importlib.abc
import importlib.util
import sys

class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "claude_agent_sdk" or fullname.startswith("claude_agent_sdk."):
            raise ModuleNotFoundError("optional SDK hidden for offline gate", name=fullname)

sys.meta_path.insert(0, NoSDK())
original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: (
    None if name == "claude_agent_sdk" else original_find_spec(name, *a, **kw)
)
import pytest
raise SystemExit(pytest.main([
    "-q", "-rs",
    "tests/test_sdk_usage.py",
]))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


@pytest.mark.parametrize("observed", ["pass", "fail"])
def test_scripted_agent_finishes_only_after_observed_passing_checks(observed):
    state = {"facts": {"offline_stage": "done"}, "checks_status": {"tests": "pass"}}
    # Historical success and persisted state must not override the latest result.
    prompt = render_state_prompt(
        state,
        f"Checks:\n  tests: {observed}   (python3 -m unittest -q)",
        [(1, "check", "Checks:\n  tests: pass   (python3 -m unittest -q)")],
    )
    result = envelope(prompt)
    assert result["state_patch"]["checks_status"] == {"tests": observed}
    assert result["action"] == ("done" if observed == "pass" else "check")
    assert result["state_patch"]["current"] == (
        "finished; all checks passed" if observed == "pass" else "checks failed; rerun tests"
    )
