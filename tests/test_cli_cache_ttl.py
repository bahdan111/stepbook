"""The headless controller asks for the 5-minute prompt cache: each step rewrites its state part anyway, so the
1-hour cache Claude Code defaults to on a subscription only raises the write rate.
Scoped to the spawned process; a value set by the operator wins."""

import json
import subprocess
from types import SimpleNamespace

from stepbook.executors.claude import ClaudeCLIExecutor

OK = {
    "type": "result",
    "result": json.dumps({"reasoning": "r", "state_patch": {}, "action": "done"}),
    "total_cost_usd": 0.01,
    "num_turns": 1,
    "modelUsage": {},
}


def _capture(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["env"] = kw.get("env")
        return SimpleNamespace(returncode=0, stdout=json.dumps(OK), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_cli_executor_defaults_to_the_five_minute_cache(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_PROMPT_CACHE_TTL", raising=False)
    seen = _capture(monkeypatch)
    ClaudeCLIExecutor().step("sys", "user", {})
    assert seen["env"]["CLAUDE_CODE_PROMPT_CACHE_TTL"] == "5m" and "CLAUDECODE" not in seen["env"]


def test_operator_override_of_the_cache_ttl_wins(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_PROMPT_CACHE_TTL", "1h")
    seen = _capture(monkeypatch)
    ClaudeCLIExecutor().step("sys", "user", {})
    assert seen["env"]["CLAUDE_CODE_PROMPT_CACHE_TTL"] == "1h"


def test_cli_executor_runs_claude_in_the_environment_workdir(monkeypatch, tmp_path):
    """The CLI's prefix names its cwd, so the subprocess must run where the environment lives."""
    seen = {}

    def run(cmd, **kw):
        seen["cwd"] = kw.get("cwd")
        return SimpleNamespace(returncode=0, stdout=json.dumps(OK), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    ClaudeCLIExecutor(cwd=str(tmp_path)).step("sys", "user", {})
    assert seen["cwd"] == str(tmp_path)
    ClaudeCLIExecutor().step("sys", "user", {})
    assert seen["cwd"] is None
