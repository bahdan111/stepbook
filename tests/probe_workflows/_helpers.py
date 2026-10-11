"""Shared helpers for the adversarial workflow probes. No model calls: ScriptedExecutor + ShellEnv in tmp dirs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from stepbook import Envelope, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv
from stepbook.storage.events import EventLog

ROOT = Path(__file__).resolve().parents[2]


def checklist() -> Skill:
    return Skill.load(ROOT / "stepbook/templates" / "checklist")


def shell_env(workdir, **kw) -> ShellEnv:
    kw.setdefault("checkpoint", False)
    kw.setdefault("checks", {})
    kw.setdefault("goal", "g")
    return ShellEnv(require_sandbox=False, workdir=workdir, **kw)


def cmd(text: str, patch: dict | None = None) -> Envelope:
    return Envelope("r", patch if patch is not None else {}, text)


class Script:
    """Scripted controller: hands out envelopes in order and records every prompt.

    ``exhausted`` is what to answer once the queue is empty: a command string, an Envelope (returned every
    time, for "the model keeps answering the same thing"), or None to fail the test on an unexpected call.
    Queue items may be callables ``fn(user_prompt) -> Envelope`` for side effects mid-call."""

    def __init__(self, *envelopes, exhausted="done"):
        self.queue = list(envelopes)
        self.prompts: list[str] = []
        self.exhausted = exhausted

    def __call__(self, system_prompt: str, user_prompt: str) -> Envelope:
        self.prompts.append(user_prompt)
        if self.queue:
            item = self.queue.pop(0)
            return item(user_prompt) if callable(item) else item
        if self.exhausted is None:
            raise AssertionError("scripted controller was called more times than the scenario allows")
        if isinstance(self.exhausted, Envelope):
            return self.exhausted
        return Envelope("exhausted", {}, self.exhausted)

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def executor(self) -> ScriptedExecutor:
        return ScriptedExecutor(self)


def events(log_dir) -> list[dict]:
    return EventLog(Path(log_dir) / "events.jsonl").read()


def kinds(log_dir) -> list[str]:
    return [e.get("kind") for e in events(log_dir)]


def steps(log_dir) -> list[dict]:
    return [e for e in events(log_dir) if e.get("kind") == "step"]


def rewrite_events(log_dir, keep) -> None:
    """Rewrite events.jsonl keeping only the lines for which ``keep(event) `` is true (simulates a kill)."""
    path = Path(log_dir) / "events.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [line for line in lines if keep(json.loads(line))]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def git_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    (repo / "README").write_text("x\n")
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], cwd=repo, check=True
    )
    return repo


def git(repo, *args) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()
