"""Zero-adapter environment: the action IS a shell command, run in a working directory with a timeout; the
observation is exit code plus truncated output. Operator-defined checks (id -> command) define done.

Safety: commands run with the working directory as cwd; a denylist blocks the obviously destructive
(recursive rm of root/home, mkfs, disk writes, curl|sh, force pushes, ...); paths in the command that resolve outside the
working directory are refused unless ``allow_outside`` is set; every command and its exit code is in the event
log. This is a guard rail, not a sandbox: run unattended only in a container or a disposable checkout.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from ..contracts.runtime import StepResult
from ..run_pointer import in_sandbox, pointer_dir
from .shell_denylist import DENY as DENY
from .shell_denylist import denylist_reason


class CommandRun(NamedTuple):
    """What one command run reports back.

    ``timed_out`` is this environment's own deadline, set where the runner kills the process group. A command
    may exit 124 by itself, so the exit code alone does not say it, and no caller re-reads the output text to
    find out.
    """

    rc: int
    output: str
    timed_out: bool = False


@dataclass
class ShellEnv:
    workdir: Path
    goal: str
    checks: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    horizon: int = 60
    cmd_timeout_s: float = 600.0
    allow_outside: bool = False
    max_output_chars: int | None = None  # Runtime archives full output and selects a prompt preview.
    checkpoint: bool = True  # if the workdir is a git repo: work on a fresh branch and commit after every changing step
    require_sandbox: bool = True  # refuse to run unless inside a container/VM or explicitly overridden
    log_dir: Path | None = None
    resuming: bool = False  # set by the runtime on resume: reuse the existing branch, never re-run actions
    branch: str | None = None
    t: int = 0
    done: bool = False
    last_checks: dict[str, str] = field(default_factory=dict)
    last_check_output: dict[str, dict] = field(default_factory=dict)  # check id -> {"rc", "tail"} from the last run
    commands: list[dict] = field(default_factory=list)
    allow_dirty: bool = False
    _preflight_done: bool = field(default=False, init=False, repr=False)

    # The preflight and the printed resume command must agree on whether this host is isolated.
    in_sandbox = staticmethod(in_sandbox)

    def preflight(self) -> None:
        """Check startup before creating run artifacts or taking a checkpoint."""
        if self.require_sandbox and not self.in_sandbox():
            raise RuntimeError(
                "ShellEnv refuses to run outside a container or VM. Run inside Docker/Podman, set "
                "STEPBOOK_SANDBOX=1 if you are already sandboxed some other way, or pass "
                "--unsandboxed-i-accept-the-risk (a denylist is not a boundary)."
            )
        self.workdir = Path(self.workdir).resolve()
        if not self.workdir.is_dir():
            raise ValueError(f"working directory does not exist: {self.workdir}")
        if self.checkpoint and not self.resuming and not self.allow_dirty:
            rc, status = self._git("status", "--porcelain", "--untracked-files=all", "--", *self._checkpoint_paths())
            if rc == 0 and status.strip():
                dirty = [line.split(maxsplit=1)[1] for line in status.splitlines()]
                summary = ", ".join(dirty[:3])
                if len(dirty) > 3:
                    summary += f" and {len(dirty) - 3} more"
                raise ValueError(f"Working folder is dirty: {summary}; commit or stash first, or pass --allow-dirty.")
        self._preflight_done = True

    def reset(self) -> str:
        if not self._preflight_done:
            self.preflight()
        self.t = 0
        self.done = False
        self.last_checks = {k: "unknown" for k in self.checks}
        if self.resuming and self.checkpoint:
            current = self._current_branch()
            if current and current.startswith("stepbook/"):
                self.branch = current  # keep working on the branch the run already made
            else:
                # Never checkpoint onto the operator's branch when the run's branch is unavailable.
                self.branch = self._start_checkpoint_branch()
        else:
            self.branch = self._start_checkpoint_branch() if self.checkpoint else None
        checks = (
            "\n".join(f"  {k}: {v}" for k, v in self.checks.items())
            or "  (none: the operator defined no checks; say 'done' when the goal is met)"
        )
        cons = "\n".join(f"  - {c}" for c in self.constraints) or "  (none)"
        return (
            f"Working directory: {self.workdir}\nGoal: {self.goal}\nConstraints:\n{cons}\n"
            f"Checks that must pass (run with the 'check' action):\n{checks}\n"
            f"Actions: a shell command line, 'check' or 'check <id>', 'done', or 'stop <reason>'. "
            f"'wait <max> until <command> [every <interval>]' re-runs <command> until it exits 0 or <max> elapses "
            f"(units s/m/h, interval default 60s, one step however long it takes; use it instead of sleep-and-poll "
            f"steps for long-running commands)."
        )

    def step(self, action: str) -> StepResult:
        self.t += 1
        a = (action or "").strip()
        low = a.lower()
        # An action word is a whole first token: 'stop now' ends the run, './stop_services.sh --all' is a command.
        if low == "done":
            return self._done_step()
        if low.split()[:1] == ["stop"]:
            return self._stop_step(a)
        if low == "check" or low.startswith("check "):
            return self._check_step(a)
        if low.split()[:1] == ["delegate"]:
            return StepResult(
                "REFUSED by the environment: delegate is not available; do the work with shell commands.",
                done=False,
                info={"ok": False, "refused": True},
            )
        if low == "wait" or low.startswith("wait "):
            return self._wait(a)
        return self._command_step(a)

    def _done_step(self) -> StepResult:
        """End the episode, reporting whether every check passes."""
        results = self._run_checks()
        ok = all(v == "pass" for v in results.values())
        self.done = True
        return StepResult(
            self._checks_text(results)
            + (
                "\nNo checks configured."
                if not results
                else "\nAll checks pass. Goal met."
                if ok
                else "\nNot all checks pass; 'done' was refused and the episode ended as incomplete."
            ),
            done=True,
            info={"ok": ok, "checks": results, "terminal": True},
        )

    def _stop_step(self, a: str) -> StepResult:
        """End the episode at the agent's request, keeping the last check results."""
        self.done = True
        return StepResult(
            "Stopped by the agent: " + a[4:].strip(),
            done=True,
            info={"ok": False, "checks": self.last_checks, "terminal": True},
        )

    def _check_step(self, a: str) -> StepResult:
        """Run the named checks, or all of them when none is named."""
        ids = [i for i in a.split()[1:] if i in self.checks] or list(self.checks)
        results = self._run_checks(ids)
        return StepResult(
            self._checks_text(results),
            done=False,
            info={"ok": all(v == "pass" for v in results.values()), "checks": results},
        )

    def _command_step(self, a: str) -> StepResult:
        """Run a shell command line unless the sandbox blocks it, and checkpoint the result."""
        blocked = self._blocked(a)
        if blocked:
            return StepResult(
                f"REFUSED by the environment: {blocked}. Choose a different command.",
                done=False,
                info={"ok": False, "refused": True},
            )
        rc, out, _ = self._run(a)
        self.commands.append({"t": self.t, "cmd": a, "rc": rc})
        ck = self._checkpoint(f"stepbook step {self.t}: {a[:60]}") if self.branch else ""
        text = (f"$ {a}\n[exit {rc}]\n{out}" if out else f"$ {a}\n[exit {rc}] (no output)") + ck
        return StepResult(
            text, done=self.horizon is not None and self.t >= self.horizon, info={"ok": rc == 0, "rc": rc}
        )

    # Poll within one interruptible step because polling wastes controller turns and delays operator intervention.
    max_wait_s: float = 6 * 3600.0
    interrupt_file: Path | None = None
    _WAIT_RE = re.compile(
        r"^wait\s+(\d+(?:\.\d+)?)\s*([smh]?)\s+(?:every\s+(\d+(?:\.\d+)?)\s*([smh]?)\s+)?until\s+(.+?)$",
        re.IGNORECASE | re.DOTALL,
    )
    _TRAIL_EVERY_RE = re.compile(r"^(.*?)\s+every\s+(\d+(?:\.\d+)?)\s*([smh]?)$", re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _secs(n: str, unit: str) -> float:
        return float(n) * {"": 1, "s": 1, "m": 60, "h": 3600}[unit.lower()]

    def parse_wait(self, action: str) -> tuple[float, float, str] | str:
        """Returns (max_s, interval_s, command) or an error string."""
        m = self._WAIT_RE.match(action.strip())
        if not m:
            return "usage: wait <max>[s|m|h] until <command> [every <interval>]"
        max_s = self._secs(m.group(1), m.group(2))
        interval = self._secs(m.group(3), m.group(4) or "") if m.group(3) else None
        cmd = m.group(5).strip()
        trail = self._TRAIL_EVERY_RE.match(cmd)
        if trail and interval is None:
            cmd, interval = trail.group(1).strip(), self._secs(trail.group(2), trail.group(3) or "")
        if len(cmd) >= 2 and cmd[0] in "\"'" and cmd[-1] == cmd[0]:
            try:
                words = shlex.split(cmd)
            except ValueError:
                pass
            else:
                if len(words) == 1 and any(char.isspace() for char in words[0]):
                    cmd = words[0]
        if not cmd:
            return "usage: wait <max>[s|m|h] until <command> [every <interval>]"
        if max_s <= 0:
            return "wait: <max> must be positive"
        if max_s > self.max_wait_s:
            return (
                f"wait: <max> exceeds this environment's limit of {self.max_wait_s / 3600:.0f}h; wait in shorter pieces"
            )
        interval_s = min(interval if interval is not None else 60.0, max_s)
        return max_s, max(interval_s, 1.0), cmd

    def _operator_waiting(self) -> bool:
        f = self.interrupt_file or (Path(self.log_dir) / "inbox.txt" if self.log_dir else None)
        try:
            return f is not None and f.exists() and bool(f.read_text(encoding="utf-8", errors="replace").strip())
        except (OSError, ValueError):
            return False

    def _wait(self, action: str) -> StepResult:
        parsed = self.parse_wait(action)
        if isinstance(parsed, str):
            return StepResult(f"REFUSED by the environment: {parsed}.", done=False, info={"ok": False, "refused": True})
        max_s, interval_s, cmd = parsed
        blocked = self._blocked(cmd)
        if blocked:
            return StepResult(
                f"REFUSED by the environment: {blocked}. Choose a different command.",
                done=False,
                info={"ok": False, "refused": True},
            )
        start = time.monotonic()
        polls = 0
        rc, out = 1, ""
        why = "timed out"
        poll_timeout = ""
        while True:
            remaining = max_s - (time.monotonic() - start)
            # A final poll starts at the deadline (possibly just after, due to scheduling).
            # Give it at most one interval; earlier polls still cannot outlive the remaining wait.
            timeout_s = min(remaining if remaining > 0 else interval_s, interval_s, self.cmd_timeout_s)
            rc, out, timed_out = self._run(cmd, timeout_s=timeout_s)
            polls += 1
            if timed_out:  # this poll's own deadline, reported by the runner; a command may exit 124 itself
                poll_timeout = f"\n[poll {polls} timed out after {timeout_s:.1f}s]"
            elapsed = time.monotonic() - start
            if rc == 0:
                why = "condition met"
                break
            if elapsed >= max_s:
                break
            sleep_s = min(interval_s, max_s - elapsed)
            slept = 0.0
            while slept < sleep_s:
                if self._operator_waiting():
                    why = "operator message waiting"
                    break
                time.sleep(min(5.0, sleep_s - slept))
                slept += min(5.0, sleep_s - slept)
            if why == "operator message waiting":
                elapsed = time.monotonic() - start
                break
        self.commands.append({"t": self.t, "cmd": action, "rc": rc})
        head = f"$ {action}\n[{why} after {elapsed:.0f}s, {polls} poll{'s' if polls != 1 else ''}, last exit {rc}]"
        if why == "operator message waiting":
            head += " (the wait ended early because a line arrived in the inbox; it is delivered with this observation)"
        text = head + poll_timeout + (f"\n{out}" if out else "")
        return StepResult(
            text,
            done=self.horizon is not None and self.t >= self.horizon,
            info={"ok": rc == 0, "rc": rc, "wait": why, "polls": polls, "elapsed_s": round(elapsed, 1)},
        )

    def score(self) -> float:
        if not self.checks:
            return 1.0 if self.done else 0.0
        return sum(1 for v in self.last_checks.values() if v == "pass") / len(self.checks)

    # ---- git checkpoints (reversibility for unattended runs)
    def _git(self, *args: str) -> tuple[int, str]:
        pr = subprocess.run(["git", *args], cwd=self.workdir, capture_output=True, text=True, timeout=120)
        return pr.returncode, (pr.stdout or pr.stderr).strip()

    def _current_branch(self) -> str | None:
        rc, out = self._git("rev-parse", "--abbrev-ref", "HEAD")
        return out.strip() if rc == 0 and out.strip() and out.strip() != "HEAD" else None

    def restore_from_events(self, events: list[dict]) -> None:
        """Resume bookkeeping from the journal instead of re-executing anything: step counter (horizon), the
        command history, and the last known check results."""
        steps = [e for e in events if e.get("kind") == "step"]
        self.t = len(steps)
        self.commands = [
            {"t": e.get("step"), "cmd": str(e.get("action", "")), "rc": (e.get("info") or {}).get("rc")}
            for e in steps
            # The same whole-token rule ``step`` dispatches by: './stop_services.sh --all' was a command.
            if str(e.get("action", "")).lower().split()[:1] not in (["check"], ["done"], ["stop"])
        ]
        for e in steps:
            checks = (e.get("info") or {}).get("checks")
            if isinstance(checks, dict):
                self.last_checks = {**self.last_checks, **checks}

    def _checkpoint_paths(self) -> list[str]:
        """Checkpoint the whole repository except the launch pointers and current logs."""
        paths = [":(top)"]
        rc, root = self._git("rev-parse", "--show-toplevel")
        if rc != 0:
            return paths
        for directory in (pointer_dir(), self.log_dir):
            if directory is None:
                continue
            try:
                relative = directory.resolve().relative_to(Path(root).resolve())
            except ValueError:
                continue
            paths.append(f":(top,exclude,literal){relative.as_posix()}")
        return list(dict.fromkeys(paths))

    def _start_checkpoint_branch(self) -> str | None:
        rc, _ = self._git("rev-parse", "--is-inside-work-tree")
        if rc != 0:
            return None
        name = f"stepbook/{time.strftime('%Y%m%d-%H%M%S')}"
        rc, out = self._git("checkout", "-q", "-b", name)
        if rc != 0:
            return None
        paths = self._checkpoint_paths()
        self._git("add", "-A", "--", *paths)
        changed, _ = self._git("diff", "--cached", "--quiet", "--", *paths)
        self._git(
            "-c",
            "user.email=stepbook@local",
            "-c",
            "user.name=stepbook",
            "commit",
            "-q",
            "-m",
            "stepbook: start",
            "--allow-empty",
            "--only",
            "--",
            *(paths if changed else []),  # An empty unborn index has no paths for Git to match.
        )
        return name

    def _checkpoint(self, message: str) -> str:
        paths = self._checkpoint_paths()
        rc, status = self._git("status", "--porcelain", "--", *paths)
        if rc != 0 or not status.strip():
            return ""
        self._git("add", "-A", "--", *paths)
        rc, _ = self._git(
            "-c",
            "user.email=stepbook@local",
            "-c",
            "user.name=stepbook",
            "commit",
            "-q",
            "-m",
            message,
            "--only",
            "--",
            *paths,
        )
        if rc != 0:
            return ""
        _, sha = self._git("rev-parse", "--short", "HEAD")
        return f"\n[checkpoint {sha} on {self.branch}: revert with `git checkout {sha}~1 -- .`]"

    # ---- internals
    def _blocked(self, cmd: str) -> str | None:
        if reason := denylist_reason(cmd):
            return reason
        if not self.allow_outside:
            for tok in shlex.split(cmd, posix=True) if self._splittable(cmd) else cmd.split():
                if tok.startswith(("/", "~")) and not tok.startswith(("/dev/null", "/usr/bin/env")):
                    target = Path(os.path.expanduser(tok.split("=")[-1] if "=" in tok else tok))
                    try:
                        target.resolve().relative_to(self.workdir)
                    except ValueError:
                        return f"path {tok!r} is outside the working directory"
        return None

    @staticmethod
    def _splittable(cmd: str) -> bool:
        try:
            shlex.split(cmd)
            return True
        except ValueError:
            return False

    def _run(self, cmd: str, *, reap_group: bool = False, timeout_s: float | None = None) -> CommandRun:
        """Reap check process groups because surviving check servers can hang later checks."""
        timeout = self.cmd_timeout_s if timeout_s is None else min(timeout_s, self.cmd_timeout_s)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0"}
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=self.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",  # Undecodable output must not leave an unresolved action.
            env=env,
            start_new_session=True,
        )

        def kill_group():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group()
            try:
                proc.communicate(timeout=5 if timeout_s is None else min(5, timeout))
            except (subprocess.TimeoutExpired, ValueError):
                pass
            duration = f"{timeout:.0f}" if timeout_s is None else f"{timeout:.1f}"
            return CommandRun(124, f"(timed out after {duration}s; the command's process group was killed)", True)
        if reap_group:
            kill_group()
        out = (stdout or "") + (("\n[stderr]\n" + stderr) if stderr else "")
        if "\x00" in out:  # NUL bytes decode but cannot travel in argv.
            out = out.replace("\x00", "\ufffd")
        if self.max_output_chars is not None and len(out) > self.max_output_chars:
            out = (
                out[: self.max_output_chars // 2]
                + f"\n... [{len(out) - self.max_output_chars} chars omitted] ...\n"
                + out[-self.max_output_chars // 2 :]
            )
        return CommandRun(proc.returncode, out.strip())

    check_output_lines: int = 40  # tail of a failing check's output shown to the agent (0 hides it)

    def _run_checks(self, ids: list[str] | None = None) -> dict[str, str]:
        results: dict[str, str] = {}
        for k in ids or list(self.checks):
            rc, out, _ = self._run(self.checks[k], reap_group=True)
            results[k] = "pass" if rc == 0 else "fail"
            self.last_checks[k] = results[k]
            # Failure output gives the agent evidence for its next decision.
            self.last_check_output[k] = {
                "rc": rc,
                "tail": "\n".join(out.splitlines()[-self.check_output_lines :]) if self.check_output_lines else "",
            }
        return results

    def _checks_text(self, results: dict[str, str]) -> str:
        lines = ["Checks:"]
        for k, v in results.items():
            lines.append(f"  {k}: {v}   ({self.checks[k]})")
            detail = self.last_check_output.get(k) or {}
            if v == "fail" and detail.get("tail"):
                lines.append(f"  {k} output (exit {detail.get('rc')}), last {self.check_output_lines} lines:")
                lines.extend("    " + line for line in detail["tail"].splitlines())
        return "\n".join(lines)
