"""Durable run pointers so a fresh chat can find a paused or interrupted run.

Standard library only: the plugin hooks load this file directly with the host's
``python3`` and must not need the harness environment. ``ACTIVE`` names the current
launcher run and can survive a crash; ``LAST_RUN`` records the most recent launch,
except a failed startup. Neither is a heartbeat: classification reads the journal and, for
new launches, the PID/start-time identity in the run's additive ``launcher.json``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

ACTIVE = "ACTIVE"
LAST_RUN = "LAST_RUN"
RESUMABLE = ("paused", "interrupted", "stopped")
UNSANDBOXED = "--unsandboxed-i-accept-the-risk"
_TERMINAL_STOPS = ("budget_stop", "max_steps_reached", "abort")


def pointer_dir(cwd: str | os.PathLike | None = None) -> Path:
    return Path(cwd or ".") / ".stepbook"


def in_sandbox() -> bool:
    """Check for an isolation boundary because instructions alone cannot enforce isolation.

    Shared by ShellEnv's preflight and the printed resume command, so both give one answer about
    whether the unsandboxed acknowledgement is needed here.
    """
    if os.environ.get("STEPBOOK_SANDBOX") == "1":
        return True
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cg = Path("/proc/1/cgroup").read_text()
        if any(k in cg for k in ("docker", "containerd", "kubepods", "lxc", "podman")):
            return True
    except OSError:
        pass
    return False


def write_launch_pointers(log_dir: str | os.PathLike, *, cwd: str | os.PathLike | None = None) -> Path:
    base = pointer_dir(cwd)
    base.mkdir(exist_ok=True)
    target = str(Path(log_dir).resolve())
    Path(target).mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    (Path(target) / "launcher.json").write_text(json.dumps({"pid": pid, "start_time": _process_start_time(pid)}))
    (base / ACTIVE).write_text(target)
    (base / LAST_RUN).write_text(target)
    return base


def _process_start_time(pid: int) -> str | None:
    """macOS/Linux start token, empty for a zombie, None when inspection is unavailable."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat=", "-o", "lstart="],
            capture_output=True,
            text=True,
            timeout=2,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    fields = result.stdout.strip().split(maxsplit=1)
    if result.returncode != 0 or len(fields) != 2:
        return None
    return "" if fields[0].startswith("Z") else fields[1]


def _launcher_dead(run_dir: Path) -> bool:
    """A missing/zombie PID or a different start token proves that the recorded launcher ended."""
    try:
        owner = json.loads((run_dir / "launcher.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(owner, dict) or type(owner.get("pid")) is not int or owner["pid"] <= 0:
        return False
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass  # Another owner prevents signaling, but its start token may still establish identity.
    except (OSError, OverflowError):
        return False
    start = _process_start_time(owner["pid"])
    if start == "":
        return True
    recorded = owner.get("start_time")
    return isinstance(recorded, str) and bool(recorded) and start is not None and start != recorded


def clear_active(cwd: str | os.PathLike | None = None) -> None:
    try:
        (pointer_dir(cwd) / ACTIVE).unlink()
    except OSError:
        pass


def read_pointer(base: str | os.PathLike, name: str) -> Path | None:
    base = Path(base)
    try:
        text = (base / name).read_text().strip()
    except (OSError, ValueError):
        return None
    if not text:
        return None
    run = Path(text)
    return run if run.is_absolute() else base.parent / run


def hook_run_allowed(cwd: Path, run: Path) -> bool:
    """Hooks only trust local, untracked runs, never a checkout's committed context."""
    try:
        root = cwd.resolve()
        resolved = run.resolve(strict=True)
        relative = resolved.relative_to(root / ".stepbook")
        if not relative.parts or not resolved.is_dir():
            return False
        # Check both the target and any tracked symlink used to reach it.
        paths = {resolved.relative_to(root)}
        lexical = Path(os.path.abspath(run))
        if lexical.is_relative_to(root):
            paths.add(lexical.relative_to(root))
        paths.update(p for path in list(paths) for p in path.parents if (root / p).is_symlink())
        prefixes = {p.as_posix().casefold() for p in paths}
    except (OSError, ValueError, RuntimeError):
        return False
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", ":(literal).stepbook"],
            capture_output=True,
            timeout=2,
        )
        if tracked.returncode != 0:
            # A damaged index or another Git error must not make a repository's
            # tracked run acceptable. Only the non-repository case is exempt.
            repository = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                timeout=2,
            )
            return repository.returncode != 0
    except OSError:
        return True  # Git unavailable, including a missing executable.
    except subprocess.TimeoutExpired:
        return False
    tracked_paths = (os.fsdecode(path).casefold() for path in tracked.stdout.split(b"\0") if path)
    return not any(path == prefix or path.startswith(prefix + "/") for path in tracked_paths for prefix in prefixes)


def recorded_launch(launch: dict) -> list[str] | None:
    """The newest recorded invocation that names a skill: a full launch, not a short resume.

    A short resume records ``run --resume <dir>``, which carries no settings of its own, so the
    launch it continued stays the one a later resume replays.
    """
    resumes = launch.get("resumes")
    records = [launch, *(resumes if isinstance(resumes, list) else [])]
    for record in reversed(records):
        argv = record.get("argv") if isinstance(record, dict) else None
        if (
            isinstance(argv, list)
            and len(argv) >= 2
            and all(isinstance(arg, str) and arg and arg.isprintable() for arg in argv)
            and argv[0] == "run"
            and not argv[1].startswith("-")
        ):
            return argv
    return None


def hook_resume_command(run: Path) -> str | None:
    """A hook hint for the short resume form; the run's own record supplies every setting.

    The printed command names nothing but the run directory the caller already verified, so no
    recorded value is ever echoed into a chat as a command. It also never adds the unsandboxed
    acknowledgement: a run that needs one is resumed by the operator typing it.
    """
    try:
        launch = json.loads((run / "launch.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(launch, dict) or recorded_launch(launch) is None:
        return None
    return shlex.join(["stepbook", "run", "--resume", str(run)])


def resume_argv(argv: list[str], log_dir: str | os.PathLike) -> list[str]:
    """The original arguments with any earlier --resume/--log-dir replaced by this run's directory."""
    out: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == "--resume" or item.startswith("--log-dir="):
            continue
        if item == "--log-dir":
            skip = True
            continue
        out.append(item)
    return out + ["--resume", "--log-dir", str(Path(log_dir).resolve())]


def write_launch_record(
    log_dir: str | os.PathLike,
    argv: list[str],
    *,
    resume: bool = False,
    program: str | list[str] = "stepbook",
    cwd: str | os.PathLike | None = None,
    skill_hash: str | None = None,
) -> dict:
    """Record how a run was launched so status/hooks can print an exact resume command.

    ``program`` is the executable as it was really invoked (an absolute console-script path or
    ``python -m stepbook.cli``), so the printed command works from a chat without PATH guesses.
    The first launch record is kept; a resume appends to ``resumes`` instead of rewriting it.
    """
    program = [program] if isinstance(program, str) else list(program)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "launch.json"
    now = time.time()
    record = None
    if path.exists():
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            record = None
        if not isinstance(record, dict):
            record = None
    if record is None:
        record = {
            "program": program,
            "argv": list(argv),
            "cwd": str(Path(cwd or os.getcwd()).resolve()),
            "started_at": now,
            "resume_command": shlex.join(program + resume_argv(list(argv), log_dir)),
        }
        if skill_hash is not None and not resume:
            record["skill_hash"] = skill_hash
    elif resume:
        resumed = {"resumed_at": now, "argv": list(argv)}
        if skill_hash is not None and record.get("skill_hash"):
            resumed["skill_hash"] = skill_hash
            if record["skill_hash"] != skill_hash:
                resumed["skill_hash_changed"] = {"from": record["skill_hash"], "to": skill_hash}
                print("skill changed since launch; resuming with the current skill", file=sys.stderr)
        record.setdefault("resumes", []).append(resumed)
    path.write_text(json.dumps(record, indent=2))
    return record


def mark_failed_launch(log_dir: Path, error: Exception) -> None:
    """Retain failed startup evidence without advertising it as resumable or discarding action recovery."""
    events = _read_events(log_dir / "events.jsonl")
    if any(e.get("kind") in ("action_started", "step") for e in events):
        return
    path = log_dir / "launch.json"
    if not path.exists():
        return
    record = json.loads(path.read_text())
    record["failed_to_start"] = {"error": str(error), "at": time.time()}
    path.write_text(json.dumps(record, indent=2))
    base = pointer_dir()
    for name in (ACTIVE, LAST_RUN):
        target = read_pointer(base, name)
        if target is not None and target.resolve() == log_dir.resolve():
            (base / name).unlink(missing_ok=True)


def _read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a torn tail is preserved separately by EventLog; never fail discovery on it
        if isinstance(event, dict):
            events.append(event)
    return events


def classify_run(run_dir: str | os.PathLike, *, active: bool = False, commands: bool = True) -> dict:
    """Read-only classification from the journal; ``active`` means an ACTIVE pointer names this run.

    Hooks pass ``commands=False`` to avoid interpreting untrusted launch commands.

    Statuses: running (ACTIVE present, launcher not known dead), paused (clean operator pause),
    completed (a step finished the episode), failed_checks (closed with named check failures),
    stopped (budget/horizon/abort record),
    interrupted (events without a terminal record: crash or kill), failed_to_start, not_started, missing.
    """
    run_dir = Path(run_dir)
    info: dict = {
        "run_dir": str(run_dir),
        "status": "missing",
        "steps": 0,
        "last_event": None,
        "last_event_age_s": None,
        "pending_actions": 0,
        "resume_command": None,
    }
    if not run_dir.exists():
        return info
    try:
        launch = json.loads((run_dir / "launch.json").read_text())
        if isinstance(launch, dict) and launch.get("failed_to_start"):
            info["status"] = "failed_to_start"
            return info
        if commands and isinstance(launch, dict):
            info["resume_command"] = _latest_resume_command(launch, run_dir)
    except (OSError, ValueError):
        pass
    events = _read_events(run_dir / "events.jsonl")
    steps = [e for e in events if e.get("kind") == "step"]
    info["steps"] = len(steps)
    last = events[-1] if events else None
    if last is not None:
        info["last_event"] = last.get("kind")
        try:
            # A missing or non-numeric ts raises here and is caught below; stdlib-only, untyped-tolerant module.
            info["last_event_age_s"] = max(0, int(time.time() - float(last.get("ts"))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    started = {e.get("action_id") for e in events if e.get("kind") == "action_started"}
    committed = {e.get("action_id") for e in steps}
    info["pending_actions"] = len(started - committed)
    info["status"], info["failed_checks"] = _event_outcome(events, active)
    reason = agent_stop_reason(terminal_step(events) or {})
    if reason is not None:
        info["stop_reason"] = reason
    if info["status"] == "running" and _launcher_dead(run_dir):
        info["status"] = "interrupted"
    return info


def terminal_step(events: list[dict]) -> dict | None:
    """The closed episode survives later records until explicitly reopened."""
    terminal = None
    for e in events:
        if e.get("kind") == "step" and e.get("done") and not e.get("horizon_stop"):
            terminal = e
        elif e.get("kind") == "reopened":
            terminal = None
    return terminal


def agent_stop_reason(event: dict) -> str | None:
    """Shell stop is an incomplete ending; an adapter's ordinary stop completes its loop."""
    action = str(event.get("action", "")).strip().split(maxsplit=1)
    info = event.get("info")
    shell_stop = isinstance(info, Mapping) and info.get("terminal") and "checks" in info
    if (
        action
        and action[0].lower() == "stop"
        and (shell_stop or str(event.get("result", "")).startswith("Stopped by the agent:"))
    ):
        return action[1] if len(action) > 1 else "no reason supplied"
    return None


def _event_outcome(events: list[dict], active: bool) -> tuple[str, list[str]]:
    """Completion survives later records unless an explicit reopen follows it."""
    last = events[-1] if events else {}
    terminal = terminal_step(events)
    if terminal is not None:
        terminal_info = terminal.get("info")
        checks = terminal_info.get("checks") if isinstance(terminal_info, Mapping) else None
        failed = [name for name, result in checks.items() if result == "fail"] if isinstance(checks, dict) else []
        if agent_stop_reason(terminal) is not None:
            return "stopped", failed
        if isinstance(checks, dict) and not failed and any(value != "pass" for value in checks.values()):
            return "stopped", []
        return ("failed_checks" if failed else "completed"), failed
    if last.get("kind") == "paused":
        return "paused", []
    if last.get("kind") in _TERMINAL_STOPS:
        return "stopped", []
    if active:
        return "running", []
    return ("interrupted" if events else "not_started"), []


def _latest_resume_command(launch: dict, run_dir: Path) -> str | None:
    """The short form: the recorded launch in this directory supplies every other option.

    The acknowledgement is never inherited, so the printed command carries it where the operator
    would have to type it again: the launch ran unsandboxed and this host is not a container.
    """
    recorded = recorded_launch(launch)
    if recorded is None or not launch.get("program"):
        return launch.get("resume_command")
    program = launch["program"]
    program = [program] if isinstance(program, str) else program
    if shutil.which("stepbook"):  # print the command the way a user types it; the full path only when needed
        program = ["stepbook"]
    safety = [UNSANDBOXED] if UNSANDBOXED in recorded and not in_sandbox() else []
    return shlex.join([*program, "run", "--resume", str(Path(run_dir).resolve()), *safety])


def resume_hint(info: dict) -> str:
    if info.get("resume_command"):
        return info["resume_command"]
    return f"stepbook run <original arguments> --resume --log-dir {shlex.quote(info['run_dir'])}"


def describe(info: dict, *, commands: bool = True) -> str:
    """One sentence for humans and the chat model; never claims liveness or execution."""
    status = info["status"]
    steps = info["steps"]
    if status == "running":
        return f"ACTIVE names this run; process liveness is not verified ({steps} steps recorded)."
    if status == "paused":
        return f"paused cleanly after {steps} recorded steps; it is not running." + (
            f" Resume with: {resume_hint(info)}" if commands else ""
        )
    if status == "completed":
        return f"completed after {steps} steps; nothing to resume."
    if status == "failed_checks":
        return (
            f"ended with failing checks: {', '.join(info['failed_checks'])}; journal closed after {steps} steps. "
            + (f"Reopen with: {resume_hint(info)} --reopen 'fix failing checks and finish'" if commands else "")
        )
    if status == "stopped":
        if "stop_reason" in info:
            return f"stopped by the agent: {info['stop_reason']} ({steps} steps recorded)."
        return (
            f"stopped by {info['last_event']} after {steps} steps; resuming needs a new allowance or horizon "
            "decision." + (f" Resume with: {resume_hint(info)}" if commands else "")
        )
    if status == "interrupted":
        warning = (
            f" {info['pending_actions']} action(s) started without a receipt; run `stepbook recover "
            f"{info['run_dir']}` and verify the outcome first."
            if info["pending_actions"]
            else ""
        )
        return f"ended without a terminal record after {steps} steps (crash or kill).{warning} " + (
            f"Resume with: {resume_hint(info)}" if commands else ""
        )
    if status == "not_started":
        return "was launched but recorded no events."
    if status == "failed_to_start":
        return "launch failed before the first action; start a new run."
    return "directory is missing."
