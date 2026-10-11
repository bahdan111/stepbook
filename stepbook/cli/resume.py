"""The short resume: ``stepbook run --resume [RUN_DIR]`` continues a run from its own launch record.

A run folder already holds the argv that started it, so continuing it needs the folder and nothing
else. The record is expanded back into an ordinary ``run`` invocation, which keeps one code path for
both spellings; only the few decisions a continuation may really make are accepted on top of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..run_pointer import (
    ACTIVE,
    LAST_RUN,
    UNSANDBOXED,
    classify_run,
    in_sandbox,
    pointer_dir,
    read_pointer,
    recorded_launch,
)

# What a short resume may still decide; every other setting comes from the record.
OVERRIDES = ("--resume", "--steps", "--executor", "--model", "--command", "--reopen", "--quiet", UNSANDBOXED)
# The record's own run directory and its spent reopen note are never replayed.
_WITH_VALUE = ("--log-dir", "--reopen")


def is_short_resume(args) -> bool:
    """A resume that names no skill folder: the record has to supply the launch."""
    return bool(getattr(args, "resume", False)) and not getattr(args, "skill_dir", None)


def expand_recorded_resume(args, raw_argv: list[str]) -> tuple[list[str], Path]:
    """The argv a short resume means, and the directory it was launched from.

    The recorded launch is aimed at its run and overridden; its relative paths mean what they meant in the
    launch directory, so the caller runs the expanded argv there. Raises ``ValueError`` carrying the
    one-sentence refusal when the run cannot be continued this way.
    """
    given = _given_options(raw_argv[1:])
    if extra := [name for name in given if name not in OVERRIDES]:
        raise ValueError(
            "a bare --resume continues a run with the settings recorded in its launch.json, so "
            f"{', '.join(extra)} cannot be given with it; name the run as 'run --resume RUN_DIR', or repeat "
            "the whole launch (the skill folder and every option) with '--resume --log-dir RUN_DIR' to change them."
        )
    run_dir, recorded, launch_cwd = _recorded_run(args)
    if UNSANDBOXED in recorded and UNSANDBOXED not in given and not in_sandbox():
        raise ValueError(
            f"the recorded launch ran outside a container and a resume never inherits {UNSANDBOXED}: "
            "pass it again, or set STEPBOOK_SANDBOX=1 if you are already sandboxed some other way."
        )
    # Replacing the agent replaces its selection as a unit, so a stale recorded command or model cannot survive it.
    agent = ("--executor", "--command", "--model")
    dropped = (*_WITH_VALUE, *agent) if {"--executor", "--command"} & set(given) else _WITH_VALUE
    launch = _without(recorded[1:], with_value=dropped, flags=(UNSANDBOXED,))
    return ["run", *launch, "--resume", "--log-dir", str(run_dir), *_without(raw_argv[1:])], launch_cwd


def _recorded_run(args) -> tuple[Path, list[str], Path]:
    """The run this resume continues, found the way ``status`` finds one, with its recorded launch."""
    base = pointer_dir()
    active = read_pointer(base, ACTIVE)
    named = args.resume if isinstance(args.resume, str) else None
    run_dir = Path(named) if named else active or read_pointer(base, LAST_RUN)
    if run_dir is None:
        raise ValueError(
            "--resume needs a run directory: none was given and neither .stepbook/ACTIVE nor "
            ".stepbook/LAST_RUN names one."
        )
    run_dir = run_dir.resolve()
    recorded, launch_cwd = _recorded_argv(run_dir)
    info = classify_run(run_dir, active=active is not None and active.resolve() == run_dir.resolve())
    if info["status"] == "running":
        raise ValueError(f"{run_dir} is running (its launcher is alive); pause or stop it before resuming it.")
    where = "" if launch_cwd == Path.cwd().resolve() else f" in {launch_cwd}, where it was launched"
    print(
        f"resuming {run_dir} ({info['status']}; {info['steps']} steps recorded) from its recorded launch{where}",
        file=sys.stderr,
    )
    return run_dir, recorded, launch_cwd


def _recorded_argv(run_dir: Path) -> tuple[list[str], Path]:
    """The arguments the run was launched with and its launch directory, or the refusal that says to supply them."""
    try:
        launch = json.loads((run_dir / "launch.json").read_text())
    except (OSError, ValueError):
        launch = None
    if not isinstance(launch, dict) or (recorded := recorded_launch(launch)) is None:
        raise ValueError(
            f"{run_dir} records no launch to resume from (no readable launch.json holding the original "
            "arguments); supply the launch again with '--resume --log-dir'."
        )
    cwd = Path(launch["cwd"]) if isinstance(launch.get("cwd"), str) else Path.cwd()
    if not cwd.is_dir():
        raise ValueError(
            f"{run_dir} was launched from {cwd}, which no longer exists; supply the launch again with "
            "'--resume --log-dir'."
        )
    return recorded, cwd.resolve()


def _given_options(tokens: list[str]) -> list[str]:
    """Every option named on the command line, however its value was attached."""
    return list(dict.fromkeys(token.partition("=")[0] for token in tokens if token.startswith("--")))


def _without(argv: list[str], *, with_value: tuple[str, ...] = (), flags: tuple[str, ...] = ()) -> list[str]:
    """Drop the named options from an argv, taking each one's value with it.

    ``--resume`` always goes: the expansion aims the launch at a run directory itself. Its value is
    optional, so it only takes a following token that is not another option.
    """
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        name, attached, _ = token.partition("=")
        index += 1
        if name in flags and not attached:
            continue
        if name in with_value:
            index += 0 if attached else 1
            continue
        if name == "--resume":
            if not attached and index < len(argv) and not argv[index].startswith("-"):
                index += 1
            continue
        out.append(token)
    return out
