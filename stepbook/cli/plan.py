"""Shared, read-only plan initialization and validation for run and validate."""

import json
from dataclasses import replace
from pathlib import Path

from ..state.skill import LockPathError, _dotted_state_path
from ..state.validator import Validator


def _merge_plan(base, patch):
    """RFC 7386, including object patches over absent or non-object targets."""
    if not isinstance(patch, dict):
        return patch
    out = dict(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = _merge_plan(out.get(key), value)
    return out


def _invalid_plan(args, exc):
    context = "; run was started with it" if getattr(args, "resume", False) else ""
    detail = " ".join(str(exc).splitlines())
    return ValueError(f"invalid --plan {str(args.plan)!r}{context}: {detail}")


def _merge_run_plan(args, skill):
    """Operator initialization may set locked values, on fresh launches and resume."""
    if not getattr(args, "plan", None):
        return ()
    try:
        patch = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        if not isinstance(patch, dict):
            raise ValueError("expected one JSON object")
        skill.initial_state = _merge_plan(skill.initial_state, patch)
        return tuple(sorted(patch))
    except (OSError, ValueError) as exc:
        raise _invalid_plan(args, exc) from exc


def validate_plan(args, skill, configure_state=None):
    """Validate the final launch state before constructing an environment or executor."""
    if not getattr(args, "plan", None):
        return
    candidate = replace(skill, initial_state=dict(skill.initial_state))
    try:
        if configure_state is not None:
            configure_state(candidate)
        for path in (*skill.protected_paths, *skill.write_once_paths):
            if _dotted_state_path(candidate.initial_state, path):
                raise LockPathError(f"lock path {path!r}: dotted keys cannot be locked; paths are dot-separated")
        Validator(skill.schema, skill.protected_paths, skill.max_deletions, skill.write_once_paths).apply(
            candidate.initial_state, {}
        )
    except Exception as exc:
        raise _invalid_plan(args, exc) from exc


def _shell_checks(args):
    checks: dict[str, str] = {}
    for value in args.check or []:
        cid, _, command = value.partition("=")
        if not command:
            raise ValueError(f"--check must be 'id=<command>', got {value!r}")
        checks[cid.strip()] = command.strip()
    return checks


def _configure_shell_state(args, skill, checks, *, only_supplied=False):
    """Keep shell CLI precedence identical in pre-build validation and launch.

    With ``only_supplied`` a flag overrides the plan only when it was given: a repeatable option is
    supplied exactly when it collected a value, whatever its default (``[]`` for run, ``None`` for validate).
    """
    state = dict(skill.initial_state)
    if not only_supplied or args.goal is not None:
        state["goal"] = args.goal
    if not only_supplied or args.constraint:
        state["constraints"] = list(args.constraint or [])
    if not only_supplied or args.check:
        state["checks"] = checks
        state["checks_status"] = {k: "unknown" for k in checks}
    merged, status = state.get("checks"), state.get("checks_status")
    if isinstance(merged, dict) and isinstance(status, dict) and set(merged) - set(status):
        # Checks and their statuses are one unit: a plan may seed checks and leave the statuses out.
        state["checks_status"] = {**dict.fromkeys(merged, "unknown"), **status}
    skill.initial_state = state

    budget = skill.initial_state.get("budget")
    if isinstance(budget, dict) and "runs_max" in budget and getattr(args, "runs_max", None) is not None:
        skill.initial_state["budget"] = {**budget, "runs_max": args.runs_max}


def _configure_shell_launch_state(args, skill, checks):
    """The state a shell run starts from: flags over the plan, and the shell fields always present."""
    _configure_shell_state(args, skill, checks, only_supplied=True)
    for key, empty in (("constraints", []), ("checks", {}), ("checks_status", {})):
        skill.initial_state.setdefault(key, empty)
