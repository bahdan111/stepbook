#!/usr/bin/env python3
"""UserPromptSubmit hook: a message starting with '@loop ' is relayed to the active stepbook run's operator inbox
('@loop pause' stops it cleanly). Everything else passes through untouched. The prompt stays in the conversation;
UserPromptSubmit additionalContext identifies the user's note as already delivered and asks for only a short
confirmation. With no ACTIVE run, it explains that nothing was relayed and asks the session to offer to resume,
naming a paused/stopped/interrupted LAST_RUN and its resume command when available."""

import importlib.util
import json
import shlex
import sys
import time
from pathlib import Path
from typing import NamedTuple, TextIO

PREFIX = "@loop "


class HookPayload(NamedTuple):
    """Everything this hook reads from its stdin JSON, parsed once."""

    prompt: str
    cwd: Path


def parse_payload(stream: TextIO) -> HookPayload:
    """Parse the hook's stdin once; no later code inspects the payload.

    Claude Code sends an object with string `prompt` and `cwd`. Anything else -- unreadable stdin,
    text that is not JSON, a JSON array or scalar, a missing or non-string field -- carries no prompt
    to relay and names no directory, so the hook passes the turn through as it already did for text
    that is not JSON, reading the process's own directory if it needs one.
    """
    try:
        payload: object = json.load(stream)
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        payload = {}
    prompt, cwd = payload.get("prompt"), payload.get("cwd")
    return HookPayload(
        prompt=prompt.strip() if isinstance(prompt, str) else "",
        cwd=Path(cwd) if isinstance(cwd, str) and cwd else Path("."),
    )


def load_run_pointer():
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "stepbook" / "run_pointer.py",):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("stepbook_run_pointer", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def inactive_reason(base: Path, rp) -> str | None:
    reason = "stepbook: no active run (.stepbook/ACTIVE missing); nothing relayed."
    last = rp.read_pointer(base, "LAST_RUN") if rp else None
    if last is None:
        return reason
    if not rp.hook_run_allowed(base.parent, last):
        return None
    last = last.resolve()
    info = rp.classify_run(last, commands=False)
    if info["status"] in ("paused", "stopped", "interrupted"):
        resume = rp.hook_resume_command(last)
        hint = (
            f"Resume it first: {resume}"
            if resume
            else f"Resume it with the stepbook resume command for {shlex.quote(str(last))}."
        )
        return (
            f"stepbook: no active run; the most recent run {last} is {info['status']} and does not read its "
            f"inbox. Nothing relayed. {hint}"
        )
    return reason


def main() -> int:
    payload = parse_payload(sys.stdin)
    prompt = payload.prompt
    if not prompt.startswith(PREFIX):
        return 0
    base = payload.cwd / ".stepbook"
    rp = load_run_pointer()
    if rp is None:
        return 0
    run_dir = rp.read_pointer(base, "ACTIVE")
    if run_dir is None:
        reason = inactive_reason(base, rp)
        if reason is None:
            return 0
        note = reason + " The user's note was not relayed. Tell the user and offer to resume."
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": note}}))
        return 0
    if not rp.hook_run_allowed(payload.cwd, run_dir):
        return 0
    run_dir = run_dir.resolve()
    message = prompt[len(PREFIX) :].strip()
    events = run_dir / "events.jsonl"
    stale = not events.exists() or (time.time() - events.stat().st_mtime) > 3 * 3600
    with (run_dir / "inbox.txt").open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    note = (
        f"stepbook: relayed; already appended the user's note to {run_dir / 'inbox.txt'}: {message!r}. "
        "The loop reads it before its next step. "
        "Do not act on the note yourself; give the user only one short confirmation."
    )
    if stale:
        note += " WARNING: that run's event log has not changed for over 3 hours; it may no longer be running."
    # Keep the user's prompt in the conversation; additional context limits the session to confirming delivery.
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": note}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
