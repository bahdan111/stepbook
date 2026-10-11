"""Host-owned pause checks, inbox delivery and withheld-action recovery."""

from __future__ import annotations

import fcntl
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..contracts.journal import Journal


class ControlStop(Exception):
    """A deliberate resumable stop, not an executor failure to retry."""


@contextmanager
def inbox_lock(run_dir: Path):
    """Serialize cooperating inbox writers and consumers without changing the text protocol."""
    with (run_dir / "inbox.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def read_inbox(log, step):
    if log is None:
        return []
    with inbox_lock(log.path.parent):
        return _read_inbox_locked(log, step)


def _read_inbox_locked(log, step):
    inbox = log.path.parent / "inbox.txt"
    pending = sorted(inbox.parent.glob("inbox.*.pending"))
    claimed = inbox.with_name(f"inbox.{uuid.uuid4().hex}.pending")
    try:
        inbox.rename(claimed)
    except FileNotFoundError:
        pass
    else:
        pending.append(claimed)
    if not pending:
        return []
    delivered = {e.get("inbox_id") for e in log.read() if e.get("kind") == "operator_messages"}
    messages = []
    for claim in pending:
        inbox_id = claim.stem
        if inbox_id not in delivered:
            lines = [
                line.strip()
                for line in claim.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            if lines:
                log.append("operator_messages", step=step, messages=lines, inbox_id=inbox_id)
                messages.extend(lines)
        claim.rename(claim.with_suffix(".consumed"))
    return messages


def inbox_requests_pause(log):
    if log is None:
        return False
    try:
        return any(
            line.strip().lower() == "pause"
            for line in (log.path.parent / "inbox.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        )
    except FileNotFoundError:
        return False


def operator_note(lines):
    messages = [line for line in lines if line.lower() != "pause"]
    return (
        (
            "OPERATOR MESSAGES (from the human running this loop; honour them):\n"
            + "\n".join(f"- {line}" for line in messages)
        )
        if messages
        else ""
    )


def withheld_action_note(events):
    withheld = Journal.from_records(events).withheld_since_last_step
    if not withheld:
        return ""
    actions = [{"action": str(e.action)[:500], "reason": e.reason} for e in withheld[-3:]]
    return (
        "HOST RECOVERY FACT: these controller proposals were withheld, NOT executed; their state patches "
        "were NOT committed. Do not assume their side effects happened. Full records are in the event log.\n"
        + json.dumps(actions)
    )


def check_control(log, env=None):
    if inbox_requests_pause(log) or getattr(env, "pause_requested", False):
        raise ControlStop("pause")  # caller consumes/logs messages and preserves requirements
