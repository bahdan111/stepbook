"""Durable action boundaries for external environments, without claiming exactly-once execution."""

from __future__ import annotations

import hashlib
import json
import shlex
import uuid
from contextlib import contextmanager
from functools import wraps

from pydantic import ValidationError

from ..contracts.errors import UnreceiptedAction
from ..contracts.journal import OperatorReceipt

try:
    import fcntl
except ImportError as error:
    fcntl = None  # type: ignore[assignment]  # optional POSIX module; run_lock re-raises the saved import error
    _fcntl_import_error = error


@contextmanager
def run_lock(log):
    """Single runtime/reconciliation writer per log (POSIX, released on process death)."""
    if log is None:
        yield
        return
    if fcntl is None:
        raise _fcntl_import_error
    with log.path.with_name("runtime.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "this run already has an active writer; do not resume or reconcile concurrently"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def exclusive_run(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with run_lock(self.log):
            return fn(self, *args, **kwargs)

    return wrapped


def pending_actions(log):
    journal = log.journal()
    return [
        {
            "action_id": e.action_id,
            "step": e.record.step,
            "action": e.record.action,
            "receipt_available": e.action_id in journal.receipts,
        }
        for e in journal.pending
    ]


def reconcile_receipt(log, action_id, receipt_path, reason):
    """Explicit operator assertion, never automatic inference from a plausible file.

    The operator must verify process termination and the action's actual outcome.
    This records what they supplied; it does not prove the external action occurred.
    """
    if not reason or not reason.strip():
        raise ValueError("reconciliation requires a reason describing how the outcome was verified")
    with run_lock(log):
        pending = [e for e in pending_actions(log) if e["action_id"] == action_id]
        if not pending or pending[0]["receipt_available"]:
            raise ValueError("action is not missing a receipt; never replace a recorded outcome")
        with receipt_path.open("rb") as f:
            raw = f.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("receipt exceeds 1 MiB")
        try:
            receipt = OperatorReceipt.model_validate(json.loads(raw))
        except ValidationError as error:
            raise ValueError("receipt must contain result (string), info (object), done (boolean)") from error
        log.append(
            "action_result",
            action_id=action_id,
            **receipt.fields,
            reconciliation={
                "authority": "explicit operator assertion; not independently verified",
                "reason": reason.strip(),
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            },
        )
        recover_actions(log)


def execute_action(log, env, **record):
    """Persist intent before dispatch and receipt before committing a step.

    A missing receipt is ambiguous, even for a seemingly read-only action: adapter
    commands may incur cost or have side effects. Never automatically redispatch it.
    """
    action_id = uuid.uuid4().hex
    if log:
        log.append("action_started", action_id=action_id, record=record)
    result = env.step(record["action"])
    horizon = getattr(env, "horizon", None)
    t = getattr(env, "t", None)
    verb = str(record.get("action", "")).strip().split(" ")[0].lower() if isinstance(record.get("action"), str) else ""
    terminal = bool((result.info or {}).get("terminal")) or verb in ("done", "stop")
    horizon_stop = bool(result.done and horizon and t is not None and t >= horizon and not terminal)
    receipt = {"result": result.observation, "info": result.info, "done": result.done}
    if horizon_stop:
        receipt["horizon_stop"] = (
            True  # the environment's step guard, not the agent's done: a resume with a higher guard continues
        )
    if log:
        log.append("action_result", action_id=action_id, **receipt)
        log.append("step", action_id=action_id, **record, **receipt)
    return result


def recover_actions(log):
    """Complete received actions from the journal; refuse unknown execution outcomes."""
    if log is None:
        raise RuntimeError("resume requires a durable event log")
    journal = log.journal()
    if not journal.count:
        raise RuntimeError("no events to resume; start a new run explicitly")
    pending = journal.pending
    ambiguous = [e for e in pending if e.action_id not in journal.receipts]
    if ambiguous:
        e = ambiguous[0]
        raise UnreceiptedAction(
            f"unresolved action {e.action_id}: {e.record.action!r} may have executed; "
            "verify the old process has stopped first, then inspect the external outcome. "
            "Save a verified receipt JSON with result (string), info (object), done (boolean); "
            "do not guess an unknown outcome. Reconcile with:\n"
            + shlex.join(
                [
                    "stepbook",
                    "recover",
                    str(log.path.parent.resolve()),
                    "--action-id",
                    e.action_id,
                    "--receipt",
                    "<receipt.json>",
                    "--reason",
                    "<how the outcome was verified>",
                ]
            )
        )
    for e in pending:
        r = journal.receipts[e.action_id]
        log.append(
            "step",
            action_id=e.action_id,
            **e.record.fields,
            result=r.result,
            info=r.info,
            done=r.done,
            recovered_from_receipt=True,
            **({"horizon_stop": True} if r.horizon_stop else {}),
        )
