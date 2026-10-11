"""Adapter subprocess transport, receipt stdout and operator inbox polling."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time as _t
from pathlib import Path
from typing import TYPE_CHECKING

from ..contracts.eval import AdapterReply
from ..runtime.controls import inbox_lock


class AdapterGlueMixin:
    """The adapter transport calls of ``AdapterEvalEnv``, which supplies the adapter
    command, the log and repository paths and the timeout."""

    if TYPE_CHECKING:
        # Supplied by ``AdapterEvalEnv``; declarations for the type checker only, never executed, so the
        # environment's dataclass fields and their order are unchanged.
        adapter: str
        evidence_roots: tuple[str, ...]
        inbox_poll_s: float
        log_dir: Path | None
        pause_requested: bool
        timeout_s: float

    def _communicate_with_inbox_poll(self, popen: subprocess.Popen) -> tuple[str, str]:
        """Wait for a long adapter call while peeking at the operator inbox every 30 s: a `pause` line sets
        `pause_requested` so the runtime stops after this step instead of 40 minutes later. Other lines are left
        for the runtime to consume normally."""
        waited = 0.0
        while True:
            try:
                return popen.communicate(timeout=min(self.inbox_poll_s, max(0.05, self.timeout_s - waited)))
            except subprocess.TimeoutExpired:
                waited += self.inbox_poll_s
                if waited >= self.timeout_s:
                    raise
                inbox = (self.log_dir / "inbox.txt") if self.log_dir else None
                if inbox and inbox.exists():
                    try:
                        with inbox_lock(inbox.parent):
                            lines = [
                                line.strip()
                                for line in inbox.read_text(encoding="utf-8", errors="replace").splitlines()
                                if line.strip()
                            ]
                            if any(line.lower() == "pause" for line in lines):
                                self.pause_requested = True
                                inbox.rename(inbox.with_name(f"inbox.{int(_t.time())}.consumed"))
                                rest = [line for line in lines if line.lower() != "pause"]
                                if rest:  # keep the other operator lines for the runtime to consume normally
                                    inbox.write_text("\n".join(rest) + "\n", encoding="utf-8")
                    except (OSError, ValueError):
                        pass

    # ---- code snapshots ----------------------------------------------------------------------------------------

    def _call(self, args: list[str]) -> AdapterReply:
        """Run the adapter and return its exit code with its stdout; stderr streams to <log_dir>/adapter.log
        (heartbeat for long evals) and only its tail reaches the model on failure.

        The exit code is the answer: callers branch on ``AdapterReply``, never on the words in ``stdout``.
        """
        cmd = [*shlex.split(self.adapter), *args]
        err_path = (self.log_dir / "adapter.log") if self.log_dir else None
        if err_path:
            err_path.parent.mkdir(parents=True, exist_ok=True)
        errf = open(err_path, "a") if err_path else None
        child_env = os.environ.copy()
        if self.evidence_roots:
            child_env["STEPBOOK_EVIDENCE_ROOTS"] = json.dumps([str(Path(p).resolve()) for p in self.evidence_roots])
        else:
            child_env.pop("STEPBOOK_EVIDENCE_ROOTS", None)
        popen = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=errf or subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=child_env,
        )
        try:
            out, err = self._communicate_with_inbox_poll(popen)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(popen.pid, signal.SIGKILL)  # the adapter and everything it spawned
            except ProcessLookupError:
                pass
            popen.wait()
            return AdapterReply(
                exit_code=-1,
                error=f"Adapter error: timed out after {self.timeout_s:.0f}s running {' '.join(args)}",
                timed_out=True,
            )
        finally:
            if errf:
                errf.close()

        reply = AdapterReply(exit_code=popen.returncode, stdout=(out or "").strip())
        if reply.ok or (reply.stdout and not reply.failed):
            return reply
        # Either the call failed, or it refused without saying why: the stderr tail is all the agent can be told.
        tail = ""
        if err_path and err_path.exists():
            tail = err_path.read_text()[-1500:]
        elif err:
            tail = err[-1500:]
        return AdapterReply(
            exit_code=reply.exit_code,
            stdout=reply.stdout,
            error=f"Adapter error (exit {reply.exit_code}): {tail}",
        )
