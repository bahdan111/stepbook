"""Caller-configured commands and deterministic scripted executors."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts.errors import ExecutorError
from .base import Envelope, _last_envelope_object, strip_nulls

# The exit code a step command uses when its own turn timeout fired; such a call must not be replayed.
EXIT_TIMED_OUT = 124  # the convention of timeout(1)


def driver_timeout(host_timeout_s: float, *, margin_s: float = 30.0, floor_s: float = 60.0) -> float:
    """A step driver's own turn timeout, shorter than the host's so the driver reports the timeout itself.

    The host gives the driver its whole controller timeout, and then kills it at exactly that moment; a driver
    that waited the same amount would be killed mid-report and its timeout retried as a crash. Shrink by
    ``margin_s``, never below ``floor_s``, and never above the host's own limit (a host timeout under the floor
    is the caller's explicit choice, as in the tests).
    """
    return min(host_timeout_s, max(host_timeout_s - margin_s, floor_s))


@dataclass
class ScriptedExecutor:
    """Deterministic executor for tests: ``fn(system_prompt, user_prompt) -> Envelope``.
    Token counts are approximated as len(text)/4 so prompt-growth assertions work offline.
    Unreported cost stays unknown; scripts that report free work must supply ``cost_usd=0.0``."""

    fn: Callable[[str, str], Envelope]

    def step(self, system_prompt: str, user_prompt: str, output_schema: dict[str, Any]) -> Envelope:
        env = self.fn(system_prompt, user_prompt)
        if env.input_tokens == 0:
            env.input_tokens = (len(system_prompt) + len(user_prompt)) // 4
        if env.output_tokens == 0:
            env.output_tokens = (len(env.reasoning) + len(env.action) + len(json.dumps(env.state_patch or {}))) // 4
        return env


@dataclass
class CommandExecutor:
    """Generic executor for any agent CLI: runs ``command`` with the rendered prompt on stdin and expects one JSON
    object on stdout matching the output schema (``{"reasoning", "state_patch", "action"}``). Placeholders in the
    command: ``{system_prompt_file}`` (path to a temp file with the system prompt) and ``{schema_file}``.
    This is how a Codex CLI or Gemini CLI executor is configured; for Codex the package ships a wrapper that makes
    the schema strict and reports usage: ``python -m stepbook.executors.codex step --schema {schema_file}``.
    Usage is unknown to the harness unless the JSON carries ``usage: {input_tokens, output_tokens}``.

    Exit ``124`` (``EXIT_TIMED_OUT``) means the agent's own turn timeout fired; it is not retried, so a paid call
    is not re-sent. Every other nonzero exit stays an ordinary retryable failure."""

    command: str
    timeout_s: float = 600.0
    name: str = "command"
    cwd: Path | str | None = None

    def step(self, system_prompt: str, user_prompt: str, output_schema: dict[str, Any]) -> Envelope:
        system_prompt, user_prompt = strip_nulls(system_prompt), strip_nulls(user_prompt)
        with (
            tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as sp,
            tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sc,
        ):
            sp.write(system_prompt)
            sc.write(json.dumps(output_schema))
            sp_path, sc_path = sp.name, sc.name
        cmd = shlex.split(self.command.format(system_prompt_file=sp_path, schema_file=sc_path))
        prompt = f"<<SYSTEM>>\n{system_prompt}\n<</SYSTEM>>\n\n{user_prompt}\n\nAnswer with one JSON object matching the schema in {sc_path}."
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout_s, cwd=self.cwd
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutorError(
                f"{self.name} timed out after {self.timeout_s:.0f}s; inspect saved work before retrying",
                retryable=False,
            ) from exc
        finally:
            for pth in (sp_path, sc_path):
                try:
                    os.unlink(pth)
                except OSError:
                    pass
        if proc.returncode != 0:
            message = f"{self.name} exited {proc.returncode}: {(proc.stderr or proc.stdout)[-1500:]}"
            if proc.returncode == EXIT_TIMED_OUT:
                # No receipt for the turn the agent abandoned: do not replay a potentially paid request.
                raise ExecutorError(message, retryable=False)
            raise ExecutorError(message)
        text = proc.stdout.strip()
        out = _last_envelope_object(text)
        if out is None:
            raise ExecutorError(f"{self.name} printed no JSON object: {text[-300:]!r}")
        usage = out.get("usage") or {}
        # Missing or null prices are unknown; only an explicit zero means free.
        # Extra usage categories are kept verbatim.
        cost_unknown = usage.get("cost_usd") is None
        cost_usd = float(usage.get("cost_usd") or 0.0) if not cost_unknown else 0.0
        return Envelope(
            reasoning=str(out.get("reasoning", "")),
            state_patch=out.get("state_patch"),
            action=str(out.get("action", "")),
            extra={k: v for k, v in out.items() if k not in ("reasoning", "state_patch", "action", "usage")},
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cost_usd=cost_usd,
            api_calls=int(out.get("api_calls") or 1),
            tool_calls=int(out.get("tool_calls") or 0),
            tool_log=list(out.get("tool_log") or []),
            usage={k: v for k, v in usage.items() if k != "cost_usd"},
            duration_s=time.time() - t0,
            raw={
                "stdout_tail": text[-500:],
                **({"cost_unknown": True} if cost_unknown else {}),
                **({"usage_unknown": True} if not usage else {}),
            },
        )
