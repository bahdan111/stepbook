"""Claude CLI execution, subscription preflight and receipt handling."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts.errors import ExecutorError, MissingExecutorBinary, RateLimited
from ..contracts.executor import CliReceipt, first_problem, reports_rate_limit
from ..reporting.usage import claude_token_totals
from .auth import SubscriptionError, check_subscription, subscription_required
from .base import Envelope, strip_nulls


def _subscription_preflight(*, cwd=None, claude_bin="claude", custom_flags=False):
    """The Claude CLI executor's own attestation; no other executor resolves a binary or asks auth status."""
    if not subscription_required():
        return None
    try:
        if custom_flags:
            raise SubscriptionError("Subscription-only calls cannot verify custom CLI flags; no model call.")
        return check_subscription(cwd=cwd, claude_bin=claude_bin)
    except SubscriptionError as error:
        # A refused preflight is known zero provider usage, not a missing paid receipt.
        raise ExecutorError(
            str(error),
            retryable=False,
            envelope=Envelope(
                "subscription preflight refused",
                None,
                "",
                cost_usd=0.0,
                api_calls=0,
                raw={"auth_preflight_refused": True},
            ),
        ) from None


def _normalize_cli_envelope(envelope: Envelope) -> Envelope:
    """Preserve CLI receipt normalization at the adapter boundary, including sparse receipts."""
    raw = envelope.raw
    envelope.raw = {**raw, "cost_unknown": raw.get("total_cost_usd") is None}
    if "modelUsage" in raw or raw.get("type") == "result" or ("model_ids" in raw and "usage" in raw):
        envelope.input_tokens, envelope.output_tokens, scope = claude_token_totals(
            raw.get("usage"), raw.get("modelUsage")
        )
        envelope.raw["token_scope"] = scope
    return envelope


def _cli_error_receipt(receipt: CliReceipt | None, elapsed: float) -> Envelope | None:
    """Retain a returned CLI receipt even when no usable action was produced."""
    if receipt is None:
        return None
    inp, out, scope = claude_token_totals(receipt.usage.document, receipt.model_usage)
    return Envelope(
        "",
        {},
        "",
        input_tokens=inp,
        output_tokens=out,
        cost_usd=receipt.cost_usd,
        duration_s=elapsed,
        api_calls=receipt.api_calls,
        usage=receipt.usage.document,
        raw={
            **receipt.document,
            "model_ids": receipt.model_ids,
            "token_scope": scope,
            "cost_unknown": receipt.cost_unknown,
        },
    )


def _tool_call_envelope(receipt: CliReceipt, elapsed: float) -> Envelope:
    """Score an attempted tool call as a no-op step rather than retrying the same prompt."""
    # The model tried to call a tool that does not exist (e.g. wrote an action as a tool call). Not
    # transient: retrying the same prompt repeats it. Return a no-op envelope so the step is scored as a
    # miss and the run continues; the log records why.
    return _normalize_cli_envelope(
        Envelope(
            reasoning=f"(model attempted a tool call instead of a text action: {str(receipt.result)[:200]})",
            state_patch={},
            action="Wait",
            input_tokens=receipt.usage.prompt_tokens,
            output_tokens=receipt.usage.completion_tokens,
            cost_usd=receipt.cost_usd,
            duration_s=elapsed,
            api_calls=receipt.api_calls,
            raw={**receipt.document, "model_error": True},
        )
    )


def _cli_envelope(receipt: CliReceipt, out: dict[str, Any], elapsed: float) -> Envelope:
    """Build the step's envelope from the response object and the receipt's usage and cost."""
    return _normalize_cli_envelope(
        Envelope(
            reasoning=str(out.get("reasoning", "")),
            state_patch=out.get("state_patch"),
            action=str(out.get("action", "")),
            extra={k: v for k, v in out.items() if k not in ("reasoning", "state_patch", "action")},
            input_tokens=receipt.usage.prompt_tokens,
            output_tokens=receipt.usage.completion_tokens,
            cost_usd=receipt.cost_usd,
            duration_s=elapsed,
            api_calls=receipt.api_calls,
            usage=receipt.usage.reported,
            raw={
                **receipt.document,
                "model_ids": receipt.model_ids,
            },
        )
    )


def _cli_failure_message(proc: subprocess.CompletedProcess[str]) -> str:
    """Keep both stream tails inside retry.py's 500-character journal error budget."""
    stdout = "\n".join((proc.stdout or "").splitlines()[-20:])[-180:]
    stderr = "\n".join((proc.stderr or "").splitlines()[-20:])[-180:]
    # A short stderr-only line already contains the entire diagnostic; preserve its existing spelling.
    if not stdout and stderr == (proc.stderr or "") and len(stderr.splitlines()) <= 1:
        return f"claude exited {proc.returncode}: {stderr}"
    lines = (proc.stderr or "").splitlines() + (proc.stdout or "").splitlines()
    # The first line the CLI printed, whatever it says: no line is recognised by its wording and skipped.
    headline = next((line.strip() for line in lines if line.strip()), "no error detail")
    headline = f"claude exited {proc.returncode}: {headline}"[:100]
    return f"{headline}\n[stdout tail]\n{stdout}\n[stderr tail]\n{stderr}"


@dataclass
class ClaudeCLIExecutor:
    """Runs ``claude -p`` once per step: fresh process, no session persistence, output constrained by
    ``--json-schema``. Each call is independent, which is the whole point. The session may read the project
    (read-only tools by default); every change goes through the returned action, which the harness runs,
    guards, journals and checkpoints."""

    model: str = "sonnet"
    tools: tuple[str, ...] = ("Read", "Grep", "Glob")
    timeout_s: float = 300.0
    claude_bin: str = "claude"
    extra_args: tuple[str, ...] = ()
    # Run in the environment directory because the controller treats the CLI's displayed cwd as its workdir.
    cwd: str | None = None

    def step(self, system_prompt: str, user_prompt: str, output_schema: dict[str, Any]) -> Envelope:
        cmd = self._command(strip_nulls(system_prompt), strip_nulls(user_prompt), output_schema)
        t0 = time.time()
        proc = self._run(cmd)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return self._exit_failure(proc, elapsed)
        receipt = self._receipt(proc)
        if receipt.is_error:
            return self._reported_error(receipt, elapsed)
        return _cli_envelope(receipt, self._structured_output(receipt, elapsed), elapsed)

    def _command(self, system_prompt: str, user_prompt: str, output_schema: dict[str, Any]) -> list[str]:
        """Assemble the headless argv, taking binary and settings from the subscription preflight when it applies."""
        auth = _subscription_preflight(claude_bin=self.claude_bin, custom_flags=bool(self.extra_args))
        return [
            auth["cli_path"] if auth else self.claude_bin,
            "-p",
            # A step loads the repository's own CLAUDE.md and .claude/settings.json, never the user's personal settings.
            "--setting-sources",
            "project",
            "--system-prompt",
            system_prompt,
            "--json-schema",
            json.dumps(output_schema),
            "--tools",
            ",".join(self.tools),
            "--no-session-persistence",
            "--output-format",
            "json",
            "--model",
            self.model,
            *self.extra_args,
            user_prompt,
        ]

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        """Run the CLI once under the step timeout, in the environment the controller's own session needs."""
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow nesting inside a CC session
        # Fresh calls rewrite state, so a longer cache TTL only increases write costs; operator overrides win.
        env.setdefault("CLAUDE_CODE_PROMPT_CACHE_TTL", "5m")
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=self.cwd or None,
            )
        except FileNotFoundError as exc:
            if exc.filename != cmd[0] or (self.cwd and not Path(self.cwd).is_dir()):
                raise
            raise MissingExecutorBinary(
                f"executor {cmd[0]!r} was not found; install it and make it available on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            # No final receipt: do not replay a potentially paid request automatically.
            raise ExecutorError(f"claude timed out after {self.timeout_s:.0f}s", retryable=False) from exc

    def _exit_failure(self, proc: subprocess.CompletedProcess[str], elapsed: float) -> Envelope:
        """Answer a nonzero exit as a scored tool-call no-op, a reported 429, or an error carrying its receipt."""
        # The CLI may exit 1 while still printing its JSON result (e.g. the model attempted a tool call with no
        # tools available). Parse it first so that case becomes a scored no-op instead of a retried "crash".
        try:
            document = json.loads(proc.stdout or "")
        except json.JSONDecodeError:
            document = None
        try:
            receipt = CliReceipt.from_document(document)
        except ValidationError as exc:
            # One unreadable field must not demote a rate limit: the 429 is read from the document, so this
            # stays the rate-limit failure the retry policy answers as one (``stepbook/runtime/retry.py``).
            cls = RateLimited if reports_rate_limit(document) else ExecutorError
            raise cls(
                f"claude exited {proc.returncode} with an unreadable JSON receipt: {first_problem(exc)}",
                retryable=False,
            ) from exc
        if receipt is not None:
            if receipt.is_error and receipt.attempted_tool_call:
                return _tool_call_envelope(receipt, elapsed)
            if receipt.rate_limited:
                raise RateLimited(
                    f"claude exited {proc.returncode}: {str(receipt.result)[:300]}",
                    envelope=_cli_error_receipt(receipt, elapsed),
                    retryable=False,
                )
        # Only the receipt's own 429 field says "rate limited"; error prose, however suggestive, is an
        # ordinary failure, and so is a silent exit.
        msg = (
            _cli_failure_message(proc)
            if receipt is None
            else f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[-2000:]}"
        )
        envelope = _cli_error_receipt(receipt, elapsed)
        raise ExecutorError(msg, envelope=envelope, retryable=envelope is None)

    def _receipt(self, proc: subprocess.CompletedProcess[str]) -> CliReceipt:
        """Parse the receipt a successful exit printed on stdout.

        Stdout that is not a receipt at all is a retryable transport-level failure, as it always was; stdout
        that is a JSON object this executor cannot read is not replayed, because the call it reports was paid.
        """
        try:
            document = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorError(f"non-JSON output from claude: {proc.stdout[:500]}") from exc
        try:
            receipt = CliReceipt.from_document(document)
        except ValidationError as exc:
            raise ExecutorError(f"unreadable JSON receipt from claude: {first_problem(exc)}", retryable=False) from exc
        if receipt is None:
            raise ExecutorError("claude printed a JSON value that is not a receipt object", retryable=False)
        return receipt

    def _reported_error(self, receipt: CliReceipt, elapsed: float) -> Envelope:
        """Answer a receipt that reports an error, scoring an attempted tool call as a no-op step.

        A successful exit reads the same structured field a failed exit does: the receipt's 429, never its prose.
        """
        msg = f"claude reported an error: {receipt.result}"
        if receipt.attempted_tool_call:
            return _tool_call_envelope(receipt, elapsed)
        cls = RateLimited if receipt.rate_limited else ExecutorError
        raise cls(msg, envelope=_cli_error_receipt(receipt, elapsed), retryable=False)

    def _structured_output(self, receipt: CliReceipt, elapsed: float) -> dict[str, Any]:
        """Take the response object, falling back to the text result for CLIs without structured output."""
        if receipt.structured_output is not None:
            return receipt.structured_output
        # Fall back to parsing the text result as JSON (older CLIs).
        try:
            out = json.loads(receipt.result or "")
        except json.JSONDecodeError as exc:
            raise ExecutorError(
                f"no structured_output in response: {str(receipt.document)[:500]}",
                envelope=_cli_error_receipt(receipt, elapsed),
                retryable=False,
            ) from exc
        if not isinstance(out, dict):  # freshly decoded text: this is the boundary, not a re-check behind one
            raise ExecutorError(
                "CLI structured output must be an object",
                envelope=_cli_error_receipt(receipt, elapsed),
                retryable=False,
            )
        return out
