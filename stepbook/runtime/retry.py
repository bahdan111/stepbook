"""Retry policy for a controller call: short backoff for transient executor failures, no retry for rate limits."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, NoReturn, Unpack

from ..contracts.errors import ExecutorError, RateLimited
from ..executors.base import Envelope, StepExecutor
from ..reporting.usage import envelope_usage
from ..state.response_validation import normalize_report_envelope
from ..storage.events import EventLog
from .settings import RetryKwargs, RetryPolicy


def call_with_retry(
    executor: StepExecutor,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    *,
    policy: RetryPolicy | None = None,
    output_check: Callable[..., Sequence[str]] | None = None,
    **overrides: Unpack[RetryKwargs],
) -> Envelope:
    """Retry transient executor failures (CLI crash, network) with short backoff; a rate limit is not retried
    at all, because waiting one out costs hours the operator did not choose to spend. A model that answers
    badly is handled by the validator, not here.

    ``overrides`` are the fields of ``RetryPolicy``, which is also accepted whole as ``policy``."""
    settings = RetryPolicy.resolve(policy, overrides)
    transient = 0
    while True:
        if settings.before_call is not None:
            settings.before_call()
        try:
            envelope = executor.step(system_prompt, user_prompt, schema)
            if "report" in schema.get("properties", {}):
                envelope = normalize_report_envelope(envelope, schema, output_check)
            return envelope
        except RateLimited as exc:  # pragma: no cover - exercised live
            _fail_rate_limit(settings, exc)
        except ExecutorError as exc:  # pragma: no cover - exercised live
            transient = _wait_out_transient_failure(settings, exc, transient)


def _log_failed_call(log: EventLog, step: int, envelope: Envelope | None) -> None:
    """Record a failed call's provider usage, or that the provider reported none."""
    if envelope is not None:
        log.append("failed_model_call", step=step, **envelope_usage(envelope))
    else:
        log.append(
            "failed_model_call",
            step=step,
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            usage_unknown=True,
        )


def _fail_rate_limit(settings: RetryPolicy, exc: RateLimited) -> NoReturn:
    """Record the failed call's receipt once, then re-raise the provider's own error unchanged.

    The run stays resumable from its event log, so the operator decides when to come back; the harness does
    not sleep through a quota window on their behalf, and invents no zero-wait ``rate_limited`` event.
    """
    if settings.log:
        _log_failed_call(settings.log, settings.step, exc.envelope)
    raise exc


def _wait_out_transient_failure(settings: RetryPolicy, exc: ExecutorError, transient: int) -> int:
    """Log the transient failure, sleep a linear backoff and return the new count, or raise when out of retries."""
    transient += 1
    if settings.log:
        _log_failed_call(settings.log, settings.step, exc.envelope)
        settings.log.append(
            "executor_error",
            step=settings.step,
            attempt=transient,
            error=str(exc)[:500],
            usage_reported=exc.envelope is not None,
        )
    if not exc.retryable:
        raise exc
    if settings.before_call is not None:
        settings.before_call()
    if transient >= settings.attempts:
        # Implicit chaining is intended: the raised error's ``__context__`` stays the handled ``ExecutorError``.
        # Not retryable: the schedule is spent, and the flag is read at `:80` to decide exactly that.
        raise ExecutorError(f"executor failed {settings.attempts} times: {exc}", retryable=False)
    time.sleep(2 * transient)
    return transient
