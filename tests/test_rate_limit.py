"""A rate limit is what the provider's receipt says it is, and it is never waited out.

Prose and empty output do not establish a rate limit; only
``api_error_status == 429`` makes a ``RateLimited``, and the controller records the failed call once and
re-raises it so the operator chooses when to resume.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from stepbook import Envelope
from stepbook.contracts.errors import ExecutorError, RateLimited
from stepbook.executors.claude import ClaudeCLIExecutor
from stepbook.runtime.loop import STATE_OUTPUT_SCHEMA, call_with_retry
from stepbook.storage.events import EventLog


def _completed(returncode, stdout="", stderr=""):
    return lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    "stderr",
    [
        "HTTP 429 rate_limit",
        "You've hit your session limit",
        "status 429",
        "the upstream is overloaded",
        "quota exhausted for this account",
        "usage limit reached",
        "processed 429 records",
        "some other crash",
        "",
    ],
    ids=[
        "http-429-prose",
        "session-limit",
        "status-429",
        "overloaded",
        "quota",
        "usage-limit",
        "429-in-data",
        "unrelated",
        "silent-exit",
    ],
)
def test_error_prose_and_silent_exits_are_ordinary_errors(monkeypatch, stderr):
    """No message, however suggestive, is evidence of a provider status; a silent exit is not one either."""
    monkeypatch.setattr(subprocess, "run", _completed(1, stderr=stderr))
    with pytest.raises(ExecutorError) as caught:
        ClaudeCLIExecutor().step("s", "u", {})
    assert type(caught.value) is ExecutorError
    assert str(caught.value) == f"claude exited 1: {stderr}"
    assert caught.value.envelope is None and caught.value.retryable is True


@pytest.mark.parametrize("exit_code", [0, 1], ids=["reported-error", "nonzero-exit"])
def test_structured_429_with_a_usable_receipt_is_a_rate_limit(monkeypatch, exit_code):
    """The receipt's own field decides at either exit status; its prose says nothing about a rate limit."""
    document = {
        "type": "result",
        "is_error": True,
        "result": "the provider refused this call",
        "api_error_status": 429,
        "total_cost_usd": 0.02,
        "num_turns": 2,
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    monkeypatch.setattr(subprocess, "run", _completed(exit_code, stdout=json.dumps(document)))
    with pytest.raises(RateLimited) as caught:
        ClaudeCLIExecutor().step("s", "u", {})
    assert caught.value.retryable is False
    envelope = caught.value.envelope
    assert envelope is not None and envelope.cost_usd == 0.02 and envelope.api_calls == 2
    assert (envelope.input_tokens, envelope.output_tokens) == (7, 3)
    assert envelope.raw["api_error_status"] == 429


@pytest.mark.parametrize("exit_code", [0, 1], ids=["reported-error", "nonzero-exit"])
def test_a_receipt_without_the_429_field_is_an_ordinary_error_whatever_it_says(monkeypatch, exit_code):
    """The converse of the pin above: a paid receipt that only *reads* like a quota failure is not one."""
    document = {
        "type": "result",
        "is_error": True,
        "result": "usage limit reached for this session",
        "total_cost_usd": 0.02,
        "num_turns": 2,
    }
    monkeypatch.setattr(subprocess, "run", _completed(exit_code, stdout=json.dumps(document)))
    with pytest.raises(ExecutorError) as caught:
        ClaudeCLIExecutor().step("s", "u", {})
    assert type(caught.value) is ExecutorError
    assert caught.value.retryable is False and caught.value.envelope.cost_usd == 0.02


def test_structured_429_survives_a_receipt_this_harness_cannot_read(monkeypatch):
    """One unreadable field must not demote a paid 429 to an ordinary failure; its cost stays unknown."""
    document = {"type": "result", "result": 123, "api_error_status": 429, "total_cost_usd": 0.02}
    monkeypatch.setattr(subprocess, "run", _completed(1, stdout=json.dumps(document)))
    with pytest.raises(RateLimited) as caught:
        ClaudeCLIExecutor().step("s", "u", {})
    assert (
        str(caught.value) == "claude exited 1 with an unreadable JSON receipt: result: Input should be a valid string"
    )
    assert caught.value.envelope is None and caught.value.retryable is False


@pytest.mark.parametrize("retryable", [False, True])
def test_rate_limit_records_one_receipt_and_re_raises_without_sleeping(tmp_path, monkeypatch, retryable):
    """Whatever the executor said about replaying the call, a 429 is never waited out or retried here."""
    slept, guards = [], []
    monkeypatch.setattr("stepbook.runtime.retry.time.sleep", slept.append)
    log = EventLog(tmp_path / "events.jsonl")
    receipt = Envelope("", {}, "", input_tokens=7, output_tokens=3, cost_usd=0.02, api_calls=2)
    limit = RateLimited("claude exited 1: rate limited", envelope=receipt, retryable=retryable)

    class Ex:
        calls = 0

        def step(self, sp, up, schema):
            Ex.calls += 1
            raise limit

    with pytest.raises(RateLimited) as caught:
        call_with_retry(
            Ex(),
            "s",
            "u",
            STATE_OUTPUT_SCHEMA,
            log=log,
            step=3,
            before_call=lambda: guards.append("checked"),
        )
    assert caught.value is limit and caught.value.envelope is receipt  # the original, not a summary of it
    assert Ex.calls == 1 and slept == [] and guards == ["checked"]
    events = log.read()
    assert [e["kind"] for e in events] == ["failed_model_call"]
    assert (events[0]["step"], events[0]["cost_usd"], events[0]["usage_unknown"]) == (3, 0.02, False)
    assert (events[0]["input_tokens"], events[0]["output_tokens"]) == (7, 3)


def test_rate_limit_without_a_receipt_records_unknown_usage_once(tmp_path, monkeypatch):
    monkeypatch.setattr("stepbook.runtime.retry.time.sleep", lambda _: pytest.fail("a rate limit is never waited out"))
    log = EventLog(tmp_path / "events.jsonl")

    class Ex:
        def step(self, sp, up, schema):
            raise RateLimited("claude exited 1: rate limited", retryable=True)

    with pytest.raises(RateLimited) as caught:
        call_with_retry(Ex(), "s", "u", STATE_OUTPUT_SCHEMA, log=log, step=1)
    # ``retryable`` is the executor's word about replaying the call, not a licence to wait out the window.
    assert caught.value.retryable is True and caught.value.envelope is None
    events = log.read()
    assert [e["kind"] for e in events] == ["failed_model_call"]
    assert events[0]["usage_unknown"] is True and events[0]["cost_usd"] is None


def test_transient_failures_keep_their_bounded_short_backoff(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr("stepbook.runtime.retry.time.sleep", slept.append)
    log = EventLog(tmp_path / "events.jsonl")

    class Ex:
        calls = 0

        def step(self, sp, up, schema):
            Ex.calls += 1
            raise ExecutorError("offline crash")

    with pytest.raises(ExecutorError) as caught:
        call_with_retry(Ex(), "s", "u", STATE_OUTPUT_SCHEMA, log=log, step=2, attempts=3)
    assert type(caught.value) is ExecutorError
    assert str(caught.value) == "executor failed 3 times: offline crash"
    # Exhausting the retry schedule must produce a non-retryable error.
    assert caught.value.retryable is False and caught.value.envelope is None
    assert Ex.calls == 3 and slept == [2, 4]
    kinds = [e["kind"] for e in log.read()]
    assert kinds == ["failed_model_call", "executor_error"] * 3


def test_pause_during_a_transient_retry_stops_before_the_next_wait(tmp_path, monkeypatch):
    slept, guards = [], []
    monkeypatch.setattr("stepbook.runtime.retry.time.sleep", slept.append)
    log = EventLog(tmp_path / "events.jsonl")

    class Paused(Exception):
        pass

    def guard():
        guards.append("checked")
        if len(guards) == 2:  # the boundary check the retry path makes after logging the failure
            raise Paused("operator paused")

    class Ex:
        calls = 0

        def step(self, sp, up, schema):
            Ex.calls += 1
            raise ExecutorError("offline crash")

    with pytest.raises(Paused):
        call_with_retry(Ex(), "s", "u", STATE_OUTPUT_SCHEMA, log=log, step=4, attempts=3, before_call=guard)
    assert Ex.calls == 1 and slept == [] and len(guards) == 2
    assert [e["kind"] for e in log.read()] == ["failed_model_call", "executor_error"]


def test_cli_timeout_is_a_non_retryable_executor_error(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    calls = []
    monkeypatch.setattr("stepbook.runtime.retry.time.sleep", calls.append)
    try:
        ClaudeCLIExecutor(timeout_s=1).step("s", "u", {"type": "object"})
        raise AssertionError
    except ExecutorError as e:
        assert "timed out" in str(e) and not isinstance(e, RateLimited)
        assert type(e) is ExecutorError and str(e) == "claude timed out after 1s"
        assert e.retryable is False and e.envelope is None

    # The policy is not to retry a timeout, so call_with_retry must re-raise it
    # after exactly one attempt and no sleep.
    executor = ClaudeCLIExecutor(timeout_s=1)
    attempts = []
    step = executor.step
    monkeypatch.setattr(executor, "step", lambda *a, **k: attempts.append(a) or step(*a, **k))
    with pytest.raises(ExecutorError, match="^claude timed out after 1s$"):
        call_with_retry(executor, "s", "u", STATE_OUTPUT_SCHEMA, attempts=5)
    assert len(attempts) == 1 and calls == []
