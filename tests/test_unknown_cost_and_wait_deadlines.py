"""Missing prices stay unknown; waits observe conditions through their deadline."""

import json
import subprocess
import time
from dataclasses import replace

import pytest

from stepbook import Runtime, Skill
from stepbook.contracts.errors import ExecutorError
from stepbook.envs.shell import CommandRun, ShellEnv
from stepbook.executors.base import Envelope
from stepbook.executors.claude import ClaudeCLIExecutor
from stepbook.executors.command import ScriptedExecutor
from stepbook.reporting.usage import envelope_usage, usage_summary
from stepbook.storage.events import EventLog
from tests._public_environment import FixtureEnv


@pytest.mark.parametrize("mode", ["success", "tool", "error"])
@pytest.mark.parametrize("price", [{}, {"total_cost_usd": None}, {"total_cost_usd": 0.0}, {"total_cost_usd": 0.2}])
def test_sparse_cli_price_is_unknown_unless_reported(monkeypatch, mode, price):
    data = {"structured_output": {"reasoning": "r", "state_patch": {}, "action": "stop"}, **price}
    if mode != "success":
        data.update(is_error=True, result="tool call attempted" if mode == "tool" else "paid failure")
    if mode == "tool":
        data["stop_reason"] = "tool_use"  # the provider's own stop reason marks a tool call, not the prose
    monkeypatch.setattr(
        ClaudeCLIExecutor, "_run", lambda *args: subprocess.CompletedProcess([], 0, json.dumps(data), "")
    )
    if mode == "error":
        with pytest.raises(ExecutorError) as caught:
            ClaudeCLIExecutor().step("", "", {})
        envelope = caught.value.envelope
    else:
        envelope = ClaudeCLIExecutor().step("", "", {})
    expected = price.get("total_cost_usd")
    receipt = envelope_usage(envelope)
    assert receipt["cost_usd"] == expected
    summary = usage_summary([{"kind": "model_call", **receipt}])
    assert summary["cost_complete"] is (expected is not None)
    assert summary["components"]["controller"]["unknown_cost_records"] == int(expected is None)


@pytest.mark.parametrize("scripted", [False, True])
@pytest.mark.parametrize("price", [{}, {"cost_usd": None}, {"cost_usd": 0.0}, {"cost_usd": 0.2}])
def test_custom_and_scripted_envelopes_do_not_invent_a_price(scripted, price):
    envelope = Envelope("reason", {}, "Wait", **price)
    if scripted:
        envelope = ScriptedExecutor(lambda system, user: envelope).step("system", "user", {})
        assert envelope.input_tokens > 0 and envelope.output_tokens > 0
    expected = price.get("cost_usd")
    receipt = envelope_usage(envelope)
    assert receipt["cost_usd"] == expected
    assert usage_summary([{"kind": "model_call", **receipt}])["cost_complete"] is (expected is not None)


@pytest.mark.parametrize("price", [0.0, 0.2])
def test_replacement_price_resolves_default_unknown_without_changing_original(price):
    raw = {"model_ids": ["scripted"]}
    original = Envelope("reason", {}, "Wait", raw=raw)
    unpriced_copy = replace(original, action="translated")
    assert envelope_usage(unpriced_copy)["cost_usd"] is None
    priced = replace(unpriced_copy, cost_usd=price)
    assert envelope_usage(priced)["cost_usd"] == price
    assert priced.raw == raw == {"model_ids": ["scripted"]}
    assert envelope_usage(original)["cost_usd"] is None
    assert envelope_usage(unpriced_copy)["cost_usd"] is None
    assert envelope_usage(replace(priced, action="again"))["cost_usd"] == price
    assert envelope_usage(replace(priced, cost_usd=None))["cost_usd"] is None
    summary = usage_summary([{"kind": "model_call", **envelope_usage(priced)}])
    assert summary["cost_complete"] is True
    assert summary["known_total_usd"] == price
    assert summary["components"]["controller"]["unknown_cost_records"] == 0


@pytest.mark.parametrize("price", [0.0, 0.2])
@pytest.mark.parametrize("flag", ["cost_unknown", "usage_unknown"])
def test_replacement_price_preserves_explicit_unknown_flags(price, flag):
    raw = {flag: True}
    original = Envelope("reason", {}, "Wait", raw=raw)
    for envelope in (original, Envelope("reason", {}, "Wait")):
        priced = replace(envelope, cost_usd=price, raw=raw)
        assert envelope_usage(priced)["cost_usd"] is None
        assert usage_summary([{"kind": "model_call", **envelope_usage(priced)}])["cost_complete"] is False
    assert envelope_usage(replace(original, cost_usd=price))["cost_usd"] is None
    assert raw == {flag: True}


@pytest.mark.parametrize("runtime_type", [Runtime])
@pytest.mark.parametrize("price", [0.0, 0.2])
def test_replacement_price_survives_runtime_finalization(tmp_path, runtime_type, price):
    result = runtime_type(
        Skill("fixture", "Inspect the fixture.", {"type": "object"}, {}),
        ScriptedExecutor(lambda system, user: replace(Envelope("offline", {}, "Wait"), cost_usd=price)),
        FixtureEnv(horizon=1, seed=1),
        log_dir=tmp_path,
    ).run(1)
    summary = result.summary()
    assert summary["cost_usd"] == price and summary["cost_complete"] is True
    events = EventLog(tmp_path / "events.jsonl").read()
    assert usage_summary(events)["components"]["controller"]["unknown_cost_records"] == 0


def test_unpriced_scripted_run_journals_null_receipt_and_incomplete_cost(tmp_path):
    result = Runtime(
        Skill("fixture", "Inspect the fixture.", {"type": "object"}, {}),
        ScriptedExecutor(lambda system, user: Envelope("offline", {}, "Wait")),
        FixtureEnv(horizon=1, seed=1),
        log_dir=tmp_path,
    ).run(1)
    events = EventLog(tmp_path / "events.jsonl").read()
    receipts = [event for event in events if event["kind"] in {"model_call", "step"}]
    assert len(receipts) == 2
    assert receipts[0]["kind"] == "model_call" and receipts[0]["cost_usd"] is None
    # The runtime still copies the numeric backing into step records (outside this slice).
    # Modern accounting uses the call receipt, never that placeholder as a known price.
    assert receipts[1]["kind"] == "step" and receipts[1]["cost_usd"] == 0.0
    assert result.cost_complete is False
    assert usage_summary(events)["cost_complete"] is False


@pytest.mark.parametrize("duration,oversleep", [(2.0, 0.0), (2.5, 0.0), (2.0, 0.01)])
@pytest.mark.parametrize("ready", [True, False])
def test_wait_polls_at_deadline_even_after_short_final_interval(tmp_path, monkeypatch, duration, oversleep, ready):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={}, checkpoint=False)
    clock = [0.0]
    polls = []
    sleeps = []
    monkeypatch.setattr("stepbook.envs.shell.time.monotonic", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds + oversleep

    def poll(command, *, timeout_s=None):
        polls.append((clock[0], timeout_s))
        return CommandRun(0, "ready") if ready and clock[0] >= duration - 0.25 else CommandRun(1, "pending")

    monkeypatch.setattr("stepbook.envs.shell.time.sleep", sleep)
    monkeypatch.setattr(env, "_run", poll)
    result = env.step(f"wait {duration}s until test -f ready every 1s")
    assert polls[-1][0] == pytest.approx(duration + oversleep)
    assert len(polls) == (3 if duration == 2 else 4)
    assert all(0 < limit <= 1 for _, limit in polls)
    assert sleeps[-1] <= 1
    assert result.info["wait"] == ("condition met" if ready else "timed out")
    assert result.info["rc"] == (0 if ready else 1)


def test_deadline_poll_has_a_bounded_timeout(tmp_path, monkeypatch):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={}, cmd_timeout_s=0.25)
    clock = [0.0]
    limits = []
    monkeypatch.setattr("stepbook.envs.shell.time.monotonic", lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds

    def poll(command, *, timeout_s=None):
        limits.append(timeout_s)
        if clock[0] < 2:
            return CommandRun(1, "pending")
        clock[0] += timeout_s
        return CommandRun(124, f"(timed out after {timeout_s:.1f}s; the command's process group was killed)", True)

    monkeypatch.setattr("stepbook.envs.shell.time.sleep", sleep)
    monkeypatch.setattr(env, "_run", poll)
    result = env.step("wait 2s until hanging-command every 1s")
    assert limits == [0.25, 0.25, 0.25]
    assert clock[0] == 2.25
    assert result.info["wait"] == "timed out" and result.info["rc"] == 124
    assert "poll 3 timed out" in result.observation


def test_wait_sees_real_file_created_between_last_interval_and_deadline(tmp_path):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={}, checkpoint=False)
    with subprocess.Popen(["sh", "-c", "sleep 1.5; touch ready"], cwd=tmp_path) as writer:
        started = time.monotonic()
        result = env.step("wait 2s until test -f ready every 1s")
        elapsed = time.monotonic() - started
        writer.wait(timeout=3)
    assert result.info["wait"] == "condition met"
    assert 1.5 <= elapsed < 3.5
