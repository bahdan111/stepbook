import json
import subprocess
from types import SimpleNamespace

import pytest

from stepbook.contracts.errors import ExecutorError
from stepbook.executors.claude import ClaudeCLIExecutor
from stepbook.reporting.usage import envelope_usage


@pytest.mark.parametrize(
    "exit_code,result",
    [
        (1, {"is_error": True, "result": "budget exceeded"}),
        (0, {"is_error": True, "result": "budget exceeded"}),
        (0, {"result": "not JSON"}),
        (0, {"result": "[]"}),
    ],
)
def test_cli_failure_preserves_complete_receipt(monkeypatch, exit_code, result):
    data = {
        **result,
        "type": "result",
        "total_cost_usd": 0.03,
        "num_turns": 2,
        "modelUsage": {
            "haiku": {"inputTokens": 10, "outputTokens": 2, "cacheReadInputTokens": 3, "cacheCreationInputTokens": 4}
        },
    }
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=exit_code, stdout=json.dumps(data), stderr="")
    )
    with pytest.raises(ExecutorError) as error:
        ClaudeCLIExecutor().step("", "", {})
    assert not error.value.retryable
    usage = envelope_usage(error.value.envelope)
    assert usage["input_tokens"] + usage["output_tokens"] == 19
    assert usage["cost_usd"] == 0.03 and usage["token_scope"] == "all_reported_models"
    assert error.value.envelope.raw["result"] == result["result"]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_error_mentioning_a_toolchain_is_not_a_tool_call(monkeypatch, exit_code):
    """A tool call is the receipt's stop reason; prose about a toolchain is an ordinary paid failure."""
    data = {"type": "result", "is_error": True, "result": "toolchain missing", "total_cost_usd": 0.03}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=exit_code, stdout=json.dumps(data), stderr="")
    )
    with pytest.raises(ExecutorError) as error:
        ClaudeCLIExecutor().step("", "", {})
    assert not error.value.retryable
    assert error.value.envelope.action != "Wait" and error.value.envelope.raw["result"] == "toolchain missing"


def test_cli_timeout_is_not_automatically_replayed(monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("claude", 120)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ExecutorError) as error:
        ClaudeCLIExecutor(timeout_s=120).step("", "", {})
    assert not error.value.retryable and error.value.envelope is None


def test_cli_error_without_usage_does_not_invent_complete_cost(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout=json.dumps({"type": "result", "is_error": True, "result": "failed"}), stderr=""
        ),
    )
    with pytest.raises(ExecutorError) as error:
        ClaudeCLIExecutor().step("", "", {})
    usage = envelope_usage(error.value.envelope)
    assert usage["cost_usd"] is None and usage["token_scope"] == "main_loop_only"


@pytest.mark.parametrize(
    "exit_code,sentence",
    [(0, "unreadable JSON receipt from claude"), (1, "claude exited 1 with an unreadable JSON receipt")],
)
def test_unreadable_receipt_is_refused_without_the_cost_it_reported(monkeypatch, exit_code, sentence):
    """A receipt whose JSON
    fails the receipt model is refused at either exit status with one sentence and no envelope, so the cost
    and usage it reported are not journaled (they are counted as unknown, never as zero)."""
    data = {"type": "result", "result": 123, "total_cost_usd": 0.03, "num_turns": 2}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=exit_code, stdout=json.dumps(data), stderr="")
    )
    with pytest.raises(ExecutorError) as error:
        ClaudeCLIExecutor().step("", "", {})
    assert str(error.value) == f"{sentence}: result: Input should be a valid string"
    assert error.value.envelope is None and not error.value.retryable


@pytest.mark.parametrize("mode", ["success", "tool", "error"])
@pytest.mark.parametrize("cost", [None, 0.0, 0.2])
@pytest.mark.parametrize("whole_query", [False, True])
def test_cli_normalizes_every_receipt_boundary(monkeypatch, mode, cost, whole_query):
    from dataclasses import asdict, replace

    data = {
        "type": "result",
        "usage": {
            "input_tokens": 1,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
            "output_tokens": 2,
        },
        "total_cost_usd": cost,
        "num_turns": 2,
        "structured_output": {"reasoning": "r", "state_patch": {"x": 1}, "action": "stop"},
    }
    if whole_query:
        data["modelUsage"] = {
            "main": {"inputTokens": 1, "cacheReadInputTokens": 3, "cacheCreationInputTokens": 4, "outputTokens": 2},
            "aux": {"inputTokens": 5, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0, "outputTokens": 6},
        }
    if mode != "success":
        data.update(is_error=True, result="tool call attempted" if mode == "tool" else "paid failure")
    if mode == "tool":
        data["stop_reason"] = "tool_use"  # the provider's own stop reason marks a tool call, not the prose
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(data)))
    if mode == "error":
        with pytest.raises(ExecutorError) as caught:
            ClaudeCLIExecutor().step("", "", {})
        assert not caught.value.retryable
        envelope = caught.value.envelope
    else:
        envelope = ClaudeCLIExecutor().step("", "", {})
    assert (envelope.input_tokens, envelope.output_tokens) == ((13, 8) if whole_query else (8, 2))
    assert envelope.cost_usd == (cost or 0.0) and envelope.raw["cost_unknown"] == (cost is None)
    assert envelope.raw["token_scope"] == ("all_reported_models" if whole_query else "main_loop_only")
    assert envelope.api_calls == 2 and envelope.action == {"success": "stop", "tool": "Wait", "error": ""}[mode]
    assert all(envelope.raw[key] == value for key, value in data.items())
    changed = replace(envelope, input_tokens=77)
    assert asdict(changed) == {**asdict(envelope), "input_tokens": 77}


@pytest.mark.parametrize("mode", ["success", "tool"])
@pytest.mark.parametrize("usage_present", [False, True])
def test_sparse_cli_receipts_mark_missing_cost_unknown_without_changing_tokens(monkeypatch, mode, usage_present):
    data = {"structured_output": {"reasoning": "r", "state_patch": {}, "action": "stop"}}
    if mode == "tool":
        data.update(is_error=True, result="tool call attempted", stop_reason="tool_use")
    if usage_present:
        data["usage"] = {"input_tokens": 5, "output_tokens": 2}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(data)))
    envelope = ClaudeCLIExecutor().step("", "", {})
    assert (envelope.input_tokens, envelope.output_tokens) == ((5, 2) if usage_present else (0, 0))
    # Missing price must be unknown in
    # every receipt, while token-scope metadata keeps its existing sparse behavior.
    normalized = mode == "success" and usage_present
    assert ("token_scope" in envelope.raw) is normalized
    assert envelope.raw["cost_unknown"] is True
    assert envelope_usage(envelope)["cost_usd"] is None
    if normalized:
        assert envelope.raw["token_scope"] == "main_loop_only" and envelope.raw["cost_unknown"]
