"""Markdown and live traces must distinguish missing usage from measured zero."""

import json
import shlex
import sys
from types import SimpleNamespace

import pytest

from stepbook import CommandExecutor, Runtime, Skill
from stepbook.cli.inspection import write_report_md
from stepbook.reporting.usage import usage_summary
from stepbook.runtime.logging_setup import trace_step
from stepbook.runtime.loop import StepResult


def _receipt(step, tokens=0, **extra):
    return {
        "kind": "model_call",
        "step": step,
        "input_tokens": tokens,
        "output_tokens": 0,
        "cost_usd": 0,
        **extra,
    }


@pytest.mark.parametrize(
    "events,total,expected",
    [
        ([_receipt(0, usage_unknown=True)], 0, "unknown"),
        ([_receipt(0, None, output_tokens=None)], 0, "unknown"),
        ([{"kind": "step", "step": 0}], 0, "unknown"),
        ([_receipt(0, 1234), _receipt(1, usage_unknown=True)], 1234, "1,234 (partial)"),
        ([_receipt(0), _receipt(1, usage_unknown=True)], 0, "0 (partial)"),
        ([_receipt(0, token_scope="main_loop_only")], 0, "0 (partial)"),
        ([_receipt(0, 1234, token_scope="main_loop_only")], 1234, "1,234 (partial)"),
        ([_receipt(0)], 0, "0"),
        ([_receipt(0, 1234), _receipt(1, 10)], 1244, "1,244"),
        ([], 0, "0"),
        ([{"kind": "executor_error"}], 0, "unknown"),
    ],
)
def test_report_token_text_and_unchanged_receipts(tmp_path, events, total, expected):
    journal = tmp_path / "events.jsonl"
    original = "".join(json.dumps(event) + "\n" for event in events)
    journal.write_text(original)
    usage_bytes = json.dumps(usage_summary(events), indent=2) + "\n"
    result = SimpleNamespace(cost_usd=0, total_tokens=total)
    report = write_report_md(tmp_path, object(), result).read_text()
    assert f"; measured tokens: {expected}\n" in report
    assert f" | agent tokens: {expected} | ended:" in report
    assert journal.read_text() == original
    assert (tmp_path / "usage.json").read_text() == usage_bytes


def test_measured_report_figures_keep_their_distinct_totals(tmp_path):
    events = [
        _receipt(0, 1234),
        {"kind": "environment_model_call", "input_tokens": 10, "output_tokens": 2, "cost_usd": 0},
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    report = write_report_md(tmp_path, object(), SimpleNamespace(cost_usd=0, total_tokens=1234)).read_text()
    assert report.splitlines()[2] == "- reported usage USD subtotal: $0.0000; measured tokens: 1,246"
    assert report.splitlines()[7] == (
        "- steps: 0 | agent cost: $0.00 | agent tokens: 1,234 | ended: no terminal event recorded"
    )


@pytest.mark.parametrize("tokens,expected", [(None, "unknown"), (0, "0"), (1234, "1234")])
def test_trace_text(capsys, tokens, expected):
    trace_step(True, 7, "echo 'hello'", "x" * 90, tokens)
    assert capsys.readouterr().out == f"""[  7] "echo 'hello'" -> {"x" * 80!r}  (prompt {expected} tok)\n"""


class _Environment:
    def reset(self):
        return "ready"

    def score(self):
        return 0.0

    def step(self, action):
        return StepResult("finished", done=True)


@pytest.mark.parametrize("runtime", [Runtime])
@pytest.mark.parametrize("tokens,expected", [(None, "unknown"), (0, "0"), (1234, "1234")])
def test_executor_usage_reaches_live_trace(tmp_path, monkeypatch, capsys, runtime, tokens, expected):
    payload = {"reasoning": "offline", "state_patch": {}, "action": "stop"}
    if tokens is not None:
        payload["usage"] = {"input_tokens": tokens, "output_tokens": 0, "cost_usd": 0}
    agent = tmp_path / "agent.py"
    agent.write_text(f"print({json.dumps(payload)!r})\n")
    executor = CommandExecutor(command=shlex.join([sys.executable, str(agent)]))
    skill = Skill("trace", "Stop.", {"type": "object", "properties": {}}, {})
    result = runtime(skill, executor, _Environment(), log_dir=tmp_path / "run", verbose=True).run(1)
    assert result.steps == 1
    assert capsys.readouterr().out == f"[  0] 'stop' -> 'finished'  (prompt {expected} tok)\n"
