"""Unknown token displays and terminal-sized report tables, without changing receipts."""

import argparse
import json
import os
import shutil
import sys

import pytest

from stepbook.cli import inspection as cli_inspect
from stepbook.reporting.usage import usage_summary


def _step(number, tokens=0, **extra):
    return {
        "kind": "step",
        "step": number,
        "action": "echo " + "a" * 50,
        "result": "r" * 80,
        "prompt_tokens": tokens,
        "output_tokens": 0,
        "cost_usd": 0,
        "rendered_prompt_chars": 123,
        **extra,
    }


@pytest.mark.parametrize("command", [cli_inspect.cmd_report, cli_inspect.cmd_status])
@pytest.mark.parametrize(
    "events,expected",
    [
        ([_step(0, usage_unknown=True)], "unknown"),
        ([_step(0, None, output_tokens=None)], "unknown"),
        ([{"kind": "step", "step": 0}], "unknown"),
        ([_step(0, 1234), _step(1, usage_unknown=True)], "1,234 (partial)"),
        ([_step(0), _step(1, usage_unknown=True)], "0 (partial)"),
        ([_step(0)], "0"),
        ([_step(0, 1234), _step(1, 10)], "1,244"),
        ([], "0"),
        ([_step(0, token_scope="main_loop_only")], "0 (partial)"),
        ([{"kind": "executor_error"}], "unknown"),
        ([{"kind": "environment_model_call", "prompt_tokens": 17, "output_tokens": 0}], "unknown"),
        (
            [
                {
                    "kind": "environment_model_call",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usage_unknown": True,
                    "token_scope": "main_loop_only",
                }
            ],
            "0 (partial)",
        ),
        (
            [
                {"kind": "failed_model_call", "step": 0, "usage_unknown": True},
                {"kind": "step", "step": 0, "prompt_tokens": 0, "output_tokens": 0, "token_scope": "main_loop_only"},
            ],
            "0 (partial)",
        ),
        ([_step(0), {"kind": "executor_error"}], "0 (partial)"),
        (
            [_step(0, usage_unknown=True, action="run baseline", result='Eval finished:\n{"cost":{"known_tokens":0}}')],
            "0 (partial)",
        ),
    ],
)
def test_token_totals_are_qualified_without_writing(tmp_path, monkeypatch, capsys, command, events, expected):
    monkeypatch.setattr(cli_inspect, "pointer_dir", lambda: tmp_path)
    path = tmp_path / "events.jsonl"
    original = "".join(json.dumps(event) + "\n" for event in events)
    path.write_text(original)
    accounting = usage_summary(events)
    assert command(argparse.Namespace(run_dir=str(tmp_path), last=20)) == 0
    assert f"; tokens: {expected}; complete:" in capsys.readouterr().out
    assert path.read_text() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["events.jsonl"]
    assert usage_summary(events) == accounting


@pytest.mark.parametrize(
    "steps,expected",
    [
        ([_step(0, usage_unknown=True)], "unknown/unknown/unknown"),
        ([_step(0, None)], "unknown/unknown/unknown"),
        ([{"kind": "step", "step": 0}], "unknown/unknown/unknown"),
        ([_step(0, 30), _step(1, 10), _step(2, 20)], "30/20/20"),
        ([_step(0, 30), _step(1, 10)], "30/30/10"),
        ([_step(0)], "0/0/0"),
        ([_step(0, 30), _step(1, usage_unknown=True), _step(2, 10)], "30/unknown/10"),
        ([_step(0, usage_unknown=True), _step(1, 0)], "unknown/unknown/0"),
    ],
)
def test_prompt_trend_preserves_measured_values(capsys, steps, expected):
    cli_inspect._print_run_tail(argparse.Namespace(run_dir="run", last=20), steps, steps)
    output = capsys.readouterr().out
    assert f"prompt tokens (provider) first/median/last: {expected} |" in output
    rows = output.splitlines()[-len(steps) :]
    for step, row in zip(steps, rows, strict=True):
        known = not step.get("usage_unknown") and step.get("prompt_tokens") is not None
        assert row.split()[-1] == (str(step["prompt_tokens"]) if known else "unknown")


@pytest.mark.parametrize("width", [100, 80, 40, 20, 10, 5, 112, 113, 140])
@pytest.mark.parametrize("tokens", [1234, None, 1234567890123])
def test_terminal_table_fits_without_truncating_numbers(monkeypatch, capsys, width, tokens):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((width, 24)))
    step = _step(12345, tokens)
    cli_inspect._print_run_tail(argparse.Namespace(run_dir="run", last=20), [step], [step])
    header, row = capsys.readouterr().out.splitlines()[-2:]
    assert len(header) <= width and len(row) <= width
    if "tok" in header.split():
        assert row.split()[-1] == ("unknown" if tokens is None else str(tokens))
    if "step" in header.split():
        assert row.split()[0] == "12345"


@pytest.mark.parametrize("terminal", [False, True])
def test_full_table_is_byte_compatible_when_wide_or_redirected(monkeypatch, capsys, terminal):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: terminal)
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((140 if terminal else 40, 24)))
    step = _step(0, 1234)
    cli_inspect._print_run_tail(argparse.Namespace(run_dir="run", last=20), [step], [step])
    assert capsys.readouterr().out.splitlines()[-2:] == [
        f"{'step':>4} {'action':40} {'result':60} {'tok':>6}",
        f"{0:>4} {step['action'][:40]:40} {step['result'][:60]:60} {1234:>6}",
    ]


def test_terminal_size_falls_back_to_100(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    def unavailable(fallback):
        assert fallback[0] == 100
        return os.terminal_size(fallback)

    monkeypatch.setattr(shutil, "get_terminal_size", unavailable)
    step = _step(0, 1234)
    cli_inspect._print_run_tail(argparse.Namespace(run_dir="run", last=20), [step], [step])
    assert all(len(line) <= 100 for line in capsys.readouterr().out.splitlines()[-2:])


@pytest.mark.parametrize("numbers,tokens", [([0, 1], [1234567, 0]), ([10000, 10001], [1234, 0])])
def test_complete_redirected_report_matches_head_text(tmp_path, monkeypatch, capsys, numbers, tokens):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    def unexpected_size_query(*args, **kwargs):
        pytest.fail("Redirected reports must not query terminal size")

    monkeypatch.setattr(shutil, "get_terminal_size", unexpected_size_query)
    steps = [_step(n, t, result="r" * 80 + "\nsecond line") for n, t in zip(numbers, tokens, strict=True)]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps))
    assert usage_summary(steps)["tokens_complete"]
    assert cli_inspect.cmd_report(argparse.Namespace(run_dir=str(tmp_path), last=20)) == 0
    # The report applies minimum numeric widths independently to each row.
    expected = (
        f"Measured cost subtotal: $0.0000; tokens: {sum(tokens):,}; complete: cost=True, tokens=True\n"
        f"run: {tmp_path}\n"
        "steps 2 | rejected patches 0 | resumes 0 | budget stop False\n"
        f"prompt tokens (provider) first/median/last: {tokens[0]}/{tokens[0]}/0 | rendered chars first/last: 123/123\n"
        "step action                                   result                                                          tok\n"
        + "".join(f"{n:>4} {'echo ' + 'a' * 35} {'r' * 60} {t:>6}\n" for n, t in zip(numbers, tokens, strict=True))
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.encode() == expected.encode()
