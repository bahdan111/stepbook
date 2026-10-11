"""Read stored evidence without creating directories, executing actions or repairing bytes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stepbook.cli.inspection import cmd_report, cmd_status
from stepbook.cli.run import _report_run
from stepbook.contracts.journal import Journal
from stepbook.run_pointer import classify_run
from stepbook.storage.events import EventLog

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("completion", [False, True])
def test_synthetic_historical_budget_events_remain_readable(tmp_path, monkeypatch, capsys, completion):
    monkeypatch.chdir(tmp_path)
    log = EventLog(tmp_path / "events.jsonl")
    log.append("budget_stop", step=0, cost_usd=0.1, max_total_usd=0.05)
    if completion:
        log.append("local_completion_allowed_after_budget", step=0, cost_usd=0.1, max_total_usd=0.05)
        log.append(
            "step",
            step=0,
            action="finish {}",
            result="saved",
            done=True,
            info={"completion_id": "completion-old", "artifact_path": "recorded-only.json"},
        )
    command = "stepbook run old --max-total-usd 0.05 --resume"
    (tmp_path / "launch.json").write_text(json.dumps({"resume_command": command}))
    records = log.read()
    assert Journal.from_records(records).count == len(records)
    info = classify_run(tmp_path, active=False)
    assert info["status"] == ("completed" if completion else "stopped")
    assert info["resume_command"] == command
    assert cmd_status(argparse.Namespace(run_dir=str(tmp_path))) == 0
    status = capsys.readouterr().out
    assert ("completion saved locally after total spending limit" if completion else "stopped by budget_stop") in status
    assert cmd_report(argparse.Namespace(run_dir=str(tmp_path), last=0)) == 0
    assert "budget stop True" in capsys.readouterr().out

    # Exercise the existing run-summary reader on synthetic evidence, never a stored run.
    result = SimpleNamespace(summary=lambda: {}, cost_usd=0.1, total_tokens=2, final_state=None, steps=1)
    _report_run(SimpleNamespace(log=log), result, SimpleNamespace(), tmp_path, "command", completion)
    summary = json.loads(capsys.readouterr().out)
    assert "report_error" not in summary
    report = (tmp_path / "report.md").read_text()
    if completion:
        assert summary["completion_after_spending_limit"] is True
        assert summary["completion_budget_excess_usd"] == pytest.approx(0.05)
        assert "Saved locally after the query reached the total spending limit" in report
    else:
        assert "ended: budget stop" in report


@pytest.mark.parametrize("kind", ["budget_stop", "paused", "operator_messages", "action_withheld", "failed_model_call"])
def test_historical_repeat_records_remain_readable(tmp_path, kind):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("run_start", measurement_repeats=2)
    log.append(
        "step",
        step=0,
        action="run baseline",
        result="receipt",
        done=False,
        action_origin="host_measurement_repeat",
        repeat_evidence={"previous_run_id": "r1"},
    )
    log.append(kind)
    records = log.read()
    assert records[0]["measurement_repeats"] == 2
    assert records[1]["action_origin"] == "host_measurement_repeat"
    assert records[1]["repeat_evidence"] == {"previous_run_id": "r1"}
    assert Journal.from_records(records).count == 3
    assert classify_run(tmp_path, active=False)["steps"] == 1
