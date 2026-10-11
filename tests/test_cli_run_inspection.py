"""status, report and recover must work on every run shape a chat may find: paused, interrupted, completed.

Offline. The fixtures are journals the runtime really writes; the CLI must not crash on any of them."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _cli(argv, cwd):
    return subprocess.run(
        [sys.executable, "-m", "stepbook.cli", *argv], capture_output=True, text=True, cwd=cwd, timeout=60
    )


def _journal(run: Path, events: list[dict], state: dict | None):
    run.mkdir(parents=True)
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    if state is not None:
        (run / "state.json").write_text(json.dumps(state))


STEP0 = {
    "ts": 2.0,
    "kind": "step",
    "step": 0,
    "action": "run baseline-1",
    "action_id": "a1",
    "result": "Eval finished:\n"
    + json.dumps(
        {
            "run_id": "r1",
            "valid": True,
            "primary_metric": {"name": "acc", "value": 0.5, "direction": "higher"},
            "deterministic_metrics": {"latency_s": 0.7},
            "cost": {"usd": 0, "tokens": 0},
        }
    ),
    "prompt_tokens": 100,
    "output_tokens": 20,
    "cost_usd": 0.01,
    "reasoning": "measure",
}
STATE = {
    "hypotheses": {"h1": {"kind": "config", "status": "todo", "rationale": "x"}},
    "kept_stack": [],
    "current": {"phase": "run"},
}


@pytest.mark.parametrize("shape", ["paused", "interrupted", "completed", "no_state", "empty_result"])
def test_report_status_and_recover_do_not_crash_on_any_run_shape(tmp_path, shape):
    run = tmp_path / ".stepbook" / shape
    events = [{"ts": 1.0, "kind": "run_start", "skill": "eval-loop-adapter"}, STEP0]
    state = STATE
    if shape == "paused":
        events.append({"ts": 3.0, "kind": "paused", "step": 1})
    elif shape == "interrupted":
        events.append(
            {"ts": 3.0, "kind": "action_started", "action_id": "a2", "record": {"step": 1, "action": "run h1-1"}}
        )
    elif shape == "completed":
        events.append(
            {
                "ts": 3.0,
                "kind": "step",
                "step": 1,
                "action": "stop enough",
                "action_id": "a2",
                "result": "Stopped.",
                "done": True,
                "prompt_tokens": 50,
                "output_tokens": 5,
                "cost_usd": 0.005,
            }
        )
    elif shape == "no_state":
        state = None
    elif shape == "empty_result":
        events = [{"ts": 1.0, "kind": "run_start"}, {**STEP0, "result": ""}]
    _journal(run, events, state)
    (tmp_path / ".stepbook" / "LAST_RUN").write_text(str(run))
    seen = []
    for argv in (["status"], ["status", "--run-dir", str(run)], ["report", str(run)], ["recover", str(run)]):
        proc = _cli(argv, tmp_path)
        assert proc.returncode == 0, (shape, argv, proc.stderr[-600:])
        assert "Traceback" not in proc.stderr
        seen.append(proc.stdout)
    recovered = json.loads(_cli(["recover", str(run)], tmp_path).stdout)
    assert len(recovered["pending_actions"]) == (1 if shape == "interrupted" else 0)

    status_line = {
        "paused": "status: paused - paused cleanly after 1 recorded steps; it is not running.",
        "interrupted": "status: interrupted - ended without a terminal record after 1 steps (crash or kill). "
        "1 action(s) started without a receipt;",
        "completed": "status: completed - completed after 2 steps; nothing to resume.",
        "no_state": "status: interrupted - ended without a terminal record after 1 steps (crash or kill). Resume with:",
        "empty_result": "status: interrupted - ended without a terminal record after 1 steps (crash or kill). "
        "Resume with:",
    }[shape]
    # Same classification whether the run is found through LAST_RUN or named explicitly.
    for status in seen[:2]:
        assert status_line in status, (shape, status)
        assert status.splitlines()[0] == f"run: {run}"
        assert f"steps: {2 if shape == 'completed' else 1} |" in status
        assert ("Resume with:" in status) is (shape != "completed")
    report = seen[2]
    assert (
        report.splitlines()[0]
        == {
            "paused": "Measured cost subtotal: $0.0100; tokens: 120; complete: cost=True, tokens=True",
            "interrupted": "Measured cost subtotal: $0.0100 (partial); tokens: 120 (partial); complete: cost=False, tokens=False",
            "completed": "Measured cost subtotal: $0.0150; tokens: 175; complete: cost=True, tokens=True",
            "no_state": "Measured cost subtotal: $0.0100; tokens: 120; complete: cost=True, tokens=True",
            "empty_result": "Measured cost subtotal: $0.0100 (partial); tokens: 120 (partial); complete: cost=False, tokens=False",
        }[shape]
    )
    assert ("== Eval loop summary" in report) is (shape != "no_state")
    # None of these shapes carries a rate_limited event, which the runtime no longer writes, so the run tail
    # prints no rate-limit figure; every other figure on the line is unchanged.
    assert f"steps {2 if shape == 'completed' else 1} | rejected patches 0 | resumes 0 | budget stop False" in report
    assert json.loads(seen[3]) == {
        "pending_actions": [{"action_id": "a2", "step": 1, "action": "run h1-1", "receipt_available": False}]
        if shape == "interrupted"
        else [],
        "note": "Read-only unless --receipt is supplied. Never infer execution from intent alone.",
    }


def test_a_stored_rate_limited_event_still_prints_its_figure(tmp_path):
    """Stored rate_limited events contribute to the report count without changing the rest of the summary."""
    run = tmp_path / ".stepbook" / "old"
    _journal(run, [{"ts": 1.0, "kind": "run_start"}, STEP0, {"ts": 3.0, "kind": "rate_limited", "step": 1}], STATE)
    proc = _cli(["report", str(run)], tmp_path)
    assert proc.returncode == 0 and "Traceback" not in proc.stderr
    assert "steps 1 | rejected patches 0 | resumes 0 | rate-limit waits 1 | budget stop False" in proc.stdout


def test_recover_without_journal_reports_cleanly(tmp_path):
    """Recovery and reporting must refuse a directory without a journal, without creating artifacts.
    Status must distinguish a missing directory from an empty run directory."""
    missing = tmp_path / "nothing"
    proc = _cli(["recover", str(missing)], tmp_path)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr and "No event log" in proc.stderr
    assert proc.stderr == "No event log at the requested run directory\n" and proc.stdout == ""
    proc = _cli(["report", str(missing)], tmp_path)
    assert "Traceback" not in proc.stderr, proc.stderr[-600:]
    assert proc.returncode == 2 and proc.stdout == ""
    assert proc.stderr == f"no run at {missing}\n"
    assert not missing.exists(), "a refused report must not create the run directory"
    proc = _cli(["status", "--run-dir", str(missing)], tmp_path)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr
    assert proc.stdout == "" and proc.stderr == f"no run at {missing}\n"
    missing.mkdir()
    proc = _cli(["status", "--run-dir", str(missing)], tmp_path)
    assert proc.returncode == 0 and "Traceback" not in proc.stderr
    assert proc.stdout.splitlines()[:2] == [
        f"run: {missing}",
        "status: not_started - was launched but recorded no events.",
    ]


@pytest.mark.parametrize(
    "extra, suffix, cells",
    [
        ({"origin": "docs", "runs": ["r1", "r2"]}, " | origin: docs | runs: 2", "docs | 2"),
        ({"origin": "agent", "runs": []}, " | origin: agent | runs: 0", "agent | 0"),
        ({}, "", " | "),
    ],
)
def test_idea_origin_and_run_count_in_status_and_generated_report(tmp_path, capsys, extra, suffix, cells):
    from stepbook.cli import main
    from stepbook.cli.inspection import write_report_md

    idea = {"status": "todo", "kind": "code", "rationale": "inspect", "evidence": "pending", **extra}
    run = tmp_path / "ideas"
    _journal(run, [{"kind": "run_start", "ts": 1}], {"hypotheses": {"h1": idea}})
    assert main(["status", "--run-dir", str(run)]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("  h1"))
    assert line == f"  {'h1':6} {'todo':12} {'code':7} {'inspect':60} | pending{suffix}"
    from types import SimpleNamespace

    report = write_report_md(run, object(), SimpleNamespace(total_tokens=0, cost_usd=0)).read_text()
    assert "| id | status | kind | rationale | evidence | origin | runs |" in report
    assert f"| h1 | todo | code | inspect | pending | {cells} |" in report
