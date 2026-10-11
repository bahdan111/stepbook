"""Offline coverage for CLI budgets, resume guidance and evidence reporting."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stepbook import Envelope, ScriptedExecutor
from stepbook.cli.inspection import write_report_md

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["stepbook"])
def cli_pair(request):
    return tuple(
        importlib.import_module(
            f"{request.param}.{name.replace('cli_', 'cli.').replace('cli.inspect', 'cli.inspection') if request.param == 'stepbook' else name}"
        )
        for name in ("cli", "cli_run")
    )


def launch(cli_pair, tmp_path, monkeypatch, skill="eval-loop", patch=None, options=()):
    cli, run = cli_pair
    monkeypatch.chdir(tmp_path)
    calls = []

    def respond(system, user):
        calls.append(user)
        return Envelope("offline", patch or {}, "done")

    monkeypatch.setattr(run, "ClaudeCLIExecutor", lambda **kw: ScriptedExecutor(respond))
    argv = [
        "run",
        str(ROOT / "stepbook/templates" / skill),
        # the research harness still selects its environment by name; the product has only --adapter or shell
        *([] if cli.__name__ == "stepbook.cli" else ["--env", "shell"]),
        "--workdir",
        str(tmp_path),
        "--goal",
        "Verify the output",
        "--executor",
        "cli",
        "--steps",
        "3",
        "--log-dir",
        str(tmp_path / "run"),
        "--no-checkpoint",
        "--unsandboxed-i-accept-the-risk",
        "--quiet",
        *options,
    ]
    assert cli.main(argv) == 0
    return argv, calls


@pytest.mark.parametrize("plan_cap, cap, expected", [(0, 7, 7), (7, 0, 0), (0, None, 0)])
def test_shell_cap_plan_precedence_and_launch_line(cli_pair, tmp_path, monkeypatch, capsys, plan_cap, cap, expected):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"budget": {"runs_max": plan_cap}}))
    opts = ["--plan", str(plan), *(["--runs-max", str(cap)] if cap is not None else [])]
    launch(cli_pair, tmp_path, monkeypatch, options=opts)
    state = json.loads((tmp_path / "run/state.json").read_text())
    assert state["budget"]["runs_max"] == expected
    err = capsys.readouterr().err
    assert err.count("runs_max:") == 1
    assert f"runs_max: {expected} (recorded for the agent; not host-enforced in shell mode)" in err


@pytest.mark.parametrize(
    "flag,value,diagnostic",
    [
        ("--runs-max", "-1", "must be nonnegative"),
        ("--runs-max", "1.5", "invalid nonnegative_int value"),
        ("--runs-max", "nope", "invalid nonnegative_int value"),
        ("--steps", "0", "must be positive"),
        ("--state-max-chars", "0", "must be positive"),
        ("--controller-timeout", "0", "must be finite and positive"),
        ("--adapter-timeout", "nan", "must be finite and positive"),
    ],
)
def test_zero_runs_allowance_keeps_other_validation(cli_pair, capsys, flag, value, diagnostic):
    with pytest.raises(SystemExit) as exc:
        cli_pair[0]._build_parser().parse_args(["run", "unused", flag, value])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert f"argument {flag}: {diagnostic}" in error


def test_shell_cap_does_not_invent_budget(cli_pair, tmp_path, monkeypatch, capsys):
    launch(cli_pair, tmp_path, monkeypatch, skill="checklist", options=["--runs-max", "7"])
    assert "budget" not in json.loads((tmp_path / "run/state.json").read_text())
    assert "runs_max: unavailable (skill has no budget.runs_max)" in capsys.readouterr().err


def test_runs_max_help_explains_enforcement(cli_pair, capsys):
    with pytest.raises(SystemExit) as exc:
        cli_pair[0].main(["run", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "enforced by the host only for adapter loops" in help_text
    assert "otherwise shown to the agent" in help_text


@pytest.mark.parametrize("reopen", [None, "   ", "continue verification"])
def test_completed_resume_explains_noop_and_reopen(cli_pair, tmp_path, monkeypatch, capsys, reopen):
    argv, calls = launch(cli_pair, tmp_path, monkeypatch, skill="checklist")
    capsys.readouterr()
    options = ["--resume", *(["--reopen", reopen] if reopen is not None else [])]
    assert cli_pair[0].main([*argv, *options]) == 0
    err = capsys.readouterr().err
    continued = bool(reopen and reopen.strip())
    assert len(calls) == (2 if continued else 1)
    assert ("no-op" in err) is not continued
    if not continued:
        assert "--reopen '<note>'" in err
    assert cli_pair[0].main(["status", "--run-dir", str(tmp_path / "run")]) == 0
    assert "--reopen '<note>'" in capsys.readouterr().out


EVIDENCE = "fixed A; broke B; " * 30 + "decisive tail: artifacts/comparison.json"
IMPROVE = {
    "baseline": {"runs": ["b1", "b2"], "primary_mean": 0.5833},
    "runs": {
        name: {
            "label": name,
            "stack": [],
            "valid": True,
            "notes": "repeat",
            "case_set": "fixed",
            "primary": 0.5833,
            "artifact": f"runs/{name}.json",
        }
        for name in ("b1", "b2")
    },
    "kept_stack": [],
    "hypotheses": {
        "cells": {
            "kind": "config",
            "status": "inconclusive",
            "origin": "user",
            "rationale": "row | column: value",
            "evidence": EVIDENCE,
            "runs": ["t1", "t2"],
        }
    },
    "notes": {"final_summary": "Restored baseline; no measured change kept."},
}
CHECKLIST = {
    "goal": "Rename the total function",
    "checklist": {"rename": {"text": "Rename callers", "status": "done", "evidence": EVIDENCE}},
    "checks_status": {"tests": "unknown"},
}


def journal(tmp_path, state):
    events = [
        {"kind": "run_start", "ts": 1},
        {
            "kind": "step",
            "step": 0,
            "ts": 2,
            "action": "check",
            "result": "old check",
            "info": {"checks": {"tests": "fail", "lint": "pass"}},
        },
        # A receipt can survive a crash before its step is committed.
        {"kind": "action_result", "ts": 3, "result": "latest check", "info": {"checks": {"tests": "pass"}}},
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (tmp_path / "state.json").write_text(json.dumps(state))


@pytest.mark.parametrize("state", [IMPROVE, CHECKLIST])
def test_shell_status_and_report_show_actual_state_and_latest_host_checks(cli_pair, tmp_path, capsys, state):
    journal(tmp_path, state)
    assert cli_pair[0].main(["status", "--run-dir", str(tmp_path)]) == 0
    console = capsys.readouterr().out
    report = write_report_md(tmp_path, object(), SimpleNamespace(total_tokens=0, cost_usd=0)).read_text()
    for output in (console, report):
        assert "Host-recorded checks" in output
        assert "tests: pass" in output and "lint: pass" in output
        assert output.index("Host-recorded checks") < output.index("Agent-maintained")
        assert "baseline: None" not in output and "reference: None" not in output
        assert EVIDENCE in output or str(tmp_path / "report.md") in output
        if state is IMPROVE:
            assert "0.5833" in output and "b1" in output and "b2" in output
            assert "runs/b1.json" in output and "kept stack: []" in output
            assert "Restored baseline; no measured change kept." in output
        else:
            assert state["goal"] in output and "Rename callers" in output
            assert "unknown" in output
    assert EVIDENCE in report


def test_report_escapes_all_decision_and_ledger_cells(tmp_path):
    from stepbook.cli.inspection import _ledger_lines

    state = {
        "hypotheses": {
            "id|x": {
                "status": "todo|later",
                "kind": "code|config",
                "rationale": "row | column",
                "evidence": "a|b\nc|d",
                "origin": "user|docs",
            }
        }
    }
    journal(tmp_path, state)
    report = write_report_md(tmp_path, object(), SimpleNamespace(total_tokens=0, cost_usd=0)).read_text()
    assert r"| id\|x | todo\|later | code\|config | row \| column | a\|b<br>c\|d | user\|docs |  |" in report
    env = SimpleNamespace(
        _runs={"h|x": [{"run_id": "r|1", "valid": False, "cost": {"note": "x|y"}}]},
        _reference_h="baseline",
        _valid_run=lambda p: False,
    )
    lines = "\n".join(_ledger_lines(env))
    assert r"h\|x" in lines and r"r\|1" in lines and r"x\|y" in lines


def test_generated_shell_report_uses_final_state(cli_pair, tmp_path, monkeypatch):
    launch(cli_pair, tmp_path, monkeypatch, patch=IMPROVE)
    report = (tmp_path / "run/report.md").read_text()
    assert EVIDENCE in report
    assert r"row \| column: value" in report
    assert "Restored baseline; no measured change kept." in report


def test_audit_table_escapes_hypothesis_pipes():
    from stepbook.reporting.decision_audit import decision_audit, render_markdown

    rows = decision_audit(
        [{"kind": "step", "step": 1, "action": "decide h|1 skip", "result": "Skipped", "reasoning": "because a|b"}]
    )
    assert r"| 1 | h\|1 | skip | none | no verdict | because a\|b |" in "\n".join(render_markdown(rows))


@pytest.mark.parametrize("state", [{"baseline": {}}, {"checklist": {}}])
def test_sparse_shell_state_omits_missing_measurements(tmp_path, capsys, state):
    from stepbook.cli import main

    journal(tmp_path, state)
    assert main(["status", "--run-dir", str(tmp_path)]) == 0
    console = capsys.readouterr().out
    report = write_report_md(tmp_path, object(), SimpleNamespace(total_tokens=0, cost_usd=0)).read_text()
    for output in (console, report):
        for absent in ("baseline: None", "reference:", "kept stack:", "goal:", "primary_mean:"):
            assert absent not in output


def test_shell_cli_cap_survives_resume(cli_pair, tmp_path, monkeypatch, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text('{"budget": {"runs_max": 0}}')
    argv, calls = launch(cli_pair, tmp_path, monkeypatch, options=["--plan", str(plan), "--runs-max", "7"])
    capsys.readouterr()
    assert cli_pair[0].main([*argv, "--resume", "--reopen", "continue", "--runs-max", "9"]) == 0
    assert len(calls) == 2
    assert json.loads((tmp_path / "run/state.json").read_text())["budget"]["runs_max"] == 9
    assert "runs_max: 9" in capsys.readouterr().err


def test_generated_checklist_report_contains_check_evidence(cli_pair, tmp_path, monkeypatch):
    launch(
        cli_pair,
        tmp_path,
        monkeypatch,
        skill="checklist",
        patch={"checklist": CHECKLIST["checklist"]},
        options=["--check", "tests=true"],
    )
    report = (tmp_path / "run/report.md").read_text()
    assert "Verify the output" in report and "tests: pass" in report and EVIDENCE in report
    assert report.index("Host-recorded checks") < report.index("Agent-maintained")
