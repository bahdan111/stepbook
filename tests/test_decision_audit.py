"""Decisions made against the computed verdict must be visible in report.md, the run summary and `report`."""

import json
import subprocess
import sys
from pathlib import Path

from stepbook.reporting.decision_audit import (
    AGAINST,
    AGREES,
    INCONCLUSIVE,
    NO_VERDICT,
    against_verdict,
    decision_audit,
    render_markdown,
)

ROOT = Path(__file__).resolve().parent.parent


def _run(hid, label, verdict_line=None, step=0):
    result = "Eval finished:\n" + json.dumps(
        {
            "run_id": f"r-{label}",
            "valid": True,
            "primary_metric": {"name": "acc", "value": 0.9, "direction": "higher"},
            "deterministic_metrics": {"latency_s": 1.12},
            "cost": {"usd": 0, "tokens": 0},
        }
    )
    if verdict_line:
        result += "\n" + verdict_line
    return {
        "ts": float(step),
        "kind": "step",
        "step": step,
        "action": f"run {label}",
        "action_id": f"a{step}",
        "result": result,
        "prompt_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.01,
    }


def _decide(hid, decision, step, reason="latency stays under the 1.2 s ceiling; accuracy +0.40", ok=True):
    return {
        "ts": float(step),
        "kind": "step",
        "step": step,
        "action": f"decide {hid} {decision}",
        "action_id": f"a{step}",
        "result": f"{hid} {decision}: kept in configuration."
        if ok
        else "Error: only the applied hypothesis can be kept.",
        "reasoning": reason,
        "prompt_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.01,
    }


REVERT = (
    "Computed verdict for h1 after 2 valid run(s) vs baseline (3 run(s)): **revert** (decisive deterministic regression in "
    "['latency_s'] (deterministic: ['latency_s -47.4%']); cost delta=n/a). You may override with a written reason."
)
PROV_REVERT = REVERT.replace("after 2 valid", "after 1 valid").replace("**revert**", "PROVISIONAL **revert**")
KEEP = "Computed verdict for h2 after 2 valid run(s) vs h1 (2 run(s)): **keep** (primary improved by +0.100 (z=3.1); cost delta=n/a). You may override with a written reason."
INCONC = "Computed verdict for h3 after 2 valid run(s) vs h1 (2 run(s)): **inconclusive (provisional: 2 of 2 runs)** (primary +0.010; nothing is outside noise). You may override with a written reason."


def test_audit_classifies_every_relation_and_uses_the_latest_verdict():
    events = [
        {"ts": 0, "kind": "run_start"},
        _run("h1", "h1-1", PROV_REVERT, 1),
        _run("h1", "h1-2", REVERT, 2),
        _decide("h1", "keep", 3),
        _run("h2", "h2-1", KEEP, 4),
        _decide("h2", "keep", 5, reason="agree"),
        _run("h3", "h3-1", INCONC, 6),
        _decide("h3", "keep", 7, reason="provisional keep"),
        _decide("h4", "skip", 8, reason="already implemented"),
        _decide("h1", "keep", 9, ok=False),
    ]
    rows = decision_audit(events)
    assert [(r["hypothesis"], r["decision"], r["verdict"], r["relation"]) for r in rows] == [
        ("h1", "keep", "revert", AGAINST),
        ("h2", "keep", "keep", AGREES),
        ("h3", "keep", "inconclusive", INCONCLUSIVE),
        ("h4", "skip", None, NO_VERDICT),
    ]
    assert (
        rows[0]["verdict_step"] == 2
        and rows[0]["verdict_provisional"] is False
        and "1.2 s ceiling" in rows[0]["reason"]
    )
    assert [r["hypothesis"] for r in against_verdict(rows)] == ["h1"]
    md = "\n".join(render_markdown(rows))
    assert "**against verdict**" in md and "| h2 | keep | keep | agrees |" in md and "inconclusive (provisional)" in md
    assert render_markdown([]) == []


def test_report_md_summary_and_report_command_surface_the_override(tmp_path):
    from stepbook.cli import write_report_md
    from stepbook.envs.eval_loop import AdapterEvalEnv

    run = tmp_path / "run"
    run.mkdir()
    events = [{"ts": 0, "kind": "run_start"}, _run("h1", "h1-1", REVERT, 1), _decide("h1", "keep", 2)]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "state.json").write_text(
        json.dumps(
            {"hypotheses": {"h1": {"kind": "config", "status": "kept", "rationale": "top_k 6"}}, "kept_stack": ["h1"]}
        )
    )

    class Result:  # minimal RunResult stand-in for the writer
        cost_usd = 0.02
        total_tokens = 30

    env = AdapterEvalEnv(adapter="offline", log_dir=run)
    text = write_report_md(run, env, Result()).read_text()
    assert "**1 decision(s) made against the computed verdict**: h1 keep vs revert" in text
    assert "## Decisions vs computed verdicts" in text and "| 2 | h1 | keep | revert | **against verdict** |" in text
    assert text.index("## Decisions vs computed verdicts") < text.index("## Computed verdicts")
    proc = subprocess.run(
        [sys.executable, "-m", "stepbook.cli", "report", str(run)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "decision: h1 keep | verdict: revert | against verdict  <-- AGAINST COMPUTED VERDICT" in proc.stdout
    assert "agent's reason: latency stays under" in proc.stdout


def test_report_on_a_pre_structured_journal_survives_a_mid_line_verdict_mention(tmp_path, capsys):
    """An old text-only journal still reports; a step that only mentions the words has no verdict sentence."""
    import argparse

    from stepbook.cli.inspection import cmd_report

    run = tmp_path / "run"
    run.mkdir()
    events = [
        {"ts": 0, "kind": "run_start"},
        _run("h1", "h1-1", REVERT, 1),
        {"ts": 2.0, "kind": "step", "step": 2, "action": "record {}", "result": "noted the Computed verdict above"},
        _decide("h1", "keep", 3),
    ]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run / "state.json").write_text(json.dumps({"hypotheses": {"h1": {"status": "kept", "kind": "config"}}}))
    assert cmd_report(argparse.Namespace(run_dir=str(run), last=10)) == 0
    out = capsys.readouterr().out
    assert "computed verdicts:\n  Computed verdict for h1 after 2 valid run(s)" in out
    assert "decision: h1 keep | verdict: revert | against verdict" in out
    assert "eval attempts consumed=1" in out


def _structured_run(hid, label, verdict, step, *, provisional=False):
    """A run step as the adapter route records it: the receipt and the computed verdict as values."""
    event = _run(hid, label, None, step)
    event["info"] = {
        "ok": True,
        "adapter_exit": 0,
        "receipt": json.loads(event["result"].split("\n", 1)[1]),
        "verdict": {
            "hypothesis": hid,
            "verdict": verdict,
            "provisional": provisional,
            "reference": "baseline",
            "runs": {"current": 2, "reference": 3},
            "reason": "recorded with the verdict, not parsed back out of the sentence",
        },
    }
    # The sentence stays in the observation for the agent; the audit must not need it.
    event["result"] += "\n" + REVERT.replace("h1", hid).replace("**revert**", f"**{verdict}**")
    return event


def _structured_decide(hid, decision, step, *, accepted=True):
    event = _decide(hid, decision, step, ok=accepted)
    event["info"] = {"ok": accepted, "adapter_exit": 0 if accepted else 1}
    if accepted:
        event["info"].update(decision=decision, hypothesis=hid, applied=None, reference=hid)
    return event


def test_audit_reads_the_recorded_verdict_and_decision_not_the_observation():
    events = [
        {"ts": 0, "kind": "run_start"},
        _structured_run("h1", "h1-1", "revert", 1, provisional=True),
        _structured_run("h1", "h1-2", "revert", 2),
        _structured_decide("h1", "keep", 3),
        _structured_run("h2", "h2-1", "keep", 4),
        _structured_decide("h2", "keep", 5),
        _structured_run("h3", "h3-1", "inconclusive", 6),
        _structured_decide("h3", "keep", 7),
        _structured_decide("h4", "skip", 8),
        _structured_decide("h1", "keep", 9, accepted=False),
    ]
    rows = decision_audit(events)
    assert [(r["hypothesis"], r["decision"], r["verdict"], r["relation"]) for r in rows] == [
        ("h1", "keep", "revert", AGAINST),
        ("h2", "keep", "keep", AGREES),
        ("h3", "keep", "inconclusive", INCONCLUSIVE),
        ("h4", "skip", None, NO_VERDICT),
    ]
    assert rows[0]["verdict_step"] == 2 and rows[0]["verdict_provisional"] is False
    assert [r["hypothesis"] for r in against_verdict(rows)] == ["h1"]
    # The same journal with its observations emptied audits identically: nothing is read from them.
    silent = [{**e, "result": ""} if e.get("kind") == "step" else e for e in events]
    assert decision_audit(silent) == rows
