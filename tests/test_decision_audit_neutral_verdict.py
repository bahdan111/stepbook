import json
from types import SimpleNamespace

import pytest

from stepbook.cli import main
from stepbook.cli.run import _report_run
from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.reporting.decision_audit import INCONCLUSIVE, against_verdict, decision_audit, render_markdown
from stepbook.storage.events import EventLog


def _review_events(earlier_revert, decision, repeats):
    def row(recall):
        return {
            "run_id": "r",
            "primary_metric": {"name": "acc", "value": 0.5, "direction": "higher"},
            "primary_deterministic": True,
            "deterministic_metrics": {"recall": recall},
        }

    env = AdapterEvalEnv(
        adapter="unused", runs_per_hypothesis=repeats, _runs={"baseline": [row(0.5)] * 3, "h1": [row(0.2)]}
    )
    prior = env._verdict("h1")
    assert "**revert**" in prior
    env._runs["h1"].append(row(0.8))
    events = [{"kind": "step", "step": 1, "result": prior}] if earlier_revert else []
    return events + [
        {"kind": "step", "step": 2, "result": env._verdict("h1")},
        {"kind": "step", "step": 3, "action": f"decide h1 {decision}", "result": f"h1 {decision}: accepted"},
    ]


@pytest.mark.parametrize("earlier_revert", [False, True], ids=["tie-only", "after-revert"])
@pytest.mark.parametrize("decision", ["keep", "revert"])
@pytest.mark.parametrize("repeats", [2, 3], ids=["final", "provisional"])
def test_no_change_replaces_prior_advice_as_neutral(earlier_revert, decision, repeats):
    rows = decision_audit(_review_events(earlier_revert, decision, repeats))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "no change"
    assert rows[0]["verdict_step"] == 2
    assert rows[0]["verdict_provisional"] == (repeats == 3)
    assert rows[0]["relation"] == INCONCLUSIVE
    assert against_verdict(rows) == []
    assert "no change" in "\n".join(render_markdown(rows))


@pytest.mark.parametrize("report_run", [_report_run], ids=["product"])
def test_neutral_verdict_reaches_summary_and_reports(tmp_path, capsys, report_run):
    log = EventLog(tmp_path / "events.jsonl")
    for event in _review_events(True, "keep", 2):
        log.append(**event)
    result = SimpleNamespace(
        summary=lambda: {}, final_state={"hypotheses": {}}, cost_usd=0, total_tokens=0, aborted=False
    )
    report_run(SimpleNamespace(log=log), result, SimpleNamespace(), tmp_path, "command", False)
    summary = json.loads(capsys.readouterr().out)
    assert summary["decisions_against_verdict"] == 0
    assert "decision_audit_error" not in summary and "report_error" not in summary
    report = (tmp_path / "report.md").read_text()
    assert "| 3 | h1 | keep | no change | inconclusive verdict |" in report
    assert "decision(s) made against" not in report
    assert main(["report", str(tmp_path)]) == 0
    assert "decision: h1 keep | verdict: no change | inconclusive verdict" in capsys.readouterr().out
