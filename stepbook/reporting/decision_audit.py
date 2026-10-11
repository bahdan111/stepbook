"""Pair every keep/revert/skip decision with the computed verdict that preceded it.

Read-only over the journal. The verdict is advice with numbers; the decision is the agent's.
This makes a disagreement visible in reports instead of buried in a step's reasoning.

Both sides are read from what the step recorded: a run step journals the verdict it computed and a decide
step the decision the adapter accepted (``stepbook/envs/eval_loop.py``). A run journaled before those fields
existed carries both only in its observation, and ``legacy_journal`` is the one place that reads it.
"""

from __future__ import annotations

from . import legacy_journal
from .report_rendering import table_cell

AGREES, AGAINST, INCONCLUSIVE, NO_VERDICT = "agrees", "against verdict", "inconclusive verdict", "no verdict"


def decision_audit(events: list[dict]) -> list[dict]:
    """One row per successful decide step: what the code recommended and what the agent chose."""
    latest: dict[str, dict] = {}
    rows: list[dict] = []
    for event in events:
        if event.get("kind") != "step":
            continue
        for computed in legacy_journal.step_verdicts(event):
            latest[computed["hypothesis"]] = {**computed, "step": event.get("step")}
        decided = legacy_journal.step_decision(event)
        if decided is None:
            continue
        hid, decision = decided["hypothesis"], decided["decision"]
        verdict = latest.get(hid)
        if verdict is None:
            relation = NO_VERDICT
        elif verdict["verdict"] in ("inconclusive", "no change"):
            relation = INCONCLUSIVE
        elif decision == "skip" or decision == verdict["verdict"]:
            relation = AGREES
        else:
            relation = AGAINST
        rows.append(
            {
                "step": event.get("step"),
                "hypothesis": hid,
                "decision": decision,
                "verdict": verdict["verdict"] if verdict else None,
                "verdict_provisional": verdict["provisional"] if verdict else None,
                "verdict_step": verdict["step"] if verdict else None,
                "relation": relation,
                "reason": str(event.get("reasoning") or "")[:400],
            }
        )
    return rows


def against_verdict(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["relation"] == AGAINST]


def render_markdown(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    lines = [
        "",
        "## Decisions vs computed verdicts",
        "",
        "The verdict is computed from receipts and is advisory; the decision is the agent's. A row marked "
        "**against verdict** needs a human look at the reason.",
        "",
        "| step | hypothesis | decision | verdict | relation | agent's reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        verdict = (r["verdict"] or "none") + (" (provisional)" if r["verdict_provisional"] else "")
        relation = f"**{r['relation']}**" if r["relation"] == AGAINST else r["relation"]
        reason = r["reason"].replace("|", "\\|").replace("\n", " ")[:220]
        cells = [table_cell(value) for value in (r["step"], r["hypothesis"], r["decision"], verdict, relation)]
        lines.append("| " + " | ".join([*cells, reason]) + " |")
    return lines
