"""Render shell checkpoints and host check receipts without adapter assumptions."""

from __future__ import annotations

import json
import math
from typing import Any

from . import legacy_journal


def table_cell(value: Any) -> str:
    """Keep text inside one Markdown table cell, including multiline evidence."""
    return str(value).replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def table_row(values: list[Any]) -> str:
    return "| " + " | ".join(table_cell(value) for value in values) + " |"


def shell_state(state: dict) -> bool:
    """Recognize layouts by their state fields, independent of the skill's name."""
    return "checklist" in state or isinstance(state.get("baseline"), dict)


def _value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def _fields(fields: dict) -> list[str]:
    return [f"- {key}: {_value(value)}" for key, value in fields.items() if value is not None]


def host_checks(events: list[dict]) -> dict:
    """Latest recorded result per check, including receipts left before a step commit."""
    checks: dict = {}
    for event in events:
        if event.get("kind") not in ("step", "action_result"):
            continue
        recorded = (event.get("info") or {}).get("checks")
        if isinstance(recorded, dict):
            checks.update(recorded)
    return checks


def host_check_lines(events: list[dict]) -> list[str]:
    lines = _fields(host_checks(events))
    return ["", "## Host-recorded checks (latest per check from journal)", "", *lines] if lines else []


def _records(title: str, records: dict) -> list[str]:
    """Preserve every recorded field, including full evidence and artifact paths."""
    rows = [
        table_row([key, field, _value(value)])
        for key, record in records.items()
        for field, value in record.items()
        if value is not None
    ]
    if not rows:
        return []
    return ["", f"### {title}", "", "| id | field | value |", "|---|---|---|", *rows]


def _checklist_rows(records: dict) -> list[str]:
    if not records:
        return []
    fields = list(dict.fromkeys(field for record in records.values() for field in record))
    return [
        "",
        "### Checklist",
        "",
        table_row(["id", *fields]),
        table_row(["---"] * (len(fields) + 1)),
        *(table_row([key, *(_value(record.get(field, "")) for field in fields)]) for key, record in records.items()),
    ]


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _recorded_measurements(events: list[dict]) -> dict:
    """Use structured execution receipts only; arbitrary shell stdout is not a metric contract."""
    runs = {}
    for event in events:
        if event.get("kind") not in ("step", "action_result"):
            continue
        row = legacy_journal.step_receipt(event)
        if row is None or not row.get("run_id"):
            continue
        metric = row.get("primary_metric") or {}
        if isinstance(metric, dict):
            runs[row["run_id"]] = {
                "primary": metric.get("value"),
                "direction": metric.get("direction"),
                "valid": row.get("valid", True),
            }
    return runs


def _primary_summary(state: dict, events: list[dict]) -> str:
    recorded = _recorded_measurements(events)
    reported = {**state.get("runs", {}), **recorded}
    host_values = _primary_values(recorded)
    agent_values = _primary_values(reported)
    values = host_values or agent_values
    source = "host-recorded" if host_values else "agent-reported"
    baseline = state.get("baseline") or {}
    baseline_ids = baseline.get("runs", [])
    baseline_values = [host_values[key] for key in baseline_ids if key in host_values]
    baseline_source = "host-recorded" if baseline_values else "agent-reported"
    if not baseline_values:
        baseline_values = [agent_values[key] for key in baseline_ids if key in agent_values]
    mean = sum(baseline_values) / len(baseline_values) if baseline_values else None
    if mean is None and _number(baseline.get("primary_mean")):
        mean = baseline["primary_mean"]
    direction = _primary_direction(recorded or reported)
    best = None
    if values and (direction or len(set(values.values())) == 1):
        best = (min if direction == "lower" else max)(values.values())
    best_text = f"{best:.6g}" if best is not None else "unknown"
    mean_text = f"{mean:.6g}" if mean is not None else "unknown"
    return f"best measured primary={best_text} ({source}) vs baseline mean={mean_text} ({baseline_source})"


def _primary_values(runs: dict) -> dict:
    return {key: row["primary"] for key, row in runs.items() if row.get("valid") and _number(row.get("primary"))}


def _primary_direction(runs: dict) -> str | None:
    """The one direction the receipts report, or None.

    Only the recorded ``direction`` field decides; the skill's prose about the metric is not read. Receipts
    that disagree, or report none, leave the direction unknown, and the caller reports no best value.
    """
    directions = {row.get("direction") for row in runs.values()} - {None}
    if len(directions) == 1 and directions <= {"higher", "lower"}:
        return directions.pop()
    return None


def shell_result_line(state: dict, events: list[dict]) -> str:
    """Headline for shell layouts only; leave adapter presentation untouched."""
    if "checklist" in state:
        items = state.get("checklist") or {}
        done = sum(row.get("status") == "done" for row in items.values())
        checks = host_checks(events)
        configured = set(state.get("checks") or {}) | set(checks)
        passing = sum(checks.get(key) == "pass" for key in configured)
        check_text = (
            f"{passing}/{len(configured)} checks passing (host-recorded)" if configured else "no checks configured"
        )
        if not items:  # a run that kept no checklist: "0/0 items done" would read as if nothing happened
            return f"Result: {check_text}"
        return f"Result: {done}/{len(items)} items done (agent-reported); {check_text}"
    if isinstance(state.get("baseline"), dict):
        ideas = state.get("hypotheses") or {}
        tried = sum(
            bool(row.get("runs")) or row.get("status") in ("testing", "kept", "reverted", "inconclusive")
            for row in ideas.values()
        )
        kept = len(state.get("kept_stack") or [])
        return f"Result: {kept}/{tried} ideas kept/tried (agent-reported); {_primary_summary(state, events)}"
    return ""


def shell_checkpoint_lines(state: dict) -> list[str]:
    lines = ["", "## Agent-maintained state", ""]
    lines += _fields({k: state[k] for k in ("goal", "current", "budget", "checks_status") if k in state})
    if "baseline" in state:
        lines += ["", "### Baseline", "", *_fields(state["baseline"])]
        lines += _records("Runs", state.get("runs") or {})
    if "kept_stack" in state:
        lines += _fields({"kept stack": state["kept_stack"]})
    lines += _records("Ideas", state.get("hypotheses") or {})
    lines += _checklist_rows(state.get("checklist") or {})
    if state.get("notes"):
        lines += ["", "### Notes", "", *_fields(state["notes"])]
    return lines


def shell_summary_lines(state: dict, events: list[dict]) -> list[str]:
    return host_check_lines(events) + shell_checkpoint_lines(state)
