"""The eval-loop-adapter action vocabulary and its report-object form.

``ACTION_RE`` is the command-string vocabulary every eval environment parses, and the one contract the product's
eval loop delivers. ``REPORT_SCHEMA`` is the same vocabulary delivered as one flat JSON object, and
``report_to_action`` converts a validated object back into the exact command string the journal records; only the
research harness's environment opts into that form. Nothing here reads state, the filesystem or the adapter, so
the same conversion can run inside format correction, before a step is approved.
"""

from __future__ import annotations

import json
import re

from stepbook.state.response_validation import parse_json_response

# `phase` is a loop-control verb an adapter may implement to advance a multi-stage experiment whose stages are
# not comparable (different eval sets); unknown verbs are still rejected, so the model cannot invent vocabulary.
ACTION_RE = re.compile(
    r"^\s*(apply|run|decide|add|stop|mine|investigate|phase|read|record|recall|propose|finish)\b\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)

REPORT_PREFIX_RE = re.compile(r"^\s*report\b\s*(.*)$", re.IGNORECASE | re.DOTALL)

# Report-object contract: the same actions as ``ACTION_RE`` delivered as one JSON object,
# ``{"action": "<verb>", ...fields}``, so the model never escapes JSON inside a command string.
# Per-verb field rules live in ``report_to_action``; the schema stays flat for native structured output.
_TOKEN_FIELDS = ("hypothesis", "label", "decision", "target", "query", "kind")
REPORT_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {  # verb -> (required, optional)
    "apply": (("hypothesis",), ()),
    "run": (("label",), ()),
    "decide": (("hypothesis", "decision"), ()),
    "add": (("hypothesis", "kind", "rationale"), ()),
    "investigate": (("hypothesis", "question"), ()),
    "mine": ((), ("target",)),
    "phase": ((), ()),
    "stop": (("reason",), ()),
    "read": (("read",), ()),
    "record": (("finding",), ()),
    "recall": ((), ("query",)),
    "finish": (("finding",), ()),
    "propose": (("proposal",), ()),
}
REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action"],
    "properties": {
        "action": {
            "type": "string",
            "enum": list(REPORT_FIELDS),
            "description": "The eval-loop-adapter action verb; supply only the fields that verb takes",
        },
        "hypothesis": {"type": "string"},
        "label": {"type": "string"},
        "decision": {"type": "string", "enum": ["keep", "revert", "skip"]},
        "kind": {"type": "string"},
        "rationale": {"type": "string"},
        "question": {"type": "string"},
        "target": {"type": "string"},
        "reason": {"type": "string"},
        "query": {"type": "string"},
        "read": {"type": "object", "description": "read locator object: path, start, end, sheet"},
        "finding": {
            "type": "object",
            "description": "record/finish finding object: summary, evidence, uncertainty, next_action (+ run_ids for finish)",
        },
        "proposal": {
            "type": "object",
            "description": "propose object: id, kind, rationale, finding, prediction, test, set or code_change",
        },
    },
}


def _report_object(report) -> dict:
    """The report as an object, decoding the host-serialized ``report {json}`` command form first."""
    if isinstance(report, str):
        m = REPORT_PREFIX_RE.match(report)
        if not m:
            raise ValueError("expected a report object or 'report {json}' command")
        report = parse_json_response(m.group(1))
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    return report


def _check_report_fields(verb: str, report: dict) -> None:
    """Reject fields this verb does not take, fields it needs and values of the wrong shape, in that order."""
    required, optional = REPORT_FIELDS[verb]
    allowed = set(required) | set(optional)
    extra = sorted(k for k in report if k != "action" and k not in allowed)
    if extra:
        raise ValueError(
            f"{verb} report does not take field(s): "
            + ", ".join(extra)
            + (f"; it takes {', '.join(required + optional)}" if required or optional else "; it takes no fields")
        )
    missing = [k for k in required if k not in report]
    if missing:
        raise ValueError(f"{verb} report needs field(s): " + ", ".join(missing))
    for key, value in report.items():
        if key == "action":
            continue
        if key in ("read", "finding", "proposal"):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a JSON object, not an encoded string")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string")
        elif key in _TOKEN_FIELDS and len(value.split()) != 1:
            raise ValueError(f"{key} must be a single token without whitespace")
    if verb == "decide" and report["decision"] not in ("keep", "revert", "skip"):
        raise ValueError("decision must be keep, revert or skip")


def _render_action(verb: str, report: dict) -> str:
    """The exact command string this verb's fields spell, object payloads dumped as compact JSON."""
    if verb == "apply":
        return f"apply {report['hypothesis']}"
    if verb == "run":
        return f"run {report['label']}"
    if verb == "decide":
        return f"decide {report['hypothesis']} {report['decision']}"
    if verb == "add":
        return f"add {report['hypothesis']} {report['kind']} {report['rationale'].strip()}"
    if verb == "investigate":
        return f"investigate {report['hypothesis']} {report['question'].strip()}"
    if verb == "mine":
        return "mine" + (f" {report['target']}" if report.get("target") else "")
    if verb == "phase":
        return "phase"
    if verb == "stop":
        return f"stop {report['reason'].strip()}"
    if verb == "recall":
        return "recall" + (f" {report['query']}" if report.get("query") else "")
    payload = report["read" if verb == "read" else "proposal" if verb == "propose" else "finding"]
    return f"{verb} {json.dumps(payload, ensure_ascii=False, allow_nan=False)}"


def report_to_action(report) -> str:
    """Convert a validated report object into the exact legacy command string, never executing it.

    Accepts the object or the host-serialized ``report {json}`` command. Pure: no state,
    adapter or file access, so the same conversion can run inside format correction.
    """
    report = _report_object(report)
    verb = report.get("action")
    if verb not in REPORT_FIELDS:
        raise ValueError("report.action must be one of: " + ", ".join(REPORT_FIELDS))
    _check_report_fields(verb, report)
    return _render_action(verb, report)
