"""Strict JSON parsing and executor-neutral report response validation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any, NamedTuple

from jsonschema import Draft202012Validator

from ..contracts.errors import ExecutorError
from .validator import PatchError, Validator


def parse_json_response(text, *, allow_json_fence=False):
    """One unambiguous JSON object; optionally unwrap exactly one outer Markdown fence.

    This never searches prose for JSON or repairs fields, numbers, or object syntax.
    """
    if allow_json_fence and isinstance(text, str):
        match = re.fullmatch(r"```(?:json)?\r?\n([\s\S]*?)\r?\n```", text.strip())
        if match:
            text = match.group(1)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON field: {key}")
            out[key] = value
        return out

    def constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number: {value}")
        return parsed

    out = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    if not isinstance(out, dict):
        raise ValueError("response must be one JSON object")
    return out


def report_output_schema(base, report_schema):
    """Opt-in object delivery; legacy command-based environments stay unchanged."""
    if report_schema is None:
        return base
    Draft202012Validator.check_schema(report_schema)
    schema = deepcopy(base)
    del schema["properties"]["action"]
    schema["properties"]["report"] = deepcopy(report_schema)
    schema["required"] = ["report" if key == "action" else key for key in schema["required"]]
    return schema


def report_command(report):
    """Serialize data, never infer missing fields, repair JSON or execute it."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object, not an encoded string")
    return "report " + json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def normalize_report_envelope(envelope, schema, check=None):
    """Authoritative check before converting a validated report to the journal command.

    Executors retain report in Envelope.extra. Preserve that original data and all
    usage/raw receipts; only the internal action field is populated by the host.
    """
    output = {**envelope.extra, "reasoning": envelope.reasoning}
    if "state_patch" in schema["properties"] or envelope.state_patch is not None:
        output["state_patch"] = envelope.state_patch
    if envelope.action:
        output["action"] = envelope.action  # reject conflicting or legacy instructions
    problems = [e.message for e in Draft202012Validator(schema).iter_errors(output)]
    if check:
        problems.extend(check(output))
    if problems:
        raise ExecutorError("Invalid report response: " + "; ".join(problems), envelope=envelope, retryable=False)
    return replace(envelope, action=report_command(output["report"]))


def translate_report_action(env, envelope):
    """Journal the environment's own command when it can derive one from a validated report.

    Environments that understand the report contract expose ``report_to_action``; the
    conversion already passed the same environment's format check, so this is a pure
    re-encoding for the journal, never a repair or an execution.
    """
    convert = getattr(env, "report_to_action", None)
    action = getattr(envelope, "action", None)
    if convert is None or not isinstance(action, str) or not action.lstrip().lower().startswith("report"):
        return envelope
    return replace(envelope, action=convert(action))


class StateCheck(NamedTuple):
    """A validator and the state it checks a patch against: one argument, because neither is usable alone.

    With a validator but no state the difference is taken against nothing, so every protected and write-once
    prefix is bypassed rather than re-based and a deletion is read as a null value.
    """

    validator: Validator
    state: dict[str, Any]


def action_response_problems(
    output: object,
    *,
    state_check: StateCheck | None = None,
    action_check: Callable[[str], Sequence[str]] | None = None,
) -> list[str]:
    """Pure format checks: no files, evidence snapshots, adapter calls or state commits."""
    if not isinstance(output, dict):
        return ["response must be a JSON object"]
    problems: list[str] = []
    if not isinstance(output.get("reasoning"), str):
        problems.append("reasoning must be a string")
    if state_check is not None:
        if "state_patch" not in output:
            problems.append(
                "Missing top-level state_patch field. Return one object with three separate JSON fields: "
                "reasoning (string), state_patch (object), and "
                "action (string). "
                'Text such as <parameter name="state_patch"> inside reasoning is only text, '
                "not a state patch. Do not use XML tags or a JSON-encoded string for this field. "
                "Supply the intended patch as an object; use {} only if no state update is needed."
            )
        else:
            try:
                state_check.validator.apply(state_check.state, output["state_patch"])
            except PatchError as error:
                problems.extend(error.problems)
    if not isinstance(output.get("action"), str) or not output["action"].strip():
        problems.append("action must be a nonempty command string")
    elif action_check is not None:
        problems.extend(action_check(output["action"]))
    return problems


def response_problems(
    output: object,
    *,
    state_check: StateCheck | None = None,
    action_check: Callable[[str], Sequence[str]] | None = None,
    report_schema: dict[str, Any] | None = None,
) -> list[str]:
    """Validate report delivery, then apply ordinary action checks."""
    if not isinstance(output, dict):
        return ["response must be a JSON object"]
    problems: list[str] = []
    if report_schema is not None:
        if "action" in output:
            problems.append("Return report as an object; do not also submit an action command")
        problems.extend(e.message for e in Draft202012Validator(report_schema).iter_errors(output.get("report")))
        try:
            output = {**output, "action": report_command(output.get("report"))}
        except (ValueError, TypeError) as error:
            return problems + [str(error)]
    generic = action_response_problems(output, state_check=state_check, action_check=action_check)
    if report_schema is not None and state_check is not None and "state_patch" not in output:
        # Generic validation reports reasoning first, then the missing patch. Only
        # this diagnostic changes; arbitrary schema/action errors stay verbatim.
        index = int(not isinstance(output.get("reasoning"), str))
        generic[index] = generic[index].replace("action (string). ", "report (object). ", 1)
    return problems + generic
