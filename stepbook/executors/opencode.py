"""OpenCode step wrapper: ``python -m stepbook.executors.opencode step --schema {schema_file}``.

Raw token categories and OpenCode's reported cost are retained; USD cost is unknown.
Permissions are OpenCode configuration, not an operating-system sandbox.
Exit code ``EXIT_TIMED_OUT`` (124) when the turn timeout fires, so that ``CommandExecutor`` does not re-send the call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import USAGE_KEYS, ParsedStep
from .command import EXIT_TIMED_OUT, driver_timeout

AGENT_NAME = "stepbook-step"

ANSWER_INSTRUCTION = "Return exactly one JSON object matching this schema, no prose and no code fence."


def build_env(permissions: str) -> dict[str, str]:
    """Keep user configuration except the wrapper's agent, update and sharing settings."""
    try:
        config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{}"))
    except json.JSONDecodeError:
        raise ValueError("OPENCODE_CONFIG_CONTENT must be a JSON object") from None
    if not isinstance(config, dict) or not isinstance(config.get("agent", {}), dict):
        raise ValueError("OPENCODE_CONFIG_CONTENT and its agent entry must be JSON objects")
    permission = "allow" if permissions == "full" else "deny"
    config.update(
        autoupdate=False,
        share="disabled",
        agent={
            **config.get("agent", {}),
            AGENT_NAME: {
                "mode": "primary",
                "description": "Choose one stepbook action and return the requested JSON object.",
                "permission": dict.fromkeys(("edit", "bash", "webfetch"), permission),
            },
        },
    )
    # Authentication stays in OpenCode's environment/configuration; never read its auth file.
    return {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(config)}


def build_argv(opencode_bin: str, *, model: str | None, variant: str | None) -> list[str]:
    """No positional message: OpenCode reads the entire message from stdin."""
    argv = [opencode_bin, "run", "--format", "json", "--agent", AGENT_NAME]
    if model:
        argv += ["-m", model]
    if variant:
        argv += ["--variant", variant]
    return argv


def _record_step_finish(out: ParsedStep, part: dict[str, Any]) -> None:
    out.api_calls += 1
    tokens = part.get("tokens") or {}
    cache = tokens.get("cache") or {}
    values = (
        tokens.get("input"),
        cache.get("read"),
        cache.get("write"),
        tokens.get("output"),
        tokens.get("reasoning"),
    )
    for key, value in zip(USAGE_KEYS, values, strict=True):
        out.usage[key] += int(value or 0)
    out.agent_cost = (out.agent_cost or 0.0) + float(part.get("cost") or 0)


def _record_tool_use(out: ParsedStep, part: dict[str, Any]) -> None:
    out.tool_calls += 1
    tool_input = (part.get("state") or {}).get("input", {})
    out.tool_log.append({"tool": part.get("tool"), "input": json.dumps(tool_input, ensure_ascii=False)[:200]})


def parse_events(jsonl: str) -> ParsedStep:
    """Fold all tool rounds and retain only the final text part as the answer."""
    out = ParsedStep(text="", agent_cost=0.0)
    for line in jsonl.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        out.session_id = event.get("sessionID") or out.session_id
        part = event.get("part") or {}
        kind = event.get("type")
        if kind == "text":
            out.text = part.get("text") or ""
        elif kind == "step_finish":
            _record_step_finish(out, part)
        elif kind == "tool_use":
            _record_tool_use(out, part)
        elif kind == "error":
            out.errors.append(event.get("error") or event.get("message") or part.get("message") or "OpenCode error")
    return out


def _last_fenced_object(text: str) -> dict[str, Any] | None:
    """Accept a bare or fenced object, retaining the last outermost object."""
    decoder = json.JSONDecoder()
    found = None
    pos = 0
    while (start := text.find("{", pos)) >= 0:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(value, dict):
            found = value
        pos = end
    return found


def _object_property(prop: Any) -> bool:
    if not isinstance(prop, dict):
        return False
    types = prop.get("type")
    return (
        types == "object"
        or isinstance(types, list)
        and "object" in types
        or "properties" in prop
        or prop.get("additionalProperties") not in (None, False)
        or any(_object_property(branch) for key in ("anyOf", "oneOf", "allOf") for branch in prop.get(key, []))
    )


def decode_answer(answer: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Decode stringified object fields; leave invalid values for the harness validator."""
    properties = schema.get("properties") or {}
    for key, value in answer.items():
        if isinstance(value, str) and (key == "state_patch" or _object_property(properties.get(key))):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                answer[key] = decoded
    return answer


def envelope_fields(folded: ParsedStep, model: str | None) -> dict[str, Any]:
    usage = folded.usage
    return {
        "input_tokens": usage["input_tokens"] + usage["cached_input_tokens"] + usage["cache_write_input_tokens"],
        "output_tokens": usage["output_tokens"] + usage["reasoning_output_tokens"],
        "usage": {
            **usage,
            "provider": model.split("/", 1)[0] if model else None,
            "model": model,
            "cost_usd": None,
            "opencode_cost": folded.agent_cost,
            "note": "Subscription logins report no USD cost; OpenCode's reported cost is retained without USD pricing.",
        },
        "api_calls": max(folded.api_calls, 1),
        "tool_calls": folded.tool_calls,
        "tool_log": folded.tool_log,
        "session_id": folded.session_id,
        "opencode_errors": folded.errors,
    }


def cmd_step(args: argparse.Namespace) -> int:
    try:
        schema = json.loads(Path(args.schema).read_text())
        if not isinstance(schema, dict):
            raise ValueError("schema must be a JSON object")
        env = build_env(args.permissions)
    except (OSError, ValueError) as exc:
        print(f"opencode configuration: {exc}", file=sys.stderr)
        return 1
    prompt = (
        sys.stdin.read()
        + "\n\nSkill output schema:\n"
        + json.dumps(schema, indent=2)
        + "\n"
        + ANSWER_INSTRUCTION
        + "\n"
    )
    argv = build_argv(args.opencode, model=args.model, variant=args.variant)
    timeout_s = driver_timeout(args.timeout)
    try:
        proc = subprocess.run(argv, input=prompt, capture_output=True, text=True, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # No receipt for the abandoned turn: the reserved code keeps the host from replaying a paid call.
        print(f"opencode timed out after {timeout_s:g}s", file=sys.stderr)
        return EXIT_TIMED_OUT
    except OSError as exc:
        print(f"opencode could not start: {exc.strerror}", file=sys.stderr)
        return 1
    folded = parse_events(proc.stdout)
    if proc.returncode or folded.errors:
        messages = json.dumps(folded.errors, ensure_ascii=False, default=str) if folded.errors else ""
        print(f"opencode exited {proc.returncode}: {messages}\n{proc.stderr[-1000:]}".strip(), file=sys.stderr)
        return proc.returncode or 1
    answer = _last_fenced_object(folded.text or "")
    if answer is None:
        print("opencode printed no JSON answer", file=sys.stderr)
        return 3
    print(json.dumps({**decode_answer(answer, schema), **envelope_fields(folded, args.model)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)
    step = sub.add_parser("step", help="step wrapper for `stepbook run --executor command`", allow_abbrev=False)
    step.add_argument("--schema", required=True, help="skill JSON schema; use {schema_file} in --command")
    step.add_argument("--model", default=None, help="provider/model; defaults to OpenCode's configured model")
    step.add_argument("--variant", default=None, help="OpenCode model variant (reasoning effort)")
    step.add_argument(
        "--permissions",
        choices=("read-only", "full"),
        default="read-only",
        help="OpenCode edit/bash/webfetch permissions (default: read-only); full is only for an externally isolated environment",
    )
    step.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="the host's per-query limit in seconds (default: 900); the turn gets 30 s less so its timeout is reported",
    )
    step.add_argument(
        "--opencode", default=os.environ.get("OPENCODE_BIN", "opencode"), help="OpenCode binary (or a fake for tests)"
    )
    args = parser.parse_args(argv)
    return cmd_step(args)


if __name__ == "__main__":
    sys.exit(main())
