"""Codex step wrapper: ``python -m stepbook.executors.codex step --schema {schema_file} [--model M]``.

Accounting: raw Codex token categories, no USD pricing; ``cost_usd`` is null.
Sandbox limitation: read-only limits tool access but cannot disable tools; tool calls are counted.
Exit code ``EXIT_TIMED_OUT`` (124) when the turn timeout fires, so that ``CommandExecutor`` does not re-send the call.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .auth import subscription_required
from .base import USAGE_KEYS, ParsedStep
from .command import EXIT_TIMED_OUT, driver_timeout

# A ChatGPT login is the only billing the plugins use, so an API credential in the environment must not
# silently become the account that pays for a step.
API_CREDENTIALS = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")


def _last_codex_object(text: str) -> dict[str, Any] | None:
    """Last outermost JSON object in ``text`` (Codex prints the schema-conforming answer last)."""
    depth, start, found = 0, None, None
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    found = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
    return found if isinstance(found, dict) else None


def strictify(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """OpenAI strict output schemas need ``additionalProperties: false`` on every object, every property required, and
    cannot express maps (``additionalProperties: <schema>``) such as the skill's id-keyed checklist. So EVERY top-level
    object-typed property (state_patch, report fields) becomes a JSON-encoded string; the returned key list says which
    fields :func:`decode_answer` must ``json.loads`` again. The harness validator still checks the decoded patch against
    the skill schema."""
    out = json.loads(json.dumps(schema))
    encoded: list[str] = []
    props = out.get("properties") or {}
    for key, prop in list(props.items()):
        if not isinstance(prop, dict):
            continue
        types = prop.get("type")
        types = types if isinstance(types, list) else [types]
        if "object" in types or prop.get("properties") or prop.get("additionalProperties") not in (None, False):
            props[key] = {
                "type": "string",
                "description": (
                    str(prop.get("description", "")) + ' Encode this object as a JSON string (e.g. "{}" for no change).'
                ).strip(),
            }
            encoded.append(key)
    out["additionalProperties"] = False
    out["required"] = list(props.keys())
    return out, encoded


def decode_answer(answer: dict[str, Any] | None, encoded: list[str]) -> dict[str, Any] | None:
    if not isinstance(answer, dict):
        return answer
    for key in encoded:
        val = answer.get(key)
        if isinstance(val, str):
            try:
                answer[key] = json.loads(val) if val.strip() else {}
            except json.JSONDecodeError:
                # Leave the undecodable string in place: the harness validator rejects it and the step is
                # retried with the reason, as it is for OpenCode and Claude. Silently reading it as "no change"
                # would run the action against state the model believed it had already updated.
                continue
    return answer


def _record_item(out: ParsedStep, ev: dict[str, Any], kind: str) -> None:
    item = ev.get("item") or {}
    itype = item.get("type") or item.get("item_type")
    if kind == "item.completed" and itype == "agent_message":
        out.text = item.get("text") or item.get("content")
    if kind == "item.completed" and itype in (
        "command_execution",
        "file_change",
        "web_search",
        "mcp_tool_call",
    ):
        out.tool_calls += 1
        out.tool_log.append({"type": itype, "input": str(item.get("command") or item.get("changes") or "")[:200]})


def parse_events(jsonl: str) -> ParsedStep:
    """Fold ``codex exec --json`` JSONL into thread id, summed usage, tool-call count and the last agent message."""
    out = ParsedStep()
    for line in jsonl.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "thread.started":
            out.session_id = ev.get("thread_id") or out.session_id
        elif kind == "turn.completed":
            out.api_calls += 1
            for k in USAGE_KEYS:
                out.usage[k] += int((ev.get("usage") or {}).get(k) or 0)
        elif kind in ("item.completed", "item.started"):
            _record_item(out, ev, kind)
        elif kind in ("turn.failed", "error"):
            out.errors.append(ev.get("error") or ev.get("message") or line[:300])
    return out


def build_argv(
    codex_bin: str,
    *,
    model: str | None,
    schema_file: str | None,
    last_file: str,
    cwd: str | None,
    resume_thread: str | None,
    ephemeral: bool,
    effort: str | None,
    sandbox: str = "read-only",
) -> list[str]:
    """No secrets on the command line: auth comes from CODEX_HOME/auth.json."""
    argv = shlex.split(codex_bin) + ["exec"]
    # sandbox "none" = --dangerously-bypass-approvals-and-sandbox, ONLY for an externally sandboxed container
    # (the flag's own help text); never on the host.
    bypass = sandbox == "none"
    if resume_thread:
        # `codex exec resume` (0.153) shares --json/-o/--output-schema/-m/-c/--skip-git-repo-check but has no
        # --sandbox or -C flag; the sandbox goes through the config override and the cwd is the thread's own.
        argv += ["resume", resume_thread]
        argv += (
            ["--dangerously-bypass-approvals-and-sandbox"] if bypass else ["-c", f"sandbox_mode={json.dumps(sandbox)}"]
        )
    else:
        argv += ["--dangerously-bypass-approvals-and-sandbox"] if bypass else ["--sandbox", sandbox]
        if cwd:
            argv += ["-C", cwd]
    argv += ["--json", "-o", last_file, "--skip-git-repo-check"]
    if schema_file:
        argv += ["--output-schema", schema_file]
    if model:
        argv += ["-m", model]
    if effort:
        argv += ["-c", f"model_reasoning_effort={json.dumps(effort)}"]
    if ephemeral and not resume_thread:
        argv.append("--ephemeral")
    argv.append("-")  # prompt on stdin
    return argv


def subscription_env() -> dict[str, str] | None:
    """The subprocess environment, without API credentials when the step must use the ChatGPT login.

    Subscription-only is the default; ``CODEX_FORCE_CHATGPT_AUTH`` asks for the same even where API-key
    billing is allowed. ``None`` means "inherit this process's environment unchanged", as before.
    """
    if not (subscription_required() or os.environ.get("CODEX_FORCE_CHATGPT_AUTH")):
        return None
    return {k: v for k, v in os.environ.items() if k not in API_CREDENTIALS}


def run_codex(argv: list[str], prompt: str, *, timeout_s: float, last_file: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            argv, input=prompt, capture_output=True, text=True, timeout=timeout_s, env=subscription_env()
        )
    except OSError as exc:
        proc = subprocess.CompletedProcess(argv, 1, "", f"codex could not start: {exc.strerror}")
    except subprocess.TimeoutExpired as exc:
        # Report the timeout as the reserved exit code instead of crashing, so the host does not replay the
        # abandoned turn; whatever Codex printed before the deadline stays available for the error tail.
        partial = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        proc = subprocess.CompletedProcess(argv, EXIT_TIMED_OUT, partial or "", f"codex timed out after {timeout_s:g}s")
    parsed = parse_events(proc.stdout)
    # Preserve the subprocess result consumed by the benchmark session driver.
    folded: dict[str, Any] = {
        "thread_id": parsed.session_id,
        "usage": parsed.usage,
        "turns": parsed.api_calls,
        "tool_calls": parsed.tool_calls,
        "tool_log": parsed.tool_log,
        "agent_message": parsed.text,
        "errors": parsed.errors,
    }
    answer = None
    last_text = ""
    try:
        last_text = Path(last_file).read_text()
        answer = _last_codex_object(last_text)
    except OSError:
        pass
    folded["last_text"] = last_text
    if answer is None and folded["agent_message"]:
        answer = _last_codex_object(str(folded["agent_message"]))
    # Codex reports failures as `error`/`turn.failed` events on stdout with an empty stderr; keep both tails.
    tail = ((proc.stderr or "")[-1000:] + "\n" + (proc.stdout or "")[-1000:]).strip()
    folded.update(
        {"answer": answer, "returncode": proc.returncode, "stderr_tail": tail, "duration_s": time.time() - t0}
    )
    return folded


def envelope_fields(folded: dict[str, Any], model: str | None) -> dict[str, Any]:
    u = folded["usage"]
    return {
        "input_tokens": u["input_tokens"],
        "output_tokens": u["output_tokens"],
        "usage": {
            **u,
            "provider": "openai",
            "model": model,
            "cost_usd": None,
            "note": "ChatGPT-login usage: plan limits/credits, no USD list price applied",
        },
        "api_calls": max(folded["turns"], 1),
        "tool_calls": folded["tool_calls"],
        "tool_log": folded["tool_log"],
    }


def cmd_step(args: argparse.Namespace) -> int:
    prompt = sys.stdin.read()
    encoded: list[str] = []
    strict_file = None
    if args.schema:
        strict, encoded = strictify(json.loads(Path(args.schema).read_text()))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
            json.dump(strict, sf)
            strict_file = sf.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as lf:
        last_file = lf.name
    try:
        argv = build_argv(
            args.codex,
            model=args.model,
            schema_file=strict_file,
            last_file=last_file,
            cwd=args.cwd,
            resume_thread=None,
            ephemeral=True,
            effort=args.effort,
            sandbox=args.sandbox,
        )
        folded = run_codex(argv, prompt, timeout_s=driver_timeout(args.timeout), last_file=last_file)
    finally:
        for p in (last_file, strict_file):
            try:
                if p:
                    os.unlink(p)
            except OSError:
                pass
    if folded["returncode"] != 0 or folded["errors"]:
        reported = folded["errors"] or folded["stderr_tail"]
        print(f"codex exited {folded['returncode']}: {reported}", file=sys.stderr)
        return int(folded["returncode"]) or 1
    answer = decode_answer(folded["answer"], encoded) or {}
    if not answer:
        print(f"codex printed no JSON answer; errors={folded['errors']}", file=sys.stderr)
        return 3
    out = {
        **answer,
        **envelope_fields(folded, args.model),
        "thread_id": folded["thread_id"],
        "codex_errors": folded["errors"],
    }
    # CommandExecutor reads reasoning/state_patch/action/usage; everything else lands in Envelope.extra.
    print(json.dumps(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("step", help="step wrapper for `stepbook run --executor command`", allow_abbrev=False)
    s.add_argument("--codex", default=os.environ.get("CODEX_BIN", "codex"), help="codex binary (or a fake for tests)")
    s.add_argument("--model", default=None)
    s.add_argument("--effort", default=None, help="model_reasoning_effort override, e.g. medium")
    s.add_argument("--schema", default=None, help="JSON schema file; use {schema_file} in --command")
    s.add_argument("--cwd", default=None)
    s.add_argument("--sandbox", default="read-only", choices=["read-only", "workspace-write", "none"])
    s.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="the host's per-query limit in seconds; the Codex turn gets 30 s less so its timeout is reported",
    )
    s.set_defaults(func=cmd_step)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
