"""Subcommands that read a run directory after the fact: the report file, status, report and recover."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time as _t
from pathlib import Path
from typing import Any

from ..envs.eval_loop import AdapterEvalEnv
from ..reporting import legacy_journal
from ..reporting.decision_audit import AGAINST, against_verdict, decision_audit, render_markdown
from ..reporting.report_rendering import (
    host_check_lines,
    shell_checkpoint_lines,
    shell_result_line,
    shell_state,
    shell_summary_lines,
    table_row,
)
from ..reporting.usage import _account_receipts, has_measured_tokens, usage_summary
from ..run_pointer import (
    ACTIVE,
    LAST_RUN,
    RESUMABLE,
    agent_stop_reason,
    classify_run,
    describe,
    pointer_dir,
    read_pointer,
    resume_hint,
)
from ..storage.events import EventLog
from ..storage.recovery import pending_actions, reconcile_receipt


def _cost_text(usd: float, complete: bool, *, known: bool = False, precision: int = 4) -> str:
    """Keep complete amounts byte-compatible; qualify every incomplete subtotal."""
    if complete:
        return f"${usd:.{precision}f}"
    return f"${usd:.{precision}f} (partial)" if known or usd else "unknown"


def _has_partial_evaluation_amount(events: list[dict]) -> bool:
    """Summary counts mark partial prices unknown, even when the reported subtotal is zero."""
    accounted, _, _ = _account_receipts(events)
    for event in accounted:
        if event.get("kind") != "step" or str(event.get("action", "")).split()[:1] != ["run"]:
            continue
        cost = (legacy_journal.step_receipt(event) or {}).get("cost") or {}
        if cost.get("known_usd") is not None:
            return True
    return False


def _usage_cost(usage: dict, events: list[dict]) -> str:
    known = any(
        row["records"] > row["unknown_cost_records"] for row in usage["components"].values()
    ) or _has_partial_evaluation_amount(events)
    return _cost_text(usage["known_total_usd"], usage["cost_complete"], known=known)


def _print_usage(usage: dict, events: list[dict]) -> None:
    print(
        f"Measured cost subtotal: {_usage_cost(usage, events)}; tokens: {_usage_tokens(usage, events)}; "
        f"complete: cost={usage['cost_complete']}, tokens={usage['tokens_complete']}"
    )


def _usage_tokens(usage: dict, events: list[dict]) -> str:
    """Qualify the existing subtotal, including measured zero in partial receipts."""
    total = usage["known_total_tokens"]
    if usage["tokens_complete"]:
        return f"{total:,}"
    if has_measured_tokens(events):
        return f"{total:,} (partial)"
    return "unknown"


def _usage_lines(usage: dict, events: list[dict]) -> list[str]:
    """Reported subtotal, completeness, the billing note."""
    return [
        f"- reported usage USD subtotal: {_usage_cost(usage, events)}; measured tokens: {_usage_tokens(usage, events)}",
        f"- accounting complete: cost={usage['cost_complete']}, tokens={usage['tokens_complete']}; see usage.json for components and gaps.",
        "",
        usage["billing_note"],
        "",
    ]


def _outcome_lines(events: list[dict], steps: list[dict], result: Any) -> list[str]:
    """How the run ended, plus the recorded completion artifact of any done step."""
    usage = usage_summary(events)
    controller = usage["components"]["controller"]
    agent_tokens = _usage_tokens({**usage, "known_total_tokens": result.total_tokens}, events)
    agent_cost = _cost_text(
        result.cost_usd if usage["cost_complete"] else controller["known_usd"],
        usage["cost_complete"],
        known=controller["records"] > controller["unknown_cost_records"],
        precision=2,
    )
    endings = {"budget_stop": "budget stop", "max_steps_reached": "max steps", "paused": "paused"}
    ending = "no terminal event recorded"
    for event in events:
        if event.get("kind") in endings:
            ending = endings[event["kind"]]
        elif event.get("kind") == "step" and event.get("done"):
            reason = agent_stop_reason(event)
            ending = f"stopped by the agent: {reason}" if reason is not None else "stop action / episode done"
    lines = [f"- steps: {len(steps)} | agent cost: {agent_cost} | agent tokens: {agent_tokens} | ended: {ending}"]
    for step in steps:
        info = step.get("info") or {}
        if step.get("done") and info.get("completion_id"):
            lines.append(f"- final evidence artifact (recorded): [{info['completion_id']}](<{info['artifact_path']}>)")
            lines.append(
                "  Persistence and receipt links are verified at execution, not the semantic correctness of the agent's conclusions."
            )
            if any(
                e.get("kind") == "local_completion_allowed_after_budget" and e.get("step") == step.get("step")
                for e in events
            ):
                lines.append(
                    "  Saved locally after the query reached the total spending limit; no additional paid operation was authorized."
                )
    return lines + host_check_lines(events)


def _flag_lines(env: Any, audit_rows: list[dict]) -> list[str]:
    """Warnings above the tables: a candidate left applied, decisions against the computed verdict."""
    lines: list[str] = []
    applied = getattr(env, "_applied", None)
    if applied:
        lines.append(
            f"- **Incomplete: candidate {applied} remains applied and undecided. It is not the accepted configuration.**"
        )
    overrides = against_verdict(audit_rows)
    if overrides:
        lines.append(
            f"- **{len(overrides)} decision(s) made against the computed verdict**: "
            + ", ".join(f"{r['hypothesis']} {r['decision']} vs {r['verdict']}" for r in overrides)
            + ". See 'Decisions vs computed verdicts'."
        )
    return lines


def _evaluation_cost(runs: dict) -> str:
    costs = [(p.get("cost") or {}) for ps in runs.values() for p in ps]
    amounts = [c.get("usd") if c.get("usd") is not None else c.get("known_usd") for c in costs]
    return _cost_text(
        sum(float(amount) for amount in amounts if amount is not None),
        all(c.get("usd") is not None for c in costs),
        known=any(amount is not None for amount in amounts),
        precision=2,
    )


def _ledger_lines(env: Any) -> list[str]:
    """Accepted reference, eval cost and the runs-by-hypothesis table from the environment's ledger."""
    runs = getattr(env, "_runs", None)
    if not runs:
        return []
    reference = getattr(env, "_reference_h", "baseline")
    lines: list[str] = []
    valid = [p for p in runs.get(reference, []) if env._valid_run(p)]
    if valid and not env._comparison_mismatch(valid):
        mean = sum(float(p["primary_metric"]["value"]) for p in valid) / len(valid)
        lines.append(
            f"- accepted reference from execution ledger: {reference}; measured mean primary={mean:.6g} over {len(valid)} valid run(s)"
        )
    lines.append(
        f"- eval runs: {sum(len(v) for v in runs.values())} | eval cost (usd, where reported): {_evaluation_cost(runs)}"
    )
    lines += [
        "",
        "## Runs by hypothesis",
        "",
        "| hypothesis | run | primary | deterministic | valid | cost |",
        "|---|---|---|---|---|---|",
    ]
    for h, ps in runs.items():
        for p in ps:
            pm = p.get("primary_metric") or {}
            lines.append(
                table_row(
                    [
                        h,
                        p.get("run_id", ""),
                        pm.get("value", ""),
                        json.dumps(p.get("deterministic_metrics") or {}),
                        p.get("valid", True),
                        json.dumps(p.get("cost") or {}),
                    ]
                )
            )
    return lines


def _checkpoint_lines(state: dict) -> list[str]:
    """The agent-maintained decision table and metric checkpoint from state.json."""
    if shell_state(state):
        return shell_checkpoint_lines(state)
    if not state.get("hypotheses"):
        return []
    lines = [
        "",
        "## Decisions",
        "",
        "| id | status | kind | rationale | evidence | origin | runs |",
        "|---|---|---|---|---|---|---|",
    ]
    for h, v in state["hypotheses"].items():
        lines.append(
            table_row(
                [
                    h,
                    v.get("status", ""),
                    v.get("kind", ""),
                    str(v.get("rationale", ""))[:80],
                    str(v.get("evidence", ""))[:160],
                    v.get("origin", ""),
                    len(v["runs"]) if isinstance(v.get("runs"), list) else "",
                ]
            )
        )
    return lines + [
        "",
        "Model checkpoint values below are agent-maintained; use the execution-ledger measurements above for arithmetic.",
        f"kept stack: {state.get('kept_stack')}",
        f"baseline: {state.get('baseline_metrics')} | reference: {state.get('reference_metrics')}",
    ]


def _verdict_sentences(step: dict) -> list[str]:
    """The verdict sentences one step showed the agent, for display only: every reader that decides anything
    from a verdict uses the step's recorded ``info["verdict"]`` (stepbook/reporting/decision_audit.py)."""
    return [line for line in str(step.get("result", "")).splitlines() if line.startswith("Computed verdict")]


def _activity_lines(env: Any, state: dict, steps: list[dict], events: list[dict]) -> list[str]:
    """Closing sections: computed verdicts, code changes, mined hypotheses, lessons, operator messages."""
    lines: list[str] = []
    verdicts = [line for e in steps for line in _verdict_sentences(e)]
    if verdicts:
        lines += ["", "## Computed verdicts", ""] + [f"- {v}" for v in verdicts]
    if getattr(env, "code_changes", None):
        lines += ["", "## Code changes", ""] + [
            f"- {c['h']}: {c.get('summary', '')[:200]} [{c.get('diff_stat', '')}]" for c in env.code_changes
        ]
    if getattr(env, "mined", None):
        lines += ["", "## Mined hypotheses", ""] + [
            f"- {m['proposal'].get('id')}: {m['proposal'].get('rationale', '')[:120]} -> {m.get('adapter_reply', '')[:80]}"
            for m in env.mined
        ]
    if state.get("lessons"):
        lines += ["", "## Lessons", ""] + [f"- {t}" for t in state["lessons"].values()]
    ops = [m for e in events if e.get("kind") == "operator_messages" for m in e.get("messages", [])]
    if ops:
        lines += ["", "## Operator messages", ""] + [f"- {m}" for m in ops]
    return lines


def write_report_md(run_dir: Path, env: Any, result: Any) -> Path:
    """Morning report as a file, from the environment's ledger and the event log. Returns the path."""
    events = EventLog(run_dir / "events.jsonl").read()
    steps = [e for e in events if e.get("kind") == "step"]
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    lines = [f"# Run report: {run_dir.name}", ""]
    lines += shell_report_header(state, events, run_dir)
    usage = usage_summary(events)
    (run_dir / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    lines += _usage_lines(usage, events)
    lines += _outcome_lines(events, steps, result)
    audit_rows = decision_audit(events)
    lines += _flag_lines(env, audit_rows)
    lines += _ledger_lines(env)
    lines += _checkpoint_lines(state)
    lines += render_markdown(audit_rows)
    lines += _activity_lines(env, state, steps, events)
    if shell_state(state):
        lines += ["", *continuation_lines(classify_active(run_dir))]
    out = run_dir / "report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def classify_active(run_dir: Path) -> dict:
    """Classify as ``status`` does: a run without a terminal record that .stepbook/ACTIVE here names is running."""
    active = read_pointer(pointer_dir(), ACTIVE)
    return classify_run(run_dir, active=active is not None and active.resolve() == run_dir.resolve())


def _status_location(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    base = pointer_dir()
    active = read_pointer(base, ACTIVE)
    positional = getattr(args, "run_dir_positional", None)
    if positional and args.run_dir and Path(positional).resolve() != Path(args.run_dir).resolve():
        raise ValueError("supply one run directory: positional or --run-dir/--log-dir")
    requested = args.run_dir or positional
    return (Path(requested) if requested else (active or read_pointer(base, LAST_RUN))), active


def cmd_status(args: argparse.Namespace) -> int:
    """Inspect an explicit directory, otherwise ACTIVE then LAST_RUN; shell outcomes precede commands."""
    try:
        run_dir, active = _status_location(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if not run_dir or not run_dir.exists():
        print(_missing_run_message(run_dir), file=sys.stderr)
        return 2
    is_active = active is not None and active.resolve() == run_dir.resolve()
    info = classify_run(run_dir, active=is_active)
    ev = EventLog(run_dir / "events.jsonl").read()
    st = read_run_state(run_dir)
    shell = shell_state(st)
    headline = shell_result_line(st, ev)
    if headline:
        print(headline)
    steps = [e for e in ev if e.get("kind") == "step"]
    last = ev[-1] if ev else {}
    age = int(_t.time() - float(last.get("ts", _t.time()))) if last else None
    print(f"run: {run_dir}\nstatus: {info['status']} - {describe(info, commands=not shell)}")
    print(
        f"steps: {len(steps)} | last event: {last.get('kind')} {age}s ago | active marker: {'yes' if is_active else 'no'}"
    )
    _print_usage(usage_summary(ev), ev)
    if not shell:
        if info["status"] == "completed":
            print(reopen_hint(info))
        if info["status"] in RESUMABLE:
            print(f"resume: {resume_hint(info)}")
    _print_last_step(steps, ev)
    if st:
        _print_status_state(st, ev)
    inbox = run_dir / "inbox.txt"
    if inbox.exists() and inbox.read_text().strip():
        print("inbox (unconsumed): " + inbox.read_text().strip().replace("\n", " | ")[:200])
    if shell:
        print("\n".join(continuation_lines(info)))
    return 0


def _print_last_step(steps: list[dict], events: list[dict]) -> None:
    if not steps:
        return
    last = steps[-1]
    print(
        f"last action: {str(last.get('action', ''))[:100]}\nlast result: {(str(last.get('result', '')).splitlines() or [''])[0][:120]}"
    )
    if (last.get("info") or {}).get("completion_id"):
        print("completion artifact (recorded): " + last["info"]["artifact_path"])
        if any(
            e.get("kind") == "local_completion_allowed_after_budget" and e.get("step") == last.get("step")
            for e in events
        ):
            print("completion saved locally after total spending limit; no additional paid operation authorized")


def reopen_hint(info: dict) -> str:
    return f"For new work, reopen with: {resume_hint(info)} --reopen '<note>'"


def read_run_state(run_dir: Path) -> dict:
    path = run_dir / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def continuation_lines(info: dict) -> list[str]:
    if info["status"] == "completed":
        return [reopen_hint(info)]
    if info["status"] == "failed_checks":
        return [f"Reopen with: {resume_hint(info)} --reopen 'fix failing checks and finish'"]
    if "stop_reason" in info:
        return [f"Reopen with: {resume_hint(info)} --reopen '<note>'"]
    if info["status"] in RESUMABLE:
        return [f"resume: {resume_hint(info)}"]
    return []


def shell_report_header(state: dict, events: list[dict], run_dir: Path) -> list[str]:
    headline = shell_result_line(state, events)
    if not headline:
        return []
    info = classify_active(run_dir)
    return [headline, f"status: {info['status']} - {describe(info, commands=False)}", ""]


def _print_status_state(st: dict, events: list[dict]) -> None:
    if shell_state(st):
        print("\n".join(shell_summary_lines(st, events)))
        return
    for k in ("current", "kept_stack", "checks_status", "budget"):
        if k in st:
            print(f"{k}: {json.dumps(st[k])[:200]}")
    _print_ideas(st)


def _print_decision_audit(ev: list[dict]) -> None:
    """One line per decision with its computed verdict, and the agent's reason where they disagree."""
    for row in decision_audit(ev):
        flag = "  <-- AGAINST COMPUTED VERDICT" if row["relation"] == AGAINST else ""
        print(
            f"decision: {row['hypothesis']} {row['decision']} | verdict: {row['verdict'] or 'none'} | {row['relation']}{flag}"
        )
        if row["relation"] == AGAINST:
            print(f"  agent's reason: {row['reason'][:300]}")


def _print_execution_ledger(steps: list[dict]) -> None:
    """Replay the steps into a ledger and print the measured reference mean and any applied candidate."""
    ledger = AdapterEvalEnv(adapter="")
    ledger.restore_from_events(steps, check_adapter=False)
    valid = [p for p in ledger._runs.get(ledger._reference_h, []) if ledger._valid_run(p)]
    if valid and not ledger._comparison_mismatch(valid):
        mean = sum(float(p["primary_metric"]["value"]) for p in valid) / len(valid)
        print(
            f"execution ledger: accepted reference={ledger._reference_h}; measured mean primary={mean:.6g} over {len(valid)} valid runs; eval attempts consumed={ledger.runs_used}"
        )
    if ledger._applied:
        print(f"Incomplete: candidate {ledger._applied} remains applied and undecided")


def _print_checkpoint(st: dict) -> None:
    """The agent-maintained checkpoint: kept stack, metrics, hypotheses, lessons, operator requirements."""
    if shell_state(st):
        print("\n".join(shell_checkpoint_lines(st)))
        return
    print(
        "Model checkpoint below is agent-maintained; execution-ledger values above are computed from recorded results:"
    )
    print(f"kept stack: {st.get('kept_stack')}")
    print(
        f"baseline: {st.get('baseline_metrics')} | reference: {st.get('reference_metrics')} | budget: {st.get('budget')}"
    )
    _print_ideas(st)
    if st.get("lessons"):
        print("lessons:")
        [print(f"  - {t}") for t in st["lessons"].values()]
    if st.get("user_requirements"):
        print("operator requirements:")
        [print(f"  - [{'x' if r.get('done') else ' '}] {r.get('text')}") for r in st["user_requirements"]]


def _print_ideas(st: dict) -> None:
    """Preserve idea wording and append the available origin and run count."""
    for h, v in (st.get("hypotheses") or {}).items():
        details = f" | origin: {v['origin']}" if "origin" in v else ""
        if isinstance(v.get("runs"), list):
            details += f" | runs: {len(v['runs'])}"
        print(
            f"  {h:6} {v.get('status', ''):12} {v.get('kind', ''):7} {str(v.get('rationale', ''))[:60]:60} | {str(v.get('evidence', ''))[:110]}{details}"
        )


def _print_computed_verdicts(steps: list[dict]) -> None:
    """The first verdict sentence of every step that showed one; a step that only mentions the words
    'Computed verdict' mid-line has no sentence to print, and used to raise IndexError here."""
    verdicts = [sentences[0] for sentences in map(_verdict_sentences, steps) if sentences]
    if verdicts:
        print("computed verdicts:")
        for line in verdicts:
            print("  " + line[:160])


def _prompt_tokens(step: dict) -> Any:
    """The recorded prompt count, unless the executor explicitly left usage unknown."""
    return None if step.get("usage_unknown") else step.get("prompt_tokens")


def _table_widths(rows: list[list[str]]) -> list[int]:
    """Preserve redirected output; shrink text, then omit columns to fit a terminal."""
    if not sys.stdout.isatty():
        return [4, 40, 60, 6]
    widths = [max(4, *(len(row[0]) for row in rows)), 40, 60, max(6, *(len(row[3]) for row in rows))]
    available = shutil.get_terminal_size(fallback=(100, 24)).columns
    while sum(widths) + sum(w > 0 for w in widths) - 1 > available:
        text_column = max((1, 2), key=lambda i: widths[i])
        if widths[text_column] > 6:
            widths[text_column] -= 1
            continue
        for column in (2, 1, 3, 0):
            if widths[column]:
                widths[column] = 0
                break
    return widths


def _print_step_table(steps: list[dict]) -> None:
    rows = [["step", "action", "result", "tok"]]
    for step in steps:
        tokens = _prompt_tokens(step)
        rows.append(
            [
                str(step["step"]),
                str(step.get("action", "")),
                (str(step.get("result", "")).splitlines() or [""])[0],
                "unknown" if tokens is None else str(tokens),
            ]
        )
    widths = _table_widths(rows)
    for row in rows:
        print(
            " ".join(
                f"{value:>{width}}" if i in (0, 3) else f"{value[:width]:{width}}"
                for i, (value, width) in enumerate(zip(row, widths, strict=True))
                if width
            )
        )


def _print_run_tail(args: argparse.Namespace, ev: list[dict], steps: list[dict]) -> None:
    """Run totals, the prompt-size trend, the last --last steps and the last rejected patches."""
    rej = [e for e in ev if e.get("kind") == "patch_rejected"]
    print(f"run: {args.run_dir}")
    waits = sum(1 for e in ev if e.get("kind") == "rate_limited")
    # The runtime no longer writes rate_limited; display the count for journals that carry it.
    rate_limit = f"rate-limit waits {waits} | " if waits else ""
    print(
        f"steps {len(steps)} | rejected patches {len(rej)} | resumes {sum(1 for e in ev if e.get('kind') == 'resumed')} | "
        f"{rate_limit}budget stop {any(e.get('kind') == 'budget_stop' for e in ev)}"
    )
    if steps:
        pt = [_prompt_tokens(e) for e in steps]
        median = sorted(pt)[len(pt) // 2] if all(value is not None for value in pt) else None
        trend = "/".join("unknown" if value is None else str(value) for value in (pt[0], median, pt[-1]))
        rc = [e.get("rendered_prompt_chars", 0) for e in steps]
        print(f"prompt tokens (provider) first/median/last: {trend} | rendered chars first/last: {rc[0]}/{rc[-1]}")
    _print_step_table(steps[-args.last :])
    for e in rej[-5:]:
        print(f"  rejected @{e['step']}: {e['problems'][0][:100]}")


def cmd_report(args: argparse.Namespace) -> int:
    """Render a run report from its event log: per-step table, totals, rejections, resumes, budget stops. For an
    eval loop with a state.json, also the morning summary: kept stack, decisions with evidence, runs, lessons.

    A directory with no journal must be refused like ``recover``, without creating artifacts."""
    run_dir = Path(args.run_dir)
    if not (run_dir / "events.jsonl").is_file():
        print(f"no run at {run_dir}", file=sys.stderr)
        return 2
    ev = EventLog(run_dir / "events.jsonl").read()
    st = read_run_state(run_dir)
    for line in shell_report_header(st, ev, run_dir):
        print(line)
    accounting = usage_summary(ev)
    _print_usage(accounting, ev)
    steps = [e for e in ev if e.get("kind") == "step"]
    state_path = run_dir / "state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        if shell_state(st):
            print("\n".join(shell_summary_lines(st, ev)))
        elif "hypotheses" in st:
            print("== Eval loop summary")
            _print_decision_audit(ev)
            _print_execution_ledger(steps)
            _print_checkpoint(st)
            _print_computed_verdicts(steps)
            print()
    _print_run_tail(args, ev, steps)
    if shell_state(st):
        print("\n".join(continuation_lines(classify_active(run_dir))))
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    path = Path(args.run_dir) / "events.jsonl"
    if not path.is_file():
        print("No event log at the requested run directory", file=sys.stderr)
        return 2
    log = EventLog(path)
    try:
        if args.receipt:
            if not args.action_id or not args.reason:
                raise ValueError("--receipt requires --action-id and --reason; verify the external outcome first")
            reconcile_receipt(log, args.action_id, Path(args.receipt), args.reason)
        elif args.action_id or args.reason:
            raise ValueError("--action-id and --reason are only used with --receipt")
        print(
            json.dumps(
                {
                    "pending_actions": pending_actions(log),
                    "note": "Read-only unless --receipt is supplied. Never infer execution from intent alone.",
                },
                indent=2,
            )
        )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _missing_run_message(run_dir: Path | None) -> str:
    return f"no run at {run_dir}" if run_dir else "no run found; start a run or supply --run-dir"
