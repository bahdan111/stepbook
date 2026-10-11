"""The ``adapter`` subcommands and the ``init`` scaffolding that writes the adapter and eval-config templates."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from ..adapters.generic import main as generic_main
from ..contracts.eval import AdapterReply
from .skill import builtin_skill_path


def _adapter_call(args: argparse.Namespace, *a: str) -> AdapterReply:
    """One adapter invocation, with the proposed evidence roots passed down in the child environment.

    The exit code decides whether the verb worked; the text is only quoted back to the user.
    """
    child_env = os.environ.copy()
    if getattr(args, "evidence_root", None):
        child_env["STEPBOOK_EVIDENCE_ROOTS"] = json.dumps([str(Path(p).resolve()) for p in args.evidence_root])
    else:
        child_env.pop("STEPBOOK_EVIDENCE_ROOTS", None)
    pr = subprocess.run([*shlex.split(args.adapter), *a], capture_output=True, text=True, timeout=600, env=child_env)
    return AdapterReply(exit_code=pr.returncode, stdout=pr.stdout.strip(), error=(pr.stdout + pr.stderr).strip())


def _status_hypotheses(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """The ids ``apply`` can be asked for, from the list ``status`` publishes: (ids, problems).

    ``describe`` is prose for the agent and is never parsed, so the only authority on what exists is the
    adapter's own ``status`` JSON.
    """
    reply = _adapter_call(args, "status")
    if not reply.ok:
        return [], [f"status failed (exit {reply.exit_code}): {reply.text[:200]}"]
    try:
        status = json.loads(reply.stdout)
    except json.JSONDecodeError:
        return [], [f"status exited 0 but printed no JSON: {reply.stdout[:200]}"]
    rows = status.get("hypotheses") if isinstance(status, dict) else None
    if not isinstance(rows, list):
        return [], ["status JSON must carry 'hypotheses': a list of {id, kind} objects (it may be empty)"]
    ids = [row["id"] for row in rows if _is_hypothesis_row(row)]
    if len(ids) != len(rows):
        return [], ["every entry of status 'hypotheses' must be an object with a non-empty 'id' and 'kind'"]
    return ids, []


def _is_hypothesis_row(row: object) -> bool:
    """One published hypothesis: the id ``apply`` is called with, and the kind the contract asks for."""
    return isinstance(row, dict) and all(
        isinstance(row.get(field), str) and row[field].strip() for field in ("id", "kind")
    )


def _check_apply_and_revert(args: argparse.Namespace, ids: list[str]) -> list[str]:
    """Apply the first hypothesis status listed that the adapter accepts, then decide revert on it."""
    if not ids:
        print(
            "note: empty hypothesis queue is valid for evidence-driven discovery; apply/revert remains untested until a proposal exists"
        )
        return []
    # try each listed hypothesis: an adapter may legitimately refuse some (unmeasurable, or blocked by a
    # dependency) with an explanatory error, and that is correct behaviour rather than a contract failure.
    applied = None
    refusals: list[str] = []
    for cand in ids:
        reply = _adapter_call(args, "apply", cand)
        if reply.ok or reply.code_change_required:  # a code hypothesis is applied too; its instructions follow
            applied = cand
            break
        refusals.append(f"{cand}: {(reply.text.splitlines() or [''])[0][:120]}")
    if applied is None:
        return ["no listed hypothesis could be applied; each refused:\n    " + "\n    ".join(refusals)]
    if refusals:
        print(
            f"note: {len(refusals)} hypothesis/es refused apply with an explanation (fine): "
            + "; ".join(r.split(":")[0] for r in refusals)
        )
    reply = _adapter_call(args, "decide", applied, "revert")
    if not reply.ok:
        return [f"decide {applied} revert did not confirm (exit {reply.exit_code}): {reply.text[:200]}"]
    return []


def _zero_spend_note(result: object) -> str | None:
    """An evaluator that declared a cost artifact and billed exactly nothing, which is what model calls
    refused before dispatch look like. Absent or null totals mean no bill was ever claimed, so they are not
    suspicious and never raise this note. Advisory only: it never becomes a problem and never changes the exit code."""
    if not isinstance(result, dict):  # a non-object body is already reported by the key checks below
        return None
    cost = result.get("cost")
    if not isinstance(cost, dict):
        return None
    totals = {name: cost[name] for name in ("tokens", "usd") if type(cost.get(name)) in (int, float)}
    if not totals or any(value != 0 for value in totals.values()):
        return None
    return (
        "note: suspicious result: the evaluator declared a cost artifact and it totals "
        + ", ".join(f"{name}={value}" for name, value in totals.items())
        + ". If this evaluator calls a model, its calls may have been refused and the cases scored without one;"
        " read the run's stderr before trusting the metric."
    )


def _check_run_result(args: argparse.Namespace) -> list[str]:
    """One real eval run, checked for the keys a result must carry before any run is trusted."""
    problems: list[str] = []
    reply = _adapter_call(args, "run", "adapter-check")
    if not reply.ok:
        return [f"run exited {reply.exit_code} instead of printing a receipt: {reply.text[:200]}"]
    try:
        m = json.loads(reply.stdout)
        note = _zero_spend_note(m)
        if note:
            print(note)
        for k in ("run_id", "cases", "primary_metric"):
            if k not in m:
                problems.append(f"run result lacks required key {k!r}")
        if "valid" not in m or "error_rate" not in m:
            problems.append("run result should carry 'valid' and 'error_rate' so invalid runs are never used")
    except json.JSONDecodeError:
        problems.append(f"run exited 0 but printed no JSON receipt: {reply.stdout[:200]}")
    return problems


def cmd_adapter_check(args: argparse.Namespace) -> int:
    """Exercise an adapter's verbs without a model: describe says something, status lists the hypotheses, the
    first one status lists is applied and reverted, and a run result's shape is validated if --run is given
    (runs cost money, so it is opt-in). Nothing here reads a sentence: describe only has to answer."""
    reply = _adapter_call(args, "describe")
    if not reply.ok or not reply.stdout:
        print(f"adapter problems:\n- describe failed (exit {reply.exit_code}): {reply.text[:200]}")
        return 1  # do not mutate or spend through an adapter that failed preflight
    ids, problems = _status_hypotheses(args)
    if problems:
        print("adapter problems:\n- " + "\n- ".join(problems))
        return 1  # without the adapter's own list there is nothing safe to apply
    problems = _check_apply_and_revert(args, ids)
    if args.run:
        problems += _check_run_result(args)
    print("adapter OK" if not problems else "adapter problems:\n- " + "\n- ".join(problems))
    return 0 if not problems else 1


def cmd_adapter_generic(args: argparse.Namespace) -> int:
    """Expose the bundled no-code adapter through the installed ``stepbook`` executable.

    This is intentionally a passthrough so an adapter command can be
    ``stepbook adapter generic /path/to/eval.json``. That works from the
    plugin's private virtualenv too; asking users to invoke ``python -m
    stepbook...`` would require the harness to be installed in their own
    interpreter.
    """
    return generic_main([str(args.eval_config), *args.adapter_args])


ADAPTER_TEMPLATE = '''#!/usr/bin/env python3
"""Eval adapter for <your system>. Keep this file in YOUR repository, not in stepbook.

THE EXIT CODE IS THE ANSWER. The harness never reads your sentences to find out what happened:

  exit 0  the verb did what it was asked (for `run`: the receipt is on stdout)
  exit 1  the verb refused; stdout is the reason, and it is shown to the agent as written. A `run` refusing
          before it started anything may print ONE JSON object instead of prose -- {"reason": "...",
          "execution_started": false, "cost": {"tokens": 0, "usd": 0}} -- which is the only way accounting
          can record that refusal as costing nothing instead of as an unknown bill
  exit 3  `apply` only: applied, AND this hypothesis needs a code change; stdout is the instructions
  anything else  your adapter failed, and the run treats the step as broken rather than refused

Contract (verified by `stepbook adapter check "python adapter.py"`):
  describe                  -> brief the agent in your own words: the system, the metrics and how noisy they
                               are, what it may try. Free text; nothing parses it, so write what helps a reader
  apply <id>                -> snapshot your config, apply the hypothesis, print what changed (exit 3 for a code one)
  run <label>               -> run your eval once, print ONE JSON object and nothing else (schema below)
  decide <id> keep|revert   -> keep: drop the snapshot; revert: restore it; print a confirmation
  reset                     -> clear applied/decided bookkeeping (called at episode start)
  status                    -> print JSON {"applied": <id or null>, "decided": {...}, "hypotheses": [{"id":
                               "h1", "kind": "config"}, ...]}: this, not describe, is where the harness and
                               `adapter check` learn which ids exist (the list may be empty). An adapter with a
                               `phase` verb also reports "phase" (the current one, or null) and
                               "phases_complete" (bool) here, because that is where the harness learns a stage
                               ended.
"""
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"          # whatever your system reads; adapt
SNAP = HERE / ".stepbook-adapter"   # add to .gitignore
HYPOTHESES = {
    "h1": {"kind": "config", "rationale": "describe the change", "set": {"top_k": 10}},
    # "h2": {"kind": "prompt", "rationale": "...", "set": {...}},
    # "h3": {"kind": "code", "rationale": "...", "code_change": "instructions for a coding step"},
}


def run_eval(label: str) -> dict:
    """Run YOUR eval and return the normalised result; it is printed as the whole of stdout.
    Required keys: run_id, cases ({case_id: "pass"|"fail"|"error"}),
    primary_metric ({name, value, direction: "higher"|"lower"}). Recommended: deterministic_metrics, metric_directions,
    secondary_metrics, cost ({usd|tokens}), duration_s, errors, error_rate, valid (false if >10% of cases errored)."""
    raise NotImplementedError("call your eval CLI here and map its output")


def main(argv):
    SNAP.mkdir(exist_ok=True)
    state = SNAP / "applied.json"
    decided = SNAP / "decided.json"
    registered = SNAP / "registered.json"
    if registered.exists():
        HYPOTHESES.update(json.loads(registered.read_text()))
    cmd, args = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "reset":
        if state.exists():
            print("Error: an experiment is still applied; resume or decide it before resetting.")
            return 1
        for f in (state, decided, registered):
            f.unlink(missing_ok=True)
        print("reset")
    elif cmd == "status":
        print(json.dumps({"applied": json.loads(state.read_text())["h"] if state.exists() else None,
                          "decided": json.loads(decided.read_text()) if decided.exists() else {},
                          # The list the harness applies from; describe's prose is for the agent only.
                          "hypotheses": [{"id": k, "kind": v["kind"]} for k, v in HYPOTHESES.items()],
                          "phase": None, "phases_complete": False}))   # this template has no `phase` verb
    elif cmd == "describe":
        print("System: <one line>. Metrics: <primary, noisy?>, <deterministic ones>.")
        print("Hypotheses you can try:")           # free text: write whatever helps the agent choose
        for k, v in HYPOTHESES.items():
            print(f"  {k} ({v['kind']}): {v['rationale']}")
    elif cmd == "apply" and len(args) == 1:
        if args[0] not in HYPOTHESES:
            print(f"Error: unknown hypothesis {args[0]}; known: {', '.join(HYPOTHESES) or 'none'}")
            return 1
        if state.exists():
            print("Error: another hypothesis is still applied; decide it first.")
            return 1
        h = HYPOTHESES[args[0]]
        shutil.copy(CONFIG, SNAP / "config.before.json")
        state.write_text(json.dumps({"h": args[0]}))
        if h["kind"] == "code":
            print(h["code_change"])     # applied; exit 3 says a coding step must carry out these instructions
            return 3
        cfg = json.loads(CONFIG.read_text()); cfg.update(h["set"]); CONFIG.write_text(json.dumps(cfg, indent=2))
        print(f"Applied {args[0]} ({h['kind']}): {h['rationale']}. Config now {json.dumps(h['set'])}.")
    elif cmd == "run":
        m = run_eval(args[0] if args else "run")
        dec = json.loads(decided.read_text()) if decided.exists() else {}
        applied = json.loads(state.read_text())["h"] if state.exists() else None
        m["applied"] = applied
        m["undecided_hypotheses"] = [h for h in HYPOTHESES if h not in dec and h != applied]
        print(json.dumps(m))            # the receipt alone; exit 0 says the eval ran
    elif cmd == "register" and len(args) == 1:   # mined hypotheses: {id, kind, rationale, set|code_change}
        h = json.loads(args[0])
        if not isinstance(h, dict) or not all(isinstance(h.get(k), str) and h[k] for k in ("id", "kind", "rationale")):
            print("Error: register needs id, kind, rationale")
            return 1
        if h["id"] in HYPOTHESES:
            print("Error: hypothesis id already exists; use a new id")
            return 1
        if h["kind"] not in ("config", "prompt", "code") or (h["kind"] == "code" and not h.get("code_change")) or (h["kind"] != "code" and not isinstance(h.get("set"), dict)):
            print("Error: hypothesis needs config/prompt set or code_change instructions")
            return 1
        HYPOTHESES[h["id"]] = h
        registered.write_text(json.dumps(HYPOTHESES))
        print(f"Registered {h['id']}")
    elif cmd == "decide" and len(args) == 2 and args[1] in ("keep", "revert"):
        if not state.exists() or json.loads(state.read_text())["h"] != args[0]:
            print("Error: not the applied hypothesis")
            return 1
        if args[1] == "revert" and (SNAP / "config.before.json").exists():
            shutil.copy(SNAP / "config.before.json", CONFIG)
        dec = json.loads(decided.read_text()) if decided.exists() else {}
        dec[args[0]] = args[1]; decided.write_text(json.dumps(dec)); state.unlink(missing_ok=True)
        print(f"{args[0]} {args[1]}: " + ("kept." if args[1] == "keep" else "config restored."))
    else:
        print(f"Error: bad adapter command {argv}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


# A valid JSON starting point for the no-code generic adapter. It deliberately contains
# representative names rather than pretending to know a project's evaluator or config.
# ``stepbook init --config-only`` is the shortest path for the common case where an
# existing eval already writes JSON or CSV results.
GENERIC_EVAL_TEMPLATE = {
    "name": "my measurable improvement loop",
    "run_command": "python -m your_project.eval --out {out}",
    "results_format": "json",
    "case_id_field": "id",
    "pass_field": "pass",
    "primary": {"name": "pass_rate", "direction": "higher"},
    "metric_fields": {},
    "config_file": "config.json",
    "protected_files": ["path/to/golden_cases.json"],
    "candidate_policy": {},
    "timeout_s": 7200,
    "hypotheses": {},
}


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold an eval-loop-adapter skill plus either a Python adapter or no-code eval.json configuration."""
    target = Path(args.dir)
    target.mkdir(parents=True, exist_ok=True)
    src = builtin_skill_path("eval-loop-adapter")
    dst = target / "skill"
    if not dst.exists():
        shutil.copytree(src, dst)
    gi = target / ".gitignore"
    if not gi.exists():
        gi.write_text(".stepbook-adapter/\n.stepbook/\n")
    if args.config_only:
        eval_config = target / "eval.json"
        if not eval_config.exists():
            eval_config.write_text(json.dumps(GENERIC_EVAL_TEMPLATE, indent=2) + "\n")
        adapter_cmd = f"stepbook adapter generic {eval_config}"
        print(
            f"scaffolded {dst} (skill) and {eval_config} (no-code generic adapter config).\n"
            "Next: set run_command, actual result fields, config_file and protected_files; configure cost_artifact and candidate_policy where applicable.\n"
            "Hypotheses are optional: leave the map empty for evidence-driven investigation and proposals. Then:\n"
            f'  stepbook adapter check "{adapter_cmd}"\n'
            f'  stepbook run {dst} --adapter "{adapter_cmd}" --runs-max 12'
        )
        return 0
    adapter = target / "adapter.py"
    if not adapter.exists():
        adapter.write_text(ADAPTER_TEMPLATE)
        adapter.chmod(0o755)
    cfg = target / "config.json"
    if not cfg.exists():
        cfg.write_text(
            json.dumps({"top_k": 3}, indent=2)
        )  # stub so `adapter check` and the template work before you adapt it
    print(
        f"scaffolded {dst} (skill) and {adapter} (adapter template).\nNext: fill run_eval(); HYPOTHESES may be empty. Then:\n"
        f'  stepbook adapter check "python {adapter}"\n  stepbook run {dst} --adapter "python {adapter}" --runs-max 20'
    )
    return 0
