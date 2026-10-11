"""Reading runs journaled before the structured adapter journal: the only module that parses adapter wording.

Every adapter-route step now records what it changed in ``StepResult.info`` -- ``ok``, ``adapter_exit`` and the
ledger facts (``applied``, ``reference``, ``receipt``, ``decision``, ``hypothesis``, ``investigation``,
``phases_complete``, ``verdict``) -- and replay, the reports and the decision audit read those fields. Runs
recorded before those fields existed carry the same facts only inside the observation the agent was shown: any
finished run of an older stepbook.

So this module exists for one reason: a finished old run must still produce a report, a status line, a usage
total and a decision audit. Resuming one on the adapter route is refused instead
(``stepbook/envs/eval_loop.py``), so nothing here ever decides what happens next in a live run.

Nothing may be added here, and no other module may read adapter wording: a reader takes the structured field
when the step has one (``structured``) and comes here only when it does not.
"""

from __future__ import annotations

import json
import re

# The observation wording the adapter environment has always journaled, as regular expressions over one line.
VERDICT_RE = re.compile(
    r"Computed verdict for (\S+) after .*?: (PROVISIONAL )?\*\*(keep|revert|inconclusive|no change)(?: \(provisional[^)]*\))?\*\*"
)
DECIDE_RE = re.compile(r"^\s*decide\s+(\S+)\s+(keep|revert|skip)\b", re.IGNORECASE)
RECEIPT_HEADER = "Eval finished:\n"
UNSTARTED_HEADER = "Error: evaluation not started:\n"


def structured(info: dict | None) -> bool:
    """Whether a step carries the structured adapter facts. Every adapter-route step records ``ok``."""
    return "ok" in (info or {})


def _result_json(result: str, header: str) -> dict | None:
    """The JSON document a legacy step printed under one of its observation headers."""
    if not result.startswith(header):
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(result.split("\n", 1)[1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def step_receipt(event: dict) -> dict | None:
    """One run step's receipt: the journaled field, or the observation of a step older than that field."""
    info = event.get("info") or {}
    if structured(info):
        receipt = info.get("receipt")
        return receipt if isinstance(receipt, dict) else None
    return _result_json(str(event.get("result") or ""), RECEIPT_HEADER)


def unstarted_cost(event: dict) -> dict | None:
    """A legacy run refusal that declared it spent nothing, which usage accounting still counts as measured."""
    if structured(event.get("info")):
        return None
    refusal = _result_json(str(event.get("result") or ""), UNSTARTED_HEADER)
    if refusal and refusal.get("execution_started") is False and refusal.get("cost") == {"tokens": 0, "usd": 0}:
        return refusal
    return None


def step_verdicts(event: dict) -> list[dict]:
    """The computed verdicts a step recorded: its ``info["verdict"]``, or its sentences on a legacy step."""
    info = event.get("info") or {}
    if structured(info):
        verdict = info.get("verdict")
        if isinstance(verdict, dict) and verdict.get("hypothesis") and verdict.get("verdict"):
            return [
                {
                    "hypothesis": str(verdict["hypothesis"]),
                    "verdict": str(verdict["verdict"]),
                    "provisional": bool(verdict.get("provisional")),
                }
            ]
        return []
    found = []
    for line in str(event.get("result") or "").splitlines():
        match = VERDICT_RE.search(line)
        if match:
            found.append(
                {
                    "hypothesis": match.group(1),
                    "verdict": match.group(3),
                    "provisional": bool(match.group(2)) or "(provisional" in line,
                }
            )
    return found


def step_decision(event: dict) -> dict | None:
    """The decision a step recorded, or None when it made none: the adapter accepted it on a legacy step iff
    the observation repeated it back without a warning."""
    info = event.get("info") or {}
    if structured(info):
        return (
            {"hypothesis": str(info["hypothesis"]), "decision": str(info["decision"])}
            if info.get("decision") and info.get("hypothesis")
            else None
        )
    match = DECIDE_RE.match(str(event.get("action") or ""))
    if not match or str(event.get("result") or "").startswith("Error"):
        return None
    return {"hypothesis": match.group(1), "decision": match.group(2).lower()}


def decision_accepted(result: str, hid: str, decision: str) -> bool:
    """Did a legacy observation record a decision the adapter accepted? Live steps read the verb's exit code."""
    return result.startswith(f"{hid} {decision}:") and "WARNING:" not in result and "Error:" not in result


def restore_step(env, verb: str, rest: list[str], result: str) -> None:
    """Apply one legacy step's effect on a read-only ledger, in the order the live step had.

    Read-only reports replay old runs here; a resume is refused before reaching this module, so no paid
    action and no adapter call ever depends on the wording below.
    """
    # Count every reserved attempt, including adapter failures and refusals: otherwise a report refunds
    # potentially paid failed runs. A local exhausted-budget refusal never reserved another attempt.
    if verb == "run" and not result.startswith("Error: run budget exhausted."):
        env.runs_used += 1
    if verb == "apply" and result.startswith("Applied") and rest:
        env._applied = rest[0]
    elif verb == "run" and (payload := _result_json(result, RECEIPT_HEADER)) is not None:
        env._restore_receipt(rest, payload)
    elif verb == "phase" and result.startswith("Advanced to"):
        env._runs.clear()
        env._reference_h = "baseline"
        env._run_dirs.clear()
    elif verb == "investigate" and rest and result.startswith("Investigation of"):
        env.investigations.append({"h": rest[0], "finding": result, "source": "event_replay"})
    elif verb == "decide" and len(rest) > 1 and rest[0] == env._applied and decision_accepted(result, *rest[:2]):
        if rest[1].lower() == "keep":
            env._reference_h = rest[0]
        env._applied = None
