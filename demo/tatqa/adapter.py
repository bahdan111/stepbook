"""Eval adapter for the TAT-QA demo RAG. Implements the eval-adapter command protocol:

    adapter.py describe                 -> hypotheses available, budget, current config
    adapter.py apply <h_id>             -> snapshot config, apply the hypothesis' settings
    adapter.py run <label>              -> run the eval once, print normalised metrics JSON and nothing else
    adapter.py decide <h_id> keep|revert-> keep: drop snapshot; revert: restore config from snapshot

The harness never sees any of this code, and never reads these sentences either: the exit code is the answer.
0 done, 1 refused (stdout is the reason), 3 from `apply` for "applied, and a code change is needed" with the
instructions on stdout; any other code is this adapter failing.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
import os
HYPS = Path(os.environ.get("TATQA_HYPS", HERE / "hypotheses.json"))
SNAP = HERE / ".snapshots"
STATE = HERE / ".snapshots" / "applied.json"


REGISTERED = SNAP / "registered.json"


class Refused(Exception):
    """A verb declined: the message is the reason, printed on stdout, and the process exits 1."""


class CodeChangeRequired(Exception):
    """`apply` applied the hypothesis and a code change is needed: the message is the instructions (exit 3)."""


def hyps() -> dict:
    base = json.loads(HYPS.read_text())
    if REGISTERED.exists():
        base.update(json.loads(REGISTERED.read_text()))
    return base


def register(payload: str) -> str:
    """Accept a mined hypothesis {id, kind, rationale, set|code_change}. Config keys must exist in config.json."""
    try:
        h = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise Refused("Error: register needs a JSON object.") from exc
    if not {"id", "kind", "rationale"} <= set(h):
        raise Refused("Error: register needs id, kind, rationale.")
    cfg = json.loads(CONFIG.read_text())
    if h["kind"] in ("config", "prompt"):
        if not isinstance(h.get("set"), dict) or not h["set"]:
            raise Refused(f"Error: {h['id']} needs a non-empty 'set' object.")
        unknown = [k for k in h["set"] if k not in cfg]
        if unknown:
            raise Refused(f"Error: {h['id']} sets unknown config keys {unknown}; known: {sorted(cfg)}.")
    elif h["kind"] == "code":
        if not h.get("code_change"):
            raise Refused(f"Error: {h['id']} needs 'code_change' instructions.")
    else:
        raise Refused(f"Error: unknown kind {h['kind']!r}.")
    SNAP.mkdir(exist_ok=True)
    reg = json.loads(REGISTERED.read_text()) if REGISTERED.exists() else {}
    reg[h["id"]] = {k: h[k] for k in ("kind", "rationale", "set", "code_change") if k in h}
    REGISTERED.write_text(json.dumps(reg, indent=1))
    return f"Registered {h['id']} ({h['kind']}): {h['rationale']}"


def describe() -> str:
    h = hyps()
    lines = [f"  {k} ({v['kind']}): {v['rationale']}" for k, v in h.items()]
    n = json.loads(CONFIG.read_text()).get("n_cases", "?")
    return (f"System: BM25 + Claude RAG over TAT-QA ({n} table-grounded questions). Metrics: accuracy (primary, noisy: "
            "about +-0.09 standard error at n=30, +-0.06 at n=60), recall_at_k and mrr (deterministic). A run with "
            "valid=false (error_rate > 10%) must not be used for a decision: rerun it. Queue ALL of the following hypotheses "
            "in state before running the baseline; they are the only ones the adapter can apply.\nAvailable hypotheses:\n"
            + "\n".join(lines) + f"\nCurrent config: {CONFIG.read_text().strip()}")


def apply(hid: str) -> str:
    h = hyps()
    if hid not in h:
        raise Refused(f"Error: unknown hypothesis {hid}. Known: {', '.join(h)}.")
    if STATE.exists():
        cur = json.loads(STATE.read_text())
        raise Refused(f"Error: {cur['h']} is still applied; decide it first.")
    SNAP.mkdir(exist_ok=True)
    shutil.copy(CONFIG, SNAP / "config.before.json")
    STATE.write_text(json.dumps({"h": hid}))
    if h[hid]["kind"] == "code":
        raise CodeChangeRequired(h[hid]["code_change"])
    cfg = json.loads(CONFIG.read_text())
    cfg.update(h[hid]["set"])
    CONFIG.write_text(json.dumps(cfg, indent=2))
    return f"Applied {hid} ({h[hid]['kind']}): {h[hid]['rationale']}. Config now {json.dumps(h[hid]['set'])}."


DECIDED = SNAP / "decided.json"


def _decided() -> dict:
    return json.loads(DECIDED.read_text()) if DECIDED.exists() else {}


def run(label: str) -> str:
    sys.path.insert(0, str(HERE))
    import rag  # noqa: E402
    m = rag.cmd_run(rag.load_config(), label)
    applied = json.loads(STATE.read_text())["h"] if STATE.exists() else None
    undecided = [h for h in hyps() if h not in _decided() and h != applied]
    payload = {k: v for k, v in m.items() if k != "config"}
    payload["applied"] = applied
    payload["undecided_hypotheses"] = undecided
    return json.dumps(payload)


def decide(hid: str, what: str) -> str:
    if not STATE.exists():
        raise Refused("Error: nothing is applied.")
    cur = json.loads(STATE.read_text())
    if cur["h"] != hid:
        raise Refused(f"Error: {hid} is not the applied hypothesis ({cur['h']}).")
    if what == "revert":
        shutil.copy(SNAP / "config.before.json", CONFIG)
    dec = _decided(); dec[hid] = what
    SNAP.mkdir(exist_ok=True); DECIDED.write_text(json.dumps(dec))
    STATE.unlink()
    return f"{hid} {what}: " + ("kept in configuration." if what == "keep" else "config restored.")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: adapter.py describe|apply <h>|run <label>|decide <h> keep|revert", file=sys.stderr)
        return 2
    cmd, args = argv[0], argv[1:]
    try:
        if cmd == "reset":
            for f in (STATE, DECIDED, REGISTERED):
                if f.exists():
                    f.unlink()
            print("reset: applied/decided/registered bookkeeping cleared")
        elif cmd == "register" and len(args) == 1:
            print(register(args[0]))
        elif cmd == "status":
            # `hypotheses`: the ids apply accepts, as values; describe's prose is for the agent only.
            # `phase`/`phases_complete`: this adapter has no `phase` verb, so there is one stage and it is open.
            print(json.dumps({"applied": json.loads(STATE.read_text())["h"] if STATE.exists() else None,
                              "decided": _decided(),
                              "hypotheses": [{"id": k, "kind": v["kind"]} for k, v in hyps().items()],
                              "phase": None, "phases_complete": False}))
        elif cmd == "describe":
            print(describe())
        elif cmd == "apply" and len(args) == 1:
            print(apply(args[0]))
        elif cmd == "run":
            print(run(args[0] if args else "run"))
        elif cmd == "decide" and len(args) == 2 and args[1] in ("keep", "revert"):
            print(decide(args[0], args[1]))
        else:
            print(f"Error: bad adapter command {argv}", file=sys.stderr)
            return 2
    except Refused as refusal:
        print(refusal)              # the reason, for the agent to read
        return 1
    except CodeChangeRequired as request:
        print(request)              # the instructions for the coding step
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
