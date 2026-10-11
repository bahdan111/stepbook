"""A small, transparent RAG over TAT-QA for the public eval-loop-adapter demo.

Pipeline: flatten each hybrid context (table + paragraphs) to text -> BM25 index -> retrieve top_k
contexts for a question -> ask Claude (JSON-constrained) for {answer, scale} -> score programmatically.
Every knob is in config.json so the eval-loop-adapter skill can tune it through adapter.py.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "tatqa_dataset_dev.json"
CASES = HERE / "cases.json"
CONFIG = HERE / "config.json"
RUNS = HERE / "runs"
DATA_URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json"

SCALE = {"": 1, "thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 1}


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def ensure_data() -> list[dict]:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(DATA), DATA_URL], check=True)
    return json.loads(DATA.read_text())


# ---------------------------------------------------------------- corpus and retrieval
def flatten_table(table: list[list[str]], fmt: str) -> str:
    if fmt == "cells":
        header = table[0] if table else []
        # find the first row that looks like column headers (years etc.)
        col_hdr = next((r for r in table[:3] if sum(bool(c.strip()) for c in r) >= 2), header)
        out = []
        for row in table:
            if row is col_hdr or not row or not row[0].strip():
                continue
            for j, cell in enumerate(row[1:], start=1):
                if cell.strip():
                    ch = col_hdr[j].strip() if j < len(col_hdr) else f"col{j}"
                    out.append(f"{row[0].strip()} | {ch}: {cell.strip()}")
        return "\n".join(out) if out else "\n".join(" | ".join(r) for r in table)
    return "\n".join(" | ".join(c.strip() for c in r) for r in table)


def build_corpus(data: list[dict], cfg: dict) -> tuple[list[str], list[str]]:
    docs, ids = [], []
    for c in data:
        text = flatten_table(c["table"]["table"], cfg["table_format"])
        if cfg["include_paragraphs"]:
            text += "\n" + "\n".join(p["text"] for p in sorted(c["paragraphs"], key=lambda p: p["order"]))
        docs.append(text)
        ids.append(c["table"]["uid"])
    return docs, ids


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.,%$-]*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self, docs: list[str], ids: list[str], cfg: dict):
        self.docs, self.ids, self.cfg = docs, ids, cfg
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    def query(self, q: str, k: int) -> list[int]:
        toks = tokenize(q)
        if self.cfg.get("number_boost"):
            toks += [t for t in toks if re.search(r"\d", t)] * 2  # triple weight on numeric tokens
        scores = self.bm25.get_scores(toks)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


# ---------------------------------------------------------------- generation
PROMPT_RULES = {
    "basic": "Answer the question using only the provided contexts. Be exact.",
    "tables": ("Answer the question using only the provided contexts. Tables are given as text rows or as "
               "'row | column: value' cells. Before answering: identify the exact row label and column label the "
               "question refers to, read the number in that cell, note the unit and scale stated in the table header "
               "or nearby text (thousand, million, percent), and for arithmetic questions show the numbers you combine. "
               "Report the number as it appears (do not convert scale) and set 'scale' separately."),
}
ANSWER_SCHEMA = {"type": "object",
                 "properties": {"reasoning": {"type": "string"}, "answer": {"type": "string"},
                                "scale": {"type": "string", "enum": ["", "thousand", "million", "billion", "percent"]}},
                 "required": ["answer", "scale"], "additionalProperties": False}


def generate(question: str, contexts: list[str], cfg: dict) -> tuple[dict, dict]:
    system = PROMPT_RULES[cfg["prompt_rules"]] + (
        " Respond with JSON: reasoning, answer (a number without units, or a short span; for multi-part answers "
        "separate with ' ; '), scale.")
    user = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)) + f"\n\nQuestion: {question}"
    cmd = ["claude", "-p", "--system-prompt", system, "--json-schema", json.dumps(ANSWER_SCHEMA), "--tools", "",
           "--no-session-persistence", "--max-budget-usd", "0.5", "--output-format", "json", "--model", cfg["gen_model"], user]
    import os
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    for attempt in range(3):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, stdin=subprocess.DEVNULL, env=env)
        try:
            data = json.loads(proc.stdout)
            out = data.get("structured_output") or json.loads(data.get("result") or "{}")
            usage = data.get("usage", {}) or {}
            return out, {"prompt_tokens": sum(int(usage.get(k, 0) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
                         "output_tokens": int(usage.get("output_tokens", 0) or 0), "cost_usd": float(data.get("total_cost_usd", 0) or 0)}
        except (json.JSONDecodeError, AttributeError):
            time.sleep(2 * (attempt + 1))
    err = (proc.stderr or proc.stdout or "").strip()[-300:] or f"generation failed (exit {proc.returncode}, no output)"
    return {"answer": "", "scale": ""}, {"prompt_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "error": err}


# ---------------------------------------------------------------- scoring
def to_number(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.fullmatch(r"\(?-?\d+(\.\d+)?\)?", s)
    if not m:
        return None
    neg = s.startswith("(") and s.endswith(")")
    v = float(s.strip("()"))
    return -v if neg else v


def norm_text(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def score_case(pred: dict, gold: dict) -> tuple[bool, str]:
    gold_answers = gold["answer"] if isinstance(gold["answer"], list) else [gold["answer"]]
    pa, ps = str(pred.get("answer", "")), pred.get("scale", "")
    if gold["answer_type"] in ("arithmetic", "count") or all(to_number(str(g)) is not None for g in gold_answers):
        gv = to_number(str(gold_answers[0]))
        pv = to_number(pa.split(";")[0])
        if gv is None or pv is None:
            return False, "non-numeric prediction for numeric gold"
        # allow either matching the raw number with correct scale, or the scaled absolute value
        if abs(pv - gv) <= max(0.011, abs(gv) * 0.005) and (ps == gold["scale"] or not gold["scale"]):
            return True, "numeric match"
        if gold["scale"] in SCALE and abs(pv * SCALE.get(ps, 1) - gv * SCALE[gold["scale"]]) <= abs(gv * SCALE[gold["scale"]]) * 0.005:
            return True, "numeric match after scale"
        return False, f"pred {pa} {ps} vs gold {gold_answers[0]} {gold['scale']}"
    # spans / multi-span: token F1 over the joined answer
    p_toks, g_toks = norm_text(pa), norm_text(" ; ".join(map(str, gold_answers)))
    if not p_toks or not g_toks:
        return False, "empty"
    common = sum(min(p_toks.count(t), g_toks.count(t)) for t in set(p_toks))
    prec, rec = common / len(p_toks), common / len(g_toks)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1 >= 0.8, f"span F1 {f1:.2f}"


# ---------------------------------------------------------------- commands
def cmd_build(cfg: dict) -> None:
    data = ensure_data()
    rng = random.Random(cfg["case_seed"])
    pool = [dict(q, gold_uid=c["table"]["uid"]) for c in data for q in c["questions"]
            if q["answer_type"] in ("arithmetic", "span", "count") and q["answer_from"] in ("table", "table-text")]
    rng.shuffle(pool)
    # stratify: half arithmetic, half span, all table-grounded
    arith = [q for q in pool if q["answer_type"] == "arithmetic"][: cfg["n_cases"] // 2]
    span = [q for q in pool if q["answer_type"] != "arithmetic"][: cfg["n_cases"] - len(arith)]
    cases = arith + span
    CASES.write_text(json.dumps([{k: q[k] for k in ("uid", "question", "answer", "answer_type", "scale", "gold_uid")} for q in cases], indent=1))
    print(f"wrote {len(cases)} cases to {CASES}")


def cmd_run(cfg: dict, label: str) -> dict:
    data = ensure_data()
    if not CASES.exists():
        cmd_build(cfg)
    cases = json.loads(CASES.read_text())
    docs, ids = build_corpus(data, cfg)
    retr = Retriever(docs, ids, cfg)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^a-zA-Z0-9_-]", "", label)[:30]
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(case: dict) -> dict:
        idx = retr.query(case["question"], cfg["top_k"])
        hit = case["gold_uid"] in [ids[i] for i in idx]
        rank = next((r + 1 for r, i in enumerate(idx) if ids[i] == case["gold_uid"]), None)
        pred, usage = generate(case["question"], [docs[i] for i in idx], cfg)
        if not str(pred.get("answer", "")).strip() and not usage.get("error"):
            usage["error"] = "empty answer"
        ok, why = score_case(pred, case)
        rec = {"uid": case["uid"], "question": case["question"], "gold": case["answer"], "gold_scale": case["scale"],
               "pred": pred, "pass": ok, "why": why, "retrieved": [ids[i] for i in idx], "gold_retrieved": hit,
               "gold_rank": rank, **usage}
        (out_dir / f"{case['uid']}.json").write_text(json.dumps(rec, indent=1))
        return rec

    with ThreadPoolExecutor(max_workers=5) as pool:
        recs = list(pool.map(one, cases))
    metrics = build_metrics(recs, cfg, run_id, label, out_dir, t0)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def gold_first(rec: dict) -> str:
    g = rec["gold"]
    return str(g[0] if isinstance(g, list) else g)


def build_metrics(recs: list[dict], cfg: dict, run_id: str, label: str, out_dir: Path, t0: float) -> dict:
    n = len(recs)
    numeric = [r for r in recs if to_number(gold_first(r)) is not None]
    return {
        "run_id": run_id, "config_label": label, "config": cfg,
        "cases": {r["uid"]: ("pass" if r["pass"] else ("error" if r.get("error") else "fail")) for r in recs},
        "primary_metric": {"name": "accuracy", "value": round(sum(r["pass"] for r in recs) / n, 4)},
        "deterministic_metrics": {"recall_at_k": round(sum(r["gold_retrieved"] for r in recs) / n, 4),
                                  "mrr": round(sum(1 / r["gold_rank"] for r in recs if r["gold_rank"]) / n, 4)},
        "secondary_metrics": {"prompt_tokens_per_case": round(sum(r.get("prompt_tokens", 0) for r in recs) / n),
                              "numeric_acc": round(sum(r["pass"] for r in numeric) / max(1, len(numeric)), 4),
                              "fails_not_retrieved": sum(1 for r in recs if not r["pass"] and not r["gold_retrieved"]),
                              "fails_retrieved": sum(1 for r in recs if not r["pass"] and r["gold_retrieved"])},
        "cost": {"usd": round(sum(r.get("cost_usd", 0) for r in recs), 4), "tokens": sum(r.get("prompt_tokens", 0) + r.get("output_tokens", 0) for r in recs)},
        "duration_s": round(time.time() - t0, 1), "artifacts": {"run_dir": str(out_dir)},
        "errors": [r["uid"] for r in recs if r.get("error")],
        "error_rate": round(sum(1 for r in recs if r.get("error")) / n, 3),
        "valid": sum(1 for r in recs if r.get("error")) / n <= 0.10,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run"); r.add_argument("--label", default="run")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "build":
        cmd_build(cfg)
    else:
        m = cmd_run(cfg, args.label)
        print(json.dumps({k: v for k, v in m.items() if k not in ("cases", "config")}))


if __name__ == "__main__":
    main()
