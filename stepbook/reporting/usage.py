"""Usage accounting from durable events; missing provider data is never free work."""

from __future__ import annotations

from . import legacy_journal

BILLING_NOTE = (
    "Reported USD is execution-usage accounting, not a verified invoice or marginal subscription charge. "
    "Claude subscription receipts may use list-price-equivalent USD. Subscription quota consumption "
    "and cash savings are not measured by token totals or these USD figures."
)

PER_MODEL_NOTE = (
    "Per-model rows split the controller component only, and only for receipts on which the provider reported "
    "whole-query per-model cost and token counts; a receipt without that is counted under unattributed, never "
    "spread across models. Nested-model and evaluation costs carry no per-model data and are outside these rows. "
    "Model rows plus unattributed account for the controller component's cost (to floating-point rounding) and "
    "its tokens; a receipt billed to more than one model is counted in each of those model rows, so the receipt "
    "counts do not add up to the component's."
)

TOKEN_KEYS = ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "outputTokens")


def controller_receipts(events):
    """Controller receipts, including failures and pre-action stops, without legacy double-counting."""
    modern = {e.get("step") for e in events if e.get("kind") in ("model_call", "failed_model_call")}
    return [
        e
        for e in events
        if e.get("kind") in ("model_call", "failed_model_call")
        or (
            e.get("kind") == "step"
            and e.get("step") not in modern
            and e.get("action_origin") != "host_measurement_repeat"
        )
    ]


def claude_token_totals(usage, model_usage):
    """Prefer whole-query model totals: top-level usage can exclude SDK auxiliary calls/compaction."""
    keys = TOKEN_KEYS
    rows = list((model_usage or {}).values())
    complete = bool(rows) and all(
        isinstance(r, dict) and all(type(r.get(k)) in (int, float) and r[k] >= 0 for k in keys) for r in rows
    )
    if complete:
        return (
            sum(int(r[k]) for r in rows for k in keys[:-1]),
            sum(int(r["outputTokens"]) for r in rows),
            "all_reported_models",
        )
    usage = usage or {}
    return (
        sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
        int(usage.get("output_tokens") or 0),
        "main_loop_only",
    )


def envelope_usage(envelope) -> dict:
    unknown = bool(envelope.raw.get("usage_unknown"))
    return {
        "input_tokens": None if unknown else envelope.input_tokens,
        "output_tokens": None if unknown else envelope.output_tokens,
        "cost_usd": None if unknown or envelope.raw.get("cost_unknown") else envelope.cost_usd,
        "api_calls": envelope.api_calls,
        "tool_calls": envelope.tool_calls,
        "tool_log": envelope.tool_log,
        # Only an executor that actually validates its output records attempts; writing an empty list for
        # every other one would claim a measurement nobody took. Readers of stored journals are unchanged.
        **({"output_validation": envelope.raw["output_validation"]} if "output_validation" in envelope.raw else {}),
        "allow_json_fence": envelope.raw.get("allow_json_fence", False),
        "response_text": envelope.raw.get("response_text"),
        "usage_unknown": unknown,
        "provider_usage": envelope.usage,
        "model_ids": envelope.raw.get("model_ids", []),
        "session_id": envelope.raw.get("session_id"),
        "memory_mode": envelope.raw.get("memory_mode"),
        "window_steps": envelope.raw.get("window_steps"),
        "window_call": envelope.raw.get("window_call"),
        "token_scope": envelope.raw.get("token_scope"),
        "provider_model_usage": envelope.raw.get("model_usage", envelope.raw.get("modelUsage")),
    }


def usage_summary(events: list[dict]) -> dict:
    """Account every durable receipt into the controller, nested-model and evaluation components."""
    return _summarize_usage(events)[0]


def has_measured_tokens(events: list[dict]) -> bool:
    """Whether accounting accepted any token value, including a partial zero; never serialized."""
    return _summarize_usage(events)[1]


def _summarize_usage(events: list[dict]) -> tuple[dict, bool]:
    """Keep measurement presence separate from the existing serialized summary contract."""
    events, unresolved, modern_steps = _account_receipts(events)
    components = {
        k: {"records": 0, "known_tokens": 0, "known_usd": 0.0, "unknown_token_records": 0, "unknown_cost_records": 0}
        for k in ("controller", "investigation_and_coding", "evaluation")
    }
    by_model = {
        "models": {},
        "unattributed": {
            "records": 0,
            "known_tokens": 0,
            "known_usd": 0.0,
            "unknown_token_records": 0,
            "unknown_cost_records": 0,
        },
        "unattributed_model_ids": [],
        "note": PER_MODEL_NOTE,
    }
    measured_tokens = False
    for e in events:
        measured_tokens |= _attribute_model_call(components, by_model, e, modern_steps)
        measured_tokens |= _attribute_evaluation(components, e)
    token_gaps, price_gaps = _usage_gaps(events, unresolved, modern_steps)
    unknown_cost = sum(c["unknown_cost_records"] for c in components.values())
    unknown_tokens = sum(c["unknown_token_records"] for c in components.values())
    summary = {
        "components": components,
        "by_model": by_model,
        "known_total_usd": sum(c["known_usd"] for c in components.values()),
        "known_total_tokens": sum(c["known_tokens"] for c in components.values()),
        "cost_complete": not unknown_cost and not token_gaps and not price_gaps,
        "tokens_complete": not unknown_tokens and not token_gaps,
        "billing_note": BILLING_NOTE,
        "actual_charge_usd": None,
        "subscription_quota_used": None,
        "gaps": token_gaps + price_gaps,
        "scope": "Controller + nested models + adapter-reported evaluation totals (generation and judge must both be included by adapter). Excludes human time and infrastructure charges.",
    }
    return summary, measured_tokens


def _account_receipts(events: list[dict]) -> tuple[list[dict], bool, set[object]]:
    """Fold durable receipts for uncommitted actions into the events, and index the modern calls."""
    # A durable receipt already contains the bill even before step commit. Account
    # for it read-only; do not require resume (or count it twice after resume).
    committed = {e.get("action_id") for e in events if e.get("kind") == "step"}
    receipts = {e.get("action_id"): e for e in events if e.get("kind") == "action_result"}
    pending = [e for e in events if e.get("kind") == "action_started" and e.get("action_id") not in committed]
    received = [
        {
            **e["record"],
            "kind": "step",
            "result": receipts[e["action_id"]]["result"],
            # The receipt carries the step's own facts as well as its observation, and accounting reads them.
            "info": receipts[e["action_id"]].get("info") or {},
        }
        for e in pending
        if e["action_id"] in receipts
    ]
    events = events + received
    unresolved = any(e["action_id"] not in receipts for e in pending)
    return events, unresolved, {e.get("step") for e in events if e.get("kind") == "model_call"}


def _add_record(row: dict, tokens, usd, tokens_partial: bool = False, cost_partial: bool = False) -> bool:
    """Add one receipt, counting gaps, and return whether a token value was accepted."""
    row["records"] += 1
    if tokens is None or tokens_partial:
        row["unknown_token_records"] += 1
    if tokens is not None:
        row["known_tokens"] += tokens
    if usd is None or cost_partial:
        row["unknown_cost_records"] += 1
    if usd is not None:
        row["known_usd"] += usd
    return tokens is not None


def _attribute_models(by_model: dict, e: dict, tokens, usd, partial: bool) -> None:
    """Split one controller receipt across the models the provider billed it to, or record it as unattributed.

    Only a whole-query per-model receipt can be split: its rows must sum to the receipt's own cost and tokens,
    which is checked here, so the split adds no figure the blended total did not already contain. Anything else
    stays unattributed rather than being guessed at, so a reader can see how much of the controller bill the
    split does not cover. The cost tolerance permits floating-point summation error and remains below the
    reporting precision."""
    rows = e.get("provider_model_usage") or {}
    priced = (
        isinstance(rows, dict)
        and bool(rows)
        and all(
            isinstance(r, dict)
            and type(r.get("costUSD")) in (int, float)
            and all(type(r.get(k)) in (int, float) and r[k] >= 0 for k in TOKEN_KEYS)
            for r in rows.values()
        )
    )
    splittable = (
        not partial
        and tokens is not None
        and usd is not None
        and priced
        and abs(sum(float(r["costUSD"]) for r in rows.values()) - usd) <= 1e-6 * max(1.0, abs(usd))
        and sum(int(r[k]) for r in rows.values() for k in TOKEN_KEYS) == tokens
    )
    if not splittable:
        _add_record(by_model["unattributed"], tokens, usd, partial)
        # Which models the unsplittable receipts named, so a report can say the blended figure covers more
        # than one model instead of printing nothing.
        named = set(by_model["unattributed_model_ids"]) | {str(m) for m in (e.get("model_ids") or [])}
        by_model["unattributed_model_ids"] = sorted(named)
        return
    for model, row in rows.items():
        target = by_model["models"].setdefault(str(model), {"records": 0, "known_tokens": 0, "known_usd": 0.0})
        target["records"] += 1
        target["known_tokens"] += sum(int(row[k]) for k in TOKEN_KEYS)
        target["known_usd"] += float(row["costUSD"])


def _attribute_model_call(components: dict, by_model: dict, e: dict, modern_steps: set[object]) -> bool:
    """Attribute a controller receipt or a nested environment model call to its component."""
    kind = e.get("kind")
    partial = e.get("token_scope") == "main_loop_only" or (
        e.get("token_scope") is None and any(str(m).startswith("claude-") for m in (e.get("model_ids") or []))
    )
    if kind in ("model_call", "failed_model_call") or (
        kind == "step" and e.get("step") not in modern_steps and e.get("action_origin") != "host_measurement_repeat"
    ):
        unknown = e.get("usage_unknown", False)
        inp = e.get("input_tokens", e.get("prompt_tokens"))
        out = e.get("output_tokens")
        tokens = None if unknown or inp is None or out is None else inp + out
        usd = None if unknown else e.get("cost_usd")
        measured_tokens = _add_record(components["controller"], tokens, usd, partial)
        _attribute_models(by_model, e, tokens, usd, partial)
        return measured_tokens
    elif kind == "environment_model_call":
        inp, out = e.get("input_tokens"), e.get("output_tokens")
        return _add_record(
            components["investigation_and_coding"],
            None if inp is None or out is None else inp + out,
            e.get("cost_usd"),
            partial,
        )
    return False


def _attribute_evaluation(components: dict, e: dict) -> bool:
    """Attribute the adapter-reported evaluation cost the run step recorded with its receipt."""
    if not (e.get("kind") == "step" and str(e.get("action", "")).split()[:1] == ["run"]):
        return False
    payload = legacy_journal.step_receipt(e) or legacy_journal.unstarted_cost(e)
    if payload is None and (e.get("info") or {}).get("evaluation_not_started") is True:
        payload = {"cost": {"tokens": 0, "usd": 0}}
    cost = (payload or {}).get("cost") or {}
    return _add_record(
        components["evaluation"],
        cost.get("tokens") if cost.get("tokens") is not None else cost.get("known_tokens"),
        cost.get("usd") if cost.get("usd") is not None else cost.get("known_usd"),
        cost.get("tokens") is None,
        cost.get("usd") is None,
    )


def _usage_gaps(events: list[dict], unresolved: bool, modern_steps: set[object]) -> tuple[list[str], list[str]]:
    """Separate gaps affecting both totals from gaps affecting only prices."""
    gaps: list[str] = []
    if unresolved:
        gaps.append("Unresolved action: execution and additional usage may have occurred before its missing receipt")
    if any(e.get("kind") == "patch_rejected" and e.get("step") not in modern_steps for e in events):
        gaps.append("Legacy log: rejected controller calls are not fully accounted for")
    if any(
        e.get("kind") == "step"
        and e.get("step") not in modern_steps
        and str(e.get("action", "")).split()[:1] in (["mine"], ["investigate"], ["apply"])
        for e in events
    ):
        gaps.append("Legacy log: nested model usage may be missing")
    if any(
        (e.get("kind") in ("executor_error", "rate_limited") and not e.get("usage_reported"))
        or (e.get("kind") == "failed_model_call" and e.get("usage_unknown"))
        for e in events
    ):
        gaps.append("Failed provider requests may have unreported usage")
    price_gaps = []
    if any(
        e.get("kind") == "failed_model_call"
        and e.get("input_tokens", e.get("prompt_tokens")) is not None
        and e.get("output_tokens") is not None
        and e.get("cost_usd") is None
        for e in events
    ):
        price_gaps.append("Failed provider requests reported usage but no price")
    return gaps, price_gaps
