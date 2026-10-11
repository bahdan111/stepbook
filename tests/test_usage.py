import json

import pytest

from stepbook.contracts.errors import RateLimited
from stepbook.executors.base import Envelope
from stepbook.reporting.usage import controller_receipts, usage_summary
from stepbook.runtime.retry import call_with_retry
from stepbook.storage.events import EventLog


def test_reported_usage_never_fabricates_subscription_invoice_or_quota():
    report = usage_summary(
        [{"kind": "model_call", "step": 0, "input_tokens": 10, "output_tokens": 2, "cost_usd": 0.01}]
    )
    assert report["known_total_usd"] == 0.01 and report["cost_complete"]
    assert report["actual_charge_usd"] is None and report["subscription_quota_used"] is None
    assert "not a verified invoice" in report["billing_note"]


def test_usage_counts_retries_nested_calls_invalid_evals_and_unknowns():
    events = [
        {"kind": "model_call", "input_tokens": 10, "output_tokens": 2, "cost_usd": 0.1},
        {"kind": "patch_rejected"},
        {"kind": "model_call", "input_tokens": 15, "output_tokens": 3, "cost_usd": 0.2},
        {"kind": "environment_model_call", "input_tokens": 20, "output_tokens": 4, "cost_usd": 0.3},
        {
            "kind": "step",
            "action": "run b",
            "prompt_tokens": 15,
            "output_tokens": 3,
            "cost_usd": 0.2,
            "result": "Eval finished:\n" + json.dumps({"valid": False, "cost": {"tokens": 50, "usd": 0.4}}),
        },
        {"kind": "step", "action": "run failed", "result": "Error: timeout"},
    ]
    usage = usage_summary(events)
    assert usage["known_total_tokens"] == 104
    assert usage["known_total_usd"] == 1.0
    assert not usage["cost_complete"] and not usage["tokens_complete"]
    assert usage["components"]["evaluation"]["unknown_cost_records"] == 1
    assert usage["components"]["controller"]["records"] == 2  # step is not double-counted


def test_legacy_usage_cannot_hide_missing_retries():
    usage = usage_summary(
        [
            {"kind": "patch_rejected"},
            {"kind": "step", "action": "stop", "cost_usd": 0, "prompt_tokens": 10, "output_tokens": 2},
        ]
    )
    assert usage["known_total_tokens"] == 12 and not usage["cost_complete"] and usage["gaps"]


def test_mixed_legacy_and_resumed_modern_log_keeps_both_costs():
    usage = usage_summary(
        [
            {"kind": "step", "step": 0, "action": "stop", "cost_usd": 0.1, "prompt_tokens": 10, "output_tokens": 2},
            {"kind": "model_call", "step": 1, "input_tokens": 20, "output_tokens": 3, "cost_usd": 0.2},
            {"kind": "step", "step": 1, "action": "stop", "cost_usd": 0.2, "prompt_tokens": 20, "output_tokens": 3},
        ]
    )
    assert usage["known_total_tokens"] == 35
    assert round(usage["known_total_usd"], 2) == 0.3


def test_partial_eval_cost_keeps_available_subtotals_without_claiming_complete():
    u = usage_summary(
        [
            {"kind": "model_call", "step": 0, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.1},
            {
                "kind": "step",
                "step": 0,
                "action": "run partial",
                "result": "Eval finished:\n"
                + json.dumps({"cost": {"tokens": None, "known_tokens": 50, "usd": None, "known_usd": 0.4}}),
            },
        ]
    )
    assert u["known_total_tokens"] == 52 and u["known_total_usd"] == 0.5
    assert not u["tokens_complete"] and not u["cost_complete"]


def test_unresolved_action_marks_usage_incomplete_until_receipt_without_double_count():
    events = [
        {"kind": "model_call", "step": 0, "input_tokens": 2, "output_tokens": 1, "cost_usd": 0.1},
        {"kind": "action_started", "action_id": "a", "record": {"step": 0, "action": "run b"}},
    ]
    u = usage_summary(events)
    assert u["known_total_tokens"] == 3 and not u["cost_complete"] and not u["tokens_complete"]
    assert "Unresolved action" in u["gaps"][0]
    receipt = "Eval finished:\n" + json.dumps({"cost": {"tokens": 10, "usd": 0.2}})
    events.append({"kind": "action_result", "action_id": "a", "result": receipt})
    before = usage_summary(events)
    assert before["known_total_tokens"] == 13 and before["cost_complete"]
    events.append({"kind": "step", "action_id": "a", "step": 0, "action": "run b", "result": receipt})
    assert usage_summary(events) == before


def test_a_receipt_less_failed_call_explains_why_the_accounting_is_incomplete(tmp_path):
    """The gap message is keyed on the failed call's unknown usage;
    a failed call whose receipt is usable leaves the accounting complete."""

    class NoReceipt:
        def step(self, system_prompt, user_prompt, schema):
            raise RateLimited("claude exited 1: rate limited")

    class WithReceipt:
        def step(self, system_prompt, user_prompt, schema):
            envelope = Envelope("r", {}, "a", input_tokens=7, output_tokens=3, cost_usd=0.02)
            raise RateLimited("claude exited 1: rate limited", envelope=envelope)

    unknown = EventLog(tmp_path / "unknown.jsonl")
    with pytest.raises(RateLimited):
        call_with_retry(NoReceipt(), "s", "u", {"type": "object"}, log=unknown, step=0)
    assert [e["kind"] for e in unknown.read()] == ["failed_model_call"]
    report = usage_summary(unknown.read())
    assert report["cost_complete"] is False and report["tokens_complete"] is False
    assert report["gaps"] == ["Failed provider requests may have unreported usage"]

    priced = EventLog(tmp_path / "priced.jsonl")
    with pytest.raises(RateLimited):
        call_with_retry(WithReceipt(), "s", "u", {"type": "object"}, log=priced, step=0)
    accounted = usage_summary(priced.read())
    assert accounted["gaps"] == [] and accounted["cost_complete"] is True
    assert accounted["known_total_usd"] == 0.02 and accounted["known_total_tokens"] == 10


def test_a_failed_call_that_reports_tokens_but_no_price_says_which_figure_is_missing(tmp_path):
    """Unknown usage gets a gap message; tokens without a price get a distinct message;
    a fully priced receipt leaves no gap at all."""

    class Unpriced:
        def step(self, system_prompt, user_prompt, schema):
            # What _cli_error_receipt builds from a structured 429 whose receipt carries no total_cost_usd.
            envelope = Envelope("", {}, "", input_tokens=7, output_tokens=3, cost_usd=0.0, raw={"cost_unknown": True})
            raise RateLimited("claude exited 1: rate limited", envelope=envelope)

    log = EventLog(tmp_path / "unpriced.jsonl")
    with pytest.raises(RateLimited):
        call_with_retry(Unpriced(), "s", "u", {"type": "object"}, log=log, step=0)
    receipt = log.read()[0]
    assert (receipt["kind"], receipt["cost_usd"], receipt["usage_unknown"]) == ("failed_model_call", None, False)
    report = usage_summary(log.read())
    assert report["gaps"] == ["Failed provider requests reported usage but no price"]
    assert report["cost_complete"] is False and report["known_total_tokens"] == 10

    unknown = {"kind": "failed_model_call", "step": 0, "cost_usd": None, "usage_unknown": True}
    assert usage_summary([unknown])["gaps"] == ["Failed provider requests may have unreported usage"]
    priced = {"kind": "failed_model_call", "step": 0, "input_tokens": 7, "output_tokens": 3, "cost_usd": 0.02}
    assert usage_summary([priced])["gaps"] == [] and usage_summary([priced])["cost_complete"] is True


def test_controller_receipts_keep_failed_queries_once_and_exclude_host_repeats():
    failed = {"kind": "failed_model_call", "step": 0, "cost_usd": 0.03}
    success = {"kind": "model_call", "step": 0, "cost_usd": 0.1}
    committed = {"kind": "step", "step": 0, "cost_usd": 0.1}
    repeat = {"kind": "step", "step": 1, "action_origin": "host_measurement_repeat"}
    legacy = {"kind": "step", "step": 2, "cost_usd": 0.2}
    assert controller_receipts([failed, success, committed, repeat, legacy]) == [failed, success, legacy]


@pytest.mark.parametrize(
    "events,measured,component,records",
    [
        (
            [{"kind": "environment_model_call", "prompt_tokens": 17, "output_tokens": 0}],
            False,
            "investigation_and_coding",
            1,
        ),
        (
            [
                {
                    "kind": "environment_model_call",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usage_unknown": True,
                    "token_scope": "main_loop_only",
                }
            ],
            True,
            "investigation_and_coding",
            1,
        ),
        (
            [
                {"kind": "failed_model_call", "step": 0, "usage_unknown": True},
                {"kind": "step", "step": 0, "prompt_tokens": 0, "output_tokens": 0, "token_scope": "main_loop_only"},
            ],
            True,
            "controller",
            2,
        ),
    ],
)
def test_token_measurement_presence_follows_accepted_accounting(events, measured, component, records):
    from stepbook.reporting.usage import has_measured_tokens

    summary = usage_summary(events)
    assert has_measured_tokens(events) is measured
    assert summary["known_total_tokens"] == 0
    assert summary["tokens_complete"] is False
    assert summary["components"][component] == {
        "records": records,
        "known_tokens": 0,
        "known_usd": 0.0,
        "unknown_token_records": records,
        "unknown_cost_records": records,
    }


def test_token_presence_is_separate_from_serialized_summary():
    from stepbook.reporting.usage import BILLING_NOTE, PER_MODEL_NOTE, has_measured_tokens

    partial = [
        {"kind": "model_call", "input_tokens": 0, "output_tokens": 0, "token_scope": "main_loop_only", "cost_usd": 0}
    ]
    unknown = [{"kind": "model_call", "input_tokens": None, "output_tokens": None, "cost_usd": 0}]
    assert has_measured_tokens(partial) is True
    assert has_measured_tokens(unknown) is False
    assert json.dumps(usage_summary(partial)) == json.dumps(usage_summary(unknown))
    empty = {"records": 0, "known_tokens": 0, "known_usd": 0.0, "unknown_token_records": 0, "unknown_cost_records": 0}
    controller = {**empty, "records": 1, "unknown_token_records": 1}
    expected = {
        "components": {"controller": controller, "investigation_and_coding": empty, "evaluation": empty},
        "by_model": {"models": {}, "unattributed": controller, "unattributed_model_ids": [], "note": PER_MODEL_NOTE},
        "known_total_usd": 0.0,
        "known_total_tokens": 0,
        "cost_complete": True,
        "tokens_complete": False,
        "billing_note": BILLING_NOTE,
        "actual_charge_usd": None,
        "subscription_quota_used": None,
        "gaps": [],
        "scope": "Controller + nested models + adapter-reported evaluation totals (generation and judge must both be included by adapter). Excludes human time and infrastructure charges.",
    }
    assert json.dumps(usage_summary(partial), indent=2) == json.dumps(expected, indent=2)
