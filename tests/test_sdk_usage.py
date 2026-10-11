from types import SimpleNamespace

import pytest

from stepbook.reporting.usage import usage_summary


def result(**kwargs):
    return SimpleNamespace(
        usage={"input_tokens": 1, "output_tokens": 2},
        model_usage={
            "main": {"inputTokens": 1, "outputTokens": 2, "cacheReadInputTokens": 3, "cacheCreationInputTokens": 4},
            "aux": {"inputTokens": 5, "outputTokens": 6, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0},
        },
        total_cost_usd=0.2,
        num_turns=2,
        session_id="persisted",
        stop_reason="tool_use",
        **kwargs,
    )


def priced_model_usage():
    """Whole-query per-model usage as the provider reports it: token counts plus that model's own cost."""
    return {
        "claude-opus-5": {
            "inputTokens": 1,
            "outputTokens": 2,
            "cacheReadInputTokens": 3,
            "cacheCreationInputTokens": 4,
            "costUSD": 0.18,
        },
        "claude-haiku-4-5": {
            "inputTokens": 5,
            "outputTokens": 6,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
            "costUSD": 0.02,
        },
    }


def test_receipts_without_per_model_usage_are_unattributed_rather_than_spread():
    usage = usage_summary(
        [
            {
                "kind": "model_call",
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": 0.1,
                "model_ids": ["claude-haiku-4-5", "claude-sonnet-5"],
            },
            {"kind": "model_call", "usage_unknown": True, "cost_usd": None},
        ]
    )
    # Naming two models on a receipt is not per-model accounting: nothing may be attributed from it.
    assert usage["by_model"]["models"] == {}
    assert usage["by_model"]["unattributed"] == usage["components"]["controller"]
    assert usage["by_model"]["unattributed"]["known_usd"] == 0.1
    assert usage["by_model"]["unattributed"]["unknown_cost_records"] == 1
    # The names are kept so a report can say two models are behind the blended figure instead of nothing.
    assert usage["by_model"]["unattributed_model_ids"] == ["claude-haiku-4-5", "claude-sonnet-5"]


def priced_receipt(models, **override):
    """One controller receipt whose provider usage prices each model separately."""
    return {
        "kind": "model_call",
        "input_tokens": sum(
            r["inputTokens"] + r["cacheReadInputTokens"] + r["cacheCreationInputTokens"] for r in models.values()
        ),
        "output_tokens": sum(r["outputTokens"] for r in models.values()),
        "cost_usd": sum(r["costUSD"] for r in models.values()),
        "token_scope": "all_reported_models",
        "provider_model_usage": models,
        "model_ids": sorted(models),
        **override,
    }


def test_the_per_model_note_claims_only_what_the_rows_carry():
    """Three receipts, each billed to both models: the note may promise neither exact USD nor summed counts."""
    usage = usage_summary([priced_receipt(priced_model_usage()) for _ in range(3)])
    controller, models = usage["components"]["controller"], usage["by_model"]["models"]
    rows_usd = sum(m["known_usd"] for m in models.values())
    assert rows_usd == pytest.approx(controller["known_usd"]) and rows_usd != controller["known_usd"]
    assert sum(m["known_tokens"] for m in models.values()) == controller["known_tokens"]
    assert controller["records"] == 3 and [m["records"] for m in models.values()] == [3, 3]
    note = usage["by_model"]["note"]
    assert "equal the controller component exactly" not in note
    assert "the receipt counts do not add up to the component's" in note


def test_a_receipt_whose_model_rows_do_not_reconcile_is_never_split():
    """A row total below the receipt's own bill would print the larger figure over the smaller rows."""
    halved = usage_summary([priced_receipt(priced_model_usage(), cost_usd=0.4)])
    assert halved["by_model"]["models"] == {}
    assert halved["by_model"]["unattributed"]["records"] == 1 and halved["by_model"]["unattributed"]["known_usd"] == 0.4
    short = usage_summary([priced_receipt(priced_model_usage(), input_tokens=99)])
    assert short["by_model"]["models"] == {} and short["by_model"]["unattributed"]["records"] == 1
    # The tolerance permits floating-point summation error.
    assert usage_summary([priced_receipt(priced_model_usage(), cost_usd=0.2 + 1e-9)])["by_model"]["models"]
    assert usage_summary([priced_receipt(priced_model_usage(), cost_usd=0.2 + 1e-3)])["by_model"]["models"] == {}
