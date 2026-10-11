import copy
import json

import pytest

from stepbook.envs.eval_loop import AdapterEvalEnv


def receipt(run_id, value=0.5, *, config="config-a", key="dataset-scorer", valid=True):
    return {
        "run_id": run_id,
        "config_label": "label-" + run_id,
        "valid": valid,
        "primary_metric": {"name": "accuracy", "value": value, "direction": "higher"},
        "cases": {"case": "pass"},
        "comparison_key": key,
        "provenance": {"config_sha256": config},
    }


def test_frontier_uses_receipts_and_preserves_original_ids_without_mutation(replay_receipts):
    env = AdapterEvalEnv(adapter="never called")
    runs = {
        "baseline": [receipt("real-1", 0.25), receipt("real-2", 0.5), receipt("bad", 1, valid=False)],
        "h1": [receipt("candidate-1", 0.75, config="config-b"), receipt("candidate-2", 0.75, config="config-b")],
    }
    replay_receipts(env, runs, applied="h1")
    # Reads the internal ledger: no public seam exposes every receipt field.
    before = copy.deepcopy(env._runs)
    summary = env.measurement_frontier()
    assert summary["reference"]["aggregate"]["mean"] == 0.375
    assert summary["reference"]["valid_count"] == 2 and summary["reference"]["invalid_count"] == 1
    assert summary["reference"]["recent"][1] == {
        "run_id": "bad",
        "label": "label-bad",
        "valid": False,
        "primary": None,
        "omitted_identifiers": [],
    }
    assert summary["candidate"]["recent"][0]["run_id"] == "candidate-1"
    assert summary["comparison_issue"] is None and env._runs == before
    assert json.loads(env.context().split("\n", 1)[1])["measurements"] == summary
    assert env.measurement_frontier() == summary
    assert summary["reference"]["receipt_count"] == 3
    assert summary["candidate"]["receipt_count"] == 2


@pytest.mark.parametrize("change", ["config", "key", "metric", "cases", "missing_identity"])
def test_incomparable_receipts_are_not_averaged(change, replay_receipts):
    env = AdapterEvalEnv(adapter="never called")
    rows = [receipt("r1"), receipt("r2")]
    if change == "config":
        rows[1]["provenance"]["config_sha256"] = "changed"
    elif change == "key":
        rows[1]["comparison_key"] = "changed"
    elif change == "metric":
        rows[1]["primary_metric"]["name"] = "different"
    elif change == "cases":
        rows[1]["cases"] = {"another": "pass"}
    else:
        for r in rows:
            r.pop("comparison_key")
    replay_receipts(env, {"baseline": rows})
    group = env.measurement_frontier()["reference"]
    assert group["aggregate"] is None and group["aggregate_unavailable_reason"]


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), "0.5"])
def test_invalid_numeric_measurements_never_become_quality_evidence(value, replay_receipts):
    env = AdapterEvalEnv(adapter="never called")
    replay_receipts(env, {"baseline": [receipt("r1", value)]})
    group = env.measurement_frontier()["reference"]
    assert group["valid_count"] == 0 and group["aggregate"] is None
    assert group["recent"][0]["primary"] is None


def test_projection_selects_recent_receipts_without_discarding_long_ids(replay_receipts):
    env = AdapterEvalEnv(adapter="never called")
    runs = {
        "baseline": [receipt("r" + str(i)) for i in range(1000)],
        "old-hypothesis": [receipt("irrelevant") for _ in range(1000)],
    }
    runs["baseline"].append(receipt("x" * 10000))
    replay_receipts(env, runs)
    summary = env.measurement_frontier()
    assert len(summary["reference"]["recent"]) == 2
    assert summary["reference"]["older_receipts_not_shown"] == 999
    assert summary["reference"]["recent"][-1]["run_id"] == "x" * 10000
    assert summary["reference"]["recent"][-1]["omitted_identifiers"] == []
    assert env.measurement_frontier()["reference"]["receipt_count"] == 1001 and "irrelevant" not in json.dumps(summary)


def test_frontier_restores_from_actual_results_not_agent_aliases_and_clears_on_phase(fake_adapter):
    events = []
    adapter = fake_adapter(
        {
            "run alias-baseline": receipt("real-baseline"),
            "apply h1": "Applied h1",
            "run alias-candidate": receipt("real-candidate", 0.75, config="b"),
            "decide h1 keep": "h1 keep: accepted",
        }
    )
    source = AdapterEvalEnv(adapter=adapter.command)
    source.runs_per_hypothesis = 1
    for action in ("run alias-baseline", "apply h1", "run alias-candidate", "decide h1 keep"):
        result = source.step(action)
        events.append(
            {
                "kind": "step",
                "action": action,
                "result": result.observation,
                "patch": {"runs": {"invented-agent-alias": {"primary": 99}}},
            }
        )
    restored = AdapterEvalEnv(adapter="must not execute")
    restored.restore_from_events(events, check_adapter=False)
    assert restored.measurement_frontier() == source.measurement_frontier()
    assert restored.measurement_frontier()["reference"]["hypothesis"] == "h1"
    assert "invented-agent-alias" not in json.dumps(restored.measurement_frontier())
    restored.restore_from_events(events, check_adapter=False)
    assert restored.measurement_frontier()["reference"]["receipt_count"] == 1
    restored.restore_from_events(
        events + [{"kind": "step", "action": "phase", "result": "Advanced to phase 2"}], check_adapter=False
    )
    assert restored.measurement_frontier()["reference"]["receipt_count"] == 0
