import json
from pathlib import Path

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.eval_loop import AdapterEvalEnv


def run_payload(valid=True):
    return {"run_id": "r", "primary_metric": {"name": "accuracy", "value": 0.5}, "cases": {"c": "pass"}, "valid": valid}


def test_failed_decision_preserves_applied_and_reference_on_resume(fake_adapter):
    adapter = fake_adapter({"apply h1": "Applied h1", "decide h1 revert": "Error: rollback failed"})
    env = AdapterEvalEnv(adapter=adapter.command)
    applied = env.step("apply h1")
    refused = env.step("decide h1 revert")
    assert applied.observation == "Applied h1" and "Error" in refused.observation
    assert env.status()["applied"] == "h1" and env.status()["reference"] == "baseline"
    fresh = AdapterEvalEnv(adapter="true")
    fresh.restore_from_events(
        [
            {"kind": "step", "action": "apply h1", "result": "Applied h1: changed", "info": applied.info},
            {"kind": "step", "action": "decide h1 revert", "result": refused.observation, "info": refused.info},
        ]
    )
    assert fresh.status()["applied"] == "h1"


def test_keep_and_skip_preconditions_run_before_any_adapter_mutation(fake_adapter, replay_receipts):
    adapter = fake_adapter({"decide h1 keep": "h1 keep: ok"})
    env = AdapterEvalEnv(adapter=adapter.command)
    replay_receipts(env, {}, applied="h1")
    assert "Error" in env.step("decide h1 keep").observation
    assert "Error" in env.step("decide h1 skip").observation
    assert "Error" in env.step("decide h1").observation
    assert "Error" in env.step("phase").observation
    assert adapter.calls == []
    runs = {"baseline": [run_payload()], "h1": [run_payload(), run_payload(False)]}
    replay_receipts(env, runs, applied="h1")
    assert "Error" in env.step("decide h1 keep").observation
    assert adapter.calls == []
    runs["h1"].append(run_payload())
    replay_receipts(env, runs, applied="h1")
    assert env.step("decide h1 keep").observation == "h1 keep: ok"
    assert adapter.calls == [["decide", "h1", "keep"]] and env.status()["reference"] == "h1"


def test_error_cases_invalidate_run_even_without_error_rate(fake_adapter):
    p = run_payload()
    p["cases"] = {"c": "error"}
    env = AdapterEvalEnv(adapter=fake_adapter({"run errors": p}).command)
    env.step("run errors")
    group = env.measurement_frontier()["reference"]
    assert group["valid_count"] == 0 and group["invalid_count"] == 1


def test_successful_repeated_evaluations_are_not_failure_loops(tmp_path, fake_adapter):
    env = AdapterEvalEnv(adapter=fake_adapter({"run baseline": run_payload()}).command, log_dir=tmp_path)
    skill = Skill.load(Path(__file__).resolve().parents[1] / "stepbook/templates/eval-loop-adapter")
    result = Runtime(
        skill, ScriptedExecutor(lambda *_: Envelope("repeat measurement", {}, "run baseline")), env, log_dir=tmp_path
    ).run(7)
    assert not result.aborted and env.runs_used == 7


def test_phase_replay_does_not_mix_comparison_sets():
    env = AdapterEvalEnv(adapter="true")
    env.restore_from_events(
        [
            {
                "kind": "step",
                "action": "run old",
                "result": "Eval finished:\n" + json.dumps(run_payload()),
                "info": {"ok": True, "receipt": run_payload()},
            },
            {
                "kind": "step",
                "action": "phase",
                "result": "Advanced to phase 2",
                "info": {"ok": True, "phase": "phase 2", "phases_complete": False},
            },
        ]
    )
    assert not env.status()["hypotheses_with_runs"] and env.runs_used == 1 and env.status()["reference"] == "baseline"


def test_changed_evaluator_identity_blocks_promotion_before_adapter_call(
    fake_adapter, replay_receipts, measure_verdict
):
    adapter = fake_adapter(startup=False)
    env = AdapterEvalEnv(adapter=adapter.command)
    runs = {
        "baseline": [{**run_payload(), "comparison_key": "checker-v1"}],
        "h1": [{**run_payload(), "comparison_key": "checker-v2"}] * 2,
    }
    assert "incomparable" in measure_verdict(runs, "h1")
    replay_receipts(env, runs, applied="h1")
    assert "comparison identity" in env.step("decide h1 keep").observation
    assert adapter.calls == []


def test_automatic_code_refusal_rollback_matches_replay(fake_adapter):
    for succeeds in (True, False):
        adapter = fake_adapter({"decide h1 revert": "h1 revert: restored" if succeeds else "Error: restore failed"})
        # Exit 3: applied, and the instructions for the coding step are the whole of stdout.
        adapter.answer("apply h1", "change code", exit_code=3)
        env = AdapterEvalEnv(adapter=adapter.command)
        result = env.step("apply h1")
        assert env.status()["applied"] == (None if succeeds else "h1")
        fresh = AdapterEvalEnv(adapter="true")
        fresh.restore_from_events(
            [{"kind": "step", "action": "apply h1", "result": result.observation, "info": result.info}]
        )
        assert fresh.status()["applied"] == env.status()["applied"]
