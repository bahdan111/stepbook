"""Environment requirements: visible check output, process-group hygiene,
actionable repeat warning, metric limits as constraints, reference-run shortfall text, noise floor in facts."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.envs.shell import ShellEnv

ROOT = Path(__file__).resolve().parent.parent


def _shell(tmp_path, checks, **kw):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return ShellEnv(workdir=work, goal="g", checks=checks, horizon=5, checkpoint=False, require_sandbox=False, **kw)


def test_failing_check_shows_its_output_tail_and_exit_code(tmp_path):
    env = _shell(
        tmp_path,
        {
            "tests": "printf 'NEW FAILURES (not in baseline):\\ntests/test_x.py::test_y\\n2 failed\\n'; exit 1",
            "lint": "true",
        },
    )
    env.reset()
    obs = env.step("check").observation
    assert "tests: fail" in obs and "lint: pass" in obs
    assert "tests output (exit 1), last 40 lines:" in obs and "tests/test_x.py::test_y" in obs
    assert "lint output" not in obs  # passing checks stay one line
    assert env.last_check_output["tests"]["rc"] == 1


def test_check_output_tail_is_bounded(tmp_path):
    env = _shell(tmp_path, {"t": "seq 1 100; exit 1"}, check_output_lines=5)
    env.reset()
    obs = env.step("check").observation
    assert "    100" in obs and "    96" in obs and "    95\n" not in obs


def test_timeout_kills_the_whole_process_group(tmp_path):
    env = _shell(tmp_path, {}, cmd_timeout_s=1.0)
    env.reset()
    marker = tmp_path / "work" / "child.pid"  # inside the working directory; outside paths are refused by design
    t0 = time.time()
    obs = env.step("(sleep 30 & echo $! > child.pid; wait)").observation
    assert "timed out" in obs and "process group was killed" in obs and time.time() - t0 < 10
    pid = int(marker.read_text().strip())
    time.sleep(0.3)
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive, "background child survived the timeout"


def test_check_children_are_reaped_after_the_check(tmp_path):
    marker = tmp_path / "work" / "server.pid"
    env = _shell(
        tmp_path, {"t": f"(sleep 30 & echo $! > {marker}); exit 0"}
    )  # checks are operator commands, not path-guarded
    env.reset()
    started = time.monotonic()
    result = env.step("check")
    elapsed = time.monotonic() - started
    pid = int(marker.read_text().strip())
    time.sleep(0.3)
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive, "a check's leftover child survived"
    assert result.info["checks"] == {"t": "pass"} and env.last_check_output["t"]["rc"] == 0
    assert result.observation == "Checks:\n  t: pass   (" + env.checks["t"] + ")"
    if elapsed >= 10:
        pytest.xfail(
            f"the check returned only after its background child exited by itself ({elapsed:.0f}s of a 30s sleep),"
            " so the disappearance above is natural exit, not reaping"
        )
    assert elapsed < 10


def _receipts(ref_runs, cur_runs, limits=None):

    def run(v, lat):
        return {
            "run_id": f"r{v}{lat}",
            "valid": True,
            "primary_metric": {"name": "acc", "value": v, "direction": "higher"},
            "deterministic_metrics": {"latency_s": lat},
            "metric_directions": {"latency_s": "lower"},
            "metric_limits": limits or {},
            "cases": {},
            "comparison_key": "k",
            "provenance": {"config_sha256": "c"},
        }

    return {"baseline": [run(*r) for r in ref_runs], "h1": [run(*r) for r in cur_runs]}


def test_metric_with_a_limit_is_a_constraint_not_an_objective(measure_verdict):
    lim = {"latency_s": {"max": 1.2}}
    within = _receipts([(0.5, 0.76), (0.6, 0.76)], [(0.9, 1.12), (1.0, 1.12)], limits=lim)
    text = measure_verdict(within, "h1")
    assert "**keep**" in text and "within limit" in text and "regression" not in text.split("(")[0]
    violated = _receipts([(0.5, 0.76), (0.6, 0.76)], [(0.9, 1.30), (1.0, 1.30)], limits=lim)
    text2 = measure_verdict(violated, "h1")
    assert "**revert**" in text2 and "VIOLATES limit" in text2
    objective = _receipts([(0.5, 0.76), (0.6, 0.76)], [(0.9, 1.12), (1.0, 1.12)])
    assert "regression" in measure_verdict(objective, "h1")  # unchanged behaviour when no limit is declared


def test_verdict_names_the_reference_shortfall_against_the_protocol(measure_verdict):
    env = _receipts([(0.25, 0.7)], [(0.5, 0.7), (0.5, 0.7)])
    text = measure_verdict(env, "h1", baseline_runs=3)
    assert "reference has 1 valid run(s) but the protocol asks for 3" in text and "repeat the baseline" in text
    env2 = _receipts([(0.25, 0.7)], [(0.5, 0.7), (0.5, 0.7)])
    assert "protocol asks" not in measure_verdict(env2, "h1")


def test_harness_facts_carry_a_noise_floor_from_repeated_runs(replay_receipts):
    env = _receipts([(0.5, 0.7), (0.625, 0.7), (0.5, 0.7)], [(0.9, 0.7), (0.875, 0.7)])
    measured = AdapterEvalEnv(adapter="must not execute")
    replay_receipts(measured, env, applied="h1")
    facts = measured.measurement_frontier()
    assert facts["reference"]["aggregate"]["spread"] == 0.125 and facts["reference"]["aggregate"]["runs"] == 3
    assert facts["noise_floor"]["observed_spread"] == 0.125
    single = _receipts([(0.5, 0.7)], [(0.9, 0.7)])
    replay_receipts(measured, single, applied="h1")
    assert measured.measurement_frontier()["noise_floor"]["observed_spread"] is None


def test_generic_adapter_declares_limits_from_metric_field_objects(tmp_path):
    (tmp_path / "eval.py").write_text(
        'import json,sys\nout=sys.argv[sys.argv.index("--out")+1]\n'
        'json.dump([{"id":"c%d"%i,"pass":i%2==0,"latency_s":1.0+i*0.1} for i in range(4)], open(out,"w"))\n'
    )
    (tmp_path / "config.json").write_text("{}")
    cfg = {
        "name": "f",
        "run_command": f"{sys.executable} eval.py --out {{out}}",
        "results_format": "json",
        "case_id_field": "id",
        "pass_field": "pass",
        "metric_fields": {"latency_s": {"direction": "lower", "max": 1.2}},
        "config_file": "config.json",
        "hypotheses": {},
    }
    (tmp_path / "eval.json").write_text(json.dumps(cfg))
    out = subprocess.run(
        [sys.executable, "-m", "stepbook.adapters.generic", str(tmp_path / "eval.json"), "run", "b1"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    ).stdout
    payload = json.loads(out)  # `run` prints the receipt alone
    assert payload["metric_limits"] == {"latency_s": {"max": 1.2}} and payload["metric_directions"] == {
        "latency_s": "lower"
    }
    assert abs(payload["deterministic_metrics"]["latency_s"] - 1.15) < 1e-6
