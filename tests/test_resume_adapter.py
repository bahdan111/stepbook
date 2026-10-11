"""Crash-resume rebuilds the adapter ledger so verdicts and reference stay right."""

import json
import stat
import sys
from pathlib import Path

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.cli import write_report_md
from stepbook.envs.eval_loop import AdapterEvalEnv

ROOT = Path(__file__).resolve().parent.parent
SKILL = Skill.load(ROOT / "stepbook/templates" / "eval-loop-adapter")

FAKE = """#!/usr/bin/env python3
import json, sys
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
cases = {f"c{i}": ("pass" if i < 50 else "fail") for i in range(100)}
if cmd == "describe": print("System: fake.\\nAvailable hypotheses:\\n  h1 (config): x")
elif cmd == "reset": print("reset")
elif cmd == "apply": print("Applied h1 (config): x. Config now {}.")
elif cmd == "run":
    label = sys.argv[2]
    better = {f"c{i}": ("pass" if i < 70 else "fail") for i in range(100)} if label.startswith("h1") else cases
    print(json.dumps({"run_id": label, "config_label": label, "cases": better,
          "primary_metric": {"name": "acc", "value": sum(v == "pass" for v in better.values()) / 100},
          "deterministic_metrics": {"recall": 0.8}, "cost": {"usd": 1.0}, "artifacts": {"run_dir": "/tmp"}}))
elif cmd == "decide": print(sys.argv[2] + " " + sys.argv[3] + ": ok")
"""


def _adapter(tmp_path):
    p = tmp_path / "a.py"
    p.write_text(FAKE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return f"{sys.executable} {p}"


def _agent(actions):
    it = iter(actions)

    def fn(system_prompt, user_prompt):
        a = next(it)
        return Envelope("r", {}, a)

    return fn


def test_resume_rebuilds_environment_ledger(tmp_path):
    adapter = _adapter(tmp_path)
    env = AdapterEvalEnv(adapter=adapter, log_dir=tmp_path)
    Runtime(SKILL, ScriptedExecutor(_agent(["run b1", "run b2", "apply h1", "run h1-1"])), env, log_dir=tmp_path).run(4)
    assert env.status()["applied"] == "h1" and "h1" in env.status()["hypotheses_with_runs"]
    # crash; fresh process
    env2 = AdapterEvalEnv(adapter=adapter, log_dir=tmp_path, resuming=True)
    seen = []

    def spy(system_prompt, user_prompt):
        if len(seen) == 1:
            frontier = env2.measurement_frontier()
            assert frontier["candidate"]["receipt_count"] == 2
            assert frontier["reference"]["receipt_count"] == 2
        seen.append(user_prompt)
        return Envelope("r", {}, "run h1-2" if len(seen) == 1 else "decide h1 keep")

    result = Runtime(SKILL, ScriptedExecutor(spy), env2, log_dir=tmp_path).run(6, resume=True, replay_env=False)
    assert env2.measurement_frontier()["reference"]["receipt_count"] == 2
    assert env2.status()["hypotheses_with_runs"] == ["baseline", "h1"]
    report = write_report_md(tmp_path, env2, result).read_text()
    assert sum(line.startswith("| baseline |") for line in report.splitlines()) == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    last_run = [e for e in events if e["kind"] == "step" and e["action"] == "run h1-2"][-1]
    assert "Computed verdict for h1 after 2 valid run(s)" in last_run["result"] and "keep" in last_run["result"]
    assert env2.status()["reference"] == "h1" and env2.runs_used == 4


def _journal(env, actions):
    """Step the environment and keep what each step recorded, as the journal does."""
    events = []
    for action in actions:
        result = env.step(action)
        events.append({"kind": "step", "action": action, "result": result.observation, "info": result.info})
    return events


def _receipt(run_id, value):
    return {
        "run_id": run_id,
        "primary_metric": {"name": "acc", "value": value},
        "cases": {"c": "pass"},
        "artifacts": {"run_dir": f"/tmp/{run_id}"},
    }


def test_resume_reads_recorded_facts_and_ignores_the_adapter_s_wording(fake_adapter):
    """The ledger comes back from each step's info, whatever words the adapter chose for the same facts."""
    adapter = fake_adapter(
        {
            "run b1": _receipt("r1", 0.5),
            "run c1": _receipt("r2", 0.7),
            "run c2": _receipt("r3", 0.7),
            "investigate h1": "h1 is already implemented in reader.py",  # no "Investigation of" prefix
            "decide h1 skip": "noted",
            "apply h2": "configuration updated",  # no "Applied" anywhere
            "decide h2 keep": "h2 is now the reference",  # no "<id> <decision>:" prefix
        }
    )
    env = AdapterEvalEnv(adapter=adapter.command)
    actions = ["run b1", "investigate h1", "decide h1 skip", "apply h2", "run c1", "run c2", "decide h2 keep"]
    events = _journal(env, actions)
    assert [e["result"] for e in events][-1] == "h2 is now the reference"
    assert env.status() == {
        "applied": None,
        "reference": "h2",
        "runs_used": 3,
        "hypotheses_with_runs": ["baseline", "h2"],
    }

    fresh = AdapterEvalEnv(adapter=adapter.command, resuming=True)
    fresh.restore_from_events(events)
    assert fresh.status() == env.status()
    assert fresh._run_dirs == {
        "b1": "/tmp/r1",
        "r1": "/tmp/r1",
        "c1": "/tmp/r2",
        "r2": "/tmp/r2",
        "c2": "/tmp/r3",
        "r3": "/tmp/r3",
        "latest": "/tmp/r3",
    }
    # The restored investigation is what lets a skip through: while it was read from the observation's first
    # words, an adapter that worded its finding differently lost it on resume and the skip was refused.
    assert [i["h"] for i in fresh.investigations] == ["h1"]
    assert fresh.step("decide h1 skip").observation == "noted"
    # A phase that advanced clears the stage's comparison set, by what `status` answered, not by "Advanced to".
    adapter.answer("phase", "stage two")
    adapter.answer("status", {"applied": None, "phase": "stage two", "phases_complete": False})
    after_phase = AdapterEvalEnv(adapter=adapter.command, resuming=True)
    after_phase.restore_from_events(events + _journal(fresh, ["phase"]))
    assert after_phase.status() == {
        "applied": None,
        "reference": "baseline",
        "runs_used": 3,
        "hypotheses_with_runs": [],
    }


def test_resume_refuses_a_journal_recorded_before_the_structured_adapter_journal(tmp_path, fake_adapter):
    """An old journal holds these facts only in its sentences: it is reported on, never resumed."""
    import pytest

    from stepbook.contracts.errors import LegacyAdapterJournal

    old = [
        {"kind": "step", "action": "run b1", "result": "Eval finished:\n{}"},
        {"kind": "step", "action": "apply h1", "result": "Applied h1: changed"},
    ]
    env = AdapterEvalEnv(adapter=fake_adapter().command, resuming=True)
    with pytest.raises(LegacyAdapterJournal, match="recorded before the structured adapter journal"):
        env.restore_from_events(old)
    # The read-only reports still replay it, through the quarantined legacy reader.
    env.restore_from_events(old, check_adapter=False)
    assert env.status()["applied"] == "h1" and env.runs_used == 1


def test_resume_refuses_on_adapter_split_brain(tmp_path):
    fake = FAKE.replace(
        'elif cmd == "decide": print(sys.argv[2] + " " + sys.argv[3] + ": ok")',
        'elif cmd == "decide": print(sys.argv[2] + " " + sys.argv[3] + ": ok")\n'
        'elif cmd == "status": print(json.dumps({"applied": None, "decided": {}}))',
    )
    p = tmp_path / "b.py"
    p.write_text(fake)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    adapter = f"{sys.executable} {p}"
    env = AdapterEvalEnv(adapter=adapter, log_dir=tmp_path)
    Runtime(SKILL, ScriptedExecutor(_agent(["run b1", "apply h1"])), env, log_dir=tmp_path).run(2)
    env2 = AdapterEvalEnv(adapter=adapter, log_dir=tmp_path, resuming=True)
    import pytest

    with pytest.raises(RuntimeError, match="resume mismatch"):
        Runtime(SKILL, ScriptedExecutor(_agent(["run h1-1"])), env2, log_dir=tmp_path).run(
            3, resume=True, replay_env=False
        )


def test_pause_during_long_adapter_call_is_consumed_and_stops_after_step(tmp_path):
    """A pause consumed before Runtime starts prevents any controller call or committed step."""
    slow = FAKE.replace('elif cmd == "run":', 'elif cmd == "run":\n    import time; time.sleep(0.6)')
    p = tmp_path / "c.py"
    p.write_text(slow)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    env = AdapterEvalEnv(adapter=f"{sys.executable} {p}", log_dir=tmp_path, inbox_poll_s=0.1)
    (tmp_path / "inbox.txt").write_text("keep top_k small\npause\n")
    # the runtime consumes the inbox before the step; simulate the file arriving mid-call instead:
    import threading

    def drop():
        import time

        time.sleep(0.2)
        (tmp_path / "inbox.txt").write_text("keep top_k small\npause\n")

    (tmp_path / "inbox.txt").unlink()
    threading.Thread(target=drop).start()
    env.reset()
    env.step("run b1")
    assert env.pause_requested
    assert not any(
        line.strip() == "pause" for line in (tmp_path / "inbox.txt").read_text().splitlines()
    )  # pause consumed
    assert "keep top_k small" in (tmp_path / "inbox.txt").read_text()  # other lines kept for the runtime
    # The environment consumed the pause before this runtime started.
    seen = []

    def spy(system_prompt, user_prompt):
        seen.append(user_prompt)
        return Envelope("r", {}, "run b2")

    log = tmp_path / "rt"
    res = Runtime(SKILL, ScriptedExecutor(spy), env, log_dir=log).run(3)
    assert res.aborted and res.steps == 0 and len(seen) == 0, "pending pause stops before any controller call"
    events = [json.loads(line) for line in (log / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "paused"
    assert events[-1]["reason"] == "operator pause at controller boundary; proposed action not dispatched"
    assert [e["kind"] for e in events].count("step") == 0


def test_pause_during_runtime_driven_adapter_call_commits_one_step_then_stops(tmp_path):
    """A pause arriving during Runtime's long action is consumed and stops after that step commits."""
    import threading
    import time

    slow = FAKE.replace('elif cmd == "run":', 'elif cmd == "run":\n    import time; time.sleep(0.6)')
    p = tmp_path / "c.py"
    p.write_text(slow)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    env = AdapterEvalEnv(adapter=f"{sys.executable} {p}", log_dir=tmp_path, inbox_poll_s=0.1)
    inbox = tmp_path / "inbox.txt"
    seen = []

    def drop():
        time.sleep(0.2)
        inbox.write_text("keep top_k small\npause\n")

    writer = threading.Thread(target=drop)

    def spy(system_prompt, user_prompt):
        assert not seen, "pause must stop the run before a second controller call"
        seen.append(user_prompt)
        writer.start()
        return Envelope("r", {}, "run b1")

    try:
        res = Runtime(SKILL, ScriptedExecutor(spy), env, log_dir=tmp_path).run(3)
    finally:
        if writer.ident is not None:
            writer.join()

    assert res.aborted and res.steps == 1 and len(seen) == 1, "one step committed, then the pause stops the run"
    assert env.pause_requested
    assert inbox.read_text().splitlines() == ["keep top_k small"]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "paused"
    assert events[-1]["reason"] == "operator pause received during a long environment call"
    assert [e["action"] for e in events if e["kind"] == "step"] == ["run b1"]
    assert [e["kind"] for e in events].count("step") == 1
