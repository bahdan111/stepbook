"""No-provider tests for run controls and their stopping conditions."""

import json

import pytest

from stepbook import Envelope, Runtime, Skill
from stepbook.contracts.errors import ExecutorError
from stepbook.reporting.usage import usage_summary
from stepbook.runtime.loop import StepResult
from stepbook.storage.events import EventLog


class Environment:
    def __init__(self):
        self.actions = []
        self.resuming = False

    def reset(self):
        return "ready"

    def step(self, action):
        self.actions.append(action)
        return StepResult("done", done=action == "stop")

    def restore_from_events(self, events):
        pass

    def score(self):
        return 0


class Model:
    session_id = None

    def __init__(self, response):
        self.response, self.calls, self.prompts = response, 0, []

    def step(self, system, prompt, schema):
        self.calls += 1
        self.prompts.append(prompt)
        self.session_id = self.session_id or "same-session"
        out = self.response(self.calls)
        out.raw["session_id"] = self.session_id
        return out


def skill():
    return Skill(
        "control-test",
        "do the work",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"memo": {"type": "string"}, "user_requirements": {"type": "array"}},
        },
        {"memo": "old", "user_requirements": []},
        protected_paths=("user_requirements",),
    )


def reply(action="work", patch=None, cost=0.1):
    return Envelope("proposed", patch or {}, action, input_tokens=10, output_tokens=2, cost_usd=cost)


@pytest.mark.parametrize("runtime", [Runtime])
def test_pause_before_query_and_resume_preserves_undelivered_requirements(tmp_path, runtime):
    (tmp_path / "inbox.txt").write_text("do not edit the evaluator\npause\n")
    ex, env = Model(lambda _: pytest.fail("pause must prevent any model query")), Environment()
    result = runtime(skill(), ex, env, log_dir=tmp_path).run(4)
    assert result.aborted and result.steps == ex.calls == 0 and not env.actions
    assert not (tmp_path / "inbox.txt").exists()
    fresh = Model(lambda _: reply("stop"))
    resumed = runtime(skill(), fresh, Environment(), log_dir=tmp_path).run(4, resume=True, replay_env=False)
    assert not resumed.aborted and fresh.calls == 1
    assert "do not edit the evaluator" in fresh.prompts[0]


@pytest.mark.parametrize("runtime", [Runtime])
def test_pause_during_query_logs_paid_response_but_never_dispatches_action(tmp_path, runtime):
    def response(_):
        (tmp_path / "inbox.txt").write_text("retain this constraint\npause\n")
        return reply("paid-eval", {"memo": "uncommitted"})

    ex, env = Model(response), Environment()
    result = runtime(skill(), ex, env, log_dir=tmp_path).run(5)
    assert result.aborted and ex.calls == 1 and result.steps == 0 and not env.actions
    if runtime is Runtime:
        assert result.final_state["memo"] == "old"
        assert result.final_state["user_requirements"] == [{"text": "retain this constraint", "done": False}]
    events = EventLog(tmp_path / "events.jsonl").read()
    assert not any(e["kind"] == "action_started" for e in events)
    assert [e["action"] for e in events if e["kind"] == "action_withheld"] == ["paid-eval"]
    assert usage_summary(events)["known_total_usd"] == 0.1
    fresh = Model(lambda _: reply("stop"))
    runtime(skill(), fresh, Environment(), log_dir=tmp_path).run(5, resume=True, replay_env=False)
    assert fresh.session_id == "same-session" and "retain this constraint" in fresh.prompts[0]
    assert "NOT executed" in fresh.prompts[0] and "paid-eval" in fresh.prompts[0]


@pytest.mark.parametrize("runtime", [Runtime])
def test_pause_between_short_actions_never_starts_next_query(tmp_path, runtime):
    env = Environment()
    step = env.step

    def pause(action):
        (tmp_path / "inbox.txt").write_text("pause\n")
        return step(action)

    env.step = pause
    ex = Model(lambda _: reply())
    result = runtime(skill(), ex, env, log_dir=tmp_path).run(6)
    assert result.aborted and ex.calls == result.steps == 1 and env.actions == ["work"]


@pytest.mark.parametrize("runtime", [Runtime])
def test_session_and_state_messages_reach_the_next_prompt(tmp_path, runtime):
    (tmp_path / "inbox.txt").write_text("inspect documents before editing\n")
    ex = Model(lambda _: reply("stop"))
    result = runtime(skill(), ex, Environment(), log_dir=tmp_path).run(1)
    assert not result.aborted and "inspect documents before editing" in ex.prompts[0]


@pytest.mark.parametrize("runtime", [Runtime])
def test_resumed_historical_budget_stop_allows_another_query(tmp_path, runtime):
    runtime(skill(), Model(lambda _: reply(cost=100)), Environment(), log_dir=tmp_path).run(1)
    log = EventLog(tmp_path / "events.jsonl")
    log.append("budget_stop", step=0, cost_usd=100, max_total_usd=1)
    ex, env = Model(lambda _: reply("stop", cost=100)), Environment()
    result = runtime(skill(), ex, env, log_dir=tmp_path).run(3, resume=True, replay_env=False)
    assert not result.aborted and result.steps == 2 and ex.calls == 1 and env.actions == ["stop"]
    assert usage_summary(log.read())["known_total_usd"] == 200
    assert sum(e["kind"] == "budget_stop" for e in log.read()) == 1


@pytest.mark.parametrize("runtime", [Runtime])
@pytest.mark.parametrize("raw", [{}, {"cost_unknown": True}, {"usage_unknown": True}])
def test_expensive_or_unpriced_controller_completes_without_dollar_stop(tmp_path, runtime, raw):
    def response(_):
        out = reply("stop", cost=100)
        out.raw.update(raw)
        return out

    ex, env = Model(response), Environment()
    result = runtime(skill(), ex, env, log_dir=tmp_path).run(2)
    assert not result.aborted and result.steps == ex.calls == 1 and env.actions == ["stop"]
    events = EventLog(tmp_path / "events.jsonl").read()
    assert not any(e["kind"] in {"budget_stop", "local_completion_allowed_after_budget"} for e in events)
    assert result.cost_complete is (not raw)
    assert usage_summary(events)["cost_complete"] is (not raw)


def test_patch_retries_are_bounded_by_retry_count(tmp_path):
    ex, env = Model(lambda _: reply(patch={"undeclared": 1}, cost=100)), Environment()
    result = Runtime(skill(), ex, env, log_dir=tmp_path, max_retries=2).run(2)
    assert result.aborted and result.rejected_patches == ex.calls == 3 and not env.actions
    assert usage_summary(EventLog(tmp_path / "events.jsonl").read())["known_total_usd"] == 300


@pytest.mark.parametrize("runtime", [Runtime])
@pytest.mark.parametrize("error", [ExecutorError])
@pytest.mark.parametrize("receipt", [None, reply(cost=100)])
def test_failed_receipt_or_unknown_cost_allows_retry(tmp_path, runtime, error, receipt, monkeypatch):
    def response(n):
        if n == 1:
            raise error("offline failure", envelope=receipt)
        return reply("stop", cost=100)

    waits = []
    monkeypatch.setattr("stepbook.runtime.loop.time.sleep", waits.append)
    ex, env = Model(response), Environment()
    result = runtime(skill(), ex, env, log_dir=tmp_path).run(2)
    assert not result.aborted and ex.calls == result.llm_calls == 2 and env.actions == ["stop"]
    assert len(waits) == 1
    usage = usage_summary(EventLog(tmp_path / "events.jsonl").read())
    assert usage["known_total_usd"] == (200 if receipt else 100)
    assert usage["cost_complete"] is (receipt is not None)


@pytest.mark.parametrize("runtime", [Runtime])
def test_pause_after_failed_query_prevents_backoff_and_retry(tmp_path, runtime, monkeypatch):
    def response(_):
        (tmp_path / "inbox.txt").write_text("retain retry constraint\npause\n")
        raise ExecutorError("offline failure")

    monkeypatch.setattr("stepbook.runtime.loop.time.sleep", lambda _: pytest.fail("pause must precede retry wait"))
    ex, env = Model(response), Environment()
    result = runtime(skill(), ex, env, log_dir=tmp_path).run(2)
    assert result.aborted and ex.calls == 1 and not env.actions
    events = EventLog(tmp_path / "events.jsonl").read()
    assert any("retain retry constraint" in e.get("messages", []) for e in events)
    assert events[-1]["kind"] == "paused"
    assert not usage_summary(events)["cost_complete"]


@pytest.mark.parametrize("runtime", [Runtime])
def test_resume_keeps_adapter_attempt_counter_not_number_of_run_actions(tmp_path, runtime, fake_adapter):
    from stepbook.envs.eval_loop import AdapterEvalEnv

    adapter = fake_adapter(
        {
            "run same-label": {
                "run_id": "r",
                "valid": True,
                "primary_metric": {"name": "acc", "value": 0.5},
                "cost": {"tokens": 2, "usd": 0.01},
            }
        }
    )

    def environment():
        return AdapterEvalEnv(adapter=adapter.command, log_dir=tmp_path, runs_max=1)

    env = environment()
    runtime(skill(), Model(lambda _: reply("run same-label")), env, log_dir=tmp_path).run(2)
    assert env.runs_used == 1  # second action was a local budget refusal
    fresh = environment()
    runtime(skill(), Model(lambda _: reply("stop")), fresh, log_dir=tmp_path).run(3, resume=True, replay_env=False)
    assert fresh.runs_used == 1
    events = EventLog(tmp_path / "events.jsonl").read()
    fresh.restore_from_events(events, check_adapter=False)
    fresh.restore_from_events(events, check_adapter=False)
    assert fresh.runs_used == 1 and len(fresh._runs["baseline"]) == 1


@pytest.mark.parametrize("runtime", [Runtime])
def test_prompts_omit_spending_facts_but_accounting_retains_evaluations(tmp_path, runtime):
    env = Environment()
    env.step = lambda _: StepResult("Eval finished:\n" + json.dumps({"cost": {"tokens": 2, "usd": 200}}))
    ex = Model(lambda _: reply("run baseline", cost=100))
    runtime(skill(), ex, env, log_dir=tmp_path).run(2)
    assert all("HOST SPENDING FACTS" not in p and "remaining_usd" not in p for p in ex.prompts)
    assert usage_summary(EventLog(tmp_path / "events.jsonl").read())["known_total_usd"] == 600
