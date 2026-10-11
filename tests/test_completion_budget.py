"""Dollar accounting does not gate completion; pause and evidence validation still do."""

import json
from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, Skill
from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.reporting.usage import usage_summary


def fixture(fake_adapter, tmp_path, *, startup=False):
    source = tmp_path / "source.txt"
    source.write_text("The measured evidence is local.\n")
    env = AdapterEvalEnv(
        adapter=fake_adapter(startup=startup).command,
        log_dir=tmp_path / "runtime",
        require_completion=True,
        evidence_cwd=str(tmp_path),
        evidence_roots=(str(source),),
    )
    env.reset = lambda: "finish the recorded investigation"
    finding = {
        "summary": "Evidence retained.",
        "evidence": [{"path": str(source)}],
        "uncertainty": "No new experiment.",
        "next_action": "Review before spending.",
        "run_ids": [],
    }
    return env, "finish " + json.dumps(finding)


class PaidResponse:
    session_id = "own-session"

    def __init__(self, action, *, during=None, patch=None, raw=None, cost=100):
        self.action, self.during, self.patch, self.raw = action, during, patch or {}, raw or {}
        self.calls = 0
        self.cost = cost

    def step(self, *_):
        self.calls += 1
        assert self.calls == 1, "unexpected additional controller query"
        if self.during:
            self.during()
        return Envelope(
            "already paid conclusion",
            self.patch,
            self.action,
            input_tokens=10,
            output_tokens=2,
            cost_usd=self.cost,
            raw={"session_id": self.session_id, **self.raw},
        )


def skill():
    return Skill("completion-budget-test", "finish", {"type": "object", "additionalProperties": False}, {})


@pytest.mark.parametrize("runtime_class", [Runtime])
@pytest.mark.parametrize("raw", [{}, {"usage_unknown": True}, {"cost_unknown": True}])
def test_expensive_or_unpriced_completion_saves_evidence(fake_adapter, tmp_path, runtime_class, raw):
    env, action = fixture(fake_adapter, tmp_path)
    executor = PaidResponse(action, raw=raw)
    runtime = runtime_class(skill(), executor, env, log_dir=env.log_dir)
    result = runtime.run(3)
    assert not result.aborted and result.steps == executor.calls == 1 and env.stopped
    events = runtime.log.read()
    assert not any(
        e["kind"] in ("budget_stop", "local_completion_allowed_after_budget", "action_withheld") for e in events
    )
    assert events[-1]["done"] and events[-1]["info"]["completion_id"]
    artifact = json.loads(Path(events[-1]["info"]["artifact_path"]).read_text())
    finding = json.loads((env.log_dir / "research" / (artifact["finding_id"] + ".json")).read_text())
    assert finding["summary"] == "Evidence retained."
    assert finding["evidence"]
    assert usage_summary(events)["known_total_usd"] == (0 if raw else 100)
    assert usage_summary(events)["cost_complete"] is (not raw)


@pytest.mark.parametrize("runtime_class", [Runtime])
@pytest.mark.parametrize("condition", ["pause", "environment_pause"])
def test_pause_withholds_completion(fake_adapter, tmp_path, runtime_class, condition):
    env, action = fixture(fake_adapter, tmp_path)

    def during():
        if condition == "pause":
            (env.log_dir / "inbox.txt").write_text("retain this requirement\npause\n")
        else:
            env.pause_requested = True

    executor = PaidResponse(action, during=during)
    runtime = runtime_class(skill(), executor, env, log_dir=env.log_dir)
    result = runtime.run(3)
    assert result.aborted and executor.calls == 1 and result.steps == 0
    assert not any(e["kind"] == "action_started" for e in runtime.log.read())
    assert not list(env.log_dir.rglob("completion-*.json"))
    if condition == "pause":
        assert any("retain this requirement" in e.get("messages", []) for e in runtime.log.read())


@pytest.mark.parametrize("runtime_class", [Runtime])
def test_incoming_operator_message_remains_durable_after_completion(fake_adapter, tmp_path, runtime_class):
    env, action = fixture(fake_adapter, tmp_path)
    message = "Recheck the source before publishing"
    executor = PaidResponse(action, during=lambda: (env.log_dir / "inbox.txt").write_text(message + "\n"))
    result = runtime_class(skill(), executor, env, log_dir=env.log_dir).run(1)
    assert not result.aborted and env.stopped
    assert (env.log_dir / "inbox.txt").read_text().strip() == message


@pytest.mark.parametrize("runtime_class", [Runtime])
@pytest.mark.parametrize("condition", ["unknown_receipt", "pending_candidate", "malformed", "outside_authority"])
def test_completion_still_validates_evidence(fake_adapter, tmp_path, runtime_class, condition):
    env, action = fixture(fake_adapter, tmp_path)
    if condition == "unknown_receipt":
        action = action.replace('"run_ids": []', '"run_ids": ["invented"]')
    elif condition == "pending_candidate":
        env._applied = "undecided"
    elif condition == "malformed":
        action = "finish {}"
    else:
        (tmp_path / "not-authorized.txt").write_text("outside scope")
        action = action.replace("source.txt", "not-authorized.txt")
    executor = PaidResponse(action)
    runtime = runtime_class(skill(), executor, env, log_dir=env.log_dir)
    result = runtime.run(1)
    assert executor.calls == result.steps == 1 and not env.stopped
    assert not list(env.log_dir.rglob("completion-*.json"))
    assert not any(e.get("done") for e in runtime.log.read())
    steps = [e for e in runtime.log.read() if e["kind"] == "step"]
    assert steps[-1]["info"]["ok"] is False
    assert "completion not saved" in steps[-1]["result"]


def test_state_validation_still_prevents_commit(fake_adapter, tmp_path):
    env, action = fixture(fake_adapter, tmp_path)
    executor = PaidResponse(action, patch={"unknown": "do not commit"})
    runtime = Runtime(skill(), executor, env, log_dir=env.log_dir, max_retries=0)
    result = runtime.run(3)
    assert result.aborted and result.steps == 0 and result.rejected_patches == 1 and executor.calls == 1
    assert not list(env.log_dir.rglob("completion-*.json"))


def test_pause_arriving_during_state_validation_is_checked_again(fake_adapter, tmp_path):
    env, action = fixture(fake_adapter, tmp_path)
    runtime = Runtime(skill(), PaidResponse(action), env, log_dir=env.log_dir)
    original = runtime.validator.apply

    def apply(*args):
        result = original(*args)
        (env.log_dir / "inbox.txt").write_text("pause\n")
        return result

    runtime.validator.apply = apply
    result = runtime.run(3)
    assert result.aborted and result.steps == 0
    assert not list(env.log_dir.rglob("completion-*.json"))


def test_cli_saves_completion_without_spending_exception(fake_adapter, tmp_path, monkeypatch, capsys, claude_on_path):
    from stepbook.cli import main
    from stepbook.executors.claude import ClaudeCLIExecutor

    env, action = fixture(fake_adapter, tmp_path, startup=True)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    executor = PaidResponse(action)
    monkeypatch.setattr(ClaudeCLIExecutor, "step", lambda *_: executor.step())
    code = main(
        [
            "run",
            str(root / "stepbook/templates/eval-loop-adapter"),
            "--adapter",
            env.adapter,
            "--executor",
            "cli",
            "--evidence-root",
            str(tmp_path / "source.txt"),
            "--require-completion",
            "--quiet",
            "--steps",
            "3",
            "--log-dir",
            str(env.log_dir),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["completion_saved"] and not result["completion_after_spending_limit"]
    assert "completion_budget_excess_usd" not in result
    assert "Saved locally after the query reached the total spending limit" not in Path(result["report"]).read_text()


@pytest.mark.parametrize("runtime_class", [Runtime])
@pytest.mark.parametrize("receipt_available", [True, False])
def test_completion_receipt_recovery_remains_safe_with_expensive_response(
    fake_adapter, tmp_path, runtime_class, receipt_available
):
    env, action = fixture(fake_adapter, tmp_path)
    executor = PaidResponse(action)
    runtime = runtime_class(skill(), executor, env, log_dir=env.log_dir)
    append = runtime.log.append

    def crash(kind, **record):
        if kind == ("step" if receipt_available else "action_result"):
            raise KeyboardInterrupt("crash at local completion boundary")
        append(kind, **record)

    runtime.log.append = crash
    with pytest.raises(KeyboardInterrupt):
        runtime.run(3)
    assert len(list(env.log_dir.rglob("completion-*.json"))) == 1
    fresh_env, _ = fixture(fake_adapter, tmp_path)
    fresh_env.restore_from_events = lambda events: AdapterEvalEnv.restore_from_events(
        fresh_env, events, check_adapter=False
    )
    resumed = runtime_class(skill(), executor, fresh_env, log_dir=env.log_dir)
    if receipt_available:
        result = resumed.run(3, resume=True, replay_env=False)
        assert not result.aborted and result.steps == executor.calls == 1
    else:
        with pytest.raises(RuntimeError, match="unresolved action"):
            resumed.run(3, resume=True, replay_env=False)
    assert executor.calls == 1 and usage_summary(resumed.log.read())["known_total_usd"] == 100
