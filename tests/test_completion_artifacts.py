import json
from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, Skill
from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.reporting.usage import usage_summary
from stepbook.storage.evidence import ResearchStore, parse_completion


def setup(fake_adapter, tmp_path, *, required=True):
    source = tmp_path / "source.txt"
    source.write_text("Table evidence.\nThe missing narrative is on page two.\n")
    log = tmp_path / "runtime"
    env = AdapterEvalEnv(
        adapter=fake_adapter().command,
        log_dir=log,
        require_completion=required,
        evidence_roots=(str(source),),
        evidence_cwd=str(tmp_path),
    )
    finding = {
        "summary": "The narrative was omitted.",
        "evidence": [{"path": str(source), "start": 2, "end": 2}],
        "uncertainty": "Page expansion is not measured yet.",
        "next_action": "Test page_limit=2.",
        "run_ids": [],
    }
    return env, finding


def test_finish_persists_source_finding_and_completion_without_native_tools(fake_adapter, tmp_path):
    env, finding = setup(fake_adapter, tmp_path)
    assert env.action_format_problems("finish " + json.dumps(finding)) == []
    assert not env.log_dir.exists()  # pure format validation never captures sources
    assert "completion artifact required" in env.action_format_problems("stop forgot report")[0]
    result = env.step("finish " + json.dumps(finding))
    assert result.done and env.stopped and result.info["ok"]
    store = ResearchStore(Path(result.info["artifact_path"]).parent)
    record = store.recall(result.info["completion_id"])
    saved = store.recall(record["finding_id"])
    source = store.recall(saved["evidence"][0])
    assert source["text"] == "2: The missing narrative is on page two."
    assert source["sha256"] and source["locator"]["start"] == 2
    assert saved["next_action"] == finding["next_action"] and record["measurements"] == []
    assert __import__("pathlib").Path(result.info["artifact_path"]).is_file()


def test_completion_attaches_host_receipts_across_phase_reset_and_replay(fake_adapter, tmp_path):
    env, finding = setup(fake_adapter, tmp_path)
    payload = {
        "run_id": "original-r1",
        "primary_metric": {"name": "accuracy", "value": 0.5},
        "cases": {"q": "pass"},
        "comparison_key": "phase1",
        "provenance": {"config_sha256": "config1"},
    }
    events = [
        {"kind": "step", "action": "run alias", "result": "Eval finished:\n" + json.dumps(payload)},
        {"kind": "step", "action": "phase", "result": "Advanced to phase 2"},
    ]
    env.restore_from_events(events, check_adapter=False)
    env.restore_from_events(events, check_adapter=False)
    assert env.status()["hypotheses_with_runs"] == [] and env.runs_used == 1
    finding["run_ids"] = ["original-r1"]
    result = env.step("finish " + json.dumps(finding))
    assert result.done
    record = json.loads(Path(result.info["artifact_path"]).read_text())
    assert record["id"] == result.info["completion_id"]
    assert len(record["measurements"]) == 1
    assert record["measurements"][0]["primary_metric"]["value"] == 0.5
    assert record["measurements"][0]["comparison_key"] == "phase1"
    assert record["measurements"][0]["config_sha256"] == "config1"


@pytest.mark.parametrize("condition", ["pending_candidate", "unknown_receipt", "outside_source", "plain_stop"])
def test_completion_cannot_bypass_authority_or_claim_unsaved_evidence(
    fake_adapter, tmp_path, condition, replay_receipts
):
    env, finding = setup(fake_adapter, tmp_path)
    if condition == "pending_candidate":
        replay_receipts(env, {}, applied="unresolved")
    elif condition == "unknown_receipt":
        finding["run_ids"] = ["invented"]
    elif condition == "outside_source":
        (tmp_path / "unallowed.txt").write_text("Outside the declared root")
        finding["evidence"] = [{"path": str(tmp_path / "unallowed.txt")}]
    action = "stop done" if condition == "plain_stop" else "finish " + json.dumps(finding)
    result = env.step(action)
    assert not result.done and result.info["ok"] is False and not env.stopped
    assert not list(tmp_path.rglob("completion-*.json"))


def test_default_plain_stop_remains_supported(fake_adapter, tmp_path):
    env, _ = setup(fake_adapter, tmp_path, required=False)
    assert env.step("stop no artifact requested").done


@pytest.mark.parametrize("bad", ['{"summary":"a","summary":"b"}', '{"x":NaN}', "[]", "x" * 16001])
def test_completion_rejects_ambiguous_unbounded_or_nonfinite_json(bad):
    with pytest.raises(ValueError) as error:
        parse_completion(bad)
    # Each input is refused for its own reason; the long one is malformed JSON, not over a size cap.
    assert (
        str(error.value)
        == {
            '{"summary":"a","summary":"b"}': "duplicate JSON field: summary",
            '{"x":NaN}': "non-finite JSON value: NaN",
            "[]": "response must be one JSON object",
            "x" * 16001: "Expecting value: line 1 column 1 (char 0)",
        }[bad]
    )
    big = {
        "summary": "s" * 16000,
        "evidence": [{"path": "/tmp/x"}],
        "uncertainty": "u",
        "next_action": "n",
        "run_ids": [],
    }
    assert parse_completion(json.dumps(big)) == big, "a valid finding over 16 kB is accepted; there is no size cap"


class Controller:
    session_id = "stable-session"

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def step(self, *_):
        self.calls += 1
        if self.calls > 1:
            pytest.fail("no extra controller query for artifact persistence")
        return Envelope(
            "final evidence",
            {},
            self.action,
            input_tokens=10,
            output_tokens=3,
            cost_usd=0.01,
            raw={"session_id": self.session_id},
        )


@pytest.mark.parametrize("runtime_class", [Runtime])
@pytest.mark.parametrize("receipt_available", [False, True])
def test_final_artifact_crash_uses_existing_receipt_recovery_without_redispatch(
    fake_adapter, tmp_path, runtime_class, receipt_available
):
    env, finding = setup(fake_adapter, tmp_path)
    skill = Skill("test", "finish the evidence", {"type": "object"}, {})
    executor = Controller("finish " + json.dumps(finding))
    runtime = runtime_class(skill, executor, env, log_dir=env.log_dir)
    append = runtime.log.append

    def crash(kind, **record):
        if kind == ("step" if receipt_available else "action_result"):
            raise KeyboardInterrupt("crash at final artifact boundary")
        append(kind, **record)

    runtime.log.append = crash
    with pytest.raises(KeyboardInterrupt):
        runtime.run(3)
    assert len(list((env.log_dir / "research").glob("completion-*.json"))) == 1
    usage = usage_summary(runtime.log.read())
    assert usage["known_total_usd"] == 0.01 and usage["cost_complete"] is receipt_available
    fresh_env, _ = setup(fake_adapter, tmp_path)
    fresh = runtime_class(skill, executor, fresh_env, log_dir=env.log_dir)
    if receipt_available:
        result = fresh.run(3, resume=True, replay_env=False)
        assert result.steps == result.llm_calls == executor.calls == 1 and not result.aborted
    else:
        fresh_env.reset = lambda: pytest.fail("unknown action must block before reset")
        with pytest.raises(RuntimeError, match="unresolved action"):
            fresh.run(3, resume=True, replay_env=False)


@pytest.mark.parametrize("runtime_class", [Runtime])
def test_required_completion_cannot_be_weakened_on_resume(fake_adapter, tmp_path, runtime_class):
    env, _ = setup(fake_adapter, tmp_path)
    skill = Skill("test", "inspect", {"type": "object"}, {})
    runtime_class(skill, Controller("add idea"), env, log_dir=env.log_dir).run(1)
    fresh_env, _ = setup(fake_adapter, tmp_path, required=False)
    fresh_env.reset = lambda: pytest.fail("must refuse changed policy before external reset")
    with pytest.raises(ValueError, match="completion policy changed"):
        runtime_class(skill, Controller("stop"), fresh_env, log_dir=env.log_dir).run(2, resume=True, replay_env=False)


def test_existing_corrupt_record_is_preserved_and_never_reported_as_saved(tmp_path):
    store = ResearchStore(tmp_path / "research")
    record = store.save("completion", {"finding_id": "example"})
    path = store.directory / (record["id"] + ".json")
    path.write_text('{"different":"do not overwrite"}')
    with pytest.raises(ValueError, match="existing research record differs"):
        store.save("completion", {"finding_id": "example"})
    assert path.read_text() == '{"different":"do not overwrite"}'


def test_symlinked_research_directory_does_not_gain_write_authority(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    directory = tmp_path / "research"
    directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ResearchStore(directory).save("completion", {"finding_id": "example"})
    assert not list(target.iterdir())


def test_final_artifact_is_linked_in_report_and_not_mislabeled_as_step_exhaustion(fake_adapter, tmp_path):
    from stepbook.cli import write_report_md

    env, finding = setup(fake_adapter, tmp_path)
    runtime = Runtime(
        Skill("test", "finish", {"type": "object"}, {}),
        Controller("finish " + json.dumps(finding)),
        env,
        log_dir=env.log_dir,
    )
    result = runtime.run(1)
    report = write_report_md(env.log_dir, env, result).read_text()
    assert "final evidence artifact" in report and "completion-" in report
    assert "ended: stop action / episode done" in report
    assert not any(e["kind"] == "max_steps_reached" for e in runtime.log.read())


def test_tampered_source_record_cannot_be_cited_as_verified_evidence(fake_adapter, tmp_path, research_action):
    env, finding = setup(fake_adapter, tmp_path)
    source = research_action(env, "read", finding["evidence"][0])
    store = ResearchStore(env.log_dir / "research")
    path = store.directory / (source["id"] + ".json")
    source["text"] = "altered after capture"
    path.write_text(json.dumps(source))
    finding["evidence"] = [source["id"]]
    result = env.step("finish " + json.dumps(finding))
    assert not result.done and result.info["ok"] is False
    assert "immutable ID" in result.observation
    assert not list(store.directory.glob("completion-*.json"))


@pytest.mark.parametrize("save", [False, True])
def test_cli_completion_contract_controls_exit_status_and_artifact_link(
    fake_adapter, tmp_path, monkeypatch, capsys, save, claude_on_path
):
    from pathlib import Path

    from stepbook.cli import main
    from stepbook.executors.claude import ClaudeCLIExecutor

    env, finding = setup(fake_adapter, tmp_path)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    action = "finish " + json.dumps(finding) if save else "stop forgot the artifact"
    monkeypatch.setattr(
        ClaudeCLIExecutor, "step", lambda *_: Envelope("", {}, action, input_tokens=10, output_tokens=3)
    )
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
            "1",
            "--log-dir",
            str(env.log_dir),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == (0 if save else 3) and result["completion_saved"] is save
    if save:
        assert Path(result["completion_artifact"]).is_file()


def test_new_cli_completion_contract_requires_source_authority_before_work(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from stepbook.cli import main
    from stepbook.executors.claude import ClaudeCLIExecutor

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ClaudeCLIExecutor, "step", lambda *_: pytest.fail("must not spend on an impossible artifact contract")
    )
    root = Path(__file__).resolve().parents[1]
    code = main(
        ["run", str(root / "stepbook/templates/eval-loop-adapter"), "--adapter", "never-run", "--require-completion"]
    )
    assert code == 2 and "needs --evidence-root" in capsys.readouterr().err


@pytest.mark.parametrize("runtime_class", [Runtime])
def test_historical_automatic_repeat_resume_refused_before_recovery(fake_adapter, tmp_path, monkeypatch, runtime_class):
    import stepbook.runtime.loop as product_runtime

    env, _ = setup(fake_adapter, tmp_path, required=False)
    skill = Skill("test", "inspect", {"type": "object"}, {})
    runtime = runtime_class(skill, Controller("stop"), env, log_dir=env.log_dir)
    runtime.log.append("run_start", measurement_repeats=2, require_completion=False)
    before = runtime.log.path.read_bytes()
    env.reset = lambda: pytest.fail("must refuse before reset")
    monkeypatch.setattr(runtime.executor, "step", lambda *a: pytest.fail("must refuse before executor dispatch"))
    module = product_runtime
    monkeypatch.setattr(module, "recover_actions", lambda *a: pytest.fail("must refuse before recovery"))
    with pytest.raises(ValueError) as error:
        runtime.run(2, resume=True, replay_env=False)
    assert str(error.value) == (
        "This journal ran with automatic measurement repeats, which no longer exist; it can be read but not resumed."
    )
    assert runtime.log.path.read_bytes() == before


@pytest.mark.parametrize("runtime_class", [Runtime])
@pytest.mark.parametrize("policy", [{}, {"measurement_repeats": 1}])
def test_historical_disabled_repeats_resume_normally(fake_adapter, tmp_path, runtime_class, policy):
    env, _ = setup(fake_adapter, tmp_path, required=False)
    skill = Skill("test", "inspect", {"type": "object"}, {})
    runtime = runtime_class(skill, Controller("stop"), env, log_dir=env.log_dir)
    runtime.log.append("run_start", require_completion=False, **policy)
    before = runtime.log.path.read_bytes()
    result = runtime.run(2, resume=True, replay_env=False)
    assert result.steps == result.llm_calls == 1 and not result.aborted
    assert runtime.log.path.read_bytes().startswith(before)
