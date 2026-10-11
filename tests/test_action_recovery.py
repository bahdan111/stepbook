import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.contracts.journal import MalformedJournal
from stepbook.runtime.loop import StepResult
from stepbook.storage.events import EventLog
from stepbook.storage.recovery import pending_actions, reconcile_receipt, recover_actions, run_lock


class External:
    def __init__(self, effects):
        self.effects = effects
        self.resuming = False
        self.restored = []

    def reset(self):
        return "ready"

    def step(self, action):
        self.effects.append(action)
        return StepResult("finished " + action, info={"ok": True})

    def restore_from_events(self, events):
        self.restored = [e["action"] for e in events if e["kind"] == "step"]

    def score(self):
        return 0


ROOT = Path(__file__).resolve().parents[1]


def skill():
    return Skill("test", "do the work", {"type": "object"}, {})


def test_receipt_recovers_patch_and_paid_action_without_redispatch(tmp_path):
    effects = []
    rt = Runtime(
        skill(),
        ScriptedExecutor(lambda *_: Envelope("r", {"finding": "remember"}, "paid-eval")),
        External(effects),
        log_dir=tmp_path,
    )
    append = rt.log.append

    def crash(kind, **data):
        if kind == "step":
            raise KeyboardInterrupt("crash after durable receipt")
        append(kind, **data)

    rt.log.append = crash
    with pytest.raises(KeyboardInterrupt):
        rt.run(1)
    assert effects == ["paid-eval"]
    fresh_env = External(effects)
    fresh = Runtime(
        skill(), ScriptedExecutor(lambda *_: pytest.fail("no replacement model call")), fresh_env, log_dir=tmp_path
    )
    result = fresh.run(1, resume=True, replay_env=False)
    assert effects == ["paid-eval"] and fresh_env.restored == ["paid-eval"]
    assert result.final_state["finding"] == "remember" and result.llm_calls == 1
    assert [e for e in fresh.log.read() if e["kind"] == "step"][0]["recovered_from_receipt"]
    recover_actions(fresh.log)
    assert len([e for e in fresh.log.read() if e["kind"] == "step"]) == 1


@pytest.mark.parametrize("after_effect", [False, True])
def test_uncertain_action_blocks_before_reset_or_model_call(tmp_path, after_effect):
    effects = []
    env = External(effects)
    rt = Runtime(skill(), ScriptedExecutor(lambda *_: Envelope("r", {}, "paid-eval")), env, log_dir=tmp_path)
    original = env.step

    def crash(action):
        if after_effect:
            original(action)
        raise KeyboardInterrupt("unknown result")

    env.step = crash
    with pytest.raises(KeyboardInterrupt):
        rt.run(1)
    fresh_env = External(effects)
    fresh_env.reset = lambda: pytest.fail("must not reset external state")
    fresh = Runtime(
        skill(), ScriptedExecutor(lambda *_: pytest.fail("must not call model")), fresh_env, log_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="unresolved action"):
        fresh.run(2, resume=True, replay_env=False)
    assert len(effects) == int(after_effect)


def test_event_tail_is_preserved_and_later_events_remain_readable(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append("first")
    with path.open("ab") as f:
        f.write(b'{"partial":"\xe2')
    assert len(log.read()) == 1
    log.append("next")
    assert [e["kind"] for e in log.read()] == ["first", "next"]
    assert next(tmp_path.glob("*.torn-*")).read_bytes() == b'{"partial":"\xe2'
    path.write_text('{"kind":"intact"}')
    log.append("last")
    assert [e["kind"] for e in log.read()] == ["intact", "last"]
    path.write_text('{broken}\n{"kind":"hidden"}\n')
    with pytest.raises(RuntimeError, match="corrupt completed event"):
        log.read()


def test_completed_run_resume_does_not_spend_again(tmp_path):
    env = External([])
    env.step = lambda _: StepResult("Stopped", done=True)
    Runtime(skill(), ScriptedExecutor(lambda *_: Envelope("r", {}, "stop")), env, log_dir=tmp_path).run(1)
    fresh = Runtime(
        skill(),
        ScriptedExecutor(lambda *_: pytest.fail("completed run must not call the model")),
        External([]),
        log_dir=tmp_path,
    )
    assert fresh.run(10, resume=True, replay_env=False).steps == 1


def test_missing_resume_log_does_not_start_a_new_run(tmp_path):
    with pytest.raises(RuntimeError, match="no events to resume"):
        Runtime(skill(), ScriptedExecutor(lambda *_: pytest.fail()), External([]), log_dir=tmp_path).run(1, resume=True)


def test_exclusive_writer_blocks_runtime_and_reconciliation(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("action_started", action_id="uncertain", record={"step": 0, "action": "run eval"})
    with run_lock(log):
        with pytest.raises(RuntimeError, match="active writer"):
            Runtime(skill(), ScriptedExecutor(lambda *_: pytest.fail()), External([]), log_dir=tmp_path).run(
                1, resume=True
            )
        with pytest.raises(RuntimeError, match="active writer"):
            reconcile_receipt(log, "uncertain", tmp_path / "missing", "verified")


def test_explicit_receipt_reconciliation_keeps_audit_and_never_overwrites(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append(
        "action_started", action_id="uncertain", record={"step": 0, "action": "run eval", "patch": {"memo": "next"}}
    )
    assert pending_actions(log)[0]["receipt_available"] is False
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"result": "Eval finished:\n{}", "info": {}, "done": False}))
    reconcile_receipt(log, "uncertain", receipt, "Confirmed process exit and inspected evaluator output")
    assert not pending_actions(log)
    events = log.read()
    assert events[-1]["kind"] == "step" and events[-1]["patch"] == {"memo": "next"}
    r = [e for e in events if e["kind"] == "action_result"][0]
    assert r["reconciliation"]["receipt_sha256"] and "not independently verified" in r["reconciliation"]["authority"]
    assert r["reconciliation"] == {
        "authority": "explicit operator assertion; not independently verified",
        "reason": "Confirmed process exit and inspected evaluator output",
        "receipt_path": str(receipt.resolve()),
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }, "the digest must be of the receipt actually read, and the operator's words kept verbatim"
    assert (r["result"], r["info"], r["done"]) == ("Eval finished:\n{}", {}, False)
    assert events[-1]["action"] == "run eval" and events[-1]["step"] == 0
    with pytest.raises(ValueError, match="never replace"):
        reconcile_receipt(log, "uncertain", receipt, "again")


def test_recovery_cli_read_only_and_bad_receipt_do_not_mutate(tmp_path, capsys):
    from stepbook.cli import main

    log = EventLog(tmp_path / "events.jsonl")
    log.append("action_started", action_id="a1", record={"action": "run baseline", "step": 0})
    before = log.path.read_bytes()
    assert main(["recover", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["pending_actions"][0]["action_id"] == "a1"
    assert log.path.read_bytes() == before
    bad = tmp_path / "receipt.json"
    bad.write_text('{"result": "guessed"}')
    assert main(["recover", str(tmp_path), "--action-id", "a1", "--receipt", str(bad), "--reason", "test"]) == 2
    assert log.path.read_bytes() == before


def test_a_malformed_action_record_is_refused_whether_or_not_a_step_committed_it(tmp_path, capsys):
    """Recovery must refuse malformed action records with one sentence and exit 2,
    whether the action is pending or already committed by a later step."""
    from stepbook.cli import main

    def journal(name, *events):
        log = EventLog(tmp_path / name / "events.jsonl")  # EventLog makes the run directory
        for kind, data in events:
            log.append(kind, **data)
        return log

    started = ("action_started", {"action_id": "a1", "record": 17})
    committed = ("step", {"action_id": "a1", "step": 0, "action": "run baseline", "result": "done"})
    readable = ("action_started", {"action_id": "a1", "record": {"step": 0, "action": "run baseline"}})
    bad_receipt = ("action_result", {"action_id": 17, "result": "done", "info": {}, "done": False})
    bad_step = ("step", {"action_id": 17, "step": 0, "action": "run baseline", "result": "done"})
    bad_record, bad_id = "record: Input should be a valid dictionary", "action_id: Input should be a valid string"
    for name, events, index, fault in (
        ("pending", (started,), 0, bad_record),
        ("committed", (started, committed), 0, bad_record),
        ("result_id", (readable, bad_receipt), 1, bad_id),
        ("step_id", (readable, bad_step), 1, bad_id),
    ):
        log = journal(name, *events)
        before = log.path.read_bytes()
        assert main(["recover", str(log.path.parent)]) == 2
        printed = capsys.readouterr()
        assert printed.out == ""
        assert printed.err == f"malformed event {index} in {log.path}: {fault}\n"
        assert log.path.read_bytes() == before, "a refused read must not touch the journal"
    assert main(["recover", str(journal("clean", readable, committed).path.parent)]) == 0
    listing = json.loads(capsys.readouterr().out)["pending_actions"]
    assert listing == [], "the step commits a1, so only the malformed field is refused above"


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("result", "result: Field required"),
        ("step", "step: Field required"),
        ("action", "action: Field required"),
        ("non_string_result", "result: Input should be a valid string"),
    ],
)
def test_a_malformed_step_is_refused_by_a_resume_the_way_recover_refuses_it(tmp_path, fault, problem):
    """Resume must refuse missing step fields and non-string observations with the same diagnostic as recovery,
    naming the record position without calling the executor or changing the journal."""
    step = {"step": 0, "action": "run baseline", "result": "finished run baseline", "action_id": "a1"}
    if fault == "non_string_result":
        step["result"] = 17
    else:
        del step[fault]
    log = EventLog(tmp_path / "events.jsonl")  # EventLog makes the run directory
    log.append("run_start", resume=False, max_steps=1, skill="test", require_completion=False)
    log.append("step", **step)
    before = log.path.read_bytes()
    executor = ScriptedExecutor(lambda *_: pytest.fail("a refused journal calls no model"))
    runtime = Runtime(skill(), executor, External([]), log_dir=tmp_path)
    with pytest.raises(MalformedJournal) as refusal:
        runtime.run(2, resume=True, replay_env=False)
    assert isinstance(refusal.value, RuntimeError)
    assert str(refusal.value) == f"malformed event 1 in {log.path}: {problem}"
    # Refused where the recovery refusal above is refused: before this attempt's run_start is recorded, so the
    # journal is byte-identical to the one the resume was handed.
    assert log.path.read_bytes() == before


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("result", "result: Field required"),
        ("step", "step: Field required"),
        ("action", "action: Field required"),
        ("non_string_result", "result: Input should be a valid string"),
        ("malformed_action_record", "action_id: Input should be a valid string"),
    ],
)
def test_the_run_command_answers_a_malformed_journal_with_recovers_sentence_and_exit(tmp_path, fault, problem):
    """The run command must report malformed step and action records with the recovery diagnostic and exit 2.
    Refusal must precede controller execution and leave the journal unchanged."""
    work, run = tmp_path / "work", tmp_path / "run"
    work.mkdir()
    run.mkdir()
    never = tmp_path / "never_called.py"
    never.write_text(
        "import sys\nsys.stderr.write('the controller was called; a refused resume must not reach it')\nsys.exit(3)\n"
    )
    if fault == "malformed_action_record":
        record = {"ts": 2.0, "kind": "action_started", "action_id": 17, "record": {"step": 0, "action": "ls"}}
    else:
        record = {"ts": 2.0, "kind": "step", "step": 0, "action": "ls", "action_id": "a1", "result": "listing"}
        if fault == "non_string_result":
            record["result"] = 17
        else:
            del record[fault]
    journal = run / "events.jsonl"
    journal.write_text(
        json.dumps({"ts": 1.0, "kind": "run_start", "skill": "checklist", "resume": False})
        + "\n"
        + json.dumps(record)
        + "\n"
    )
    before = journal.read_bytes()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "stepbook.cli",
            "run",
            str(ROOT / "stepbook/templates/checklist"),
            "--workdir",
            str(work),
            "--goal",
            "resume a journal no writer produces",
            "--executor",
            "command",
            "--command",
            f"{sys.executable} {never}",
            "--steps",
            "2",
            "--log-dir",
            str(run),
            "--resume",
            "--quiet",
            "--unsandboxed-i-accept-the-risk",
            "--no-checkpoint",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-600:])
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == f"malformed event 1 in {journal}: {problem}\n"
    assert journal.read_bytes() == before, "a refused resume must not touch the journal"


def test_a_consistent_non_string_action_id_is_refused_rather_than_recovered(tmp_path, capsys):
    """Recovery must refuse non-string action IDs even when they match across started, result and step records."""
    from stepbook.cli import main

    log = EventLog(tmp_path / "events.jsonl")
    log.append("action_started", action_id=5, record={"step": 0, "action": "run baseline"})
    log.append("action_result", action_id=5, result="done", info={}, done=False)
    sentence = f"malformed event 0 in {log.path}: action_id: Input should be a valid string"
    with pytest.raises(RuntimeError) as refusal:
        recover_actions(log)
    assert str(refusal.value) == sentence
    assert [e["kind"] for e in log.read()] == ["action_started", "action_result"], "no step was recovered"
    # The step the old recovery appended for this triple, in its key order; that journal is refused the same way.
    recovered = {"step": 0, "action": "run baseline", "result": "done", "info": {}, "done": False}
    log.append("step", action_id=5, **recovered, recovered_from_receipt=True)
    assert main(["recover", str(tmp_path)]) == 2
    assert capsys.readouterr().err == sentence + "\n"
