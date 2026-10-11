"""Failed terminal checks change reporting, never the journal's close/reopen contract."""

import json
import shlex
from types import SimpleNamespace

import pytest

from stepbook.cli import main
from stepbook.run_pointer import RESUMABLE, classify_run, describe
from stepbook.storage.events import EventLog
from tests.probe_workflows._helpers import ROOT, Script, cmd, kinds, steps


def _done(checks):
    return {"kind": "step", "done": True, "info": {"ok": False, "checks": checks, "terminal": True}}


@pytest.mark.parametrize("info", ["legacy text", ["legacy text"], None], ids=["string", "list", "null"])
@pytest.mark.parametrize("active", [False, True])
def test_non_mapping_terminal_info_is_completed(tmp_path, info, active):
    journal = tmp_path / "events.jsonl"
    journal.write_text(json.dumps({"kind": "step", "done": True, "info": info}) + "\n")
    before = journal.read_bytes()
    outcome = classify_run(tmp_path, active=active)
    assert outcome["status"] == "completed"
    assert outcome["failed_checks"] == []
    assert journal.read_bytes() == before


def test_run_help_documents_failed_checks_exit(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["run", "--help"])
    assert caught.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "1 checks failed;" in help_text
    assert "3 incomplete (agent stop, step horizon reached, abort or required completion missing)" in help_text


@pytest.mark.parametrize("local", [False], ids=["product"])
@pytest.mark.parametrize(
    "event, aborted, required, product_expected, local_expected",
    [
        ({"kind": "paused"}, True, True, 3, 1),
        ({"kind": "paused"}, True, False, 3, 0),
        ({"kind": "paused"}, False, False, 3, 0),
        (_done({"unit": "fail"}), False, False, 1, 1),
        (_done({"unit": "pass"}), False, False, 0, 0),
        ({"kind": "max_steps_reached"}, False, True, 3, 3),
        ({"kind": "max_steps_reached"}, False, False, 3, 3),
        ({"kind": "abort"}, True, False, 3, 3),
    ],
)
def test_reporters_preserve_exit_precedence_and_fallback(
    tmp_path, capsys, local, event, aborted, required, product_expected, local_expected
):
    from stepbook.cli import run as product_run

    log = EventLog(tmp_path / "events.jsonl")
    log.append(**event)
    result = SimpleNamespace(summary=lambda: {}, final_state=None, cost_usd=0, total_tokens=0, aborted=aborted)
    report = product_run._report_run
    code = report(SimpleNamespace(log=log), result, SimpleNamespace(), tmp_path, "command", required)
    assert code == (local_expected if local else product_expected)
    output = json.loads(capsys.readouterr().out)
    assert output["executor"] == "command"
    assert output["log_dir"] == str(tmp_path)
    assert "report_error" not in output
    if required:
        assert output["completion_required"] is True and output["completion_saved"] is False


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize(
    ("events", "status", "failed"),
    [
        ([_done({"unit": "fail", "lint": "pass", "types": "fail"})], "failed_checks", ["unit", "types"]),
        ([_done({"unit": "fail"}), {"kind": "resumed"}, {"kind": "paused"}], "failed_checks", ["unit"]),
        ([_done({"unit": "fail"}), {"kind": "reopened"}, {"kind": "paused"}], "paused", []),
        ([_done({"unit": "fail"}), {"kind": "reopened"}, _done({"unit": "pass"})], "completed", []),
        ([_done({"unit": "pass"}), {"kind": "reopened"}, _done({"unit": "fail"})], "failed_checks", ["unit"]),
        ([_done({}), {"kind": "resumed"}], "completed", []),
        ([{"kind": "step", "done": True}], "completed", []),
        ([_done({"unit": "unknown"})], "stopped", []),
        ([{**_done({"unit": "fail"}), "horizon_stop": True}, {"kind": "max_steps_reached"}], "stopped", []),
        ([{**_done({"unit": "fail"}), "done": False}, {"kind": "paused"}], "paused", []),
    ],
)
def test_only_current_terminal_check_failures_change_classification(tmp_path, active, events, status, failed):
    journal = tmp_path / "events.jsonl"
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    before = journal.read_bytes()
    info = classify_run(tmp_path, active=active)
    assert info["status"] == status
    assert info.get("failed_checks", []) == failed
    if failed:
        assert status not in RESUMABLE
        message = describe(info)
        assert "ended with failing checks: " + ", ".join(failed) in message
        assert "--resume --log-dir" in message and "--reopen" in message
        assert "nothing to resume" not in message
    assert journal.read_bytes() == before


@pytest.fixture
def closed_cli_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "log with spaces"
    argv = [
        "run",
        str(ROOT / "stepbook/templates/checklist"),
        "--workdir",
        str(tmp_path),
        "--goal",
        "g",
        "--check",
        "unit=test -f fixed",
        "--check",
        "lint=false",
        "--check",
        "passing=true",
        "--executor",
        "cli",
        "--steps",
        "5",
        "--log-dir",
        str(log),
        "--quiet",
        "--no-checkpoint",
        "--unsandboxed-i-accept-the-risk",
    ]

    def create(check):
        if check == "true":
            (tmp_path / "fixed").touch()
            argv[argv.index("lint=false")] = "lint=true"
        script = Script(cmd("done"), exhausted=None)
        monkeypatch.setattr("stepbook.cli.run.ClaudeCLIExecutor", lambda **_: script.executor())
        expected = 0 if check == "true" else 1
        assert main(argv) == expected
        assert script.calls == 1
        capsys.readouterr()
        return SimpleNamespace(log=log, argv=argv, expected=expected)

    return create


@pytest.mark.parametrize("check", ["true", "false"])
def test_cli_exit_status_and_terminal_checks(closed_cli_run, check):
    log = closed_cli_run(check).log
    assert steps(log)[-1]["done"] is True
    assert steps(log)[-1]["info"] == {
        "ok": check == "true",
        "checks": {
            "unit": "pass" if check == "true" else "fail",
            "lint": "pass" if check == "true" else "fail",
            "passing": "pass",
        },
        "terminal": True,
    }


@pytest.mark.parametrize("check", ["true", "false"])
def test_cli_status_describes_closed_journal(closed_cli_run, capsys, check):
    run = closed_cli_run(check)
    log = run.log
    assert main(["status", "--run-dir", str(log)]) == 0
    status = capsys.readouterr().out
    if run.expected:
        assert "status: failed_checks" in status
        assert "ended with failing checks: unit, lint" in status
        assert "--reopen" in status
        command = status.split("Reopen with: ", 1)[1].splitlines()[0]
        tokens = shlex.split(command)
        assert tokens[tokens.index("--resume") + 1] == str(log)
        assert tokens[tokens.index("--reopen") + 1].strip()
    else:
        assert "status: completed" in status
        assert "--reopen '<note>'" in status
        command = status.split("For new work, reopen with: ", 1)[1].splitlines()[0]
        tokens = shlex.split(command)
        assert tokens[tokens.index("--resume") + 1] == str(log)
        assert tokens[tokens.index("--reopen") + 1] == "<note>"


def _resume_closed_run(run, monkeypatch):
    plain = Script(exhausted=None)
    monkeypatch.setattr("stepbook.cli.run.ClaudeCLIExecutor", lambda **_: plain.executor())
    assert main(run.argv + ["--resume"]) == run.expected
    assert plain.calls == 0 and len(steps(run.log)) == 1


@pytest.mark.parametrize("check", ["true", "false"])
def test_cli_resume_preserves_closed_journal(closed_cli_run, monkeypatch, check):
    _resume_closed_run(closed_cli_run(check), monkeypatch)


def test_cli_reopen_repairs_failed_checks(closed_cli_run, tmp_path, monkeypatch):
    run = closed_cli_run("false")
    _resume_closed_run(run, monkeypatch)
    (tmp_path / "fixed").touch()
    run.argv[run.argv.index("lint=false")] = "lint=true"
    repair = Script(cmd("done"), exhausted=None)
    monkeypatch.setattr("stepbook.cli.run.ClaudeCLIExecutor", lambda **_: repair.executor())
    assert main(run.argv + ["--resume", "--reopen", "checks repaired; finish"]) == 0
    assert repair.calls == 1 and kinds(run.log).count("reopened") == 1
    assert classify_run(run.log)["status"] == "completed"
