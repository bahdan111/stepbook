"""CLI outcome coverage: offline CLI runs and read-only inspection of recorded evidence."""

import contextlib
import importlib
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stepbook import Envelope, ScriptedExecutor
from stepbook.cli import main as stepbook_main
from stepbook.contracts.errors import UnreceiptedAction
from stepbook.run_pointer import classify_run, resume_hint, write_launch_record
from stepbook.storage.events import EventLog
from stepbook.storage.recovery import recover_actions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["stepbook"])
def modules(request):
    return tuple(
        importlib.import_module(
            f"{request.param}.{name.replace('cli_', 'cli.').replace('cli.inspect', 'cli.inspection') if request.param == 'stepbook' else name}"
        )
        for name in ("cli", "cli_run", "cli_inspect")
    )


def launch(modules, tmp_path, monkeypatch, action, checks=()):
    cli, runner, _ = modules
    monkeypatch.chdir(tmp_path)
    calls = []

    def respond(*_):
        calls.append(action)
        return Envelope("offline", {}, action)

    monkeypatch.setattr(runner, "_select_run_executor", lambda *a: (ScriptedExecutor(respond), "command"))
    argv = [
        "run",
        str(ROOT / "stepbook/templates/checklist"),
        # the research harness still selects its environment by name; the product has only --adapter or shell
        *([] if cli.__name__ == "stepbook.cli" else ["--env", "shell"]),
        "--workdir",
        str(tmp_path),
        "--goal",
        "verify",
        "--steps",
        "1",
        "--log-dir",
        str(tmp_path / "run"),
        "--no-checkpoint",
        "--unsandboxed-i-accept-the-risk",
        "--quiet",
    ]
    for check in checks:
        argv += ["--check", check]
    return cli.main(argv), argv, calls


@pytest.mark.parametrize(
    "action,checks,expected",
    [
        ("done", (), 0),
        ("done", ("unit=true",), 0),
        ("done", ("unit=false",), 1),
        ("stop missing credentials", (), 3),
        ("echo waiting", (), 3),
    ],
)
def test_exit_mapping_and_summary_contract(modules, tmp_path, monkeypatch, capsys, action, checks, expected):
    code, _, _ = launch(modules, tmp_path, monkeypatch, action, checks)
    assert code == expected
    output = json.loads(capsys.readouterr().out)
    assert "report_error" not in output
    assert set(output) == {
        "runtime",
        "steps",
        "llm_calls",
        "rejected_patches",
        "score",
        "avg_prompt_tokens",
        "max_prompt_tokens",
        "total_tokens",
        "cost_usd",
        "cost_complete",
        "wall_s",
        "aborted",
        "assisted_steps",
        "first_attempt_score",
        "rendered_prompt_tokens_est",
        "rendered_prompt_tokens_est_total",
        "model_ids",
        "executor",
        "log_dir",
        "report",
        "decisions_against_verdict",
        "state_file",
    }
    if action.startswith("stop"):
        assert output["score"] == 1.0  # Preserve the existing score contract.
        info = classify_run(tmp_path / "run")
        assert info["status"] == "stopped" and info["stop_reason"] == "missing credentials"
        for verb in ("status", "report"):
            assert modules[0].main([verb, str(tmp_path / "run")]) == 0
            text = capsys.readouterr().out
            assert "stopped by the agent: missing credentials" in text
            assert "--reopen" in text.splitlines()[-1]
    if action == "done" and not checks:
        event = EventLog(tmp_path / "run/events.jsonl").read()[-1]
        assert "no checks configured" in event["result"].lower()
        assert "All checks pass" not in event["result"]


def test_exhausted_resume_explains_total_horizon_without_call(modules, tmp_path, monkeypatch, capsys):
    _, argv, calls = launch(modules, tmp_path, monkeypatch, "echo waiting")
    capsys.readouterr()
    assert modules[0].main([*argv, "--resume"]) == 3
    err = capsys.readouterr().err
    assert "1 steps already done" in err and "horizon" in err
    assert "--steps 2" in err and "total" in err
    assert len(calls) == 1
    assert modules[0].main([*argv, "--resume", "--steps", "2"]) == 3
    assert len(calls) == 2


def test_help_explains_total_steps_and_exit_mapping(modules, capsys):
    with pytest.raises(SystemExit) as caught:
        modules[0].main(["run", "--help"])
    assert caught.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "total steps including earlier invocations" in text
    assert "0 goal met" in text and "1 checks failed" in text
    assert "3 incomplete" in text and "horizon" in text and "agent stop" in text


def test_printed_resume_command_is_the_short_form_and_rewrites_no_launch(tmp_path, monkeypatch):
    """Whatever the record holds, the printed command names the run and the run supplies the rest."""
    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")  # isolated host: no acknowledgement to repeat
    monkeypatch.setattr("stepbook.run_pointer.shutil.which", lambda name: None)  # the recorded program is printed
    write_launch_record(tmp_path, ["run", "skill", "--model", "old"], program=["python", "-m", "stepbook.cli"])
    write_launch_record(
        tmp_path,
        ["run", "skill", "--executor", "command", "--command", "new agent", "--steps", "7", "--reopen", "old note"],
        resume=True,
    )
    before = (tmp_path / "launch.json").read_bytes()
    argv = shlex.split(resume_hint(classify_run(tmp_path)))
    assert argv == ["python", "-m", "stepbook.cli", "run", "--resume", str(tmp_path)]
    assert (tmp_path / "launch.json").read_bytes() == before


def test_recovery_refusal_supplies_exact_command_and_receipt_contract(tmp_path):
    run = tmp_path / "run with spaces"
    log = EventLog(run / "events.jsonl")
    log.append("action_started", action_id="pending-id", record={"step": 0, "action": "touch output"})
    before = log.path.read_bytes()
    with pytest.raises(UnreceiptedAction) as caught:
        recover_actions(log)
    text = str(caught.value)
    assert "verify the old process has stopped first" in text
    assert "result (string), info (object), done (boolean)" in text
    command = next(line for line in text.splitlines() if line.startswith("stepbook recover "))
    argv = shlex.split(command)
    assert argv[:3] == ["stepbook", "recover", str(run)]
    assert argv[argv.index("--action-id") + 1] == "pending-id"
    assert "--receipt" in argv and "--reason" in argv
    assert log.path.read_bytes() == before


def journal(run, state, events):
    run.mkdir(exist_ok=True)
    (run / "state.json").write_text(json.dumps(state))
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def views(modules, run, capsys):
    outputs = []
    for verb in ("status", "report"):
        assert modules[0].main([verb, str(run)]) == 0
        outputs.append(capsys.readouterr().out)
    outputs.append(modules[2].write_report_md(run, object(), SimpleNamespace(total_tokens=0, cost_usd=0)).read_text())
    return outputs


def test_checklist_headline_rows_and_commands_last(modules, tmp_path, capsys):
    state = {
        "goal": "Rename",
        "checks": {"unit": "true", "lint": "true"},
        "checks_status": {"unit": "pass", "lint": "pass"},
        "checklist": {
            "one": {"text": "first | item", "status": "done", "evidence": "proof\nnext"},
            "two": {"text": "second", "status": "todo"},
        },
    }
    journal(
        tmp_path,
        state,
        [
            {"kind": "step", "step": 0, "action": "check", "result": "Checks:", "info": {"checks": {"unit": "pass"}}},
            {"kind": "max_steps_reached"},
        ],
    )
    before = (tmp_path / "events.jsonl").read_bytes()
    for output in views(modules, tmp_path, capsys):
        headline = next(line for line in output.splitlines() if line and not line.startswith("#"))
        assert headline.startswith("Result: 1/2 items done (agent-reported); 1/2 checks passing (host-recorded)")
        assert output.count("| one |") == 1 and output.count("| two |") == 1
        assert r"first \| item" in output and "proof<br>next" in output
        assert "--resume" in output.splitlines()[-1]
        assert output.index("--resume") > output.index("proof")
    assert (tmp_path / "events.jsonl").read_bytes() == before


def test_report_agrees_with_status_while_active_names_the_run(tmp_path, monkeypatch, capsys):
    """A live run has no terminal record yet; report must not call it a crash while status calls it running."""
    monkeypatch.chdir(tmp_path)
    run = tmp_path / ".stepbook/run"
    run.parent.mkdir()
    state = {"goal": "Fix", "checks": {"unit": "true"}, "checklist": {"one": {"text": "first", "status": "todo"}}}
    journal(run, state, [{"kind": "step", "step": 0, "action": "ls", "result": "ok"}])
    (run.parent / "ACTIVE").write_text(str(run))
    for verb in ("status", "report"):
        assert stepbook_main([verb, str(run)]) == 0
        output = capsys.readouterr().out
        assert "status: running" in output and "interrupted" not in output, verb
    (run.parent / "ACTIVE").unlink()
    assert stepbook_main(["report", str(run)]) == 0
    assert "status: interrupted" in capsys.readouterr().out


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("direction,best", [("higher", "0.8"), ("lower", "0.4")])
def test_improvement_headline_counts_tried_and_labels_measurements(modules, tmp_path, capsys, host, direction, best):
    state = {
        "baseline": {"runs": ["b1", "b2"], "primary_mean": 999},
        "facts": {"eval": {"primary": f"accuracy, {direction} is better"}},
        "runs": {key: {"primary": value, "valid": True} for key, value in {"b1": 0.4, "b2": 0.6, "t1": 0.8}.items()},
        "hypotheses": {
            "h1": {"status": "kept", "runs": ["t1"]},
            "h2": {"status": "inconclusive", "runs": ["failed"]},
            "h3": {"status": "todo"},
            "h4": {"status": "skipped"},
        },
        "kept_stack": ["h1"],
    }
    state["runs"]["failed"] = {"valid": False, "primary": 1000}
    events = [{"kind": "run_start"}]
    if host:
        for i, (key, row) in enumerate(state["runs"].items()):
            events.append(
                {
                    "kind": "step",
                    "step": i,
                    "action": "run " + key,
                    "result": "Eval finished:\n"
                    + json.dumps(
                        {
                            "run_id": key,
                            "valid": row["valid"],
                            "primary_metric": {"value": row["primary"], "direction": direction},
                        }
                    ),
                }
            )
        state["runs"]["t1"]["primary"] = 500  # Host receipt must win over contradictory state.
    journal(tmp_path, state, events)
    # Only a receipt's own direction field orders the values; the eval fact's prose about the metric is not read,
    # so agent-reported rows that carry no direction leave the best value unknown.
    expected = best if host else "unknown"
    for output in views(modules, tmp_path, capsys):
        headline = next(line for line in output.splitlines() if line and not line.startswith("#"))
        assert "1/2 ideas kept/tried (agent-reported)" in headline
        assert f"best measured primary={expected}" in headline and "baseline mean=0.5" in headline
        assert ("host-recorded" if host else "agent-reported") in headline.split("best measured")[1]
        assert "999" not in headline and "1000" not in headline


@pytest.mark.parametrize("flag", [None, "--run-dir", "--log-dir"])
def test_status_directory_forms(modules, tmp_path, capsys, flag):
    journal(tmp_path, {}, [{"kind": "run_start"}])
    assert modules[0].main(["status", *([flag] if flag else []), str(tmp_path)]) == 0
    assert f"run: {tmp_path}" in capsys.readouterr().out


def adapter_outputs(run):
    state = {
        "hypotheses": {"h1": {"status": "kept", "kind": "config", "rationale": "candidate", "runs": ["r1"]}},
        "kept_stack": ["h1"],
    }
    journal(
        run,
        state,
        [
            {
                "kind": "step",
                "step": 0,
                "ts": 0,
                "action": "stop enough",
                "result": "Stopped.",
                "done": True,
                "cost_usd": 0,
                "prompt_tokens": 10,
                "output_tokens": 1,
            }
        ],
    )
    outputs = {}
    for module in ("stepbook",):
        cli = importlib.import_module(module + ".cli")
        inspect = importlib.import_module(module + (".cli.inspection" if module == "stepbook" else ".cli_inspect"))
        for verb in ("status", "report"):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                assert cli.main([verb, *(["--run-dir"] if verb == "status" else []), str(run)]) == 0
            outputs[module + "." + verb] = stream.getvalue().replace(str(run), "<run>")
        outputs[module + ".saved"] = (
            inspect.write_report_md(run, object(), SimpleNamespace(total_tokens=11, cost_usd=0))
            .read_text()
            .replace(run.name, "<name>")
        )
    return outputs


def test_adapter_output_byte_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("stepbook.cli.inspection._t.time", lambda: 0)
    actual = adapter_outputs(tmp_path)
    assert actual == json.loads((ROOT / "tests/fixtures/persona_adapter_output.json").read_text())


@pytest.mark.parametrize(
    "action,check,expected",
    [
        ("done", "true", 0),
        ("done", "false", 1),
        ("stop blocked", "true", 3),
        ("echo pending", "true", 3),
        ("pause", "true", 3),
    ],
)
def test_an_installed_cli_propagates_local_run_outcome(tmp_path, action, check, expected):
    """Run an installed console script with a scripted controller in its own process; no auth/network."""
    binary = tmp_path / "venv/bin/stepbook"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from stepbook import Envelope, ScriptedExecutor\n"
        "from stepbook import cli; from stepbook.cli import run as cli_run\n"
        "cli.preflight_command = lambda args: None\n"
        f"cli_run._select_run_executor = lambda *args: (ScriptedExecutor(lambda *_: Envelope('offline', {{}}, {action!r})), 'command')\n"
        "sys.argv = ['stepbook', *sys.argv[1:]]\n"
        "sys.exit(cli.main())\n"
    )
    binary.chmod(0o755)
    run = tmp_path / "run"
    if action == "pause":
        run.mkdir()
        (run / "inbox.txt").write_text("pause\n")
    proc = subprocess.run(
        [
            str(binary),
            "run",
            str(ROOT / "stepbook/templates/checklist"),
            "--workdir",
            str(tmp_path),
            "--goal",
            "offline",
            "--steps",
            "1",
            "--log-dir",
            str(run),
            "--check",
            "unit=" + check,
            "--no-checkpoint",
            "--unsandboxed-i-accept-the-risk",
            "--quiet",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == expected, proc.stderr
    assert "report_error" not in json.loads(proc.stdout)


def test_stop_survives_resume_and_requires_explicit_reopen(modules, tmp_path, monkeypatch, capsys):
    _, argv, calls = launch(modules, tmp_path, monkeypatch, "stop missing credentials")
    capsys.readouterr()
    assert modules[0].main([*argv, "--resume"]) == 3
    assert len(calls) == 1 and classify_run(tmp_path / "run")["stop_reason"] == "missing credentials"
    assert modules[0].main([*argv, "--resume", "--reopen", "credentials supplied", "--steps", "2"]) == 3
    assert len(calls) == 2


def test_conflicting_status_paths_are_refused(modules, tmp_path, capsys):
    assert modules[0].main(["status", str(tmp_path), "--run-dir", str(tmp_path / "other")]) == 2
    assert "supply one run directory" in capsys.readouterr().err


def test_mixed_measurement_sources_are_labelled_separately(modules, tmp_path, capsys):
    state = {
        "baseline": {"runs": ["b1", "b2"]},
        "runs": {"b1": {"primary": 0.4, "valid": True}, "b2": {"primary": 0.6, "valid": True}},
        "hypotheses": {},
        "kept_stack": [],
    }
    journal(
        tmp_path,
        state,
        [
            {
                "kind": "step",
                "step": 0,
                "action": "run trial",
                "result": "Eval finished:\n"
                + json.dumps(
                    {"run_id": "trial", "valid": True, "primary_metric": {"value": 0.8, "direction": "higher"}}
                ),
            }
        ],
    )
    for output in views(modules, tmp_path, capsys):
        headline = next(line for line in output.splitlines() if line and not line.startswith("#"))
        assert "best measured primary=0.8 (host-recorded)" in headline
        assert "baseline mean=0.5 (agent-reported)" in headline


@pytest.mark.parametrize("eval_fact", ["accuracy, higher is better", ["eval.py"], None])
def test_blocked_improvement_without_measurements_or_structured_eval_fact(modules, tmp_path, capsys, eval_fact):
    state = {"baseline": {"runs": []}, "runs": {}, "facts": {"eval": eval_fact}}
    journal(
        tmp_path,
        state,
        [
            {
                "kind": "step",
                "step": 0,
                "action": "stop model unavailable",
                "result": "Stopped by the agent: model unavailable",
                "done": True,
                "info": {"terminal": True, "checks": {}, "ok": False},
            }
        ],
    )
    for output in views(modules, tmp_path, capsys):
        assert "best measured primary=unknown" in output
        assert "baseline mean=unknown" in output
        assert "stopped by the agent: model unavailable" in output
        assert "--reopen" in output.splitlines()[-1]


def test_result_headline_without_checklist_items_shows_only_checks():
    from stepbook.reporting.report_rendering import shell_result_line

    state = {"checklist": {}, "checks": {"tests": "python -m pytest -q"}}
    events = [
        {"kind": "step", "step": 0, "action": "check", "result": "Checks:", "info": {"checks": {"tests": "pass"}}}
    ]
    assert shell_result_line(state, events) == "Result: 1/1 checks passing (host-recorded)"
