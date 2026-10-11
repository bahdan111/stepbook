"""The short resume: `stepbook run --resume [RUN_DIR]` continues a run from its own launch record.

Offline: a scripted controller stands in for the agent, so no model is ever called.
"""

import importlib
import json
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook import Envelope, ScriptedExecutor, cli
from stepbook.cli import run as cli_run
from stepbook.run_pointer import classify_run, resume_hint, write_launch_pointers, write_launch_record

ROOT = Path(__file__).resolve().parent.parent
CHECKLIST = ROOT / "stepbook/templates/checklist"


@pytest.fixture
def launch(tmp_path, monkeypatch):
    """Launch runs through main() with a scripted controller, keeping the namespace each run received."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    seen = []

    def select(args, env, roots):
        seen.append(args)
        return ScriptedExecutor(lambda *_: Envelope("offline", {}, "echo hi")), "command"

    monkeypatch.setattr(cli_run, "_select_run_executor", select)

    def run(argv):
        return cli.main(argv), seen

    return run


def _launch_argv(tmp_path):
    return [
        "run",
        str(CHECKLIST),
        "--workdir",
        str(tmp_path),
        "--goal",
        "verify the short resume",
        "--check",
        "unit=true",
        "--constraint",
        "never force push",
        "--steps",
        "1",
        "--log-dir",
        str(tmp_path / ".stepbook" / "run"),
        "--no-checkpoint",
        "--allow-dirty",
        "--allow-outside",
        "--quiet",
    ]


def test_resume_from_the_record_needs_no_other_option(launch, tmp_path, capsys):
    code, seen = launch(_launch_argv(tmp_path))
    assert code == 3 and classify_run(tmp_path / ".stepbook" / "run")["status"] == "stopped"
    capsys.readouterr()

    assert launch(["run", "--resume", str(tmp_path / ".stepbook" / "run")])[0] == 3
    resumed = seen[-1]
    assert (resumed.skill_dir, resumed.goal, resumed.steps) == (str(CHECKLIST), "verify the short resume", 1)
    assert (resumed.workdir, resumed.log_dir) == (str(tmp_path), str(tmp_path / ".stepbook" / "run"))
    assert (resumed.check, resumed.constraint) == (["unit=true"], ["never force push"])
    assert (resumed.allow_dirty, resumed.allow_outside, resumed.quiet, resumed.resume) == (True, True, True, True)
    assert "resuming" in capsys.readouterr().err
    assert len(json.loads((tmp_path / ".stepbook" / "run" / "launch.json").read_text())["resumes"]) == 1


def test_resume_without_a_directory_uses_the_pointer_and_steps_override(launch, tmp_path):
    code, seen = launch(_launch_argv(tmp_path))
    assert code == 3
    assert launch(["run", "--resume", "--steps", "3"])[0] == 3
    assert (seen[-1].log_dir, seen[-1].steps) == (str(tmp_path / ".stepbook" / "run"), 3)
    assert classify_run(tmp_path / ".stepbook" / "run")["steps"] == 3  # the journal really continued


def test_reopen_works_with_the_short_form(launch, tmp_path):
    code, seen = launch(_launch_argv(tmp_path))
    assert code == 3
    assert launch(["run", "--resume", "--reopen", "one more command, then stop", "--steps", "2"])[0] == 3
    assert seen[-1].reopen == "one more command, then stop" and seen[-1].resume is True


def test_an_option_the_record_owns_is_refused(launch, tmp_path, capsys):
    code, seen = launch(_launch_argv(tmp_path))
    assert code == 3
    capsys.readouterr()
    assert launch(["run", "--resume", "--goal", "something else", "--workdir", str(tmp_path)])[0] == 2
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and "--goal, --workdir cannot be given with it" in error
    assert "settings recorded in its launch.json" in error
    assert len(seen) == 1  # nothing was launched
    assert "resumes" not in json.loads((tmp_path / ".stepbook" / "run" / "launch.json").read_text())


def test_the_acknowledgement_is_never_taken_from_the_record(launch, tmp_path, monkeypatch, capsys):
    code, _ = launch(_launch_argv(tmp_path) + ["--unsandboxed-i-accept-the-risk"])
    assert code == 3
    capsys.readouterr()
    monkeypatch.delenv("STEPBOOK_SANDBOX")
    monkeypatch.setattr(cli.resume, "in_sandbox", lambda: False)
    assert launch(["run", "--resume"])[0] == 2
    error = capsys.readouterr().err
    assert "never inherits --unsandboxed-i-accept-the-risk" in error
    assert "pass it again, or set STEPBOOK_SANDBOX=1" in error
    assert launch(["run", "--resume", "--unsandboxed-i-accept-the-risk", "--steps", "2"])[0] == 3


def test_a_live_run_is_not_continued_by_a_second_launcher(launch, tmp_path, capsys):
    run = tmp_path / ".stepbook" / "live"
    write_launch_record(run, _launch_argv(tmp_path), cwd=tmp_path)
    (run / "events.jsonl").write_text(json.dumps({"ts": 1.0, "kind": "run_start"}) + "\n")
    write_launch_pointers(run, cwd=tmp_path)  # this process stands in for the live launcher
    assert launch(["run", "--resume"])[0] == 2
    assert "is running (its launcher is alive)" in capsys.readouterr().err


def test_a_run_without_a_record_says_to_supply_the_launch(launch, tmp_path, capsys):
    (tmp_path / "bare").mkdir()
    assert launch(["run", "--resume", str(tmp_path / "bare")])[0] == 2
    assert "records no launch to resume from" in capsys.readouterr().err


def test_the_long_form_is_unchanged(launch, tmp_path, capsys):
    """Every option repeated with --resume --log-dir keeps working, record or no record."""
    argv = _launch_argv(tmp_path)
    code, seen = launch(argv)
    assert code == 3
    (tmp_path / ".stepbook" / "run" / "launch.json").unlink()
    assert launch([*argv, "--steps", "2", "--resume"])[0] == 3
    assert (seen[-1].skill_dir, seen[-1].steps, seen[-1].resume) == (str(CHECKLIST), 2, True)
    assert classify_run(tmp_path / ".stepbook" / "run")["steps"] == 2


def test_status_and_hook_hints_print_the_short_form(launch, tmp_path, monkeypatch, capsys):
    code, _ = launch(_launch_argv(tmp_path))
    assert code == 3
    capsys.readouterr()
    run = tmp_path / ".stepbook" / "run"
    assert cli.main(["status", str(run)]) == 0
    status = capsys.readouterr().out
    command = shlex.split(status.split("resume: ", 1)[1].splitlines()[0])
    assert command[-3:] == ["run", "--resume", str(run)]
    assert shlex.split(resume_hint(classify_run(run)))[-3:] == ["run", "--resume", str(run)]
    hint = subprocess.run(
        ["python3", str(ROOT / "hooks/reinject_state.py")],
        input=json.dumps({"cwd": str(tmp_path)}),
        capture_output=True,
        text=True,
        check=True,
    )
    context = json.loads(hint.stdout)["hookSpecificOutput"]["additionalContext"]
    assert f"stepbook run --resume {run}" in context


def test_short_resume_runs_in_the_launch_directory(launch, tmp_path, monkeypatch, capsys):
    """Relative paths in the record mean what they meant at launch, wherever the resume is typed."""
    argv = ["." if token == str(tmp_path) else token for token in _launch_argv(tmp_path)]
    argv[argv.index("--log-dir") + 1] = ".stepbook/run"
    assert launch(argv)[0] == 3
    capsys.readouterr()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    code, seen = launch(["run", "--resume", str(tmp_path / ".stepbook" / "run")])
    assert code == 3
    assert Path.cwd() == tmp_path.resolve() and seen[-1].workdir == "."
    assert f"in {tmp_path.resolve()}, where it was launched" in capsys.readouterr().err
    assert len(json.loads((tmp_path / ".stepbook" / "run" / "launch.json").read_text())["resumes"]) == 1


def test_switching_the_agent_on_resume_drops_the_recorded_model(launch, tmp_path):
    assert launch([*_launch_argv(tmp_path), "--model", "sonnet"])[0] == 3

    code, seen = launch(["run", "--resume", str(tmp_path / ".stepbook" / "run"), "--command", "python agent.py"])
    assert code == 3
    assert (seen[-1].executor, seen[-1].command, seen[-1].model) == ("command", "python agent.py", "opus")


ADAPTER = """#!/usr/bin/env python3
import json, sys
from pathlib import Path
Path(__file__).with_name("adapter-calls.txt").open("a").write(" ".join(sys.argv[1:]) + "\\n")
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
cases = {f"c{i}": "pass" if i < 6 else "fail" for i in range(10)}
if cmd == "describe": print("System: fake.\\nAvailable hypotheses:\\n  h1 (config): x")
elif cmd == "reset": print("reset")
elif cmd == "run":
    print(json.dumps({"run_id": sys.argv[2], "config_label": sys.argv[2], "cases": cases,
          "primary_metric": {"name": "acc", "value": 0.6}, "cost": {"usd": 0.0}, "artifacts": {"run_dir": "/tmp"}}))
"""


def test_short_resume_continues_an_adapter_run_without_rerunning_its_evals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER)
    adapter.chmod(adapter.stat().st_mode | stat.S_IEXEC)
    actions = iter(["run b1", "run b2"])
    seen = []

    def select(args, env, roots):
        seen.append(args)
        return ScriptedExecutor(lambda *_: Envelope("offline", {}, next(actions))), "command"

    monkeypatch.setattr(cli_run, "_select_run_executor", select)
    log = tmp_path / ".stepbook" / "adapter-run"
    launch = ["run", str(ROOT / "stepbook/templates/eval-loop-adapter"), "--adapter", f"{sys.executable} {adapter}"]
    launch += ["--goal", "raise accuracy", "--runs-max", "5", "--evidence-root", str(tmp_path)]
    assert cli.main([*launch, "--steps", "1", "--quiet", "--log-dir", str(log)]) == 3

    assert cli.main(["run", "--resume", str(log), "--steps", "2"]) == 3
    resumed = seen[-1]
    assert (resumed.adapter, resumed.runs_max, resumed.evidence_root) == (
        f"{sys.executable} {adapter}",
        5,
        [str(tmp_path)],
    )
    assert (resumed.resume, resumed.quiet, resumed.log_dir) == (True, True, str(log))
    journal = [json.loads(line) for line in (log / "events.jsonl").read_text().splitlines()]
    assert [event["action"] for event in journal if event["kind"] == "step"] == ["run b1", "run b2"]
    calls = (tmp_path / "adapter-calls.txt").read_text().splitlines()
    assert sum(call.startswith("run b1") for call in calls) == 1, "a resume must not rerun a recorded eval"


@pytest.mark.parametrize("module", ["stepbook", "bench.local_harness"])
def test_both_harnesses_continue_a_run_from_its_record(module, tmp_path, monkeypatch, claude_on_path):
    main = pytest.importorskip(f"{module}.cli", reason="the research harness is absent from the public export").main
    run = importlib.import_module("stepbook.cli.run" if module == "stepbook" else "bench.local_harness.cli_run")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    respond = ScriptedExecutor(lambda *_: Envelope("offline", {}, "echo hi"))
    monkeypatch.setattr(run, "ClaudeCLIExecutor", lambda **_: respond)
    log = tmp_path / ".stepbook" / "run"
    # the research harness still selects its environment by name; the product has only --adapter or shell
    environment = [] if module == "stepbook" else ["--env", "shell"]
    launch = ["run", str(CHECKLIST), *environment, "--workdir", str(tmp_path), "--goal", "verify both harnesses"]
    assert main([*launch, "--steps", "1", "--no-checkpoint", "--log-dir", str(log)]) == 3

    assert main(["run", "--resume", str(log), "--steps", "2"]) == 3
    assert classify_run(log)["steps"] == 2
    assert [entry["argv"] for entry in json.loads((log / "launch.json").read_text())["resumes"]] == [
        ["run", "--resume", str(log), "--steps", "2"]
    ]


@pytest.mark.parametrize("on_path", [True, False], ids=["on-path", "not-on-path"])
def test_printed_resume_uses_the_plain_name_when_stepbook_is_on_path(launch, tmp_path, monkeypatch, capsys, on_path):
    """A user types `stepbook`; the recorded full path is printed only where that name would not be found."""
    assert launch(_launch_argv(tmp_path))[0] == 3
    capsys.readouterr()
    run = tmp_path / ".stepbook" / "run"
    monkeypatch.setattr(
        "stepbook.run_pointer.shutil.which", lambda name: "/usr/local/bin/stepbook" if on_path else None
    )
    program = shlex.split(resume_hint(classify_run(run)))[0]
    if on_path:
        assert program == "stepbook"
    else:
        assert program == json.loads((run / "launch.json").read_text())["program"][0]
