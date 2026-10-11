"""End-to-end through the CLI with a fake agent CLI (CommandExecutor) and the shell environment: no model."""

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_command_shell_environment_and_child_share_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")
    work = tmp_path / "shell work"
    work.mkdir()
    agent = tmp_path / "pwd_agent.py"
    agent.write_text(
        "import json, os, shlex\n"
        'print(json.dumps(dict(reasoning="inspect directories", state_patch={}, '
        'action="echo " + shlex.quote(os.getcwd()) + " > child-cwd.txt; pwd > shell-cwd.txt")))\n'
    )
    run = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "stepbook.cli",
            "run",
            str(ROOT / "stepbook/templates/checklist"),
            "--workdir",
            work.name,
            "--goal",
            "inspect working directories",
            "--executor",
            "command",
            "--command",
            shlex.join([sys.executable, str(agent)]),
            "--steps",
            "1",
            "--log-dir",
            str(run),
            "--quiet",
            "--no-checkpoint",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 3, proc.stderr
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    step = next(event for event in events if event["kind"] == "step")
    assert Path(shlex.split(step["action"])[1]) == work.resolve()
    assert (work / "child-cwd.txt").read_text().strip() == str(work.resolve())
    assert (work / "shell-cwd.txt").read_text().strip() == str(work.resolve())


FAKE_AGENT = r"""#!/usr/bin/env python3
import json, re, sys
prompt = sys.stdin.read()
state = json.loads(re.search(r"```json\n(.*?)\n```", prompt, re.S).group(1))
obs = prompt.split("<<<OBSERVATION\n", 1)[1].split("\nOBSERVATION>>>", 1)[0]
if not state.get("checklist"):
    out = {
        "reasoning": "plan", "state_patch": {"checklist": {"c1": {"text": "write the file", "status": "doing"}}},
        "action": "echo hello > out.txt",
    }
elif "Checks:" in obs and "pass" in obs:
    out = {"reasoning": "all good", "state_patch": {"checks_status": {"f": "pass"}}, "action": "done"}
elif state["checklist"]["c1"]["status"] == "doing":
    out = {
        "reasoning": "verify", "state_patch": {"checklist": {"c1": {"status": "done", "evidence": "exit 0"}}},
        "action": "check",
    }
else:
    out = {"reasoning": "stop", "state_patch": {}, "action": "stop confused"}
print("thinking...")
print(json.dumps(out))
"""


@pytest.mark.parametrize("spelling", ["explicit", "inferred"])
def test_cli_checklist_with_fake_agent_and_shell_env(tmp_path, spelling):
    agent = tmp_path / "agent.py"
    agent.write_text(FAKE_AGENT)
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    work = tmp_path / "work"
    work.mkdir()
    log = tmp_path / "log"
    cmd = [
        sys.executable,
        "-m",
        "stepbook.cli",
        "run",
        str(ROOT / "stepbook/templates" / "checklist"),
        "--workdir",
        str(work),
        "--goal",
        "create out.txt containing hello",
        "--check",
        "f=grep -q hello out.txt",
        "--steps",
        "6",
        "--command",
        f"{sys.executable} {agent}",
        "--log-dir",
        str(log),
        "--quiet",
        "--no-checkpoint",
        "--unsandboxed-i-accept-the-risk",
    ]
    # Shell is the environment without --adapter, and a --command is the agent to call; the older
    # spelling that says both must keep working.
    if spelling == "explicit":
        cmd += ["--executor", "command"]
    # The child runs from tmp_path, so name this checkout explicitly instead of whichever stepbook
    # the interpreter has installed.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr[-800:]
    summary = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert summary["score"] == 1.0 and summary["steps"] == 3 and summary["rejected_patches"] == 0
    assert (work / "out.txt").read_text().strip() == "hello"
    assert (log / "state.json").exists() and (log / "report.md").exists() and (log / "events.jsonl").exists()
    state = json.loads((log / "state.json").read_text())
    assert state["checklist"]["c1"]["status"] == "done" and state["goal"].startswith("create out.txt")


def test_adapter_cli_preserves_explicit_goal_and_human_constraints(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        'import sys\nprint("System: fixture; no hypotheses" if sys.argv[1] == "describe" else "reset")\n'
    )
    agent = tmp_path / "agent.py"
    agent.write_text(
        "import json,sys\nsys.stdin.read()\n"
        'print(json.dumps({"reasoning":"stop", "state_patch":{}, "action":"stop inspected"}))\n'
    )
    log = tmp_path / "log"
    command = [
        sys.executable,
        "-m",
        "stepbook.cli",
        "run",
        str(ROOT / "stepbook/templates/eval-loop-adapter"),
        "--adapter",
        f"{sys.executable} {adapter}",
        "--executor",
        "command",
        "--command",
        f"{sys.executable} {agent}",
        "--goal",
        "Improve extraction reliability",
        "--constraint",
        "Do not edit gold answers",
        "--constraint",
        "Use only development evidence",
        "--steps",
        "1",
        "--log-dir",
        str(log),
        "--quiet",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=30)
    assert proc.returncode == 0, proc.stderr
    state = json.loads((log / "state.json").read_text())
    assert state["goal"] == "Improve extraction reliability"
    assert [r["text"] for r in state["user_requirements"]] == [
        "Do not edit gold answers",
        "Use only development evidence",
    ]


def test_explicit_forty_steps_is_not_replaced_by_eval_count_heuristic(tmp_path, monkeypatch, capsys):
    from stepbook import Envelope, ScriptedExecutor
    from stepbook.cli import main

    adapter = tmp_path / "adapter.py"
    adapter.write_text('import sys\nprint("ready" if sys.argv[1] in ("reset", "describe") else "available")\n')
    calls = []

    def response(*_):
        calls.append(1)
        # Successful local evidence reads avoid loop-failure detection, with no paid eval.
        return Envelope("inspect", {}, 'read {"path":"evidence.txt"}')

    monkeypatch.setattr("stepbook.cli.run.ClaudeCLIExecutor", lambda **_: ScriptedExecutor(response))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evidence.txt").write_text("fact")
    assert (
        main(
            [
                "run",
                str(ROOT / "stepbook/templates/eval-loop-adapter"),
                "--adapter",
                f"{sys.executable} {adapter}",
                "--evidence-root",
                str(tmp_path),
                "--steps",
                "40",
                "--runs-max",
                "6",
                "--quiet",
                "--log-dir",
                str(tmp_path / "run"),
            ]
        )
        == 3
    )
    assert len(calls) == 40


def test_adapter_check_does_not_run_paid_eval_after_describe_error(tmp_path):
    from stepbook.cli import main

    adapter = tmp_path / "adapter.py"
    calls = tmp_path / "calls.txt"
    adapter.write_text(
        "import sys\nfrom pathlib import Path\n"
        f'with Path({str(calls)!r}).open("a") as f: f.write(sys.argv[1] + "\\n")\n'
        'print("Error: missing evaluation inputs")\nsys.exit(1)\n'  # exit 1: refused, not an answer
    )
    assert main(["adapter", "check", f"{sys.executable} {adapter}", "--run"]) == 1
    assert calls.read_text() == "describe\n"


@pytest.mark.parametrize(
    "options, sentence",
    [(["--model", "opus"], "--model is not supported by --executor command; put the model in the command string")],
)
def test_command_refuses_unsupported_options_before_environment(tmp_path, monkeypatch, capsys, options, sentence):
    from stepbook.cli import main

    built = []

    def environment(*args):
        built.append(args)
        return None, ()  # Stop the old implementation before any controller or helper can run.

    monkeypatch.setattr("stepbook.cli.run._build_run_environment", environment)
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    status = main(
        [
            "run",
            str(ROOT / "stepbook/templates/checklist"),
            "--executor",
            "command",
            "--command",
            "unused-offline-stub",
            "--log-dir",
            str(run),
            "--quiet",
            *options,
        ]
    )
    captured = capsys.readouterr()
    assert status == 2
    assert captured.err == sentence + "\n"
    assert captured.out == ""
    assert built == []
    assert not run.exists()
    assert not (tmp_path / ".stepbook").exists()


@pytest.mark.parametrize("flag", ["--max-total-usd", "--max-budget-usd", "--helper-budget-usd"])
@pytest.mark.parametrize("executor", ["command", "cli"])
def test_removed_dollar_flags_fail_before_environment(tmp_path, monkeypatch, capsys, flag, executor):
    from stepbook.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "stepbook.cli.run._build_run_environment",
        lambda *args: pytest.fail("removed flag reached environment construction"),
    )
    with pytest.raises(SystemExit) as error:
        main(["run", "unused", "--executor", executor, flag, "1"])
    assert error.value.code == 2
    assert f"unrecognized arguments: {flag} 1" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
