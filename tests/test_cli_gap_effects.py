"""CLI dispatch/effect locks using local executable stand-ins, never a provider."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_cli(tmp_path):
    work, binary = tmp_path / "work", tmp_path / "bin"
    work.mkdir()
    binary.mkdir()
    # No real provider executable can be resolved through this PATH. Git is
    # exposed solely for the disposable repository's checkpoint tests.
    (binary / "git").symlink_to(shutil.which("git"))
    (binary / "claude_agent_sdk.py").write_text("raise RuntimeError('provider SDK forbidden in offline test')\n")
    script = binary / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
root = Path(os.environ["GAP_ROOT"])
responses = json.loads((root / "responses.json").read_text())
counter = root / "calls.txt"
n = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(n + 1))
prompt = sys.argv[-1] if "-p" in sys.argv else sys.stdin.read()
(root / f"prompt-{n}.txt").write_text(prompt)
response = responses[min(n, len(responses) - 1)]
if "-p" in sys.argv:
    print(json.dumps({"result": json.dumps(response), "total_cost_usd": 0, "num_turns": 1}))
else:
    print(json.dumps(response))
"""
    )
    script.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(binary),
        "PYTHONPATH": str(binary) + os.pathsep + str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "GAP_ROOT": str(tmp_path),
        "STEPBOOK_ALLOW_API_BILLING": "1",
    }

    def run(responses, *options, executor="command", expected_exit=0):
        (tmp_path / "responses.json").write_text(json.dumps(responses))
        args = [
            sys.executable,
            "-m",
            "stepbook.cli",
            "run",
            str(ROOT / "stepbook/templates/checklist"),
            "--workdir",
            str(work),
            "--goal",
            "fixture",
            "--log-dir",
            str(tmp_path / "log"),
            "--quiet",
            "--unsandboxed-i-accept-the-risk",
        ]
        if executor != "auto":
            args += ["--executor", executor]
        if executor == "command":
            args += ["--command", shlex.join([sys.executable, str(script)])]
        proc = subprocess.run(
            args + list(options), cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15
        )
        assert proc.returncode == expected_exit, proc.stderr
        events = [json.loads(line) for line in (tmp_path / "log/events.jsonl").read_text().splitlines()]
        return json.loads(proc.stdout), events

    return run, work


def response(action, patch=None):
    return {"reasoning": "fixture", "state_patch": patch or {}, "action": action}


@pytest.mark.parametrize("executor,expected", [("auto", "cli"), ("cli", "cli"), ("command", "command")])
def test_executor_selection_and_unbounded_steps_default(local_cli, tmp_path, executor, expected):
    """Bug: CLI selects the wrong executor or introduces a default shell step horizon."""
    run, _ = local_cli
    summary, events = run([response("printf marker"), response("done")], executor=executor)
    assert summary["executor"] == expected and summary["steps"] == 2
    assert next(e for e in events if e["kind"] == "run_start")["max_steps"] is None
    assert (tmp_path / "calls.txt").read_text() == "2"
    assert [e["action"] for e in events if e["kind"] == "step"] == ["printf marker", "done"]


def test_explicit_steps_bound_dispatch(local_cli, tmp_path):
    """Bug: --steps stops limiting actual controller calls and shell actions."""
    run, work = local_cli
    summary, events = run([response("printf x >> marks.txt")], "--steps", "2", expected_exit=3)
    assert summary["steps"] == 2 and (tmp_path / "calls.txt").read_text() == "2"
    assert (work / "marks.txt").read_text() == "xx"
    assert [e for e in events if e["kind"] == "step"][-1]["horizon_stop"] is True


@pytest.mark.parametrize("disabled", [False, True])
def test_checkpoint_default_has_git_effect(local_cli, disabled):
    """Bug: CLI reverses the default checkpoint policy or ignores --no-checkpoint."""
    run, work = local_cli
    subprocess.run(["git", "init", "-q", "-b", "operator"], cwd=work, check=True, capture_output=True, timeout=1)
    run(
        [response("printf saved > saved.txt")],
        "--steps",
        "1",
        *(["--no-checkpoint"] if disabled else []),
        expected_exit=3,
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=work, check=True, capture_output=True, text=True, timeout=1
    ).stdout.strip()
    if disabled:
        assert branch == "operator"
    else:
        assert branch.startswith("stepbook/")
        saved = subprocess.run(
            ["git", "show", "HEAD:saved.txt"], cwd=work, check=True, capture_output=True, text=True, timeout=1
        )
        assert saved.stdout == "saved"


def test_exhausted_default_aborts_without_dispatch(local_cli):
    """Exhausted product validation aborts without a synthetic or rejected action."""
    run, work = local_cli
    summary, events = run(
        [response("printf forbidden > forbidden.txt", {"goal": "unauthorized"})], "--steps", "1", expected_exit=3
    )
    assert summary["steps"] == 0 and summary["rejected_patches"] == 4 and summary["aborted"]
    assert any(e["kind"] == "abort" and e["reason"] == "max retries exceeded" for e in events)
    assert not any(e["kind"] in {"step", "skip", "action_intent"} for e in events)
    assert not (work / "forbidden.txt").exists()


def test_cmd_timeout_reaches_shell_process(local_cli, tmp_path):
    """Bug: --cmd-timeout is parsed but never bounds the shell process."""
    run, work = local_cli
    (work / "block.py").write_text("import threading\nthreading.Event().wait()\n")
    command = shlex.join([sys.executable, "block.py"])
    _, events = run([response(command)], "--steps", "1", "--cmd-timeout", "0.02", "--allow-outside", expected_exit=3)
    step = next(e for e in events if e["kind"] == "step")
    assert step["info"] == {"ok": False, "rc": 124}
    assert step["result"] == f"$ {command}\n[exit 124]\n(timed out after 0s; the command's process group was killed)"
    assert (tmp_path / "calls.txt").read_text() == "1"


@pytest.mark.parametrize("cap", [None, 2000])
def test_state_max_chars_changes_prompt_but_not_saved_state(local_cli, tmp_path, cap):
    """Bug: --state-max-chars fails to cap rendering or truncates durable state instead."""
    run, _ = local_cli
    facts = {f"f{i:03}": "evidence " + "x" * 180 for i in range(100)}
    run(
        [response("printf saved", {"facts": facts}), response("done")],
        *([] if cap is None else ["--state-max-chars", str(cap)]),
    )
    prompt = (tmp_path / "prompt-1.txt").read_text()
    assert ("abbreviated for size" in prompt) is (cap is not None)
    assert json.loads((tmp_path / "log/state.json").read_text())["facts"] == facts


@pytest.mark.parametrize(
    "flag,value,diagnostic",
    [
        ("--steps", "0", "must be positive"),
        ("--steps", "-1", "must be nonnegative"),
        ("--runs-max", "-1", "must be nonnegative"),
        ("--max-turns", "0", "unrecognized arguments: --max-turns 0"),
        ("--max-tool-calls", "0", "unrecognized arguments: --max-tool-calls 0"),
        ("--max-repairs", "-1", "unrecognized arguments: --max-repairs -1"),
        ("--repeat-warning", "0", "unrecognized arguments: --repeat-warning 0"),
        ("--controller-timeout", "nan", "must be finite and positive"),
        ("--adapter-timeout", "inf", "must be finite and positive"),
        ("--max-source-bytes", "0", "unrecognized arguments: --max-source-bytes 0"),
        ("--measurement-repeats", "0", "unrecognized arguments: --measurement-repeats 0"),
        ("--helper-budget-usd", "0", "unrecognized arguments: --helper-budget-usd 0"),
    ],
)
def test_numeric_diagnostic_names_option(flag, value, diagnostic, capsys):
    """Bug: invalid numeric CLI input loses its option-specific diagnostic or reaches dispatch."""
    with pytest.raises(SystemExit) as error:
        main(["run", "unused", flag, value])
    assert error.value.code == 2
    output = capsys.readouterr()
    if diagnostic.startswith("unrecognized arguments:"):
        assert output.err.endswith(f"stepbook: error: {diagnostic}\n")
    else:
        assert output.err.endswith(f"stepbook run: error: argument {flag}: {diagnostic}\n")
    assert output.out == ""
