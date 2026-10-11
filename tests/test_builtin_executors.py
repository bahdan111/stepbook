"""Offline built-in driver selection, authoring, launch recording and resume."""

import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from stepbook.cli import _build_parser, main
from stepbook.cli.run import _preflight_executor, _select_run_executor
from stepbook.cli.run_args import executor_command, executor_option_problem
from stepbook.executors import codex, opencode
from stepbook.executors.base import USAGE_KEYS, ParsedStep
from stepbook.executors.command import CommandExecutor
from tests.test_author_cli_contract import draft_executor as draft_executor
from tests.test_codex_step import fake_codex as fake_codex
from tests.test_codex_step import run_step as codex_step
from tests.test_codex_step import runtime_schema as runtime_schema
from tests.test_opencode_step import fake_opencode as fake_opencode
from tests.test_opencode_step import fixture_events, text_event
from tests.test_opencode_step import run_step as opencode_step

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("verb", ["run", "author"])
@pytest.mark.parametrize("kind", ["codex", "opencode"])
@pytest.mark.parametrize("model_flags", [[], ["--model", "opus"], ["--model=provider/model with {braces}"]])
def test_builtin_arguments_and_exact_command(verb, kind, model_flags):
    argv = [verb, "skill", "--executor", kind, *model_flags]
    args = _build_parser().parse_args(argv)
    args.raw_argv = argv
    assert args.executor == kind and executor_option_problem(args) is None
    expected = [sys.executable, "-m", f"stepbook.executors.{kind}", "step"]
    if model_flags:
        expected += ["--model", args.model]
    if verb == "run":  # author has no controller timeout to pass on
        expected += ["--timeout", f"{args.controller_timeout:g}"]
    expected += ["--schema", "schema.json"]
    assert shlex.split(executor_command(args).format(schema_file="schema.json")) == expected


@pytest.mark.parametrize("kind", ["codex", "opencode"])
def test_builtin_selected_command_reaches_subprocess(tmp_path, monkeypatch, kind):
    argv = ["run", "skill", "--executor", kind, "--model", "fixture", "--workdir", str(tmp_path)]
    args = _build_parser().parse_args(argv)
    args.raw_argv = argv
    executor, selected = _select_run_executor(args, None, ())
    assert type(executor) is CommandExecutor and selected == kind
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert json.loads(Path(cmd[-1]).read_text()) == {"type": "object"}
        return subprocess.CompletedProcess(cmd, 0, '{"action": "done"}', "")

    monkeypatch.setattr(subprocess, "run", run)
    assert executor.step("system", "observation", {"type": "object"}).action == "done"
    cmd, kwargs = calls[0]
    assert cmd[:-1] == [
        sys.executable,
        "-m",
        f"stepbook.executors.{kind}",
        "step",
        "--model",
        "fixture",
        "--timeout",
        f"{args.controller_timeout:g}",
        "--schema",
    ]
    assert kwargs["cwd"] == tmp_path.resolve() and kwargs["timeout"] == args.controller_timeout
    assert "<<SYSTEM>>\nsystem\n<</SYSTEM>>" in kwargs["input"]


@pytest.mark.parametrize("kind", ["codex", "opencode"])
def test_controller_timeout_bounds_the_builtin_driver_turn(kind):
    """The host's per-query allowance reaches the driver, whose own 900s default would cut a longer step short."""
    argv = ["run", "skill", "--executor", kind, "--controller-timeout", "5400"]
    args = _build_parser().parse_args(argv)
    args.raw_argv = argv
    command = shlex.split(executor_command(args).format(schema_file="schema.json"))
    assert command[command.index("--timeout") + 1] == "5400"
    helped = subprocess.run(
        [sys.executable, "-m", f"stepbook.executors.{kind}", "step", "--help"], capture_output=True, text=True
    )
    assert helped.returncode == 0 and "--timeout" in helped.stdout


@pytest.mark.parametrize("kind", ["codex", "opencode"])
@pytest.mark.parametrize("from_source", [False, True])
def test_author_builtin_driver(tmp_path, draft_executor, kind, from_source):
    draft, calls = draft_executor
    options = ["--executor", kind, "--model", "fixture"]
    if from_source:
        source = tmp_path / "source.txt"
        source.write_text("Review this procedure.\n")
        del draft["skill_md"]
        draft["run_section"] = "Run one check per step."
        options += ["--from", str(source)]
    target = tmp_path / "draft"
    assert main(["author", str(target), "review", *options]) == 0
    assert main(["validate", str(target)]) == 0
    assert len(calls) == 1
    # author carries no controller timeout, so the driver keeps its own default turn limit.
    assert calls[0][0][:-1] == [
        sys.executable,
        "-m",
        f"stepbook.executors.{kind}",
        "step",
        "--model",
        "fixture",
        "--schema",
    ]


def test_codex_launch_and_recorded_resume(tmp_path, monkeypatch, fake_codex, capsys):
    fake, log = fake_codex
    monkeypatch.setenv("CODEX_BIN", fake)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work, check=True, capture_output=True)
    run = tmp_path / "run"
    argv = [
        "run",
        str(ROOT / "stepbook/templates/checklist"),
        "--executor",
        "codex",
        "--model",
        "fixture",
        "--workdir",
        str(work),
        "--goal",
        "test resume",
        "--steps",
        "1",
        "--log-dir",
        str(run),
        "--no-checkpoint",
        "--unsandboxed-i-accept-the-risk",
        "--quiet",
    ]
    assert main(argv) == 3
    assert json.loads(capsys.readouterr().out)["executor"] == "codex"
    launch = json.loads((run / "launch.json").read_text())
    assert launch["argv"] == argv
    resume = shlex.split(launch["resume_command"])
    resume = resume[resume.index("run") :]
    assert resume[resume.index("--executor") + 1] == "codex" and "--command" not in resume
    resume[resume.index("--steps") + 1] = "2"
    assert main(resume) == 3
    assert json.loads(capsys.readouterr().out)["executor"] == "codex"
    launch = json.loads((run / "launch.json").read_text())
    assert launch["argv"] == argv and launch["resumes"][-1]["argv"] == resume
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "step"]) == 2
    called = json.loads(log.read_text())["argv"]
    assert called[called.index("-m") + 1] == "fixture"


def test_both_parsers_share_model_and_categories():
    first, second = codex.parse_events(""), opencode.parse_events("")
    assert type(first) is type(second) is ParsedStep
    assert asdict(first).keys() == asdict(second).keys()
    assert tuple(first.usage) == tuple(second.usage) == USAGE_KEYS
    first.usage["input_tokens"] = 1
    first.errors.append("first only")
    assert second.usage["input_tokens"] == 0 and second.errors == []


def test_wrappers_keep_common_keys_and_provider_specific_keys(
    tmp_path,
    monkeypatch,
    fake_codex,
    fake_opencode,
    runtime_schema,
):
    fixture_events(tmp_path, monkeypatch, [text_event({"reasoning": "ok", "state_patch": {}, "action": "done"})])
    first = codex_step(tmp_path, fake_codex[0], runtime_schema)
    second = opencode_step(tmp_path, runtime_schema)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    a, b = json.loads(first.stdout), json.loads(second.stdout)
    assert a.keys() - {"thread_id", "codex_errors"} == b.keys() - {"session_id", "opencode_errors"}
    assert a["usage"].keys() == b["usage"].keys() - {"opencode_cost"}
    assert list(a["usage"])[:5] == list(b["usage"])[:5] == list(USAGE_KEYS)


def test_codex_wrapper_preserves_json_bytes(tmp_path, fake_codex, runtime_schema):
    result = codex_step(tmp_path, fake_codex[0], runtime_schema)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        '{"reasoning": "fake codex", "state_patch": {"current": "step done", '
        '"checklist": {"verify": {"text": "Check output", "status": "done"}}}, "action": "echo hi", '
        '"input_tokens": 1000, "output_tokens": 50, "usage": {"input_tokens": 1000, '
        '"cached_input_tokens": 800, "cache_write_input_tokens": 30, "output_tokens": 50, '
        '"reasoning_output_tokens": 20, "provider": "openai", "model": null, "cost_usd": null, '
        '"note": "ChatGPT-login usage: plan limits/credits, no USD list price applied"}, '
        '"api_calls": 1, "tool_calls": 1, "tool_log": [{"type": "command_execution", "input": "echo hi"}], '
        '"thread_id": "thread-step", "codex_errors": []}\n'
    )


def test_opencode_wrapper_preserves_json_bytes(tmp_path, fake_opencode, runtime_schema):
    result = opencode_step(tmp_path, runtime_schema)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        '{"ok": true, "n": 3, "input_tokens": 5919, "output_tokens": 15, '
        '"usage": {"input_tokens": 2335, "cached_input_tokens": 3584, "cache_write_input_tokens": 0, '
        '"output_tokens": 15, "reasoning_output_tokens": 0, "provider": null, "model": null, '
        '"cost_usd": null, "opencode_cost": 0.0, '
        '"note": "Subscription logins report no USD cost; OpenCode\'s reported cost is retained without USD pricing."}, '
        '"api_calls": 1, "tool_calls": 0, "tool_log": [], '
        '"session_id": "ses_000000000000Example0000018", "opencode_errors": []}\n'
    )


@pytest.mark.parametrize("verb", ["run", "author"])
@pytest.mark.parametrize("kind", ["codex", "opencode"])
@pytest.mark.parametrize("command", ["ignored-agent", ""])
def test_builtin_rejects_command_before_creating_output(tmp_path, capsys, verb, kind, command):
    target = tmp_path / "output"
    argv = [verb, str(target), "--executor", kind, "--command", command]
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"--command is not supported by --executor {kind}; use --executor command\n"
    assert not target.exists()


@pytest.mark.parametrize("kind", ["codex", "opencode"])
@pytest.mark.parametrize("override", [False, True])
def test_builtin_missing_binary_refuses_before_run_directory(tmp_path, monkeypatch, capsys, kind, override):
    from tests.test_launch_recovery import launch_args
    from tests.test_shell_env_resume import _git_repo

    work = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    binary = "missing-agent" if override else kind
    if override:
        monkeypatch.setenv(f"{kind.upper()}_BIN", binary)
    else:
        monkeypatch.delenv(f"{kind.upper()}_BIN", raising=False)
    checked = []

    def which(candidate, **kwargs):
        checked.append(candidate)
        return sys.executable if candidate == sys.executable else None

    monkeypatch.setattr("stepbook.cli.run.shutil.which", which)
    argv = launch_args(tmp_path, work)
    index = argv.index("--executor")
    argv[index + 1] = kind
    del argv[index + 2 : index + 4]
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"executor {binary!r} was not found; install it and make it available on PATH\n"
    assert checked == [binary]
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / ".stepbook").exists()


@pytest.mark.parametrize("kind", ["codex", "opencode"])
@pytest.mark.parametrize("via_path", [False, True])
def test_builtin_preflight_resolves_override_in_executor_directory(tmp_path, monkeypatch, kind, via_path):
    work = tmp_path / "work"
    binary_dir = work / "bin"
    binary_dir.mkdir(parents=True)
    binary = binary_dir / "fake agent"
    binary.write_text("unused: preflight must not execute this file")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", "bin")
    configured = binary.name if via_path else "bin/" + binary.name
    if kind == "codex":
        configured = shlex.join([configured, "--fixture-flag"])
    monkeypatch.setenv(f"{kind.upper()}_BIN", configured)
    _preflight_executor(CommandExecutor("unused", cwd=work), kind)
