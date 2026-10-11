"""Parse synthetic stored commands without executing runs."""

import argparse
import shlex
from pathlib import Path

import pytest

from stepbook import cli

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "cli" / "resume_commands.txt"
RESUME_LINES = [line for line in FIXTURE.read_text().splitlines() if line and not line.startswith("#")]
# This independent option table describes the stored commands, not argparse internals.
VALUE_OPTIONS = {
    "--steps": (int, None),
    "--state-max-chars": (int, None),
    "--cmd-timeout": (float, 7200.0),
    "--model": (str, "opus"),
    "--log-dir": (str, None),
    "--workdir": (str, None),
    "--goal": (str, None),
    "--executor": (str, "cli"),
    "--command": (str, None),
}
BOOL_OPTIONS = {"--resume", "--allow-outside", "--unsandboxed-i-accept-the-risk", "--allow-dirty"}


@pytest.fixture
def parser(monkeypatch):
    # main() builds argparse inline; intercept its public parse_args seam before
    # auth preflight or command dispatch, then restore it for real parsing below.
    class ParserReady(Exception):
        pass

    def capture(parser, *args, **kwargs):
        raise ParserReady(parser)

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(ParserReady) as ready:
            cli.main([])
    return ready.value.args[0]


def _stored_values(argv):
    expected = {flag[2:].replace("-", "_"): default for flag, (_, default) in VALUE_OPTIONS.items()}
    expected.update({flag[2:].replace("-", "_"): False for flag in BOOL_OPTIONS})
    tokens = iter(argv[2:])
    for flag in tokens:
        name = flag[2:].replace("-", "_")
        if flag in BOOL_OPTIONS:
            expected[name] = True
        else:
            assert flag in VALUE_OPTIONS, f"fixture option needs a value assertion: {flag}"
            expected[name] = VALUE_OPTIONS[flag][0](next(tokens))
    return expected


def test_stored_command_fixture_integrity():
    assert len(RESUME_LINES) == 3
    for line in RESUME_LINES:
        program, *argv = shlex.split(line)
        assert program.endswith("/.venv/bin/stepbook")
        assert argv[0] == "run"
        _stored_values(argv)  # No supplied option may escape the value lock.
    for line in RESUME_LINES[:-1]:
        values = _stored_values(shlex.split(line)[1:])
        assert values["resume"] and values["goal"] and values["log_dir"]
    assert "INCOMPLETE SYNTHETIC COMMAND" in FIXTURE.read_text()
    assert shlex.split(RESUME_LINES[-1])[1:] == [
        "run",
        "checklist",
        "--workdir",
        "/tmp/example/incomplete",
    ]


@pytest.mark.parametrize("line", RESUME_LINES, ids=["cli-resume", "command-resume", "incomplete-synthetic"])
def test_stored_command_parsed_values(parser, line):
    _, *argv = shlex.split(line)
    args = parser.parse_args(argv)
    assert args.cmd == "run"
    assert args.skill_dir == argv[1]
    for name, expected in _stored_values(argv).items():
        assert getattr(args, name) == expected, f"stored option {name}"
        assert type(getattr(args, name)) is type(expected), f"stored option {name} type"


def test_incomplete_synthetic_command_parses_but_dispatch_requires_goal(parser, monkeypatch, capsys):
    args = parser.parse_args(shlex.split(RESUME_LINES[-1])[1:])
    assert args.goal is None
    assert args.resume is False, "this fragment is NOT a usable resume command"

    def refuse_environment(*args, **kwargs):
        pytest.fail("incomplete synthetic command must be refused before constructing ShellEnv")

    monkeypatch.setattr("stepbook.cli.run.ShellEnv", refuse_environment)
    assert args.fn(args) == 2
    assert capsys.readouterr().err == "shell mode needs --workdir and --goal\n"


def test_shell_dispatch_requires_a_goal_when_neither_flag_nor_skill_has_one(parser, capsys, tmp_path):
    # No --env: shell is the environment without --adapter. checklist's initial state has an empty goal.
    args = parser.parse_args(["run", str(ROOT / "stepbook/templates/checklist"), "--workdir", str(tmp_path)])
    assert args.fn(args) == 2
    assert capsys.readouterr().err == "shell mode needs --workdir and --goal\n"
    assert not list(tmp_path.iterdir())


def test_run_defaults(parser):
    args = parser.parse_args(["run", "skills/checklist"])
    expected = {
        "model": "opus",
        "executor": "cli",
        "cmd_timeout": 7200.0,
        "state_max_chars": None,
    }
    assert {name: getattr(args, name) for name in expected} == expected
