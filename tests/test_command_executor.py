import argparse
import json
import os
import shlex
import stat
import sys
from pathlib import Path

import pytest

from stepbook import CommandExecutor, Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.cli.inspection import cmd_report
from stepbook.contracts.errors import ExecutorError
from stepbook.executors.command import EXIT_TIMED_OUT
from stepbook.reporting.usage import envelope_usage, usage_summary
from stepbook.runtime.controls import check_control
from stepbook.runtime.loop import STATE_OUTPUT_SCHEMA
from stepbook.runtime.retry import call_with_retry
from stepbook.storage.events import EventLog
from tests._public_environment import FixtureEnv

FAKE_AGENT = """#!/usr/bin/env python3
import json, sys
prompt = sys.stdin.read()
print("progress: thinking...")
assert "<<SYSTEM>>" in prompt and "Latest Observation" in prompt
print(json.dumps({"reasoning": "fake", "state_patch": {"inventory": {"shelf_0": "item_0"}}, "action": "Store item_0 shelf_0",
                  "usage": {"input_tokens": 123, "output_tokens": 45}}))
"""


def test_command_executor_runs_any_cli(tmp_path):
    agent = tmp_path / "agent.py"
    agent.write_text(FAKE_AGENT)
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    ex = CommandExecutor(command=f"{sys.executable} {agent} --schema {{schema_file}}", name="fake")
    env = ex.step("SYS", "Skill Execution State:\n{}\nLatest Observation: x", STATE_OUTPUT_SCHEMA)
    assert env.action == "Store item_0 shelf_0" and env.state_patch == {"inventory": {"shelf_0": "item_0"}}
    assert env.input_tokens == 123 and env.output_tokens == 45


@pytest.mark.parametrize("path_type", [Path, str])
def test_command_executor_cwd_reaches_pwd(tmp_path, monkeypatch, path_type):
    repo = tmp_path / "work tree"
    repo.mkdir()
    agent = tmp_path / "pwd_agent.py"
    agent.write_text(
        "import json, subprocess\n"
        'print(json.dumps(dict(reasoning="pwd", state_patch={}, '
        'action=subprocess.check_output(["pwd"], text=True).strip())))\n'
    )
    executor = CommandExecutor(command=shlex.join([sys.executable, str(agent)]), cwd=path_type(repo))
    envelope = executor.step("SYS", "Latest Observation: ready", STATE_OUTPUT_SCHEMA)
    assert Path(envelope.action) == repo.resolve()


@pytest.mark.parametrize(
    "usage, expected",
    [
        (None, (None, True, False, False, None)),
        (
            {"input_tokens": 123, "output_tokens": 45},
            (None, False, False, True, None),
        ),
        (
            {"input_tokens": 123, "output_tokens": 45, "cost_usd": None},
            (None, False, False, True, None),
        ),
        (
            {"input_tokens": 123, "output_tokens": 45, "cost_usd": 40.0},
            (40.0, False, True, True, None),
        ),
        ({"input_tokens": 123, "output_tokens": 45, "cost_usd": 0.0}, (0.0, False, True, True, None)),
    ],
    ids=["no-usage", "tokens-no-price", "null-price", "priced", "reported-free"],
)
def test_wrapper_usage_reaches_journal_and_controls(tmp_path, monkeypatch, usage, expected):
    payload = {"reasoning": "offline", "state_patch": {}, "action": "Wait"}
    if usage is not None:
        payload["usage"] = usage
    response = tmp_path / "response.json"
    response.write_text(json.dumps(payload))
    agent = tmp_path / "wrapper.py"
    agent.write_text("import pathlib, sys\nprint(pathlib.Path(sys.argv[1]).read_text())\n")
    executor = CommandExecutor(command=shlex.join([sys.executable, str(agent), str(response)]))
    envelope = executor.step("SYS", "Latest Observation: ready", STATE_OUTPUT_SCHEMA)
    log = EventLog(tmp_path / "events.jsonl")
    log.append("model_call", step=0, **envelope_usage(envelope))
    receipt = log.read()[0]
    accounting = usage_summary(log.read())
    check_control(log)
    stop = next((e["reason"] for e in log.read() if e["kind"] == "budget_stop"), None)
    assert (
        receipt["cost_usd"],
        receipt["usage_unknown"],
        accounting["cost_complete"],
        accounting["tokens_complete"],
        stop,
    ) == expected


@pytest.mark.parametrize("runtime", [Runtime])
@pytest.mark.parametrize("unpriced", [True, False], ids=["unpriced", "priced"])
def test_run_result_cost_completeness_matches_journal_and_report(tmp_path, capsys, runtime, unpriced):
    skill = Skill("fixture", "Inspect the fixture.", {"type": "object"}, {})
    replies = iter(
        [
            Envelope("priced", {}, "Wait", cost_usd=1.23456),
            # The complete-cost case must explicitly report zero; omission is unknown.
            Envelope("next", {}, "Wait", cost_usd=None if unpriced else 0.0),
        ]
    )
    result = runtime(
        skill,
        ScriptedExecutor(lambda system, prompt: next(replies)),
        FixtureEnv(horizon=2, seed=1),
        log_dir=tmp_path,
    ).run(2)
    accounting = usage_summary(EventLog(tmp_path / "events.jsonl").read())
    summary = result.summary()
    assert (summary["cost_usd"], summary["cost_complete"], result.cost_complete) == (
        1.2346,
        accounting["cost_complete"],
        not unpriced,
    )
    cmd_report(argparse.Namespace(run_dir=str(tmp_path), last=0))
    cost_line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("Measured cost"))
    assert f"complete: cost={summary['cost_complete']}" in cost_line


def test_a_failed_command_call_is_accounted_as_unknown_usage_not_as_free(tmp_path):
    """CommandExecutor.step raises without an envelope on a nonzero
    exit, so the retry loop can only journal that the provider reported nothing, and the accounting says the
    cost is incomplete with the unknown-usage sentence rather than the new missing-price one."""
    agent = tmp_path / "exit3.py"
    agent.write_text("import sys\nsys.stdin.read()\nsys.stderr.write('boom\\n')\nsys.exit(3)\n")
    executor = CommandExecutor(command=shlex.join([sys.executable, str(agent)]), name="fake")
    log = EventLog(tmp_path / "events.jsonl")
    with pytest.raises(ExecutorError) as caught:
        call_with_retry(executor, "SYS", "Latest Observation: ready", STATE_OUTPUT_SCHEMA, log=log, attempts=1, step=0)
    assert "fake exited 3" in str(caught.value) and caught.value.__context__.envelope is None
    records = log.read()
    assert [e["kind"] for e in records] == ["failed_model_call", "executor_error"]
    assert (records[0]["cost_usd"], records[0]["input_tokens"], records[0]["usage_unknown"]) == (None, None, True)
    assert records[1]["usage_reported"] is False
    accounting = usage_summary(records)
    assert accounting["gaps"] == ["Failed provider requests may have unreported usage"]
    assert accounting["cost_complete"] is False and accounting["tokens_complete"] is False


@pytest.mark.parametrize(
    "code, error",
    [(EXIT_TIMED_OUT, ExecutorError), (3, None)],
    ids=["timed-out", "ordinary"],
)
def test_a_timed_out_turn_is_not_replayed_and_other_failures_retry(tmp_path, monkeypatch, code, error):
    """Any ``--command`` agent can report that its own turn timed out (exit 124): that call is not retried.
    Every other nonzero exit keeps today's retryable failure."""
    agent = tmp_path / "exits.py"
    agent.write_text(f"import sys\nsys.stdin.read()\nsys.stderr.write('agent said why\\n')\nsys.exit({code})\n")
    executor = CommandExecutor(command=shlex.join([sys.executable, str(agent)]), name="fake")
    slept, calls = [], []
    monkeypatch.setattr("stepbook.runtime.retry.time.sleep", slept.append)
    step = executor.step
    monkeypatch.setattr(executor, "step", lambda *a, **k: calls.append(a) or step(*a, **k))
    with pytest.raises(ExecutorError) as caught:
        call_with_retry(executor, "SYS", "Latest Observation: ready", STATE_OUTPUT_SCHEMA, attempts=3)
    if error is None:  # unchanged: retried on its ordinary schedule, then reported as spent
        assert type(caught.value) is ExecutorError and len(calls) == 3 and slept == [2, 4]
        assert str(caught.value) == "executor failed 3 times: fake exited 3: agent said why\n"
    else:
        assert type(caught.value) is error and caught.value.retryable is False
        assert str(caught.value) == f"fake exited {code}: agent said why\n"
        assert len(calls) == 1 and slept == []


def test_the_timeout_exit_code_is_documented_for_a_custom_command():
    """A ``--command`` author reads the contract from the executor's own docstring."""
    doc = CommandExecutor.__doc__
    assert f"``{EXIT_TIMED_OUT}``" in doc and EXIT_TIMED_OUT == 124


FATAL_CLAUDE = "no Claude binary lookup or auth status for a caller-configured command"


def _no_claude(monkeypatch):
    """Make every Claude lookup/status function fatal, so any use of them fails the test loudly."""
    from stepbook.executors import auth
    from stepbook.executors import claude as executor

    def fatal(*_args, **_kwargs):
        raise AssertionError(FATAL_CLAUDE)

    for name in ("cli_path", "check_subscription", "check_environment"):
        monkeypatch.setattr(auth, name, fatal)
    monkeypatch.setattr(auth.shutil, "which", fatal)
    monkeypatch.setattr(executor, "check_subscription", fatal)


def test_guarded_run_executes_a_custom_command_without_any_claude_check(tmp_path, monkeypatch):
    """The subscription guard covers the harness's own Claude calls, not another provider's CLI."""
    monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
    _no_claude(monkeypatch)
    agent = tmp_path / "agent.py"
    agent.write_text(
        "import json, sys\n"
        "schema = json.load(open(sys.argv[2]))\n"
        "system = open(sys.argv[4]).read()\n"
        "sys.stdin.read()\n"
        'print(json.dumps({"reasoning": "offline", "state_patch": {}, '
        '"action": f"Wait {sorted(schema)[0]} {system}"}))\n'
    )
    executor = CommandExecutor(
        command=f"{sys.executable} {agent} --schema {{schema_file}} --system {{system_prompt_file}}",
        name="fake",
    )
    envelope = executor.step("SYS", "Latest Observation: ready", STATE_OUTPUT_SCHEMA)
    assert envelope.action == "Wait additionalProperties SYS"  # placeholders substituted, prompt files written and read
    assert envelope.raw["usage_unknown"] is True


def test_guarded_preflight_attests_the_claude_cli_only(monkeypatch):
    """The product CLI no longer refuses --executor command outright; it just has nothing to attest."""
    from stepbook.executors import auth

    monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
    _no_claude(monkeypatch)
    assert auth.preflight_command(argparse.Namespace(cmd="run", executor="command")) is None
    assert auth.preflight_command(argparse.Namespace(cmd="adapter", executor="cli")) is None
    checked = []
    monkeypatch.setattr(auth, "check_subscription", lambda **kw: checked.append(kw))
    assert auth.preflight_command(argparse.Namespace(cmd="run", executor="cli")) is None
    assert checked == [{}]


def test_the_api_billing_opt_out_skips_the_preflight_and_rides_into_children(monkeypatch):
    """Both spellings of the opt-out turn the guard off; the flag exports it for adapters and drivers."""
    from stepbook.executors import auth

    monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
    _no_claude(monkeypatch)  # any attempt to attest a login fails this test loudly
    assert auth.subscription_required() is True
    assert auth.preflight_command(argparse.Namespace(cmd="run", executor="cli", allow_api_billing=True)) is None
    assert os.environ["STEPBOOK_ALLOW_API_BILLING"] == "1"  # the children of this process inherit the choice
    assert auth.subscription_required() is False
    monkeypatch.setenv("STEPBOOK_ALLOW_API_BILLING", "1")  # the environment variable alone is enough
    assert auth.preflight_command(argparse.Namespace(cmd="run", executor="cli")) is None
