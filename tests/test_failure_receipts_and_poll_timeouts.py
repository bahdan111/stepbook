import subprocess
import time
from types import SimpleNamespace

import pytest

from stepbook.contracts.errors import ExecutorError
from stepbook.envs.shell import CommandRun, ShellEnv
from stepbook.executors.claude import ClaudeCLIExecutor, _cli_failure_message
from stepbook.reporting.usage import usage_summary
from stepbook.runtime.retry import call_with_retry
from stepbook.storage.events import EventLog


@pytest.mark.parametrize("tokens", [(7, 3), (0, 0)])
def test_unpriced_failed_call_keeps_complete_tokens(tokens):
    reported = {
        "kind": "failed_model_call",
        "step": 0,
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "cost_usd": None,
        "usage_unknown": False,
    }
    summary = usage_summary([reported])
    assert summary["tokens_complete"] is True
    assert summary["cost_complete"] is False
    assert summary["gaps"] == ["Failed provider requests reported usage but no price"]


@pytest.mark.parametrize("tokens", [{}, {"input_tokens": 7}, {"output_tokens": 3}])
def test_price_gap_requires_both_token_fields(tokens):
    missing_tokens = {"kind": "failed_model_call", "step": 0, "cost_usd": None, **tokens}
    summary = usage_summary([missing_tokens])
    assert "Failed provider requests reported usage but no price" not in summary["gaps"]
    assert summary["tokens_complete"] is False


def _failure_record(tmp_path, monkeypatch, stdout, stderr):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr),
    )
    log = EventLog(tmp_path / "events.jsonl")
    with pytest.raises(ExecutorError):
        call_with_retry(ClaudeCLIExecutor(), "system", "user", {}, log=log, step=4, attempts=1)
    return next(event for event in log.read() if event["kind"] == "executor_error")


def test_cli_failure_journal_keeps_both_stream_tails_under_the_first_printed_line(tmp_path, monkeypatch):
    """The headline is the first non-empty line the CLI printed; no line is skipped for how it reads."""
    warning = "Ignoring 2 permissions.allow entries from .claude/settings.json"
    stdout = "startup detail\nstdout diagnostic"
    stderr = f"\n{warning}\nCan't reach the API server (ENOTFOUND)"
    record = _failure_record(tmp_path, monkeypatch, stdout, stderr)
    assert record["error"].startswith(f"claude exited 1: {warning}")
    assert "stdout diagnostic" in record["error"]
    assert "Can't reach the API server (ENOTFOUND)" in record["error"]
    assert len(record["error"]) <= 500


@pytest.mark.parametrize("prefix", ["x\n" * 30, "x" * 3000 + "\n"], ids=["many-lines", "long-line"])
def test_cli_failure_journal_bounds_both_tails(tmp_path, monkeypatch, prefix):
    record = _failure_record(
        tmp_path, monkeypatch, "OLD-STDOUT\n" + prefix + "stdout end", "OLD-STDERR\n" + prefix + "stderr end"
    )
    headline, stdout_tail = record["error"].split("\n[stdout tail]\n")
    stdout_tail, stderr_tail = stdout_tail.split("\n[stderr tail]\n")
    assert headline.startswith("claude exited 1:")
    for tail, ending in [(stdout_tail, "stdout end"), (stderr_tail, "stderr end")]:
        assert tail.endswith(ending)
        assert "OLD-" not in tail
        assert len(tail.splitlines()) <= 20
        assert len(tail) <= 180
    assert len(record["error"]) <= 500


def test_cli_failure_headline_comes_from_stdout_when_stderr_printed_nothing(tmp_path, monkeypatch):
    record = _failure_record(tmp_path, monkeypatch, "connection failed\nmore detail", "")
    assert record["error"].splitlines()[0] == "claude exited 1: connection failed"


@pytest.mark.parametrize(
    "stderr",
    [
        "Error: Ignoring 2 permissions.allow entries failed: malformed configuration",
        "Error: Ignoring 2 permissions.allow entries from .claude/settings.json",
        "Ignoring 2 permissions.allow entries from .claude/settings.json failed",
    ],
)
def test_cli_failure_keeps_error_quoting_trust_warning(tmp_path, monkeypatch, stderr):
    proc = subprocess.CompletedProcess([], 1, "startup detail", stderr)
    expected = f"claude exited 1: {stderr}"
    assert _cli_failure_message(proc).splitlines()[0] == expected
    record = _failure_record(tmp_path, monkeypatch, proc.stdout, proc.stderr)
    assert record["error"].splitlines()[0] == expected


def test_cli_long_headline_cannot_truncate_either_journal_tail(tmp_path, monkeypatch):
    record = _failure_record(tmp_path, monkeypatch, "x" * 3000 + "\nstdout end", "y" * 3000 + "\nstderr end")
    assert "stdout end" in record["error"]
    assert record["error"].endswith("stderr end")
    assert len(record["error"]) <= 500


def test_wait_exit_124_without_a_timeout_is_not_labeled_a_poll_timeout(tmp_path):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={})
    env.reset()
    result = env.step("wait 0.2s until exit 124 every 1s")
    assert result.info["rc"] == 124
    assert "poll 1 timed out" not in result.observation


def test_wait_bounds_a_hanging_poll_and_reports_that_poll_timeout(tmp_path, monkeypatch):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={}, cmd_timeout_s=30)
    env.reset()
    clock = [100.0]
    poll_limits = []

    monkeypatch.setattr("stepbook.envs.shell.time.monotonic", lambda: clock[0])

    def hanging_poll(command, *, reap_group=False, timeout_s=None):
        poll_limits.append(timeout_s)
        elapsed = env.cmd_timeout_s if timeout_s is None else timeout_s
        clock[0] += elapsed
        return CommandRun(124, f"(timed out after {elapsed:.1f}s; the command's process group was killed)", True)

    monkeypatch.setattr(env, "_run", hanging_poll)
    result = env.step("wait 0.2s until hanging-command every 0.1s")

    assert poll_limits == pytest.approx([0.2])
    assert clock[0] <= 101.2  # deadline plus the parser's one-second minimum interval
    assert result.info["wait"] == "timed out"
    assert "poll 1 timed out after 0.2s" in result.observation


def test_wait_limits_each_poll_by_interval_and_remaining_time(tmp_path, monkeypatch):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={}, cmd_timeout_s=30)
    env.reset()
    clock = [100.0]
    limits = []
    monkeypatch.setattr("stepbook.envs.shell.time.monotonic", lambda: clock[0])

    def sleep(seconds):
        clock[0] += seconds

    def poll(command, *, reap_group=False, timeout_s=None):
        limits.append(timeout_s)
        seconds = env.cmd_timeout_s if timeout_s is None else timeout_s
        clock[0] += seconds
        return CommandRun(124, f"(timed out after {seconds:.1f}s; the command's process group was killed)", True)

    monkeypatch.setattr("stepbook.envs.shell.time.sleep", sleep)
    monkeypatch.setattr(env, "_run", poll)
    result = env.step("wait 2.5s until hanging-command every 1s")
    assert limits == pytest.approx([1.0, 0.5])
    assert clock[0] <= 103.5
    assert "poll 2 timed out after 0.5s" in result.observation
    assert env.commands == [{"t": 1, "cmd": "wait 2.5s until hanging-command every 1s", "rc": 124}]


def test_wait_real_hanging_process_respects_deadline(tmp_path):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={}, cmd_timeout_s=2)
    env.reset()
    started = time.monotonic()
    result = env.step("wait 0.2s until sleep 10 every 1s")
    assert time.monotonic() - started <= 1.2
    assert result.info["wait"] == "timed out"
    assert result.info["rc"] == 124
    assert "poll 1 timed out" in result.observation
