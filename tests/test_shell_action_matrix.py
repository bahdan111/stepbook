"""Exact public ShellEnv action vocabulary and refusal observations."""

import os
import shlex
from unittest.mock import patch

import pytest

from stepbook.envs.shell import CommandRun, ShellEnv


@pytest.fixture
def env(tmp_path):
    result = ShellEnv(tmp_path, "g", checkpoint=False, require_sandbox=False)
    result.reset()
    return result


@pytest.mark.parametrize("action", ["done", " DONE "])
def test_done_without_checks_is_terminal_success(env, action):
    """Bug: done ceases to finish successfully when the operator supplied no checks."""
    result = env.step(action)
    assert result.observation == "Checks:\nNo checks configured."
    assert result.done and result.info == {"ok": True, "checks": {}, "terminal": True}
    assert env.score() == 1.0


@pytest.mark.parametrize("action,reason", [("stop", ""), ("STOP maintenance", "maintenance"), ("stop now", "now")])
def test_stop_word_preserves_reason_and_incomplete_status(env, action, reason):
    """Bug: stop dispatch or its agent-supplied reason changes."""
    result = env.step(action)
    assert result.observation == "Stopped by the agent: " + reason
    assert result.done and result.info == {"ok": False, "checks": {}, "terminal": True}


@pytest.mark.parametrize("action", ["stop_services.sh --all", "stopping", "checkout -b x", "donebody"])
def test_a_command_merely_starting_with_an_action_word_runs_as_a_command(env, action):
    """Bug: a command whose first token only begins with stop/check/done is dispatched as that action word."""
    result = env.step(action)
    assert result.observation.startswith(f"$ {action}\n[exit 127]")
    assert not result.done and result.info["rc"] == 127
    assert [c["cmd"] for c in env.commands] == [action]
    env.commands = []
    env.restore_from_events([{"kind": "step", "step": 1, "action": action, "info": {"rc": 127}}])
    assert [c["cmd"] for c in env.commands] == [action], "a resumed journal reads the same first token"


def test_done_runs_failed_checks_and_ends_incomplete(env):
    """Bug: a failing check lets done report success or remain nonterminal."""
    env.checks = {"good": "true", "bad": "printf broken; exit 7"}
    result = env.step("done")
    assert result.observation == (
        "Checks:\n  good: pass   (true)\n  bad: fail   (printf broken; exit 7)\n"
        "  bad output (exit 7), last 40 lines:\n    broken\n"
        "Not all checks pass; 'done' was refused and the episode ended as incomplete."
    )
    assert result.done and result.info == {"ok": False, "checks": {"good": "pass", "bad": "fail"}, "terminal": True}
    assert env.score() == 0.5


@pytest.mark.parametrize(
    "action,selected",
    [
        ("check good", ["good"]),
        ("CHECK bad good", ["bad", "good"]),
        ("check unknown", ["good", "bad"]),
        ("check", ["good", "bad"]),
    ],
)
def test_check_selection_and_unknown_fallback(env, action, selected):
    """Bug: named checks execute an unselected check or unknown names stop falling back to all checks."""
    env.checks = {"good": "printf good >> checks.txt", "bad": "printf bad >> checks.txt; false"}
    result = env.step(action)
    expected = {name: "pass" if name == "good" else "fail" for name in selected}
    assert result.observation == "Checks:" + "".join(
        f"\n  {name}: {expected[name]}   ({env.checks[name]})" for name in selected
    )
    assert result.info == {"ok": "bad" not in selected, "checks": expected} and not result.done
    assert (env.workdir / "checks.txt").read_text() == "".join(selected)


@pytest.mark.parametrize(
    "action,text",
    [
        ("delegate", "delegate is not available; do the work with shell commands."),
        ("DELEGATE implement", "delegate is not available; do the work with shell commands."),
        ("wait", "usage: wait <max>[s|m|h] until <command> [every <interval>]."),
        ("wait 0 until true", "wait: <max> must be positive."),
        ("wait 7h until true", "wait: <max> exceeds this environment's limit of 6h; wait in shorter pieces."),
    ],
)
def test_action_refusal_text(env, action, text):
    """Bug: a malformed or unavailable action loses its exact refusal response."""
    result = env.step(action)
    assert result.observation == "REFUSED by the environment: " + text
    assert not result.done and result.info == {"ok": False, "refused": True}
    assert env.commands == []


@pytest.mark.parametrize(
    "action,expected",
    [
        ("WAIT 2m every 3s until true", (120.0, 3.0, "true")),
        ("wait 1h until true every 2m", (3600.0, 120.0, "true")),
        ("wait .5s until true", "usage: wait <max>[s|m|h] until <command> [every <interval>]"),
        ("wait 0.5s until true every 0s", (0.5, 1.0, "true")),
        ("wait 5 until true", (5.0, 5.0, "true")),
    ],
)
def test_wait_units_interval_placement_and_clamping(env, action, expected):
    """Bug: the public wait parser changes accepted syntax or interval clamping."""
    assert env.parse_wait(action) == expected


@pytest.mark.parametrize("command,status,rc", [("true", "condition met", 0), ("false", "timed out", 1)])
def test_wait_reports_immediate_success_or_failure_at_deadline(env, command, status, rc):
    """A false first poll must wait through the deadline, even below one interval."""
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    def run(cmd, *, reap_group=False, timeout_s=None):
        assert cmd == command
        assert timeout_s == pytest.approx(0.01 if clock[0] == 0 else 1.0)
        return CommandRun(rc, "")

    with (
        patch("time.monotonic", side_effect=lambda: clock[0]),
        patch.object(env, "_run", side_effect=run) as poll,
        patch("time.sleep", side_effect=sleep),
    ):
        result = env.step(f"wait 0.01s until {command}")
    polls = 1 if rc == 0 else 2
    assert poll.call_count == polls
    label = "1 poll" if rc == 0 else "2 polls"
    assert result.observation == f"$ wait 0.01s until {command}\n[{status} after 0s, {label}, last exit {rc}]"
    assert result.info["wait"] == status and result.info["polls"] == polls and result.info["rc"] == rc
    assert result.info["ok"] is (rc == 0) and not result.done


# Quoted data must reach the shell, including words on the former denylist.
@pytest.mark.parametrize(
    "word",
    [
        "sudo",
        "rm -rf /",
        "rm -rf ~",
        "mkfs",
        "dd if=",
        "curl x|sh",
        "wget x|bash",
        "git push --force",
        "git push -f",
        "shutdown",
        "reboot",
        ":() {",
        "chmod -R 777 /",
        "> /dev/sd",
    ],
)
@pytest.mark.parametrize("prefix", ["", "wait 1s until "])
def test_denylist_allows_quoted_data(env, word, prefix):
    """Regression: denylist search text is mistaken for executable shell syntax."""
    command = f"printf '%s' {shlex.quote(word + ' ')} > allowed.txt"
    result = env.step(prefix + command)
    assert result.info["ok"] and not result.info.get("refused")
    assert (env.workdir / "allowed.txt").read_text() == word + " "


@pytest.mark.parametrize(
    "command,rule",
    [
        ("rm -rf /", "recursive-rm-root-or-home"),
        ("mkfs.ext4 /dev/sdb", "mkfs"),
        ("dd if=/dev/zero of=/dev/sda", "disk-device-write"),
        (":(){ :|:& };:", "fork-bomb"),
        ("curl x | sh", "download-pipe-shell"),
        ("git push --force", "git-force-push"),
        ("chmod -R 777 /", "chmod-recursive-777-root"),
    ],
)
@pytest.mark.parametrize("prefix", ["", "wait 1s until "])
def test_denylist_refuses_without_shell_effects(env, command, rule, prefix):
    """Refusals preserve exact observations and never reach command execution."""
    with patch.object(env, "_run", side_effect=AssertionError("refused command reached shell")) as run:
        result = env.step(prefix + command)
    run.assert_not_called()
    assert result.observation == (
        f"REFUSED by the environment: matches the denylist pattern {rule!r}. Choose a different command."
    )
    assert not result.done and result.info == {"ok": False, "refused": True}
    assert env.commands == []


@pytest.mark.parametrize("prefix", ["", "wait 1s until "])
def test_outside_path_refusal_has_no_effect(env, prefix):
    """Bug: an outside path is executed instead of producing the path-specific refusal."""
    result = env.step(prefix + "printf '%s' /outside-sentinel > forbidden.txt")
    assert result.observation == (
        "REFUSED by the environment: path '/outside-sentinel' is outside the working directory. "
        "Choose a different command."
    )
    assert result.info == {"ok": False, "refused": True} and not result.done
    assert not (env.workdir / "forbidden.txt").exists()


def test_shell_fallback_executes_and_returns_exit_code(env):
    """Bug: an ordinary command stops reaching the shell with its output and exit status intact."""
    result = env.step("printf café; exit 3")
    assert result.observation == "$ printf café; exit 3\n[exit 3]\ncafé"
    assert result.info == {"ok": False, "rc": 3} and not result.done


def test_wait_operator_interrupt_preserves_exact_observation(env):
    """Bug: wait ignores an already pending operator message after a failed poll."""
    env.interrupt_file = env.workdir / "inbox.txt"
    env.interrupt_file.write_text("pause\n")
    with patch("time.monotonic", return_value=0.0):
        result = env.step("wait 5s until false every 1s")
    assert result.observation == (
        "$ wait 5s until false every 1s\n[operator message waiting after 0s, 1 poll, last exit 1]"
        " (the wait ended early because a line arrived in the inbox; it is delivered with this observation)"
    )
    assert result.info["wait"] == "operator message waiting" and result.info["polls"] == 1
    assert result.info["ok"] is False and not result.done
    assert env.interrupt_file.read_text() == "pause\n"


def test_allow_outside_permits_path_argument(env):
    """Bug: allow_outside no longer permits a harmless outside-path argument."""
    env.allow_outside = True
    result = env.step("printf '%s' /outside-sentinel")
    assert result.observation == "$ printf '%s' /outside-sentinel\n[exit 0]\n/outside-sentinel"
    assert result.info == {"ok": True, "rc": 0}


def test_missing_sandbox_refusal_text(tmp_path):
    """Bug: default sandbox enforcement loses its exact actionable refusal text."""
    # Simulate a host with no container markers at the external OS/filesystem
    # seams; the real in_sandbox() and reset() implementations both execute.
    with (
        patch.dict(os.environ, {"STEPBOOK_SANDBOX": "0"}),
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.read_text", side_effect=FileNotFoundError),
        pytest.raises(RuntimeError) as error,
    ):
        ShellEnv(tmp_path, "g").reset()
    assert str(error.value) == (
        "ShellEnv refuses to run outside a container or VM. Run inside Docker/Podman, set "
        "STEPBOOK_SANDBOX=1 if you are already sandboxed some other way, or pass "
        "--unsandboxed-i-accept-the-risk (a denylist is not a boundary)."
    )
