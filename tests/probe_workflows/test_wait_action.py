"""The 'wait <max> until <command> [every <interval>]' action."""

import threading
import time

from stepbook.storage.events import EventLog
from stepbook.storage.recovery import execute_action

from ._helpers import shell_env, steps


def _env(tmp_path, **kw):
    env = shell_env(tmp_path, **kw)
    env.reset()
    return env


def test_echo_wait_is_a_shell_command_not_a_wait(tmp_path):
    r = _env(tmp_path).step("echo wait 5s until x")
    assert r.observation.startswith("$ echo wait 5s until x\n[exit 0]\nwait 5s until x") and "wait" not in r.info


def test_every_inside_a_quoted_condition_does_not_steal_the_interval(tmp_path):
    env = _env(tmp_path)
    assert env.parse_wait("wait 2s until grep -q 'runs every 5' log.txt every 1s") == (
        2.0,
        1.0,
        "grep -q 'runs every 5' log.txt",
    )
    assert env.parse_wait("wait 2s until echo 'every 5'") == (2.0, 2.0, "echo 'every 5'")
    assert env.parse_wait("wait 2s until ./poll.sh --every 5") == (2.0, 2.0, "./poll.sh --every 5")


def test_fractional_units_and_max_below_interval(tmp_path):
    env = _env(tmp_path)
    assert env.parse_wait("wait 0.5s until true") == (0.5, 1.0, "true")
    assert env.parse_wait("wait 1.5m every 0.5m until true") == (90.0, 30.0, "true")
    assert env.parse_wait("wait 1s until false every 10s") == (1.0, 1.0, "false")
    t0 = time.monotonic()
    r = env.step("wait 0.5s until true")
    assert r.info["wait"] == "condition met" and r.info["polls"] == 1
    r = env.step("wait 1s until false every 10s")
    # Failure gets a deadline poll even when the requested interval exceeds the wait.
    assert r.info["wait"] == "timed out" and r.info["polls"] == 2 and 1 <= time.monotonic() - t0 < 1.5


def test_inbox_line_arriving_between_polls_ends_the_wait(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    env = _env(tmp_path, log_dir=log)
    threading.Timer(0.3, lambda: (log / "inbox.txt").write_text("look at this\n")).start()
    t0 = time.monotonic()
    r = env.step("wait 10s until false every 1s")
    # the inbox is checked before each sleep chunk, not after it, so a line landing mid-sleep costs one more poll
    assert r.info["wait"] == "operator message waiting" and time.monotonic() - t0 < 2.5 and r.info["polls"] <= 2


def test_wait_on_the_horizon_step_sets_done_and_is_journaled_as_a_horizon_stop(tmp_path):
    env = _env(tmp_path, horizon=1)
    log = EventLog(tmp_path / "log" / "events.jsonl")
    r = execute_action(log, env, step=0, action="wait 1s until true")
    assert r.done and r.info["ok"]
    assert steps(tmp_path / "log")[-1].get("horizon_stop") is True


def test_condition_command_timing_out_under_cmd_timeout_counts_as_a_failed_poll(tmp_path):
    env = _env(tmp_path, cmd_timeout_s=0.3)
    t0 = time.monotonic()
    r = env.step("wait 1s until sleep 5 every 1s")
    assert time.monotonic() - t0 < 1.5
    assert r.info["wait"] == "timed out" and r.info["rc"] == 124 and "process group was killed" in r.observation


def test_wait_survives_undecodable_inbox_bytes(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    env = _env(tmp_path, log_dir=log)
    (log / "inbox.txt").write_bytes(b"don\x92t stop\n")  # cp1252 apostrophe
    r = env.step("wait 3s until false every 1s")
    assert r.info["wait"] == "operator message waiting"
