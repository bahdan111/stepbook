"""The 'wait <max> until <command> [every <interval>]' action: one step for a long poll."""

import time

import pytest

from stepbook.envs.shell import ShellEnv


def _env(tmp_path, **kw):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={}, **kw)
    env.reset()
    return env


def test_wait_parses_units_and_interval_positions(tmp_path):
    env = _env(tmp_path)
    assert env.parse_wait("wait 90m until test -f done.flag every 2m") == (5400.0, 120.0, "test -f done.flag")
    assert env.parse_wait("wait 2h every 30s until curl -sf localhost/health") == (
        7200.0,
        30.0,
        "curl -sf localhost/health",
    )
    assert env.parse_wait("wait 45 until true") == (45.0, 45.0, "true")  # default 60s interval capped at max
    assert isinstance(env.parse_wait("wait until true"), str)
    assert isinstance(env.parse_wait("wait 10m"), str)
    assert "limit" in env.parse_wait("wait 7h until true")
    usage = "usage: wait <max>[s|m|h] until <command> [every <interval>]"
    assert env.parse_wait("wait until true") == usage
    assert env.parse_wait("wait 10m") == usage
    assert env.parse_wait("wait 7h until true") == (
        "wait: <max> exceeds this environment's limit of 6h; wait in shorter pieces"
    )
    assert env.parse_wait("wait 0 until true") == "wait: <max> must be positive"
    assert env.parse_wait("wait 90m until true every 0s") == (5400.0, 1.0, "true")


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize(
    "template,interval",
    [
        ("wait 180 until {condition}", 60.0),
        ("wait 180 until {condition} every 5", 5.0),
        ("wait 180 every 5 until {condition}", 5.0),
    ],
)
def test_wait_unwraps_whole_quoted_condition(tmp_path, quote, template, interval):
    env = _env(tmp_path, checkpoint=False)
    command = "test -f .stepbook/runs/r4_top_k5/exit.status"
    action = template.format(condition=f"{quote}{command}{quote}")
    assert env.parse_wait(action) == (180.0, interval, command)


@pytest.mark.parametrize(
    "command",
    [
        "test -f done.flag",
        "test -f 'a b'",
        "grep -q 'x y' f",
        "'true'",
        '"true"',
        "'test -f done.flag",
        '"test -f done.flag',
        "'test -f done.flag\"",
        "'test -f 'a b'",
        "'test' '-f done.flag'",
        "''",
    ],
)
@pytest.mark.parametrize("suffix,interval", [("", 60.0), (" every 5", 5.0)])
def test_wait_keeps_other_conditions_unchanged(tmp_path, command, suffix, interval):
    env = _env(tmp_path, checkpoint=False)
    assert env.parse_wait(f"wait 180 until {command}{suffix}") == (180.0, interval, command)


@pytest.mark.parametrize("quote", ["'", '"'])
def test_wait_quoted_condition_meets_real_file_in_one_step(tmp_path, quote):
    env = _env(tmp_path, checkpoint=False)
    (tmp_path / "done.flag").touch()
    action = f"wait 0.1 until {quote}test -f done.flag{quote} every 1"
    result = env.step(action)
    assert result.info["ok"] and result.info["wait"] == "condition met"
    assert result.info["polls"] == 1 and result.info["rc"] == 0
    assert result.observation == f"$ {action}\n[condition met after 0s, 1 poll, last exit 0]"
    assert env.commands == [{"t": 1, "cmd": action, "rc": 0}]
    assert env.t == 1 and not result.done


@pytest.mark.parametrize("quote", ["'", '"'])
def test_wait_quoted_blocked_condition_is_refused(tmp_path, monkeypatch, quote):
    env = _env(tmp_path, checkpoint=False)

    def unexpected_run(*args, **kwargs):
        pytest.fail("a blocked condition must never execute")

    monkeypatch.setattr(env, "_run", unexpected_run)
    result = env.step(f"wait 180 until {quote}rm -rf /{quote} every 5")
    assert result.info == {"ok": False, "refused": True}
    assert result.observation == (
        "REFUSED by the environment: matches the denylist pattern "
        "'recursive-rm-root-or-home'. Choose a different command."
    )
    assert env.commands == [] and env.t == 1 and not result.done


def test_wait_returns_when_condition_becomes_true(tmp_path):
    env = _env(tmp_path)
    flag = tmp_path / "done.flag"
    env.step(f"(sleep 1.2; touch {flag.name}) > /dev/null 2>&1 &")
    t0 = time.monotonic()
    r = env.step(f"wait 20s until test -f {flag.name} every 1s")
    assert r.info["ok"] and r.info["wait"] == "condition met" and r.info["polls"] >= 2
    assert time.monotonic() - t0 < 15 and "condition met" in r.observation
    assert env.t == 2 and not r.done


def test_wait_times_out_with_last_output_and_nonzero_exit(tmp_path):
    env = _env(tmp_path)
    r = env.step("wait 2s until sh -c 'echo still-running; exit 3' every 1s")
    assert not r.info["ok"] and r.info["wait"] == "timed out" and r.info["rc"] == 3
    assert "still-running" in r.observation and "timed out after" in r.observation


def test_wait_ends_early_when_an_operator_line_arrives(tmp_path):
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    env = _env(tmp_path, log_dir=log_dir)
    (log_dir / "inbox.txt").write_text("pause\n")
    t0 = time.monotonic()
    r = env.step("wait 5m until false every 4m")
    assert r.info["wait"] == "operator message waiting" and time.monotonic() - t0 < 30
    assert "inbox" in r.observation and not r.info["ok"]


def test_wait_condition_goes_through_the_denylist(tmp_path):
    env = _env(tmp_path)
    started = time.monotonic()
    r = env.step("wait 1m until rm -rf / every 10s")
    assert r.info.get("refused") and "REFUSED" in r.observation
    assert r.info == {"ok": False, "refused": True}
    assert r.observation == (
        "REFUSED by the environment: matches the denylist pattern "
        "'recursive-rm-root-or-home'. Choose a different command."
    )
    # A parser failure would answer with the "usage:" text and no `refused` flag; a
    # polled condition would take at least one interval and report `polls`.
    assert "usage:" not in r.observation and "polls" not in r.info and "wait" not in r.info
    assert time.monotonic() - started < 1.0
    assert env.t == 1 and not r.done


def test_wait_is_advertised_in_the_reset_text(tmp_path):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={})
    assert "'wait <max> until <command> [every <interval>]'" in env.reset()
