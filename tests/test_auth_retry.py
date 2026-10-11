"""Subscription status retries are offline, bounded, and never retry a known refusal."""

import json
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from stepbook.executors import auth

UNVERIFIABLE = "Claude subscription status could not be verified; no model call or fallback."
UNAVAILABLE = "Claude subscription status unavailable; no model call or fallback."
LOGIN_REQUIRED = (
    "Existing first-party Claude subscription login is required; this login would bill an API key, "
    "so log in with your subscription, or pass --allow-api-billing."
)
LOGIN = {
    "loggedIn": True,
    "authMethod": "claude.ai",
    "apiProvider": "firstParty",
    "subscriptionType": "max",
}


def response(status=LOGIN, *, returncode=0):
    return SimpleNamespace(returncode=returncode, stdout=json.dumps(status), stderr="offline stderr")


def glitch(kind):
    if kind == "timeout":
        return subprocess.TimeoutExpired("offline status", 15, output="offline stdout", stderr="offline stderr")
    if kind == "oserror":
        return OSError("offline startup failure")
    if kind == "exit":
        return response(returncode=1)
    return SimpleNamespace(returncode=0, stdout="invalid JSON", stderr="offline stderr")


@pytest.fixture
def status_check(monkeypatch):
    for name in auth.OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    binary = Mock(return_value="offline-claude")
    run = Mock()
    pause = Mock()
    monkeypatch.setattr(auth, "cli_path", binary)
    monkeypatch.setattr(auth.subprocess, "run", run)
    monkeypatch.setattr(time, "sleep", pause)
    return binary, run, pause


@pytest.mark.parametrize("kind", ["timeout", "oserror", "exit", "json"])
def test_transient_status_failure_then_success(status_check, kind, capsys):
    binary, run, pause = status_check
    run.side_effect = [glitch(kind), response({**LOGIN, "email": "private@example.test", "token": "private"})]

    assert auth.check_subscription(claude_bin="offline-claude", cwd=".") == {
        "cli_path": "offline-claude",
        "auth": LOGIN,
    }

    binary.assert_called_once_with("offline-claude")
    expected = call(
        ["offline-claude", "--setting-sources", "", "auth", "status", "--json"],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert run.call_args_list == [expected, expected]
    pause.assert_called_once_with(5)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("kind", ["timeout", "oserror", "exit", "json"])
def test_three_transient_failures_keep_existing_error(status_check, kind, capsys):
    _, run, pause = status_check
    run.side_effect = [glitch(kind) for _ in range(3)]

    with pytest.raises(auth.SubscriptionError) as caught:
        auth.check_subscription()

    assert str(caught.value) == (UNAVAILABLE if kind == "exit" else UNVERIFIABLE)
    assert run.call_count == 3
    assert pause.call_args_list == [call(5), call(5)]
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("name", auth.OVERRIDES)
def test_environment_override_is_refused_without_retry(status_check, monkeypatch, name):
    binary, run, pause = status_check
    monkeypatch.setenv(name, "offline-override")

    with pytest.raises(auth.SubscriptionError) as caught:
        auth.check_subscription()

    assert str(caught.value) == (
        f"Subscription-only call refused: API/provider override present ({name}) would bill an API key; "
        "log in with your subscription, or pass --allow-api-billing. "
        "No model call or fallback; values are not logged."
    )
    binary.assert_not_called()
    run.assert_not_called()
    pause.assert_not_called()


def test_missing_cli_is_refused_without_retry(monkeypatch):
    for name in auth.OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    which, run, pause = Mock(return_value=None), Mock(), Mock()
    monkeypatch.setattr(auth.shutil, "which", which)
    monkeypatch.setattr(auth.subprocess, "run", run)
    monkeypatch.setattr(time, "sleep", pause)

    with pytest.raises(auth.SubscriptionError) as caught:
        auth.check_subscription()

    assert str(caught.value) == "Claude CLI unavailable for subscription preflight; no model call."
    which.assert_called_once_with("claude")
    run.assert_not_called()
    pause.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        {**LOGIN, "loggedIn": False},
        {**LOGIN, "authMethod": "api_key"},
        {**LOGIN, "apiProvider": "bedrock"},
        {},
        [],
        None,
    ],
)
@pytest.mark.parametrize("after_glitch", [False, True])
def test_parsed_refusal_stops_immediately(status_check, status, after_glitch):
    _, run, pause = status_check
    run.side_effect = ([glitch("timeout")] if after_glitch else []) + [response(status)]

    with pytest.raises(auth.SubscriptionError) as caught:
        auth.check_subscription()

    assert str(caught.value) == LOGIN_REQUIRED
    assert run.call_count == (2 if after_glitch else 1)
    assert pause.call_args_list == ([call(5)] if after_glitch else [])


def test_retry_limits_and_pause_are_patchable(status_check, monkeypatch):
    _, run, pause = status_check
    monkeypatch.setattr(auth, "SUBSCRIPTION_STATUS_ATTEMPTS", 2, raising=False)
    monkeypatch.setattr(auth, "SUBSCRIPTION_STATUS_RETRY_SECONDS", 0.25, raising=False)
    run.side_effect = [glitch("timeout"), glitch("json")]

    with pytest.raises(auth.SubscriptionError) as caught:
        auth.check_subscription()

    assert str(caught.value) == UNVERIFIABLE
    assert run.call_count == 2
    pause.assert_called_once_with(0.25)
