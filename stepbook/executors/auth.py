"""Subscription preflight for harness-owned Claude calls, on unless API-key billing is allowed.

The guard is the default; ``--allow-api-billing`` (or ``STEPBOOK_ALLOW_API_BILLING=1``) opts out of it.
It never extracts credentials, changes login, clears the user's environment, or falls back to another
provider. This is a technical preflight for the installed Claude CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SUBSCRIPTION_STATUS_ATTEMPTS = 3
SUBSCRIPTION_STATUS_RETRY_SECONDS = 5

OVERRIDES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
)


class SubscriptionError(RuntimeError):
    pass


ALLOW_API_BILLING = "STEPBOOK_ALLOW_API_BILLING"


def subscription_required():
    """Subscription-only is the default; only an explicit opt-out turns the guard off."""
    return os.environ.get(ALLOW_API_BILLING) != "1"


def accept_billing_choice(args):
    """``--allow-api-billing`` opts this process out, through the environment its children inherit."""
    if getattr(args, "allow_api_billing", False):
        os.environ[ALLOW_API_BILLING] = "1"


def check_environment():
    names = [name for name in OVERRIDES if os.environ.get(name)]
    if names:
        raise SubscriptionError(
            "Subscription-only call refused: API/provider override present ("
            + ", ".join(names)
            + ") would bill an API key; log in with your subscription, or pass --allow-api-billing. "
            "No model call or fallback; values are not logged."
        )


def cli_path(claude_bin="claude"):
    found = shutil.which(claude_bin)
    if not found:
        raise SubscriptionError("Claude CLI unavailable for subscription preflight; no model call.")
    return str(Path(found).resolve())


def check_subscription(*, claude_bin="claude", cwd=None):
    check_environment()  # before invoking even an authentication-status subprocess
    binary = cli_path(claude_bin)
    for attempt in range(SUBSCRIPTION_STATUS_ATTEMPTS):
        try:
            result = subprocess.run(
                [binary, "--setting-sources", "", "auth", "status", "--json"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode:
                error = "Claude subscription status unavailable; no model call or fallback."
            else:
                status = json.loads(result.stdout)
                break
        except (OSError, subprocess.TimeoutExpired, ValueError):
            error = "Claude subscription status could not be verified; no model call or fallback."
        if attempt == SUBSCRIPTION_STATUS_ATTEMPTS - 1:
            raise SubscriptionError(error) from None
        time.sleep(SUBSCRIPTION_STATUS_RETRY_SECONDS)
    if (
        not isinstance(status, dict)
        or status.get("loggedIn") is not True
        or status.get("authMethod") != "claude.ai"
        or status.get("apiProvider") != "firstParty"
    ):
        raise SubscriptionError(
            "Existing first-party Claude subscription login is required; this login would bill an API key, "
            "so log in with your subscription, or pass --allow-api-billing."
        )
    # No email, account/organization ID, token, raw stdout/stderr or credentials.
    return {
        "cli_path": binary,
        "auth": {key: status.get(key) for key in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")},
    }


def preflight_command(args):
    """Early CLI refusal before adapter setup; the Claude executor rechecks immediately before its calls.

    Only ``--executor cli`` is attested. A caller-configured agent command is not this guard's to verify, and
    refusing it outright would make the guard a provider lock rather than a subscription check.
    """
    accept_billing_choice(args)
    if subscription_required() and args.cmd == "run" and args.executor == "cli":
        check_subscription()
