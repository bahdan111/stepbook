"""No runner may place a credential value on a command line (argv is visible in ps, docker inspect and logs).

Secrets must be exported and passed by name (`-e VAR`), or read inside the child."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET = re.compile(r"(TOKEN|SECRET|API_KEY|PASSWORD)\w*=\s*[\"']?\$", re.IGNORECASE)
FLAG_VALUE = re.compile(r"(?:-e|--env|--build-arg)\s+\w*(TOKEN|SECRET|API_KEY|PASSWORD)\w*=", re.IGNORECASE)


def _scripts():
    for folder in ("scripts", "integrations"):
        for p in (ROOT / folder).rglob("*"):
            if p.suffix in (".sh", "") and p.is_file() and "results" not in p.parts and p.stat().st_size < 200_000:
                try:
                    yield p, p.read_text()
                except UnicodeDecodeError:
                    continue


def test_no_secret_value_is_passed_as_a_command_line_argument():
    offenders = []
    for path, text in _scripts():
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if FLAG_VALUE.search(line) or ("docker run" in line and SECRET.search(line)):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:100]}")
    assert not offenders, "secrets on command lines:\n" + "\n".join(offenders)


def test_live_auth_child_inherits_token_without_putting_it_on_argv(monkeypatch):
    import os
    from types import SimpleNamespace

    from stepbook.executors import auth

    sentinel = "offline-oauth-sentinel"
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", sentinel)
    for key in auth.OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(auth, "cli_path", lambda *_args, **_kwargs: "/mock/claude")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert sentinel not in " ".join(argv)
        assert kwargs.get("env", os.environ)["CLAUDE_CODE_OAUTH_TOKEN"] == sentinel
        return SimpleNamespace(
            returncode=0,
            stdout='{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty"}',
        )

    monkeypatch.setattr(auth.subprocess, "run", run)
    auth.check_subscription()
    assert calls == [["/mock/claude", "--setting-sources", "", "auth", "status", "--json"]]
