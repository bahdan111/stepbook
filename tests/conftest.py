"""Scripted adapters exercise the same subprocess/stdout boundary as real adapters."""

import json
import os
import re
import shlex
import sys
import tomllib
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from stepbook.envs.eval_loop import AdapterEvalEnv

# Fixture projects carry their own tests (the offline toy repo); they are run by the tests that use them
collect_ignore_glob = ["fixtures/*"]


@pytest.fixture(autouse=True)
def allow_api_billing(monkeypatch):
    """Subscription-only is the product default; the offline suite uses a fake ``claude`` and opts out.

    A test of the guard itself removes this variable (``monkeypatch.delenv``) and asserts the refusal.
    """
    monkeypatch.setenv("STEPBOOK_ALLOW_API_BILLING", "1")


def package_version():
    """The one version, read where it lives: no test spells it, so a bump needs no test edit."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["project"]["version"]


ADAPTER_SCRIPT = r"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
args = sys.argv[1:]
with (root / "requests.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\n")
script = json.loads((root / "replies.json").read_text())
counts_path = root / "counts.json"
counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
key = json.dumps(args)
if key not in script:
    key = json.dumps(args[:1])
route = script.get(key)
index = counts.get(key, 0)
if route is None or (not route["repeat"] and index >= len(route["responses"])):
    with (root / "unexpected.jsonl").open("a") as log:
        log.write(json.dumps(args) + "\n")
    sys.exit("Unscripted adapter request: " + repr(args))
response = route["responses"][0 if route["repeat"] else index]
counts[key] = index + 1
counts_path.write_text(json.dumps(counts))
for name, text in response["writes"].items():
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
payload = response["payload"]
if response["label_from_argv"]:
    payload["config_label"] = args[1]
print(json.dumps(payload) if isinstance(payload, dict) else payload)
code = response["exit_code"]
if code is None:
    # This stand-in chooses its own exit code the way the shipped adapters do: a reason sentence is a
    # refusal, anything else is done. Tests that pin the contract pass exit_code= explicitly.
    code = 1 if isinstance(payload, str) and payload.startswith("Error:") else 0
sys.exit(code)
"""


class ScriptedAdapter:
    """Exact argv or verb replies; lists queue receipts, single values repeat.

    Dict replies emit JSON, which for `run` is the receipt the contract asks for. The exit code is the
    adapter's answer: pass ``exit_code=`` to script one, or let a reply that opens with "Error:" exit 1 and
    everything else exit 0, as the bundled adapter does.
    Optional writes happen in the child before stdout, for inbox timing tests.
    Request logs and queue positions survive fresh environment instances.
    """

    def __init__(self, directory, replies, startup):
        self.directory = directory
        directory.mkdir()
        script = directory / "adapter.py"
        script.write_text(ADAPTER_SCRIPT)
        self.command = shlex.join([sys.executable, str(script)])
        self.routes = {}
        defaults = {"reset": "ready", "describe": "ready", "status": "ready"} if startup else {}
        for command, reply in {**defaults, **replies}.items():
            self.answer(command, reply)

    def answer(self, command, payload, *, writes=None, label_from_argv=False, exit_code=None):
        """Script one repeatable reply or a finite list of replies for a verb/argv."""
        args = shlex.split(command) if isinstance(command, str) else list(command)
        queued = isinstance(payload, list)
        self.routes[json.dumps(args)] = {
            "repeat": not queued,
            "responses": [
                {
                    "payload": value,
                    "writes": {str(path): text for path, text in (writes or {}).items()},
                    "label_from_argv": label_from_argv,
                    "exit_code": exit_code,
                }
                for value in (payload if queued else [payload])
            ],
        }
        (self.directory / "replies.json").write_text(json.dumps(self.routes))

    @property
    def calls(self):
        path = self.directory / "requests.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    @property
    def runs(self):
        return [args[1] for args in self.calls if args[0] == "run"]


@pytest.fixture
def fake_adapter(tmp_path):
    """Make isolated executable adapters and fail on every unscripted request."""
    adapters = []

    def make(replies=None, *, startup=True):
        adapter = ScriptedAdapter(tmp_path / f"fake-adapter-{len(adapters)}", replies or {}, startup)
        adapters.append(adapter)
        return adapter

    yield make
    for adapter in adapters:
        unexpected = adapter.directory / "unexpected.jsonl"
        assert not unexpected.exists(), f"Unscripted adapter requests: {unexpected.read_text()}"


class ScriptedSDK:
    """Offline query boundary with finite scripts and observable SDK inputs."""

    def __init__(self):
        self.calls = []
        self.scripts = deque()
        self.module = ModuleType("claude_agent_sdk")

        class Options(SimpleNamespace):
            def __init__(self, **kwargs):
                super().__init__(
                    **{
                        "setting_sources": None,
                        "skills": None,
                        "extra_args": {},
                        "env": {},
                        "thinking": None,
                        "resume": None,
                        **kwargs,
                    }
                )

        class Result(SimpleNamespace):
            def __init__(self, **kwargs):
                output = {"reasoning": "r", "state_patch": {}, "action": "stop"}
                super().__init__(
                    **{
                        "subtype": "success",
                        "duration_ms": 1,
                        "duration_api_ms": 1,
                        "is_error": False,
                        "num_turns": 1,
                        "session_id": "offline-session",
                        "usage": {},
                        "model_usage": {},
                        "total_cost_usd": 0.01,
                        "stop_reason": None,
                        "errors": None,
                        "structured_output": output,
                        "result": json.dumps(output),
                        **kwargs,
                    }
                )

        self.module.ClaudeAgentOptions = Options
        self.module.ResultMessage = Result
        self.module.HookMatcher = SimpleNamespace
        self.module.query = self.query
        self.result = Result

    def enqueue(self, *messages):
        """One query yields these messages/raises these errors, or runs an async script."""
        self.scripts.append(messages)

    async def query(self, *, prompt, options):
        self.calls.append(SimpleNamespace(prompt=prompt, options=options))
        if not self.scripts:
            pytest.fail("Unscripted SDK query")
        for message in self.scripts.popleft():
            if isinstance(message, BaseException):
                raise message
            if callable(message):
                async for item in message(prompt=prompt, options=options):
                    yield item
            else:
                yield message

    @staticmethod
    async def hooks(options, event, data):
        """Dispatch actual callbacks using the SDK's tool-name matcher contract."""
        replies = []
        for matcher in options.hooks.get(event, []):
            pattern = getattr(matcher, "matcher", None)
            if pattern and not re.fullmatch(pattern, data.get("tool_name", "")):
                continue
            for hook in matcher.hooks:
                replies.append(await hook(data, "offline-tool", None))
        return replies


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install an optional-dependency stub per test; never connect to a provider."""
    sdk = ScriptedSDK()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk.module)
    yield sdk
    assert not sdk.scripts, "Unused SDK query scripts"


@pytest.fixture
def replay_receipts():
    """Restore documented step receipts, including historical accepted decisions."""

    def replay(env, runs, *, reference="baseline", applied=None):
        events = []
        for hypothesis, rows in runs.items():
            if hypothesis != "baseline":
                events.append({"kind": "step", "action": f"apply {hypothesis}", "result": f"Applied {hypothesis}"})
            for row in rows:
                events.append(
                    {"kind": "step", "action": "run measurement", "result": "Eval finished:\n" + json.dumps(row)}
                )
            if hypothesis != "baseline":
                decision = "keep" if hypothesis == reference else "revert"
                events.append(
                    {
                        "kind": "step",
                        "action": f"decide {hypothesis} {decision}",
                        "result": f"{hypothesis} {decision}: accepted",
                    }
                )
        if applied:
            events.append({"kind": "step", "action": f"apply {applied}", "result": f"Applied {applied}"})
        env.restore_from_events(events, check_adapter=False)
        return events

    return replay


@pytest.fixture
def measure_verdict(fake_adapter):
    """Deliver scripted measurements and return the verdict text in the observation."""

    def measure(runs, hypothesis, *, reference="baseline", **options):
        adapter = fake_adapter()
        env = AdapterEvalEnv(adapter=adapter.command, **options)
        events = []

        def step(action):
            result = env.step(action)
            events.append({"kind": "step", "action": action, "result": result.observation})
            return result

        if reference != "baseline":
            adapter.answer(f"apply {reference}", f"Applied {reference}")
            step(f"apply {reference}")
        adapter.answer("run reference", runs[reference])
        for _ in runs[reference]:
            step("run reference")
        if reference != "baseline":
            # Old journals can start with an already accepted reference, without its predecessor's runs.
            env.restore_from_events(
                events
                + [{"kind": "step", "action": f"decide {reference} keep", "result": f"{reference} keep: accepted"}],
                check_adapter=False,
            )
        adapter.answer(f"apply {hypothesis}", f"Applied {hypothesis}")
        step(f"apply {hypothesis}")
        adapter.answer("run candidate", runs[hypothesis])
        for _ in runs[hypothesis]:
            result = step("run candidate")
        text = result.observation.split("\n", 2)[2]
        assert text.startswith(f"Computed verdict for {hypothesis}")
        groups = options.get("alternative_groups", {})
        if not any(hypothesis in members for members in groups.values()):
            assert " NOTE:" not in text
        return text

    return measure


@pytest.fixture
def research_action():
    """Read persisted research through the environment's action receipts."""

    def action(env, verb, value=""):
        result = env.step(verb + " " + (json.dumps(value) if verb in ("read", "record") else value))
        assert result.info["ok"], result.observation
        return json.loads(result.observation.split("\n", 1)[1])

    return action


@pytest.fixture
def claude_on_path(tmp_path_factory, monkeypatch):
    """A stand-in `claude` on PATH, so a CLI run with a stubbed executor passes the binary preflight without the real CLI."""
    bin_dir = tmp_path_factory.mktemp("fake-bin")
    fake = bin_dir / "claude"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
