"""Offline coverage of the shipped Codex step command with a schema-checking fake CLI."""

import copy
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from stepbook import CommandExecutor, Envelope, Skill
from stepbook.executors.command import EXIT_TIMED_OUT
from stepbook.runtime.loop import STATE_OUTPUT_SCHEMA
from stepbook.state.patch_schema import state_output_schema
from stepbook.state.validator import PatchError, Validator

ROOT = Path(__file__).resolve().parents[1]
PATCH = {"current": "step done", "checklist": {"verify": {"text": "Check output", "status": "done"}}}
USAGE = {
    "input_tokens": 1000,
    "cached_input_tokens": 800,
    "cache_write_input_tokens": 30,
    "output_tokens": 50,
    "reasoning_output_tokens": 20,
}
FAKE_CODEX = r"""
import json, os, sys
from pathlib import Path
from jsonschema import Draft202012Validator

argv = sys.argv[1:]
prompt = sys.stdin.read()
assert argv[0] == "exec" and argv[-1] == "-", argv
schema = json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text())
def check_strict(node):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False, node
            assert set(node.get("required", [])) == set(node.get("properties", {})), node
        for value in node.values():
            check_strict(value)
    elif isinstance(node, list):
        for value in node:
            check_strict(value)
check_strict(schema)
mode = os.environ.get("FAKE_STEP_MODE", "ok")
answer = {"reasoning": "fake codex", "state_patch": json.dumps(json.loads(os.environ["FAKE_STEP_PATCH"])),
          "action": "echo hi"}
if mode == "badpatch":
    answer["state_patch"] = "{not json"
Draft202012Validator(schema).validate(answer)
Path(os.environ["FAKE_STEP_LOG"]).write_text(json.dumps({"argv": argv, "prompt": prompt, "schema": schema}))
answer_text = "no JSON answer" if mode == "prose" else json.dumps(answer)
events = [
    {"type": "thread.started", "thread_id": "thread-step"},
    {"type": "item.completed", "item": {"type": "command_execution", "command": "echo hi"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": answer_text}},
    {"type": "turn.completed", "usage": json.loads(os.environ["FAKE_STEP_USAGE"])},
]
if mode == "error":
    events.append({"type": "turn.failed", "error": "fake failure"})
for event in events:
    print(json.dumps(event))
if mode != "fallback":
    Path(argv[argv.index("-o") + 1]).write_text(answer_text)
sys.exit(7 if mode == "exit" else 0)
"""


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    fake = tmp_path / "fake codex.py"
    fake.write_text(FAKE_CODEX)
    log = tmp_path / "codex.json"
    monkeypatch.setenv("FAKE_STEP_LOG", str(log))
    monkeypatch.setenv("FAKE_STEP_PATCH", json.dumps(PATCH))
    monkeypatch.setenv("FAKE_STEP_USAGE", json.dumps(USAGE))
    monkeypatch.delenv("FAKE_STEP_MODE", raising=False)
    return shlex.join([sys.executable, str(fake)]), log


@pytest.fixture
def runtime_schema():
    skill = Skill.load(ROOT / "stepbook/templates/checklist")
    return state_output_schema(STATE_OUTPUT_SCHEMA, skill.schema)


def run_step(tmp_path, codex, schema, *flags):
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(schema))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "stepbook.executors.codex",
            "step",
            "--codex",
            codex,
            "--schema",
            str(schema_file),
            *flags,
        ],
        input="<<SYSTEM>>\nsystem\n<</SYSTEM>>\nLatest Observation: x",
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )


@pytest.mark.parametrize("model", [None, "example-model"])
def test_module_step_decodes_answer_and_reports_usage(fake_codex, tmp_path, runtime_schema, model):
    codex, log = fake_codex
    proc = run_step(tmp_path, codex, runtime_schema, *(["--model", model] if model else []))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["state_patch"] == PATCH and out["action"] == "echo hi" and out["reasoning"] == "fake codex"
    assert {key: out["usage"][key] for key in USAGE} == USAGE
    assert out["usage"]["cost_usd"] is None and out["usage"]["provider"] == "openai"
    assert out["usage"]["model"] == model
    assert out["input_tokens"] == 1000 and out["output_tokens"] == 50
    assert out["api_calls"] == 1 and out["tool_calls"] == 1 and out["thread_id"] == "thread-step"
    assert out["codex_errors"] == []
    logged = json.loads(log.read_text())
    argv = logged["argv"]
    assert "--ephemeral" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert (argv[argv.index("-m") + 1] if "-m" in argv else None) == model
    assert "Latest Observation: x" in logged["prompt"]
    assert not Path(argv[argv.index("-o") + 1]).exists()
    assert not Path(argv[argv.index("--output-schema") + 1]).exists()


def test_command_executor_decodes_patch_and_keeps_usage(fake_codex, runtime_schema):
    codex, log = fake_codex
    command = shlex.join([sys.executable, "-m", "stepbook.executors.codex", "step", "--codex", codex])
    executor = CommandExecutor(command=command + " --schema {schema_file}", cwd=ROOT)
    envelope = executor.step("SYS", "Latest Observation: x", runtime_schema)
    assert isinstance(envelope, Envelope)
    assert envelope.state_patch == PATCH and envelope.action == "echo hi"
    assert {key: envelope.usage[key] for key in USAGE} == USAGE
    assert envelope.input_tokens == 1000 and envelope.output_tokens == 50
    assert envelope.usage["provider"] == "openai" and envelope.raw["cost_unknown"] is True
    assert envelope.api_calls == 1 and envelope.tool_calls == 1
    assert "<<SYSTEM>>\nSYS\n<</SYSTEM>>" in json.loads(log.read_text())["prompt"]
    Draft202012Validator(runtime_schema).validate(
        {"reasoning": envelope.reasoning, "state_patch": envelope.state_patch, "action": envelope.action}
    )


def test_real_runtime_schema_round_trip(runtime_schema):
    from stepbook.executors.codex import decode_answer, strictify

    original = copy.deepcopy(runtime_schema)
    strict, encoded = strictify(runtime_schema)
    assert encoded == ["state_patch"]
    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])
    assert strict["properties"]["state_patch"]["type"] == "string"
    answer = {"reasoning": "checked", "state_patch": json.dumps(PATCH), "action": "stop"}
    Draft202012Validator.check_schema(strict)
    Draft202012Validator(strict).validate(answer)
    decoded = decode_answer(answer, encoded)
    assert decoded["state_patch"] == PATCH
    Draft202012Validator(runtime_schema).validate(decoded)
    assert runtime_schema == original


@pytest.mark.parametrize("mode,code", [("exit", 7), ("error", 1), ("prose", 3), ("fallback", 0)])
def test_step_exit_codes_and_agent_message_fallback(fake_codex, tmp_path, runtime_schema, monkeypatch, mode, code):
    codex, _ = fake_codex
    monkeypatch.setenv("FAKE_STEP_MODE", mode)
    proc = run_step(tmp_path, codex, runtime_schema)
    assert proc.returncode == code, proc.stderr
    if code:
        assert proc.stdout == "" and proc.stderr
        assert "codex printed no JSON answer" in proc.stderr if mode == "prose" else "codex exited" in proc.stderr
        if mode == "error":
            assert "fake failure" in proc.stderr
    else:
        assert json.loads(proc.stdout)["state_patch"] == PATCH


def test_json_helpers_keep_their_parsing_semantics():
    from stepbook.executors.base import _last_envelope_object
    from stepbook.executors.codex import _last_codex_object

    text = 'progress\n{"action": "old"}\n{"state_patch": {"current": "new"}}'
    assert _last_codex_object(text) == {"state_patch": {"current": "new"}}
    assert _last_envelope_object(text) == {"action": "old"}


@pytest.mark.parametrize("exists", [False, True])
def test_codex_start_failure_is_one_line_without_traceback(tmp_path, runtime_schema, exists):
    binary = tmp_path / "unavailable codex"
    if exists:
        binary.write_text("not executable")
        binary.chmod(0o600)
    proc = run_step(tmp_path, shlex.quote(str(binary)), runtime_schema)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert len(proc.stderr.splitlines()) == 1
    assert proc.stderr.startswith("codex exited 1: codex could not start: ")


def test_the_turn_timeout_is_shorter_than_the_host_limit(tmp_path, runtime_schema, monkeypatch, capsys):
    """The host kills the driver at its own limit, so the Codex turn must give up 30 s earlier and say so."""
    from stepbook.executors import codex as driver

    seen = []

    def fake_run(argv, **kwargs):
        seen.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"], output="")

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "stdin", io.StringIO("prompt"))
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(runtime_schema))
    assert driver.main(["step", "--schema", str(schema_file), "--timeout", "900"]) == EXIT_TIMED_OUT
    assert seen == [870.0]
    assert f"codex exited {EXIT_TIMED_OUT}: codex timed out after 870s" in capsys.readouterr().err


def test_an_unparseable_patch_is_rejected_by_the_validator_not_read_as_no_change(
    fake_codex, tmp_path, runtime_schema, monkeypatch
):
    """A patch the wrapper cannot decode stays the string Codex sent, so the step is retried with the reason
    instead of running its action against state the model believed it had updated."""
    from stepbook.executors.codex import decode_answer

    codex, _ = fake_codex
    monkeypatch.setenv("FAKE_STEP_MODE", "badpatch")
    proc = run_step(tmp_path, codex, runtime_schema)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["state_patch"] == "{not json"
    assert not Draft202012Validator(runtime_schema).is_valid(
        {"reasoning": "fake codex", "state_patch": "{not json", "action": "echo hi"}
    )
    with pytest.raises(PatchError, match="state_patch must be a JSON object, got str"):
        Validator(Skill.load(ROOT / "stepbook/templates/checklist").schema).apply({}, "{not json")
    # An empty encoded field still means "no change", as the strict-schema description tells the model.
    assert decode_answer({"state_patch": "not json"}, ["state_patch"]) == {"state_patch": "not json"}
    assert decode_answer({"state_patch": "  "}, ["state_patch"]) == {"state_patch": {}}


def test_api_credentials_are_dropped_for_a_subscription_only_step(monkeypatch, tmp_path):
    """Subscription-only is the default, so a Codex step keeps API key variables only when billing is
    allowed; the older forcing flag still drops them even then."""
    from stepbook.executors.codex import API_CREDENTIALS, run_codex, subscription_env

    for key in API_CREDENTIALS:
        monkeypatch.setenv(key, "sk-SENTINEL-0123456789")
    monkeypatch.delenv("CODEX_FORCE_CHATGPT_AUTH", raising=False)
    monkeypatch.setenv("STEPBOOK_ALLOW_API_BILLING", "1")
    assert subscription_env() is None  # inherit, exactly as before
    for forcing in (False, True):
        if forcing:  # the bare-CLI flag asks for the ChatGPT login even where API billing is allowed
            monkeypatch.setenv("STEPBOOK_ALLOW_API_BILLING", "1")
            monkeypatch.setenv("CODEX_FORCE_CHATGPT_AUTH", "1")
        else:  # the default: nothing opted out of the subscription, so no API key rides into the step
            monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
            monkeypatch.delenv("CODEX_FORCE_CHATGPT_AUTH", raising=False)
        env = subscription_env()
        assert env is not None and not [key for key in API_CREDENTIALS if key in env]
        assert env["PATH"] == os.environ["PATH"]  # only the credentials are removed

    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
    monkeypatch.delenv("CODEX_FORCE_CHATGPT_AUTH", raising=False)
    run_codex(["codex"], "prompt", timeout_s=5, last_file=str(tmp_path / "absent"))
    assert not [key for key in API_CREDENTIALS if key in seen["env"]]


def test_run_codex_start_failure_preserves_result_contract(tmp_path, monkeypatch):
    from stepbook.executors.codex import run_codex

    def fail(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(subprocess, "run", fail)
    folded = run_codex(["unused"], "prompt", timeout_s=1, last_file=str(tmp_path / "absent"))
    assert folded == {
        "thread_id": None,
        "usage": dict.fromkeys(USAGE, 0),
        "turns": 0,
        "tool_calls": 0,
        "tool_log": [],
        "agent_message": None,
        "errors": [],
        "last_text": "",
        "answer": None,
        "returncode": 1,
        "stderr_tail": "codex could not start: Permission denied",
        "duration_s": folded["duration_s"],
    }
    assert folded["duration_s"] >= 0
