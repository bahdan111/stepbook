"""Offline OpenCode step and full command-executor loop, using a fixture-replaying CLI."""

import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from stepbook import CommandExecutor, Skill
from stepbook.executors.command import EXIT_TIMED_OUT
from stepbook.runtime.loop import STATE_OUTPUT_SCHEMA
from stepbook.state.patch_schema import state_output_schema

ROOT = Path(__file__).resolve().parents[1]
# Sanitised OpenCode 1.18.33 JSON event fixtures
SAMPLES = ROOT / "tests/fixtures/opencode"
PROMPT = "<<SYSTEM>>\nsystem\n<</SYSTEM>>\nLatest Observation: ready"
PATCH = {"current": "checked", "facts": {"result": {"ok": True}}}


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    fake = tmp_path / "fake opencode"
    fake.write_text(f"#!{sys.executable}\n" + (ROOT / "tests/fixtures/fake_opencode.py").read_text())
    fake.chmod(0o755)
    log = tmp_path / "opencode.jsonl"
    for key in tuple(os.environ):
        if key.startswith("FAKE_OPENCODE_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("FAKE_OPENCODE_LOG", str(log))
    monkeypatch.setenv("FAKE_OPENCODE_FIXTURE", str(SAMPLES / "answer-only.jsonl"))
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    monkeypatch.setenv("OPENCODE_BIN", str(fake))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # the smoke repo never sees an enclosing checkout
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    return fake, log


@pytest.fixture
def runtime_schema():
    return state_output_schema(STATE_OUTPUT_SCHEMA, Skill.load(ROOT / "stepbook/templates/checklist").schema)


def run_step(tmp_path, schema, *flags):
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(schema))
    return subprocess.run(
        [sys.executable, "-m", "stepbook.executors.opencode", "step", "--schema", str(schema_file), *flags],
        input=PROMPT,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )


def fixture_events(tmp_path, monkeypatch, events):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    monkeypatch.setenv("FAKE_OPENCODE_FIXTURE", str(path))


def text_event(answer):
    return {"type": "text", "sessionID": "test-session", "part": {"text": json.dumps(answer)}}


@pytest.mark.parametrize("model", [None, "openai/gpt-5.6-sol"])
def test_answer_sample_stdin_config_and_options(fake_opencode, tmp_path, runtime_schema, monkeypatch, model):
    fake, log = fake_opencode
    flags = ["--opencode", str(fake), "--variant", "high"] if model else []
    if model:
        flags += ["--model", model]
        monkeypatch.setenv("OPENCODE_BIN", str(tmp_path / "must-not-run"))
    proc = run_step(tmp_path, runtime_schema, *flags)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] is True and out["n"] == 3
    assert out["input_tokens"] == 5919 and out["output_tokens"] == 15
    assert out["usage"] == {
        "input_tokens": 2335,
        "cached_input_tokens": 3584,
        "cache_write_input_tokens": 0,
        "output_tokens": 15,
        "reasoning_output_tokens": 0,
        "provider": "openai" if model else None,
        "model": model,
        "cost_usd": None,
        "opencode_cost": 0,
        "note": out["usage"]["note"],
    }
    assert "subscription" in out["usage"]["note"].lower() and "USD" in out["usage"]["note"]
    assert out["api_calls"] == 1 and out["tool_calls"] == 0 and out["tool_log"] == []
    assert out["session_id"] == "ses_000000000000Example0000018" and out["opencode_errors"] == []
    logged = json.loads(log.read_text())
    argv = logged["argv"]
    assert (argv[argv.index("-m") + 1] if "-m" in argv else None) == model
    assert (argv[argv.index("--variant") + 1] if "--variant" in argv else None) == ("high" if model else None)
    assert logged["prompt"] == (
        PROMPT
        + "\n\nSkill output schema:\n"
        + json.dumps(runtime_schema, indent=2)
        + "\nReturn exactly one JSON object matching this schema, no prose and no code fence.\n"
    )
    assert json.loads((tmp_path / "schema.json").read_text()) == runtime_schema


def test_multistep_sample_sums_usage_and_counts_tool(fake_opencode, tmp_path, runtime_schema, monkeypatch):
    fake, _ = fake_opencode
    monkeypatch.setenv("FAKE_OPENCODE_FIXTURE", str(SAMPLES / "read-tool-then-answer.jsonl"))
    proc = run_step(tmp_path, runtime_schema)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["top_k"] == 3 and out["ran_ls"] is False
    assert (out["input_tokens"], out["output_tokens"], out["api_calls"], out["tool_calls"]) == (8836, 133, 2, 1)
    assert out["usage"]["input_tokens"] == 4612 and out["usage"]["cached_input_tokens"] == 4224
    assert out["usage"]["output_tokens"] == 44 and out["usage"]["reasoning_output_tokens"] == 89
    assert out["tool_log"] == [{"tool": "read", "input": '{"filePath": "/work/example/config.json"}'}]


def test_cache_write_cost_last_text_and_tool_truncation(fake_opencode, tmp_path, runtime_schema, monkeypatch):
    events = [text_event({"action": "old"})]
    for i in (1, 2):
        events.append(
            {
                "type": "step_finish",
                "part": {
                    "tokens": {"input": 10 * i, "output": 3 * i, "reasoning": i, "cache": {"read": i, "write": 2 * i}},
                    "cost": 0.125 * i,
                },
            }
        )
    events += [
        {
            "type": "tool_use",
            "part": {
                "tool": "bash",
                "state": {"status": "error", "input": {"command": "x" * 300}, "output": "must not be logged"},
            },
        },
        text_event({"action": "done", "state_patch": json.dumps(PATCH), "reasoning": "checked"}),
    ]
    fixture_events(tmp_path, monkeypatch, events)
    proc = run_step(tmp_path, runtime_schema, "--model", "example/model")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["action"] == "done" and out["state_patch"] == PATCH
    assert (out["input_tokens"], out["output_tokens"]) == (39, 12)
    assert out["usage"]["cache_write_input_tokens"] == 6
    assert out["usage"]["opencode_cost"] == 0.375 and out["usage"]["cost_usd"] is None
    assert out["usage"]["provider"] == "example"
    assert out["tool_calls"] == 1 and out["tool_log"][0]["tool"] == "bash"
    assert len(out["tool_log"][0]["input"]) == 200 and "must not be logged" not in proc.stdout


def test_command_executor_decodes_all_object_fields(fake_opencode, tmp_path, runtime_schema, monkeypatch):
    runtime_schema["properties"]["report"] = {"type": ["object", "null"]}
    fixture_events(
        tmp_path,
        monkeypatch,
        [
            text_event(
                {
                    "reasoning": "checked",
                    "state_patch": json.dumps(PATCH),
                    "action": "check",
                    "report": '{"ok": true}',
                }
            )
        ],
    )
    command = shlex.join(
        [sys.executable, "-m", "stepbook.executors.opencode", "step", "--opencode", str(fake_opencode[0])]
    )
    envelope = CommandExecutor(command=command + " --schema {schema_file}", cwd=ROOT).step(
        "SYS", "ready", runtime_schema
    )
    assert envelope.state_patch == PATCH and envelope.extra["report"] == {"ok": True}
    assert envelope.api_calls == 1 and envelope.raw["cost_unknown"] is True
    Draft202012Validator(runtime_schema).validate(
        {
            "reasoning": envelope.reasoning,
            "state_patch": envelope.state_patch,
            "action": envelope.action,
            "report": envelope.extra["report"],
        }
    )


@pytest.mark.parametrize(
    "mode,code,message",
    [
        ("prose", 3, "no JSON answer"),
        ("error", 1, "fixture failure"),
        ("exit", 7, "process failed"),
        ("both", 9, "fixture failure"),
    ],
)
def test_failures_keep_messages_and_exit_codes(
    fake_opencode, tmp_path, runtime_schema, monkeypatch, mode, code, message
):
    events = [text_event({"action": "old"})]
    if mode == "prose":
        events.append({"type": "text", "part": {"text": "no answer here"}})
    if mode in ("error", "both"):
        events.append({"type": "error", "error": {"name": "APIError", "data": {"message": "fixture failure"}}})
    if mode in ("exit", "both"):
        monkeypatch.setenv("FAKE_OPENCODE_EXIT", str(code))
        monkeypatch.setenv("FAKE_OPENCODE_STDERR", "process failed")
    fixture_events(tmp_path, monkeypatch, events)
    proc = run_step(tmp_path, runtime_schema)
    assert proc.returncode == code and proc.stdout == ""
    assert message in proc.stderr


@pytest.mark.parametrize("permissions,value", [("read-only", "deny"), ("full", "allow")])
def test_permissions_merge_only_owned_config(fake_opencode, tmp_path, runtime_schema, monkeypatch, permissions, value):
    # Learn the wrapper-owned name from an initial call without fixing its spelling in the fake.
    fake, log = fake_opencode
    assert run_step(tmp_path, runtime_schema).returncode == 0
    first = json.loads(log.read_text())
    name = first["argv"][first["argv"].index("--agent") + 1]
    original = {
        "autoupdate": True,
        "share": "auto",
        "model": "configured/model",
        "permission": {"read": "allow"},
        "agent": {name: {"mode": "subagent", "permission": {"bash": "ask"}}, "user-agent": {"description": "keep"}},
        "provider": {"example": {"options": {"baseURL": "https://example.invalid"}}},
    }
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(original))
    monkeypatch.setenv("FAKE_OPENCODE_PERMISSION", value)
    proc = run_step(tmp_path, runtime_schema, "--permissions", permissions)
    assert proc.returncode == 0, proc.stderr
    merged = json.loads(log.read_text().splitlines()[-1])["config"]
    expected = {
        **original,
        "autoupdate": False,
        "share": "disabled",
        "agent": {
            **original["agent"],
            name: {**first["config"]["agent"][name], "permission": dict.fromkeys(("edit", "bash", "webfetch"), value)},
        },
    }
    assert merged == expected
    assert json.loads(os.environ["OPENCODE_CONFIG_CONTENT"]) == original


@pytest.mark.parametrize("config", ['{"private": "do-not-print",', "[]", '{"agent": []}'])
def test_invalid_config_fails_without_launch_or_config_leak(
    fake_opencode, tmp_path, runtime_schema, monkeypatch, config
):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", config)
    proc = run_step(tmp_path, runtime_schema)
    assert proc.returncode != 0 and proc.stdout == ""
    assert "OPENCODE_CONFIG_CONTENT" in proc.stderr and "do-not-print" not in proc.stderr
    assert not fake_opencode[1].exists()


def test_timeout_and_help(fake_opencode, tmp_path, runtime_schema, monkeypatch):
    monkeypatch.setenv("FAKE_OPENCODE_SLEEP", "2")
    proc = run_step(tmp_path, runtime_schema, "--timeout", "0.1")
    assert proc.returncode == EXIT_TIMED_OUT and "timed out" in proc.stderr and proc.stdout == ""
    help_proc = subprocess.run(
        [sys.executable, "-m", "stepbook.executors.opencode", "step", "--help"], capture_output=True, text=True
    )
    assert help_proc.returncode == 0 and "externally isolated" in help_proc.stdout


def test_a_plain_failure_is_still_an_ordinary_retryable_exit(fake_opencode, tmp_path, runtime_schema, monkeypatch):
    """A failure other than the turn timeout keeps its own exit code and stays an ordinary retryable failure."""
    fixture_events(tmp_path, monkeypatch, [{"type": "error", "error": {"data": {"message": "provider crashed"}}}])
    monkeypatch.setenv("FAKE_OPENCODE_EXIT", "5")
    proc = run_step(tmp_path, runtime_schema)
    assert proc.returncode == 5 and "provider crashed" in proc.stderr


def test_the_turn_timeout_is_shorter_than_the_host_limit(tmp_path, runtime_schema, monkeypatch, capsys):
    """The host kills the driver at its own limit, so the OpenCode turn must give up 30 s earlier and say so."""
    from stepbook.executors import opencode as driver

    seen = []

    def fake_run(argv, **kwargs):
        seen.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "stdin", io.StringIO("prompt"))
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(runtime_schema))
    assert driver.main(["step", "--schema", str(schema_file), "--timeout", "900"]) == EXIT_TIMED_OUT
    assert seen == [870.0]
    assert "opencode timed out after 870s" in capsys.readouterr().err


def test_object_decoding_preserves_invalid_values_for_validation():
    from stepbook.executors.opencode import decode_answer

    schema = {
        "properties": {
            "report": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "map": {"additionalProperties": {"type": "string"}},
            "prose": {"type": "string"},
        }
    }
    answer = {"state_patch": "not JSON", "report": '{"ok":true}', "map": "{}", "prose": '{"leave":"alone"}'}
    assert decode_answer(answer, schema) == {
        "state_patch": "not JSON",
        "report": {"ok": True},
        "map": {},
        "prose": '{"leave":"alone"}',
    }
    assert decode_answer({"state_patch": "[]"}, schema) == {"state_patch": "[]"}


def test_parser_noise_fences_empty_object_and_error_message():
    from stepbook.executors.opencode import _last_fenced_object, parse_events

    assert _last_fenced_object('progress {broken\n{"old": 1}\n```json\n{"state_patch": {"nested": "}"}}\n```') == {
        "state_patch": {"nested": "}"},
    }
    assert _last_fenced_object("{}") == {}
    assert _last_fenced_object("no JSON") is None
    folded = parse_events('progress\n{broken\n[]\nnull\n{"type":"error","message":"failed"}\n')
    assert folded.errors == ["failed"]


def test_cli_checklist_reaches_done_in_temporary_git_repo(fake_opencode, tmp_path, monkeypatch):
    fake, log = fake_opencode
    monkeypatch.setenv("FAKE_OPENCODE_SMOKE", "1")
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work, check=True, capture_output=True)
    (work / "result.txt").write_text("verified\n")
    run = tmp_path / "run"
    command = shlex.join([sys.executable, "-m", "stepbook.executors.opencode", "step", "--opencode", str(fake)])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "stepbook.cli",
            "run",
            str(ROOT / "stepbook/templates/checklist"),
            "--workdir",
            str(work),
            "--goal",
            "verify result.txt",
            "--check",
            "verified=grep -qx verified result.txt",
            "--steps",
            "2",
            "--executor",
            "command",
            "--command",
            command + " --schema {schema_file}",
            "--log-dir",
            str(run),
            "--quiet",
            "--no-checkpoint",
            "--unsandboxed-i-accept-the-risk",
        ],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    summary = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert summary["score"] == 1.0 and summary["steps"] == 2 and summary["rejected_patches"] == 0
    steps = [
        json.loads(line)
        for line in (run / "events.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "step"
    ]
    assert [step["action"] for step in steps] == ["check", "done"]
    assert steps[-1]["done"] is True and steps[-1]["info"]["checks"] == {"verified": "pass"}
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2 and all(Path(call["cwd"]) == work for call in calls)
