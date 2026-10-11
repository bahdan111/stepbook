"""Offline acceptance checks for flexible research; no providers or application evals."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.state.validator import PatchError, Validator
from stepbook.storage.evidence import EvidenceError, ResearchStore, parse_completion, parse_proposal

ROOT = Path(__file__).resolve().parents[1]


def note(ref):
    return {
        "summary": "Evidence-backed conclusion. " * 1000,
        "evidence": [ref],
        "uncertainty": "Unverified prediction. " * 200,
        "next_action": "Inspect then test. " * 200,
    }


def test_long_working_memory_and_new_research_metadata_preserve_authority():
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop-adapter")
    v = Validator(skill.schema, skill.protected_paths, skill.max_deletions, skill.write_once_paths)
    patch = {
        "hypotheses": {
            f"h{i}": {
                "kind": "data-preparation",
                "status": "todo",
                "rationale": "long rationale " * 100,
                "extra_research": {"pdf_pages": list(range(30))},
            }
            for i in range(45)
        },
        "lessons": {"l": "lesson " * 1000},
        "investigations": {"f": "finding " * 1000},
        "custom_research": {"question": "why?"},
    }
    new, changed = v.apply(skill.initial_state, patch)
    assert new["hypotheses"] == patch["hypotheses"] and new["custom_research"] == patch["custom_research"]
    assert new["lessons"] == patch["lessons"] and new["investigations"] == patch["investigations"]
    assert set(new) - set(skill.initial_state) == {"custom_research"}
    assert new["goal"] == skill.initial_state["goal"] and new["protocol"] == skill.initial_state["protocol"]
    assert sorted(changed) == sorted(
        [*(f"hypotheses.h{i}" for i in range(45)), "lessons.l", "investigations.f", "custom_research"]
    )
    with pytest.raises(PatchError, match="protected") as blocked:
        v.apply(new, {"goal": "secretly change goal"})
    assert str(blocked.value) == "'goal' is protected and cannot be changed by the model"
    with pytest.raises(PatchError) as typed:
        v.apply(new, {"current": {"phase": 42}})
    assert str(typed.value) == (
        "schema violation at 'current.phase': 42 is not of type 'string'; "
        "schema violation at 'current.phase': 42 is not one of "
        "['research', 'select', 'apply', 'run', 'decide', 'done']"
    )


def test_large_findings_many_citations_and_complete_source_survive_restart(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("\n".join("evidence " * 30 for _ in range(350)))
    store = ResearchStore(tmp_path / "records", (str(tmp_path),))
    source = store.read({"path": str(path), "start": 1, "end": 350})
    assert len(source["text"]) > 12000 and "350:" in source["text"] and not source["truncated"]
    finding = note(source["id"])
    finding["evidence"] *= 25
    finding["run_ids"] = ["original-run-" + str(i) for i in range(40)]
    assert parse_completion(json.dumps(finding)) == finding
    h = {"id": "h", "kind": "custom", "rationale": "r", "prediction": "p", "test": "t", "finding": finding}
    assert parse_proposal(json.dumps(h)) == h
    saved = store.record(finding)
    path.write_text("changed later")
    fresh = ResearchStore(tmp_path / "records", (str(tmp_path),))
    assert fresh.recall(saved["id"])["summary"] == finding["summary"]
    assert fresh.recall(source["id"])["text"] == source["text"]
    finding["evidence"] = ["source-" + "0" * 24]
    with pytest.raises(EvidenceError, match="unknown"):
        fresh.record(finding)


def test_explicit_resource_guard_and_line_citations_preserve_scope(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    (root / "source.py").write_text("one\ntwo\nthree\n")
    (tmp_path / "outside.py").write_text("secret")
    store = ResearchStore(tmp_path / "records", (str(root),), str(root))
    assert store.read({"path": "source.py:2-3"})["text"] == "2: two\n3: three"
    with pytest.raises(EvidenceError, match="outside"):
        store.read({"path": "../outside.py:1"})
    guarded = ResearchStore(tmp_path / "records", (str(root),), str(root), max_source_bytes=4)
    with pytest.raises(EvidenceError, match="max_source_bytes"):
        guarded.read({"path": "source.py"})


@pytest.mark.parametrize("runtime", [Runtime])
def test_no_implicit_horizon_until_agent_finishes_and_resume_does_not_replay(tmp_path, runtime, fake_adapter):
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop-adapter")
    calls = []
    env = AdapterEvalEnv(adapter=fake_adapter().command, log_dir=tmp_path, evidence_roots=(str(tmp_path),))

    def respond(*_):
        calls.append(1)
        return Envelope("inspect", {}, "add h config inspected" if len(calls) < 90 else "stop complete")

    model = ScriptedExecutor(respond)
    model.session_id = "offline-session"
    result = runtime(skill, model, env, log_dir=tmp_path).run(None)
    assert result.steps == 90 and not result.aborted and env.runs_max is None
    before = len(calls)
    runtime(skill, model, env, log_dir=tmp_path).run(None, resume=True, replay_env=False)
    assert len(calls) == before


def test_compaction_recovery_keeps_late_constraints_and_pending_work(tmp_path):
    run = tmp_path / ".stepbook" / "run"
    run.mkdir(parents=True)
    state = {
        "notes": {"old": "noise " * 4000},
        "goal": "Improve extraction",
        "user_requirements": [{"text": "Never change golden cases", "done": False}],
        "current": {"phase": "research", "h": "h1"},
        "hypotheses": {"h1": {"rationale": "Inspect page 11", "status": "testing"}},
        "pending_actions": {"a": "receipt needs reconciliation"},
    }
    (run / "state.json").write_text(json.dumps(state))
    active = tmp_path / ".stepbook"
    active.mkdir(exist_ok=True)
    (active / "ACTIVE").write_text(str(run))
    result = subprocess.run(
        [sys.executable, str(ROOT / "hooks/reinject_state.py")],
        input=json.dumps({"cwd": str(tmp_path)}),
        capture_output=True,
        text=True,
        check=True,
    )
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Never change golden cases" in context and "Inspect page 11" in context
    assert "receipt needs reconciliation" in context and "noise noise" not in context
    assert json.loads((run / "state.json").read_text()) == state


def test_schema_accepts_long_memory_with_explicit_custom_caps_still_supported():
    from tests._shared import MEMORY_SCHEMA

    v = Validator(MEMORY_SCHEMA)
    state = {"documents": {}, "allowed_changes": [], "open_questions": ""}
    patch = {
        "documents": {
            "doc": {
                "run_id": "r",
                "passed_case_ids": [str(i) for i in range(100)],
                "finding": "x" * 601,
                "evidence": ["path"] * 30,
            }
        },
        "open_questions": "q" * 2000,
    }
    new, changed = v.apply(state, patch)
    assert len(new["documents"]["doc"]["finding"]) == 601
    assert new["documents"]["doc"] == patch["documents"]["doc"]
    assert new["open_questions"] == patch["open_questions"] and new["allowed_changes"] == []
    assert changed == ["documents.doc", "open_questions"]
    custom = Validator({"type": "object", "properties": {"field": {"type": "string", "maxLength": 3}}})
    with pytest.raises(PatchError) as capped:
        custom.apply({}, {"field": "long"})
    assert str(capped.value) == (
        "schema violation at 'field': string has 4 characters (Unicode code points); maxLength is 3, excess 1."
        " Shorten this field while preserving supported facts; do not invent evidence or remove a needed update."
        " No text was trimmed or applied."
    )
    assert custom.apply({}, {"field": "abc"}) == ({"field": "abc"}, ["field"])


def test_pdf_more_than_ten_pages_can_be_read_and_recalled(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(12):
        writer.add_blank_page(width=100, height=100)
    path = tmp_path / "source.pdf"
    writer.write(path)
    store = ResearchStore(tmp_path / "records", (str(tmp_path),))
    saved = store.read({"path": str(path), "start": 1, "end": 12})
    assert "PAGE 12" in store.recall(saved["id"])["text"]


def test_no_default_source_file_ceiling(tmp_path):
    path = tmp_path / "large.txt"
    with path.open("wb") as f:
        f.write(b"first line\n")
        f.seek(33 * 1024 * 1024)
        f.write(b"\nlast line")
    store = ResearchStore(tmp_path / "records", (str(tmp_path),))
    assert store.read({"path": str(path), "end": 1})["text"] == "1: first line"


def test_plugin_subscription_guard_refuses_api_before_any_subprocess(monkeypatch):
    from stepbook.contracts.errors import ExecutorError
    from stepbook.executors.claude import ClaudeCLIExecutor

    monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-secret")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("must refuse before dispatch"))
    with pytest.raises(ExecutorError) as caught:
        ClaudeCLIExecutor().step("system", "user", {})
    assert not caught.value.retryable and caught.value.envelope.api_calls == 0
    assert "not-a-real-secret" not in str(caught.value)


def test_report_persistence_accepts_long_prose_and_line_citations(tmp_path):
    from tests import _shared as shared
    from tests._shared import answer

    workspace = shared.report_workspace(tmp_path)
    report = answer(1)
    report.update(
        summary="Detailed analysis. " * 200,
        next_test="A supported next step. " * 200,
        uncertainty="An unresolved question. " * 200,
        supporting_quote="verbatim evidence " * 30,
    )
    (workspace / "common/dev/doc.txt").write_text(report["supporting_quote"])
    report["evidence"] = ["episodes/001/results.json:1", "common/dev/doc.txt:1"]
    assert shared.source_quote_present(workspace, report)
    stream = shared.ReportEnvironment(workspace)
    report["summary"] *= 3
    action = "report " + json.dumps(report)
    assert len(action) > 14000
    result = stream.step(action)
    assert result.done and all(stream.answers[0]["checks"].values())
    saved = stream.store.recall(stream.answers[0]["finding_id"])
    assert saved["summary"] == report["summary"]
    report["supporting_quote"] = "invented quotation"
    assert not shared.source_quote_present(workspace, report)


@pytest.mark.parametrize("method, accepted", [("claude.ai", True), ("api_key", False)])
def test_bundled_auth_preflight_is_isolated_and_retains_no_identity(monkeypatch, method, accepted):
    from types import SimpleNamespace

    from stepbook.executors import auth

    for key in auth.OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(auth, "cli_path", lambda *_a, **_k: "/mock/claude")
    commands = []

    def status(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": method,
                    "apiProvider": "firstParty",
                    "subscriptionType": "max",
                    "email": "not-to-be-recorded@example.test",
                    "token": "never-record-this",
                }
            ),
        )

    monkeypatch.setattr(auth.subprocess, "run", status)
    if accepted:
        result = auth.check_subscription()
        assert set(result["auth"]) == {"loggedIn", "authMethod", "apiProvider", "subscriptionType"}
        assert "never-record-this" not in json.dumps(result)
        assert result == {
            "cli_path": "/mock/claude",
            "auth": {
                "loggedIn": True,
                "authMethod": "claude.ai",
                "apiProvider": "firstParty",
                "subscriptionType": "max",
            },
        }
        assert "not-to-be-recorded@example.test" not in json.dumps(result)
    else:
        with pytest.raises(auth.SubscriptionError) as refused:
            auth.check_subscription()
        assert str(refused.value) == (
            "Existing first-party Claude subscription login is required; this login would bill an API key, "
            "so log in with your subscription, or pass --allow-api-billing."
        )
    assert [command for command, _ in commands] == [
        ["/mock/claude", "--setting-sources", "", "auth", "status", "--json"]
    ]
    assert commands[0][1] == {"cwd": None, "capture_output": True, "text": True, "timeout": 15}


def test_shell_output_is_not_lost_before_runtime_can_archive_it(tmp_path, monkeypatch):
    from stepbook.envs.shell import ShellEnv

    output = "begin\n" + "evidence " * 2000 + "\nend"
    (tmp_path / "big.txt").write_text(output)  # a real command: _run now uses Popen in its own process group
    env = ShellEnv(tmp_path, "inspect", checkpoint=False, require_sandbox=False)
    assert env._run("cat big.txt")[1] == output


def test_generic_adapter_retains_full_stdout_and_stderr(tmp_path):
    from tests.test_generic_adapter import _adapter, _setup

    cfg = _setup(tmp_path)
    script = tmp_path / "eval.py"
    script.write_text(
        script.read_text() + '\nprint("out-begin" + "x" * 25000 + "out-end")\n'
        'print("err-begin" + "y" * 25000 + "err-end", file=sys.stderr)\n'
    )
    receipt = json.loads(_adapter(cfg, "run", "offline-fixture"))
    folder = Path(receipt["artifacts"]["run_dir"])
    assert (folder / "stdout.txt").read_text() == "out-begin" + "x" * 25000 + "out-end\n"
    assert (folder / "stderr.txt").read_text() == "err-begin" + "y" * 25000 + "err-end\n"
