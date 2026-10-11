import hashlib
import json
from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.storage.evidence import EvidenceError, ResearchStore


def finding(source):
    return {
        "summary": "Expected value is in another section",
        "evidence": [source["id"]],
        "uncertainty": "Need to verify the prompt context",
        "next_action": "Read the truncation code",
    }


def test_source_identity_survives_changes_and_restart(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("first\nsecond\nthird")
    store = ResearchStore(tmp_path / "research", (str(tmp_path),))
    source = store.read({"path": str(path), "start": 2, "end": 2})
    assert source["text"] == "2: second"
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = store.record(finding(source))
    assert store.record(finding(source))["id"] == saved["id"]
    path.write_text("different")
    fresh = ResearchStore(tmp_path / "research", (str(tmp_path),))
    assert fresh.recall(source["id"])["text"] == "2: second"
    assert fresh.recall(saved["id"])["next_action"] == "Read the truncation code"
    assert fresh.read({"path": str(path)})["id"] != source["id"]


def test_record_file_is_indented_and_keeps_the_body_key_order_the_id_cannot_pin(tmp_path):
    """The id hashes a sort_keys encoding (stepbook/storage/evidence.py:77), so it is blind to key order; the file is
    written with indent=2 in insertion order (evidence.py:89), which is what a reader of the record sees."""
    path = tmp_path / "input.txt"
    path.write_text("first\nsecond\nthird")
    store = ResearchStore(tmp_path / "research", (str(tmp_path),))
    source = store.read({"path": str(path), "start": 2, "end": 2})
    body = finding(source)
    saved = store.record(dict(body))
    on_disk = (tmp_path / "research" / (saved["id"] + ".json")).read_text(encoding="utf-8")
    assert list(json.loads(on_disk)) == ["id", "kind", "summary", "evidence", "uncertainty", "next_action"]
    assert on_disk.startswith(
        f'{{\n  "id": "{saved["id"]}",\n  "kind": "finding",\n  "summary": "{body["summary"]}",\n'
        f'  "evidence": [\n    "{source["id"]}"\n  ],\n'
    )
    # Same fields in the reverse order get the same id, and the bytes stay put only because save() skips the write
    # when the record file already exists (evidence.py:87) and compares parsed dicts (evidence.py:100). In a fresh
    # store the same reversed body writes different bytes under that same id, so only the assertions above catch it.
    again = store.record({k: body[k] for k in reversed(list(body))})
    assert again["id"] == saved["id"] and list(again)[2:] == ["next_action", "uncertainty", "evidence", "summary"]
    assert (tmp_path / "research" / (saved["id"] + ".json")).read_text(encoding="utf-8") == on_disk


@pytest.mark.parametrize("accepted", [True, False])
def test_inline_proposal_preserves_source_finding_and_registration_in_one_action(
    tmp_path, accepted, fake_adapter, research_action
):
    (tmp_path / "source.txt").write_text("Inspect the second page")
    adapter = fake_adapter({"register": "Registered h1" if accepted else "Error: candidate policy"})
    env = AdapterEvalEnv(
        adapter=adapter.command, log_dir=tmp_path, evidence_roots=(str(tmp_path),), evidence_cwd=str(tmp_path)
    )
    h = {
        "id": "h1",
        "kind": "config",
        "rationale": "more evidence",
        "prediction": "higher accuracy",
        "test": "two paired evals",
        "set": {"limit": 2},
        "finding": {
            "summary": "Context is truncated",
            "evidence": ["source.txt"],
            "uncertainty": "may not fix all cases",
            "next_action": "measure",
        },
    }
    result = env.step("propose " + json.dumps(h))
    assert result.info["ok"] == accepted and len(adapter.calls) == 1
    submitted = json.loads(adapter.calls[0][1])
    finding = research_action(env, "recall", submitted["finding"])
    assert research_action(env, "recall", finding["evidence"][0])["sha256"]
    registration = research_action(env, "recall", result.info["research_id"])
    assert registration["accepted"] == accepted and registration["finding_id"] == finding["id"]
    assert research_action(env, "recall", registration["proposal_id"])["proposal"]["finding"] == finding["id"]


def test_read_cannot_escape_roots_even_through_symlink(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    (root / "link").symlink_to(outside)
    store = ResearchStore(tmp_path / "research", (str(root),))
    for path in (outside, root / ".." / "secret.txt", root / "link"):
        with pytest.raises(EvidenceError, match="outside") as refused:
            store.read({"path": str(path)})
        assert str(refused.value) == "source is outside --evidence-root; reads never grant themselves new roots"
    with pytest.raises(EvidenceError) as bad_id:
        store.recall("../../secret")
    assert str(bad_id.value) == "invalid research record id"
    with pytest.raises(EvidenceError, match="unknown research") as unknown:
        store.record(finding({"id": "source-" + "0" * 24}))
    assert str(unknown.value) == "unknown research record source-" + "0" * 24
    # None of the refusals leaked the secret into a record, and a file inside a root still reads.
    assert "private" not in json.dumps([path.read_text() for path in (tmp_path / "research").rglob("*.json")])
    (root / "inside.txt").write_text("visible")
    assert store.read({"path": str(root / "inside.txt")})["text"] == "1: visible"


def test_index_paginates_without_losing_old_findings(tmp_path):
    store = ResearchStore(tmp_path / "research")
    ids = {store.save("source", {"text": str(n)})["id"] for n in range(24)}
    page = store.recall()
    second = store.recall(page["next"])
    assert page["total"] == 24 and second["next"] is None
    assert {r["id"] for r in page["records"] + second["records"]} == ids


def test_native_tool_locator_is_snapshotted_when_recording(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("limit = 1\n")
    store = ResearchStore(tmp_path / "research", (str(tmp_path),))
    record = store.record(
        {
            "summary": "Context is limited",
            "evidence": [{"path": str(source), "start": 1, "end": 1}],
            "uncertainty": "Effect needs measurement",
            "next_action": "Try another page",
        }
    )
    assert store.recall(record["evidence"][0])["text"] == "1: limit = 1"


def test_relative_source_uses_explicit_tool_cwd_without_expanding_authority(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("limit = 1")
    (tmp_path / "private.txt").write_text("hidden")
    store = ResearchStore(tmp_path / "research", (str(workspace),), str(workspace))
    assert store.read({"path": "app.py"})["text"] == "1: limit = 1"
    with pytest.raises(EvidenceError, match="outside"):
        store.read({"path": "../private.txt"})


def test_record_file_strings_use_same_authority_and_snapshot(tmp_path):
    workspace = tmp_path / "allowed"
    workspace.mkdir()
    (workspace / "app.py").write_text("limit = 1")
    (tmp_path / "secret.txt").write_text("private")
    store = ResearchStore(tmp_path / "research", (str(workspace),), str(workspace))
    saved = store.record(finding({"id": "app.py"}))
    source = store.recall(saved["evidence"][0])
    assert source["text"] == "1: limit = 1" and source["sha256"]
    (workspace / "app.py").unlink()
    assert store.recall(source["id"])["text"] == "1: limit = 1"
    with pytest.raises(EvidenceError, match="outside"):
        store.record(finding({"id": "../secret.txt"}))
    # An absent canonical id is an absent record, never interpreted as a filename.
    with pytest.raises(EvidenceError, match="unknown research"):
        store.record(finding({"id": "source-" + "0" * 24}))


def test_authoritative_frontier_survives_without_a_model_state_patch(tmp_path, research_action):
    (tmp_path / "source.txt").write_text("a fact")
    env = AdapterEvalEnv(adapter="true", log_dir=tmp_path, evidence_roots=(str(tmp_path),), evidence_cwd=str(tmp_path))
    source = research_action(env, "read", {"path": "source.txt"})
    saved = research_action(env, "record", finding(source))
    fresh = AdapterEvalEnv(
        adapter="true", log_dir=tmp_path, evidence_roots=(str(tmp_path),), evidence_cwd=str(tmp_path)
    )
    assert saved["id"] in fresh.context()
    frontier = json.loads(fresh.context().split("\n", 1)[1])
    assert not {"mine_enabled", "code_changes_enabled"} & frontier.keys()
    assert research_action(fresh, "recall", saved["id"])["evidence"] == [source["id"]]


def test_pdf_and_xlsx_evidence_locators(tmp_path):
    from openpyxl import Workbook
    from pypdf import PdfWriter

    pdf = tmp_path / "document.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(pdf)
    wb = Workbook()
    wb.active.title = "Field checks"
    wb.active.append(["field", "expected", "actual"])
    wb.active.append(["date", "2025-01-02", None])
    xlsx = tmp_path / "report.xlsx"
    wb.save(xlsx)
    store = ResearchStore(tmp_path / "research", (str(tmp_path),))
    page = store.read({"path": str(pdf), "start": 1, "end": 1})
    assert page["locator"]["unit"] == "page" and "OCR/visual" in page["text"]
    assert store.read({"path": str(xlsx)})["sheets"] == ["Field checks"]
    row = store.read({"path": str(xlsx), "sheet": "Field checks", "start": 2, "end": 2})
    assert "2025-01-02" in row["text"] and row["locator"]["start"] == 2


def test_controller_investigates_and_registers_without_initial_hypotheses(tmp_path, fake_adapter, research_action):
    adapter = fake_adapter({"register": ["Registered next1", "Error: duplicate hypothesis"]})
    env = AdapterEvalEnv(adapter=adapter.command, log_dir=tmp_path, evidence_roots=(str(tmp_path),))
    (tmp_path / "code.txt").write_text("pages = document[:1]")
    source = research_action(env, "read", {"path": str(tmp_path / "code.txt")})
    saved = research_action(env, "record", finding(source))
    proposal = {
        "id": "next1",
        "kind": "config",
        "rationale": "Check context coverage",
        "set": {"pages": 2},
        "finding": saved["id"],
        "prediction": "Recover missing fields",
        "test": "Repeat dev eval; check regressions",
    }
    result = env.step("propose " + json.dumps(proposal))
    assert result.info["ok"] and adapter.calls[-1][0] == "register"
    assert research_action(env, "recall", result.info["research_id"])["accepted"]
    result = env.step("propose " + json.dumps(proposal))
    assert result.info["ok"] is False and "not registered" in result.observation


def test_runtime_restart_recalls_evidence_without_reexecuting_reads(tmp_path, research_action):
    path = tmp_path / "doc.txt"
    path.write_text("A source fact")
    skill = Skill.load(Path(__file__).resolve().parents[1] / "stepbook/templates/eval-loop-adapter")
    env = AdapterEvalEnv(adapter="true", log_dir=tmp_path, evidence_roots=(str(tmp_path),))
    rt = Runtime(
        skill,
        ScriptedExecutor(lambda *_: Envelope("read", {}, "read " + json.dumps({"path": str(path)}))),
        env,
        log_dir=tmp_path,
    )
    rt.run(1)
    source = research_action(env, "recall")["records"][0]["id"]
    path.unlink()
    env2 = AdapterEvalEnv(adapter="true", log_dir=tmp_path, evidence_roots=(str(tmp_path),), resuming=True)
    result = Runtime(
        skill, ScriptedExecutor(lambda *_: Envelope("recall", {}, "recall " + source)), env2, log_dir=tmp_path
    ).run(2, resume=True, replay_env=False)
    assert result.steps == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "A source fact" in [e for e in events if e["kind"] == "step"][-1]["result"]


def test_a_non_string_supersedes_escapes_the_step_when_truthy_and_is_stored_when_falsy(tmp_path, research_action):
    """`supersedes` is opaque: a truthy non-string reaches `recall` and must produce an ordinary step error.
    A falsy value bypasses `recall` and must be stored and read back unchanged."""
    (tmp_path / "source.txt").write_text("a fact")
    env = AdapterEvalEnv(adapter="true", log_dir=tmp_path, evidence_roots=(str(tmp_path),), evidence_cwd=str(tmp_path))
    source = research_action(env, "read", {"path": "source.txt"})
    refused = env.step("record " + json.dumps(finding(source) | {"supersedes": 5}))
    assert refused.info == {"ok": False} and not refused.done
    assert refused.observation == "Error: 'int' object has no attribute 'startswith'"
    assert not list((tmp_path / "research").glob("finding-*.json")), "the refused document saved no finding"
    for falsy in (0, [], {}):
        saved = research_action(env, "record", finding(source) | {"supersedes": falsy})
        on_disk = json.loads((tmp_path / "research" / (saved["id"] + ".json")).read_text(encoding="utf-8"))
        # json.dumps so 0 is not read as false; the id digests the body, so each falsy value files its own record.
        assert json.dumps(on_disk["supersedes"]) == json.dumps(falsy)
        assert json.dumps(research_action(env, "recall", saved["id"])["supersedes"]) == json.dumps(falsy)
