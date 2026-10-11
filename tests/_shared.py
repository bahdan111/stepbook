"""Offline fixtures shared by live tests.

Provide synthetic source documents, expected answers, and report and memory schemas without a live runner."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from stepbook.runtime.loop import Skill, StepResult
from stepbook.storage.events import EventLog
from stepbook.storage.evidence import ResearchStore


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fixture(tmp_path):
    source = tmp_path / "original/combined"
    workspace = source / "workspace"
    common = {
        "application_reference.py": "application code",
        "scoring_reference.py": "scorer code",
        "dev/doc.txt": "source evidence",
        "dev/doc.pdf": "PDF fixture",
    }
    for name, text in common.items():
        p = workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    write_json(workspace / "dev/cases.json", [{"uid": "q", "context_id": "doc", "pdf": "doc.pdf"}])
    common["dev/cases.json"] = ""
    (workspace / "notes.md").write_text("OLD AGENT HINT")
    write_json(source / "final-config.json", {"prompt": "FUTURE CHOICE"})
    log = EventLog(source / "runtime/events.jsonl")
    for i in range(3):
        d = workspace / ".stepbook-adapter/runs" / f"r{i}"
        write_json(d / "config.json", {"page_limit": i + 1, "prompt": "Answer"})
        write_json(
            d / "results.json",
            [
                {
                    "id": "q",
                    "context_id": "doc",
                    "pass": None if i == 1 else True,
                    "prediction": None if i == 1 else {"answer": "evidence"},
                }
            ],
        )
        write_json(d / "metrics.json", {"accuracy": 0 if i == 1 else 1})
        (d / "stdout.txt").write_text("UNNEEDED EXECUTION TRACE")
        row = {
            "run_id": f"r{i}",
            "applied": None if i == 0 else f"h{i}",
            "comparison_key": "same",
            "primary_metric": {"name": "accuracy", "value": 0 if i == 1 else 1},
            "cases": {"q": "error" if i == 1 else "pass"},
            "valid": i != 1,
            "cost": {"usd": 0.2, "tokens": 30},
            "provenance": {
                "config_sha256": digest(d / "config.json"),
                "protected_files": {n: digest(workspace / n) for n in common},
            },
        }
        log.append(
            "step",
            action=f"run r{i}",
            result="Eval finished:\n" + json.dumps(row),
            reasoning="HISTORICAL REASONING",
            state_patch={"answer": "DO NOT COPY"},
        )
    write_json(
        source / "FROZEN.json",
        {"completed": True, "undecided": None, "event_count": len(log.read()), "journal_sha256": digest(log.path)},
    )
    return source


def answer(n):
    return {
        "run_id": f"r{n - 1}",
        "document_id": "doc",
        "passed_case_ids": [] if n == 2 else ["q"],
        "failed_case_ids": ["q"] if n == 2 else [],
        "prior_run_id": None if n == 1 else f"r{n - 2}",
        "newly_passing_ids": ["q"] if n == 3 else [],
        "newly_failing_ids": ["q"] if n == 2 else [],
        "evaluation_valid": n != 2,
        "allowed_changes": ["page_limit", "prompt"],
        "summary": "Saved predictions include missing output in the second evaluation.",
        "supporting_quote": "source evidence",
        "next_test": "Inspect response parsing before changing the prompt.",
        "uncertainty": "No new test performed",
        "proposal_status": "untested",
        "evidence": [f"episodes/{n:03d}/results.json", "common/dev/doc.txt"],
    }


REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        **{
            key: {"type": "string"}
            for key in (
                "run_id",
                "document_id",
                "summary",
                "supporting_quote",
                "next_test",
                "uncertainty",
                "proposal_status",
            )
        },
        **{
            key: {"type": "array", "items": {"type": "string"}}
            for key in (
                "passed_case_ids",
                "failed_case_ids",
                "newly_passing_ids",
                "newly_failing_ids",
                "allowed_changes",
                "evidence",
            )
        },
        "prior_run_id": {"type": ["string", "null"]},
        "evaluation_valid": {"type": "boolean"},
    },
    "required": [
        "run_id",
        "document_id",
        "summary",
        "supporting_quote",
        "next_test",
        "uncertainty",
        "proposal_status",
        "passed_case_ids",
        "failed_case_ids",
        "newly_passing_ids",
        "newly_failing_ids",
        "allowed_changes",
        "evidence",
        "prior_run_id",
        "evaluation_valid",
    ],
}

REPORT_SCHEMA["properties"]["supporting_quote"]["description"] = (
    "Verbatim source text only. Put filenames, page/line citations and commentary in evidence or summary, "
    "not inside this quotation. Whitespace from line wrapping may differ; words, numbers and punctuation must match."
)
REPORT_SCHEMA["properties"]["evidence"]["description"] = (
    "Source paths, separate from supporting_quote. Include the current results file and cited document. "
    "Optional :line or :start-end suffixes refer to text/code lines, not PDF page numbers."
)


MEMORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "documents": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "run_id": {"type": "string"},
                    "passed_case_ids": {"type": "array", "items": {"type": "string"}},
                    "finding": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["run_id", "passed_case_ids", "finding", "evidence"],
            },
        },
        "allowed_changes": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "string"},
    },
    "required": ["documents", "allowed_changes", "open_questions"],
}


def report_workspace(tmp_path):
    """Expose a single offline source and result through ordinary files."""
    source = fixture(tmp_path) / "workspace"
    workspace = tmp_path / "workspace"
    shutil.copytree(source / "dev", workspace / "common/dev")
    write_json(
        workspace / "episodes/001/results.json",
        json.loads((source / ".stepbook-adapter/runs/r0/results.json").read_text()),
    )
    return workspace


def report_skill():
    return Skill(
        "offline-report",
        "Inspect the source and return a report object.",
        MEMORY_SCHEMA,
        {"documents": {}, "allowed_changes": ["page_limit", "prompt"], "open_questions": ""},
    )


def source_quote_present(workspace, report):
    quote = report.get("supporting_quote")
    if not isinstance(quote, str) or not quote.strip():
        return False
    for citation in report.get("evidence", []):
        if re.fullmatch(r"common/dev/doc\.txt(?::[1-9][0-9]*(?:-[1-9][0-9]*)?)?", citation):
            source = (workspace / "common/dev/doc.txt").read_text()
            if " ".join(quote.split()) in " ".join(source.split()):
                return True
    return False


class ReportEnvironment:
    """Single-report test environment using the public evidence persistence API."""

    response_report_schema = REPORT_SCHEMA

    def __init__(self, workspace):
        self.workspace = workspace
        self.answers = []
        self.store = ResearchStore(workspace / "findings", roots=(str(workspace),), base=str(workspace))

    def reset(self):
        return "Inspect common/dev/doc.txt and episodes/001/results.json, then return a report."

    def step(self, action):
        assert action.startswith("report ")
        report = json.loads(action.removeprefix("report "))
        finding = self.store.record(
            {
                "summary": report["summary"],
                "evidence": report["evidence"],
                "uncertainty": report["uncertainty"],
                "next_action": report["next_test"],
            }
        )
        self.answers.append(
            {
                "answer": report,
                "finding_id": finding["id"],
                "checks": {"source_quote": source_quote_present(self.workspace, report), "finding_saved": True},
            }
        )
        return StepResult("Report received.", done=True)

    def score(self):
        return float(bool(self.answers) and all(self.answers[0]["checks"].values()))
