"""Offline author CLI contracts through both product executors."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from stepbook.cli import author, main
from stepbook.cli.author import AUTHOR_FROM_SCHEMA, AUTHOR_FROM_SYSTEM, AUTHOR_SCHEMA, AUTHOR_SYSTEM

DRAFT = {
    "reasoning": "draft",
    "action": "review",
    "skill_md": "Review the procedure.\n",
    "schema_json": {
        "type": "object",
        "properties": {"goal": {"type": "string"}, "notes": {"type": "object"}},
        "required": ["goal"],
        "x-stepbook": {"protected": ["goal"]},
    },
    "initial_state": {"goal": "review"},
}


ROOT = Path(__file__).resolve().parents[1]


def completed_schema(schema):
    expected = copy.deepcopy(schema)
    checklist = json.loads((ROOT / "stepbook/templates/checklist/schema.json").read_text())
    for key in ("goal", "constraints", "checks", "checks_status"):
        expected.setdefault("properties", {}).setdefault(key, checklist["properties"][key])
    protected = expected.setdefault("x-stepbook", {}).setdefault("protected", [])
    for key in ("goal", "constraints", "checks"):
        if key not in protected:
            protected.append(key)
    return expected


@pytest.fixture
def draft_executor(monkeypatch):
    calls = []
    draft = copy.deepcopy(DRAFT)

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        result = {"structured_output": draft} if cmd[0] == "claude" else draft
        return subprocess.CompletedProcess(cmd, 0, json.dumps(result), "")

    monkeypatch.setattr("subprocess.run", run)
    return draft, calls


@pytest.mark.parametrize(
    "options",
    [
        ["--executor", "cli", "--model", "fixture"],
        ["--executor", "command", "--command", "offline-agent"],
        ["--command", "offline-agent"],  # --command alone implies --executor command
    ],
)
def test_author_validates_draft(tmp_path, draft_executor, options):
    draft, calls = draft_executor
    target = tmp_path / "new"
    assert main(["author", str(target), "review a procedure", *options]) == 0
    assert main(["validate", str(target)]) == 0
    assert len(calls) == 1
    assert sorted(p.name for p in target.iterdir()) == ["SKILL.md", "initial_state.json", "schema.json"]
    assert (target / "SKILL.md").read_text() == draft["skill_md"]
    assert json.loads((target / "schema.json").read_text()) == completed_schema(draft["schema_json"])
    assert json.loads((target / "initial_state.json").read_text()) == draft["initial_state"]
    if options[1] == "cli":
        assert calls[0][0][calls[0][0].index("--model") + 1] == "fixture"
    else:
        assert "review a procedure" in calls[0][1]["input"]


@pytest.mark.parametrize("kind", ["folder", "file", "symlink", "dangling"])
def test_author_refuses_existing_target_without_call(tmp_path, draft_executor, kind):
    _, calls = draft_executor
    target = tmp_path / "existing"
    if kind == "folder":
        target.mkdir()
        (target / "keep").write_text("original")
    elif kind == "file":
        target.write_text("original")
    else:
        other = tmp_path / "other"
        if kind == "symlink":
            other.mkdir()
        target.symlink_to(other)
    assert main(["author", str(target), "review"]) == 2
    assert calls == []
    if kind == "folder":
        assert (target / "keep").read_text() == "original"
    elif kind == "file":
        assert target.read_text() == "original"
    else:
        assert target.is_symlink()


@pytest.mark.parametrize("change", ["missing", "state", "schema", "lock"])
def test_author_refuses_invalid_draft(tmp_path, draft_executor, change):
    draft, calls = draft_executor
    if change == "missing":
        del draft["skill_md"]
    elif change == "state":
        draft["initial_state"] = {"goal": 7}
    elif change == "schema":
        draft["schema_json"]["properties"]["goal"]["type"] = "bogus"
    else:
        draft["schema_json"]["x-stepbook"]["protected"] = ["goal..x"]
    target = tmp_path / "new"
    assert main(["author", str(target), "review"]) == 2
    assert len(calls) == 1
    assert not target.exists()


@pytest.mark.parametrize(
    "options", [["--executor", "command"], ["--executor", "command", "--command", "offline-agent", "--model", "opus"]]
)
def test_author_executor_refusals(tmp_path, draft_executor, options):
    _, calls = draft_executor
    assert main(["author", str(tmp_path / "new"), "review", *options]) == 2
    assert calls == []


@pytest.mark.parametrize("from_source", [False, True])
def test_author_refuses_target_created_during_draft(tmp_path, from_source):
    from stepbook import Envelope, ScriptedExecutor
    from stepbook.cli.author import author_skill, author_skill_from

    target = tmp_path / "new"
    source = tmp_path / "SKILL.md"
    source.write_bytes(b"Keep this text.\r\n")

    def respond(*_):
        target.mkdir()
        (target / "keep").write_text("concurrent writer")
        return Envelope("draft", None, "review", extra={**DRAFT, "run_section": RUN_SECTION})

    with pytest.raises(FileExistsError):
        if from_source:
            author_skill_from(source, target, ScriptedExecutor(respond))
        else:
            author_skill("review", target, ScriptedExecutor(respond))
    assert source.read_bytes() == b"Keep this text.\r\n"
    assert sorted(p.name for p in target.iterdir()) == ["keep"]
    assert (target / "keep").read_text() == "concurrent writer"


def test_author_failed_executor_creates_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "offline failure"))
    target = tmp_path / "new"
    assert main(["author", str(target), "review", "--executor", "command", "--command", "offline-agent"]) == 2
    assert not target.exists()


FROM_OPTIONS = [["--executor", "cli", "--model", "fixture"], ["--executor", "command", "--command", "offline-agent"]]
ENVIRONMENT = (
    "Environment: one action per step: a shell command line run in the project's working directory, "
    "`check` or `check <id>`, `wait <max> until <command> [every <interval>]`, "
    "`done` (runs the configured checks and ends the run) or `stop <reason>`. "
    "The agent sees the action output as the observation. `delegate` is refused by the environment."
)
RUN_SECTION = "Use one shell command per step. Record progress in notes; return done or stop <reason>."


@pytest.fixture
def from_executor(draft_executor):
    draft, calls = draft_executor
    del draft["skill_md"]
    draft["run_section"] = "\n  " + RUN_SECTION + "  \n"
    return draft, calls


def assert_from_prompt(call, source, original, options, notes=None):
    cmd, kwargs = call
    prompt = f"Existing skill (kept verbatim; source: {source.resolve()}):\n<<<SKILL\n{original.decode()}\nSKILL>>>\n\n"
    if notes is not None:
        prompt += f"Notes from the user:\n{notes}\n\n"
    prompt += ENVIRONMENT
    if options[1] == "cli":
        assert cmd[-1] == prompt
        assert cmd[cmd.index("--system-prompt") + 1] == AUTHOR_FROM_SYSTEM
        assert json.loads(cmd[cmd.index("--json-schema") + 1]) == AUTHOR_FROM_SCHEMA
        assert "run_section" in cmd[cmd.index("--json-schema") + 1]
    else:
        assert prompt in kwargs["input"]
        assert AUTHOR_FROM_SYSTEM in kwargs["input"]
        assert "verbatim as the first part of the instructions and must not be rewritten or repeated" in kwargs["input"]


@pytest.mark.parametrize("options", FROM_OPTIONS)
@pytest.mark.parametrize("kind", ["file", "folder"])
@pytest.mark.parametrize(
    "original",
    [
        b"Review the procedure.\n",
        b"Review the procedure.",
        "---\nname: review\ndescription: Résumé\n---\nKeep this text.\n\n".encode(),
        b"---\r\nname: review\r\n---\r\nKeep CRLF.\r\n",
    ],
)
def test_author_from_preserves_source_and_validates(tmp_path, from_executor, capsys, options, kind, original):
    draft, calls = from_executor
    folder = tmp_path / "source"
    folder.mkdir()
    source = folder / ("instructions.txt" if kind == "file" else "SKILL.md")
    source.write_bytes(original)
    mtime = source.stat().st_mtime_ns
    target = tmp_path / "new"
    assert main(["author", str(target), "--from", str(source if kind == "file" else folder), *options]) == 0
    assert capsys.readouterr().out == (
        f"drafted skill at {target} from {source.resolve()}. Your SKILL.md text is unchanged at the top of {target}/SKILL.md; "
        f"review the added section and schema.json, then: stepbook validate {target}\n"
    )
    expected = original + (b"" if original.endswith(b"\n") else b"\n")
    expected += (
        "\n## Running this skill one step at a time\n"
        f"This skill's own files (scripts, references) are in {source.resolve().parent}. "
        "Shell commands run in the project's working directory, not in that folder.\n"
        f"\n{RUN_SECTION}\n"
    ).encode()
    assert (target / "SKILL.md").read_bytes().startswith(original)
    assert (target / "SKILL.md").read_bytes() == expected
    assert json.loads((target / "schema.json").read_bytes()) == completed_schema(draft["schema_json"])
    assert json.loads((target / "initial_state.json").read_bytes()) == draft["initial_state"]
    assert main(["validate", str(target)]) == 0
    assert len(calls) == 1
    assert_from_prompt(calls[0], source, original, options)
    assert source.read_bytes() == original
    assert source.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("options", FROM_OPTIONS)
def test_author_from_description_is_notes(tmp_path, from_executor, options):
    _, calls = from_executor
    source = tmp_path / "SKILL.md"
    original = b"Keep the instructions."
    source.write_bytes(original)
    notes = "Track the review status."
    assert main(["author", str(tmp_path / "new"), notes, "--from", str(source), *options]) == 0
    assert len(calls) == 1
    assert_from_prompt(calls[0], source, original, options, notes)


def test_author_needs_description_or_source(tmp_path, draft_executor, capsys):
    _, calls = draft_executor
    assert main(["author", str(tmp_path / "parent" / "new")]) == 2
    assert capsys.readouterr().err == "author refused: one of DESCRIPTION or --from is needed\n"
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kind", ["missing", "folder", "directory_skill", "fifo", "fifo_skill", "utf8", "empty", "space", "dangling"]
)
def test_author_from_refuses_bad_source_before_call(tmp_path, from_executor, capsys, kind):
    _, calls = from_executor
    source = tmp_path / "source"
    if kind in ("folder", "directory_skill", "fifo_skill"):
        source.mkdir()
        if kind == "directory_skill":
            (source / "SKILL.md").mkdir()
        elif kind == "fifo_skill":
            os.mkfifo(source / "SKILL.md")
    elif kind == "dangling":
        source.symlink_to(tmp_path / "missing")
    elif kind == "fifo":
        os.mkfifo(source)
    elif kind != "missing":
        source.write_bytes({"utf8": b"\xff", "empty": b"", "space": b" \t\r\n"}[kind])
    before = set(tmp_path.rglob("*"))
    target = tmp_path / "parent" / "new"
    assert main(["author", str(target), "--from", str(source)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("author refused: ")
    assert len(err.splitlines()) == 1
    assert calls == []
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("kind", ["folder", "file", "symlink", "dangling"])
def test_author_from_refuses_existing_target(tmp_path, from_executor, capsys, kind):
    _, calls = from_executor
    source = tmp_path / "SKILL.md"
    source.write_bytes(b"Keep this text.")
    target = tmp_path / "existing"
    if kind == "folder":
        target.mkdir()
        (target / "keep").write_text("original")
    elif kind == "file":
        target.write_text("original")
    else:
        target.symlink_to(source if kind == "symlink" else tmp_path / "missing")
    before = set(tmp_path.rglob("*"))
    assert main(["author", str(target), "--from", str(source)]) == 2
    assert capsys.readouterr().err == f"author refused: refusing to overwrite existing skill target: {target}\n"
    assert calls == []
    assert set(tmp_path.rglob("*")) == before
    assert source.read_bytes() == b"Keep this text."
    if kind == "folder":
        assert (target / "keep").read_text() == "original"
    elif kind == "file":
        assert target.read_text() == "original"
    else:
        assert target.is_symlink()


@pytest.mark.parametrize("options", FROM_OPTIONS)
@pytest.mark.parametrize("change", ["missing", "state", "schema", "space"])
def test_author_from_refuses_invalid_draft(tmp_path, from_executor, capsys, options, change):
    draft, calls = from_executor
    if change == "missing":
        del draft["run_section"]
    elif change == "space":
        draft["run_section"] = " \t\r\n"
    elif change == "state":
        draft["initial_state"] = {"goal": 7}
    else:
        draft["schema_json"]["properties"]["goal"]["type"] = "bogus"
    source = tmp_path / "SKILL.md"
    source.write_bytes(b"Keep this text.")
    target = tmp_path / "new"
    assert main(["author", str(target), "--from", str(source), *options]) == 2
    assert capsys.readouterr().err.startswith("author refused: ")
    assert len(calls) == 1
    assert not target.exists()
    assert source.read_bytes() == b"Keep this text."
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("source_args", [[], ["--from", "missing"]])
@pytest.mark.parametrize(
    "options,reason",
    [
        (["--executor", "command"], "--executor command needs --command"),
        (
            ["--executor", "command", "--command", "offline-agent", "--model", "opus"],
            "--model is not supported by --executor command; put the model in the command string",
        ),
    ],
)
def test_author_option_errors_precede_input_refusals(tmp_path, draft_executor, capsys, source_args, options, reason):
    _, calls = draft_executor
    assert main(["author", str(tmp_path / "new"), *source_args, *options]) == 2
    assert capsys.readouterr().err == reason + "\n"
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_author_original_prompt_contract():
    assert (
        hashlib.sha256(AUTHOR_SYSTEM.encode()).hexdigest()
        == "be65a8ed89a996551616a1c034f3c57fc81a64fc657ad5807ec549ff3c774d3e"
    )
    assert AUTHOR_SCHEMA == {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "skill_md": {"type": "string"},
            "schema_json": {"type": "object"},
            "initial_state": {"type": "object"},
            "action": {"type": "string"},
        },
        "required": ["skill_md", "schema_json", "initial_state", "action"],
        "additionalProperties": False,
    }
    rules = AUTHOR_SYSTEM.split("and (2) schema.json: ")[1].split("Also produce")[0]
    assert rules in AUTHOR_FROM_SYSTEM
    assert AUTHOR_FROM_SCHEMA == {
        **AUTHOR_SCHEMA,
        "properties": {k: v for k, v in AUTHOR_SCHEMA["properties"].items() if k != "skill_md"}
        | {"run_section": {"type": "string"}},
        "required": ["run_section", "schema_json", "initial_state", "action"],
    }


@pytest.mark.parametrize("options", FROM_OPTIONS)
def test_author_original_bytes_and_messages(tmp_path, draft_executor, capsys, options):
    draft, calls = draft_executor
    draft["skill_md"] = "Résumé.\n  \n"
    target = tmp_path / "new"
    assert main(["author", str(target), "review", *options]) == 0
    assert capsys.readouterr().out == (
        f"drafted skill at {target}. Review SKILL.md and schema.json, then: stepbook validate {target}\n"
    )
    assert (target / "SKILL.md").read_bytes() == "Résumé.\n".encode()
    for name, value in (("schema", completed_schema(draft["schema_json"])), ("initial_state", draft["initial_state"])):
        assert (target / f"{name}.json").read_bytes() == json.dumps(value, indent=2).encode()
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    prompt = "Procedure:\nreview\n\n" + ENVIRONMENT
    if options[1] == "cli":
        assert cmd[-1] == prompt
        assert cmd[cmd.index("--system-prompt") + 1] == AUTHOR_SYSTEM
        assert json.loads(cmd[cmd.index("--json-schema") + 1]) == AUTHOR_SCHEMA
    else:
        assert f"<<SYSTEM>>\n{AUTHOR_SYSTEM}\n<</SYSTEM>>\n\n{prompt}\n\nAnswer with one JSON object" in kwargs["input"]


def test_author_help_lists_from(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["author", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--from" in help_text
    assert "[description]" in help_text
    assert (
        "an existing skill's SKILL.md or its folder: keep its text as the instructions and draft only the state "
        "schema, the initial state and a section on running it step by step"
    ) in help_text

    assert "the procedure in plain words; with --from, optional notes for drafting the state" in help_text
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "draft and validate a skill folder from a procedure or an existing skill" in " ".join(
        capsys.readouterr().out.split()
    )


@pytest.mark.parametrize("kind", ["file_link", "folder_link", "skill_link", "relative"])
def test_author_source_paths_resolve_once(tmp_path, from_executor, monkeypatch, capsys, kind):
    _, calls = from_executor
    folder = tmp_path / "source"
    folder.mkdir()
    source = folder / "SKILL.md"
    original = b"Keep this text.\r\n"
    source.write_bytes(original)
    link = tmp_path / "link"
    if kind == "file_link":
        link.symlink_to(source)
    elif kind == "folder_link":
        link.symlink_to(folder, target_is_directory=True)
    elif kind == "skill_link":
        link.mkdir()
        (link / "SKILL.md").symlink_to(source)
    else:
        monkeypatch.chdir(tmp_path)
        link = Path("source")
    resolved = []
    resolve = author.source_skill_path

    def record(path):
        resolved.append(path)
        return resolve(path)

    monkeypatch.setattr(author, "source_skill_path", record)
    target = tmp_path / "new"
    assert main(["author", str(target), "--from", str(link)]) == 0
    assert resolved == [link]
    assert f"from {source.resolve()}." in capsys.readouterr().out
    assert_from_prompt(calls[0], source, original, FROM_OPTIONS[0])
    assert (target / "SKILL.md").read_bytes().startswith(original)
    assert source.read_bytes() == original


@pytest.mark.parametrize("from_source", [False, True])
def test_author_multiline_refusal_is_one_line(tmp_path, draft_executor, capsys, from_source):
    draft, calls = draft_executor
    draft["schema_json"]["properties"]["goal"]["type"] = "bogus"
    draft["run_section"] = RUN_SECTION
    source = tmp_path / "SKILL.md"
    source.write_text("Review.")
    target = tmp_path / "new"
    inputs = ["--from", str(source)] if from_source else ["review"]
    assert main(["author", str(target), *inputs]) == 2
    error = capsys.readouterr().err
    assert error.startswith("author refused: ") and len(error.splitlines()) == 1
    assert len(calls) == 1 and not target.exists()
