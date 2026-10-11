"""Plan preflight uses launch semantics without constructing or writing a run."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from stepbook import cli
from stepbook.cli import run as cli_run
from stepbook.state.validator import Validator

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "stepbook/templates/eval-loop"


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("plan validation must not construct a run or launch a process")

    for name in ("_build_run_environment", "_select_run_executor", "_prepare_run_directory", "_build_runtime"):
        monkeypatch.setattr(cli_run, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def validated_states(monkeypatch):
    states = []
    apply = Validator.apply

    def record(self, state, patch):
        states.append(copy.deepcopy(state))
        return apply(self, state, patch)

    monkeypatch.setattr(Validator, "apply", record)
    return states


def validate(plan, *options, skill=SKILL):
    return cli.main(["validate", str(skill), "--plan", str(plan), *options])


def test_valid_plan_merges_initial_state_and_reports_keys(tmp_path, capsys, validated_states):
    plan = tmp_path / "plan.json"
    plan.write_text('{"facts": {"eval": "python eval.py"}, "baseline": {"notes": null}}')
    assert validate(plan) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 2 and lines[0].startswith("ok: skill 'eval-loop',")
    assert lines[1] == "plan valid: top-level keys: baseline, facts"
    initial = json.loads((SKILL / "initial_state.json").read_text())
    assert validated_states[-1] == {**initial, "facts": {"eval": "python eval.py"}}


def test_empty_plan_reports_no_keys(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    assert validate(plan) == 0
    assert capsys.readouterr().out.splitlines()[1] == "plan valid: top-level keys: (none)"


def test_plan_may_initialize_protected_fields(tmp_path, validated_states):
    plan = tmp_path / "plan.json"
    plan.write_text('{"budget": {"runs_max": 5}, "protocol": {"baseline_runs": 3}}')
    assert validate(plan) == 0
    state = validated_states[-1]
    assert state["budget"] == {"runs_used": 0, "runs_max": 5}
    assert state["protocol"]["baseline_runs"] == 3


@pytest.mark.parametrize(
    "payload",
    [
        "missing",
        "directory",
        "bad_utf8",
        "{",
        "[]",
        "null",
        '"text"',
        '{"facts": 1}',
        '{"protocol": null}',
        '{"budget": {"runs_max": -1}}',
        '{"budget.runs_max": 4}',
        '{"hypotheses": {"h": {"kind": "config"}}}',
    ],
)
def test_invalid_plan_matches_run_refusal(tmp_path, capsys, payload):
    plan = tmp_path / "plan\nname.json"
    if payload == "directory":
        plan.mkdir()
    elif payload == "bad_utf8":
        plan.write_bytes(b"\xff")
    elif payload != "missing":
        plan.write_text(payload)
    assert validate(plan) == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert len(refused.err.splitlines()) == 1
    assert refused.err.startswith(f"invalid --plan {str(plan)!r}:")
    args = cli._build_parser().parse_args(["run", str(SKILL), "--plan", str(plan)])
    assert cli_run.cmd_run(args) == 2
    assert capsys.readouterr().err == refused.err


def test_cli_fields_override_plan_with_run_precedence(tmp_path, capsys, validated_states):
    plan = tmp_path / "plan.json"
    plan.write_text('{"goal": 99, "constraints": 99, "checks": 99, "checks_status": 99}')
    assert validate(plan) == 2
    assert "invalid --plan" in capsys.readouterr().err
    options = [
        "--goal",
        "CLI goal",
        "--constraint",
        "first",
        "--constraint",
        "second",
        "--check",
        " old = false ",
        "--check",
        "old=true",
        "--check",
        "new=echo a=b",
    ]
    assert validate(plan, *options) == 0
    state = validated_states[-1]
    assert state["goal"] == "CLI goal"
    assert state["constraints"] == ["first", "second"]
    assert state["checks"] == {"old": "true", "new": "echo a=b"}
    assert state["checks_status"] == {"old": "unknown", "new": "unknown"}
    args = cli._build_parser().parse_args(["run", str(SKILL), "--plan", str(plan), *options])
    cli_run._load_run_skill(args)
    assert validated_states[-1] == state


@pytest.mark.parametrize("options", [["--goal", "CLI"], ["--constraint", "CLI"], ["--check", "cli=true"]])
def test_omitted_cli_fields_keep_plan_values(tmp_path, validated_states, options):
    payload = {
        "goal": "plan",
        "constraints": ["plan"],
        "checks": {"plan": "false"},
        "checks_status": {"plan": "unknown"},
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload))
    assert validate(plan, *options) == 0
    expected = {**json.loads((SKILL / "initial_state.json").read_text()), **payload}
    if options[0] == "--goal":
        expected["goal"] = "CLI"
    elif options[0] == "--constraint":
        expected["constraints"] = ["CLI"]
    else:
        expected.update(checks={"cli": "true"}, checks_status={"cli": "unknown"})
    assert validated_states[-1] == expected


def test_run_keeps_the_plan_values_validate_reported(tmp_path, validated_states):
    payload = {
        "goal": "plan goal",
        "constraints": ["plan rule"],
        "checks": {"plan": "false"},
        "checks_status": {"plan": "unknown"},
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload))
    assert validate(plan) == 0
    reported = validated_states[-1]
    assert {key: reported[key] for key in payload} == payload
    # The launch must validate the same state it will run with: no flag, no override.
    args = cli._build_parser().parse_args(["run", str(SKILL), "--plan", str(plan)])
    cli_run._load_run_skill(args)
    assert validated_states[-1] == reported


def test_invalid_check_matches_run_refusal(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    assert validate(plan, "--check", "missing-command") == 2
    refused = capsys.readouterr()
    args = cli._build_parser().parse_args(
        [
            "run",
            str(SKILL),
            "--plan",
            str(plan),
            "--goal",
            "goal",
            "--check",
            "missing-command",
        ]
    )
    assert cli_run.cmd_run(args) == 2
    assert capsys.readouterr().err == refused.err


@pytest.mark.parametrize("payload", ["{}", '{"facts": 1}'])
def test_validation_writes_no_files(tmp_path, monkeypatch, payload):
    folder = tmp_path / "skill"
    folder.mkdir()
    for name in ("SKILL.md", "schema.json", "initial_state.json"):
        (folder / name).symlink_to(SKILL / name)
    plan = tmp_path / "plan.json"
    plan.write_text(payload)
    monkeypatch.chdir(tmp_path)

    def snapshot():
        return {str(p.relative_to(tmp_path)): p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}

    before = snapshot()
    assert validate(plan, "--goal", "goal", "--check", "never=touch should-not-exist", skill=folder) == (
        0 if payload == "{}" else 2
    )
    assert snapshot() == before


def test_plain_validate_keeps_existing_output(capsys):
    assert cli.main(["validate", str(SKILL)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    assert captured.out.startswith("ok: skill 'eval-loop',")


@pytest.mark.parametrize("lock", ["protected", "write_once"])
def test_dotted_lock_alias_refused_even_with_permissive_schema(tmp_path, capsys, lock):
    folder = tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").symlink_to(SKILL / "SKILL.md")
    (folder / "schema.json").write_text(json.dumps({"type": "object", "x-stepbook": {lock: ["nested.key"]}}))
    plan = tmp_path / "plan.json"
    plan.write_text('{"nested": {"key": "seed"}}')
    assert validate(plan, skill=folder) == 0
    capsys.readouterr()
    plan.write_text('{"nested.key": "ambiguous"}')
    assert validate(plan, skill=folder) == 2
    refused = capsys.readouterr()
    assert "dotted keys cannot be locked" in refused.err
    args = cli._build_parser().parse_args(["run", str(folder), "--plan", str(plan)])
    assert cli_run.cmd_run(args) == 2
    assert capsys.readouterr().err == refused.err
