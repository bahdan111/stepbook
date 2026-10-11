"""Public skill validation and an offline shell loop in a disposable repository."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from stepbook import Envelope, ScriptedExecutor, Skill
from stepbook.state.validator import PatchError, Validator

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("origin", ["user", "docs", "agent"])
@pytest.mark.parametrize("status", ["todo", "investigating", "testing", "kept", "reverted", "inconclusive", "skipped"])
def test_hypothesis_bookkeeping_accepts_sources_and_lifecycle(origin, status):
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop")
    idea = {
        "kind": "code",
        "rationale": "A case fails",
        "how": "Inspect its parser",
        "origin": origin,
        "status": status,
        "runs": [],
        "evidence": "",
        "commit": "",
    }
    state, _ = Validator(skill.schema, skill.protected_paths).apply(skill.initial_state, {"hypotheses": {"h1": idea}})
    assert state["hypotheses"]["h1"] == idea
    with pytest.raises(PatchError, match="not one of"):
        Validator(skill.schema).apply(state, {"hypotheses": {"h1": {"origin": "invented"}}})


def test_new_hypothesis_needs_no_execution_placeholders():
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop")
    idea = {"kind": "config", "rationale": "Retrieve more contexts", "origin": "user", "status": "todo"}
    validator = Validator(skill.schema, skill.protected_paths)
    state, _ = validator.apply(skill.initial_state, {"hypotheses": {"h1": idea}})
    assert state["hypotheses"]["h1"] == idea
    for field in idea:
        with pytest.raises(PatchError, match=f"'{field}' is a required property"):
            validator.apply(state, {"hypotheses": {"h1": {field: None}}})
    with pytest.raises(PatchError, match="unexpected"):
        validator.apply(state, {"hypotheses": {"h1": {"unexpected": True}}})


@pytest.mark.parametrize("field,value", [("how", []), ("runs", "b1"), ("runs", [1]), ("evidence", {}), ("commit", 1)])
def test_optional_hypothesis_fields_remain_typed(field, value):
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop")
    idea = {
        "kind": "config",
        "rationale": "Retrieve more contexts",
        "origin": "user",
        "status": "todo",
        field: value,
    }
    with pytest.raises(PatchError, match="not of type"):
        Validator(skill.schema).apply(skill.initial_state, {"hypotheses": {"h1": idea}})


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def test_goal_user_requirements_and_cap_are_protected():
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop")
    assert set(skill.protected_paths) == {"goal", "user_requirements", "budget.runs_max"}
    assert skill.initial_state["budget"] == {"runs_used": 0, "runs_max": None}


def test_cap_changes_are_rejected_even_before_first_eval():
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop")
    validator = Validator(skill.schema, skill.protected_paths, skill.max_deletions, skill.write_once_paths)
    for cap in (None, 4):
        skill.initial_state["budget"]["runs_max"] = cap
        with pytest.raises(PatchError, match="budget.runs_max.*protected"):
            validator.apply(skill.initial_state, {"budget": {"runs_max": 5}})


@pytest.mark.parametrize("module", ["stepbook"])
def test_seeded_shell_first_action_is_baseline_and_resume_keeps_plan(tmp_path, monkeypatch, module):
    import importlib

    cli = importlib.import_module(module + ".cli")
    cli_run = importlib.import_module(module + (".cli.run" if module == "stepbook" else ".cli_run"))
    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _git(repo, "init", "-q")
    metrics = {"primary": 0.5, "cases": {"a": True, "b": False}}
    (repo / "fake_eval.py").write_text(f"import json\nprint(json.dumps({metrics!r}))\n")
    _git(repo, "add", "fake_eval.py")
    _git(repo, "-c", "user.name=test", "-c", "user.email=test@local", "commit", "-qm", "fixture")
    original = _git(repo, "rev-parse", "HEAD")
    log = tmp_path / "log"
    command = f"{shlex.quote(sys.executable)} fake_eval.py"
    idea = {
        "kind": "code",
        "rationale": "User suggests parser handling",
        "origin": "user",
        "status": "todo",
    }
    plan = {
        "facts": {
            "eval": {"commands": [command], "primary": "stdout JSON primary, higher", "per_case": "cases booleans"},
            "fixed_files": ["fake_eval.py"],
            "case_set": {"count": 2, "fingerprint": "a,b"},
        },
        "hypotheses": {"h1": idea},
        "budget": {"runs_max": 2},
        "protocol": {"baseline_runs": 2, "runs_per_hypothesis": 2},
    }
    plan_file = repo / ".stepbook/plan.json"
    plan_file.parent.mkdir()
    plan_file.write_text(json.dumps(plan))
    prompts = []

    def respond(system, user):
        prompts.append(user)
        state = json.loads(user.split("```json\n", 1)[1].split("\n```", 1)[0])
        assert state["facts"] == plan["facts"]
        assert state["hypotheses"]["h1"]["origin"] == "user"
        assert state["budget"]["runs_max"] == 2
        n = len(prompts)
        if n == 1:
            assert state["budget"]["runs_used"] == 0
            assert state["hypotheses"]["h1"] == idea
        else:
            assert "[exit 0]" in user and json.dumps(metrics) in user
        measured = {"primary": 0.5, "outcomes": metrics["cases"], "valid": True, "notes": "completed"}
        if n <= 2:
            run_id = f"b{n}"
            pending = {
                "label": f"baseline-{n}",
                "stack": [],
                "valid": False,
                "notes": "pending",
                "case_set": "a,b",
                "artifact": "stdout in journal",
            }
            return Envelope(
                "Reserve baseline before launching.",
                {
                    "runs": {**({"b1": measured} if n == 2 else {}), run_id: pending},
                    "baseline": {"runs": [f"b{i}" for i in range(1, n + 1)]},
                    "budget": {"runs_used": n},
                    "current": {"phase": "run", "run": run_id},
                },
                command,
            )
        assert n == 3
        return Envelope(
            "Both baseline repeats complete; cap spent before any trial.",
            {
                "runs": {"b2": measured},
                "baseline": {"primary_mean": 0.5},
                "protocol": {"noise_cases": 0},
                "current": {"phase": "done"},
                "hypotheses": {"h1": {"status": "skipped", "evidence": "Budget spent on two baseline repeats."}},
                "notes": {"summary": "Baseline 0.5; two attempts; no kept changes."},
            },
            "done",
        )

    def interrupt_after_baseline(system, user):
        if prompts:
            raise KeyboardInterrupt("simulate interruption after first baseline receipt")
        return respond(system, user)

    monkeypatch.setattr(
        cli_run, "_select_run_executor", lambda *a: (ScriptedExecutor(interrupt_after_baseline), "command")
    )
    argv = [
        "run",
        str(ROOT / "stepbook/templates/eval-loop"),
        # the research harness names its environment; the product has only --adapter or shell
        *([] if module == "stepbook" else ["--env", "shell"]),
        "--workdir",
        str(repo),
        "--goal",
        "Improve accuracy; at most two evals",
        "--constraint",
        "Do not edit fake_eval.py",
        "--plan",
        str(plan_file),
        "--allow-outside",
        "--steps",
        "3",
        "--quiet",
        "--log-dir",
        str(log),
    ]
    with pytest.raises(KeyboardInterrupt, match="first baseline receipt"):
        cli.main(argv)
    checkpoint_head = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "diff", "--name-only", original, checkpoint_head) == ""
    saved = json.loads((log / "state.json").read_text())
    assert saved["budget"] == {"runs_used": 1, "runs_max": 2}
    launch = json.loads((log / "launch.json").read_text())
    assert "--plan" in launch["resume_command"] and str(plan_file) in launch["resume_command"]
    monkeypatch.setattr(cli_run, "_select_run_executor", lambda *a: (ScriptedExecutor(respond), "command"))
    assert cli.main([*argv, "--resume"]) == 0
    state = json.loads((log / "state.json").read_text())
    assert len(prompts) == 3
    assert state["baseline"]["runs"] == ["b1", "b2"]
    assert state["runs"]["b1"]["valid"] and state["runs"]["b2"]["valid"]
    assert state["budget"] == {"runs_used": 2, "runs_max": 2}
    skill = Skill.load(ROOT / "stepbook/templates/eval-loop")
    with pytest.raises(PatchError, match="budget.runs_max.*protected"):
        Validator(skill.schema, skill.protected_paths).apply(state, {"budget": {"runs_max": 3}})
    assert state["kept_stack"] == [] and state["hypotheses"]["h1"]["status"] == "skipped"
    events = [json.loads(line) for line in (log / "events.jsonl").read_text().splitlines()]
    steps = [event for event in events if event["kind"] == "step"]
    assert [event["action"] for event in steps] == [command, command, "done"]
    assert not [event for event in events if event["kind"] == "patch_rejected"]
    assert _git(repo, "rev-parse", "HEAD") == checkpoint_head
    assert ".stepbook/plan.json" not in _git(repo, "ls-files")
