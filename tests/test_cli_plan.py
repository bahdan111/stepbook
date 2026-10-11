"""Launch plans are validated before construction; all controllers here are offline."""

import importlib
import json
from pathlib import Path

import pytest

from stepbook import Envelope, ScriptedExecutor
from stepbook.run_pointer import classify_run
from stepbook.state.validator import PatchError, Validator
from tests.probe_workflows._helpers import git_repo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["stepbook"])
def cli_pair(request):
    return tuple(
        importlib.import_module(
            f"{request.param}.{name.replace('cli_', 'cli.').replace('cli.inspect', 'cli.inspection') if request.param == 'stepbook' else name}"
        )
        for name in ("cli", "cli_run")
    )


def args_for(cli, plan, *options, skill="eval-loop"):
    return cli._build_parser().parse_args(
        ["run", str(ROOT / "stepbook/templates" / skill), "--plan", str(plan), *options]
    )


def test_checklist_plan_accepts_seeded_checklist_and_structured_facts(cli_pair, tmp_path, monkeypatch):
    cli, run = cli_pair
    payload = {
        "checklist": {
            "create": {
                "text": "Create out.txt containing ready",
                "status": "todo",
                "done_when": "Command output shows out.txt contains ready",
            },
            "verify": {"text": "Run the operator check", "status": "todo"},
        },
        "facts": {"output": {"path": "out.txt", "expected": "ready"}, "inputs": ["source.txt"], "count": 1},
        "checks": {"plan_only": "false"},
        "constraints": ["plan rule"],
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload))
    args = args_for(
        cli,
        plan,
        "--workdir",
        str(tmp_path),
        "--goal",
        "Create and verify the output",
        "--check",
        "output=test -f out.txt",
        "--constraint",
        "Keep source.txt unchanged",
        skill="checklist",
    )
    skill = run._load_run_skill(args)
    assert skill.initial_state["checklist"] == payload["checklist"]
    assert skill.initial_state["facts"] == payload["facts"]
    monkeypatch.setattr(run.ShellEnv, "preflight", lambda self: None)
    assert run._build_shell_environment(args, skill) is not None
    assert skill.initial_state["checks"] == {"output": "test -f out.txt"}
    assert skill.initial_state["checks_status"] == {"output": "unknown"}
    assert skill.initial_state["constraints"] == ["Keep source.txt unchanged"]


@pytest.mark.parametrize("include_how", [False, True])
def test_eval_loop_shell_structured_plan_needs_only_known_values(cli_pair, tmp_path, monkeypatch, include_how):
    cli, run = cli_pair
    idea = {"kind": "config", "rationale": "retrieve 8 contexts", "origin": "user", "status": "todo"}
    if include_how:
        idea["how"] = "config.json top_k=8"
    payload = {
        "facts": {
            "eval": {
                "commands": ["python3 rag.py run --label <name>"],
                "results": "runs/<id>/metrics.json",
                "primary": "primary_metric.value, higher is better",
                "per_case": "metrics.json cases",
            },
            "fixed_files": ["cases.json", "data/", "score.py"],
            "case_set": {"count": 12, "fingerprint": "sha256:7e29ac95c57d1f00", "file": "cases.json"},
        },
        "protocol": {"baseline_runs": 2, "runs_per_hypothesis": 2},
        "budget": {"runs_max": 5},
        "hypotheses": {"top_k8": idea},
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload))
    args = args_for(
        cli,
        plan,
        "--workdir",
        str(tmp_path),
        "--goal",
        "Improve accuracy; at most five evals",
        "--constraint",
        "Keep cases fixed",
    )
    skill = run._load_run_skill(args)
    monkeypatch.setattr(run.ShellEnv, "preflight", lambda self: None)
    assert run._build_shell_environment(args, skill) is not None
    state = skill.initial_state
    assert state["facts"] == payload["facts"]
    assert state["hypotheses"] == payload["hypotheses"]
    assert state["goal"] == args.goal and state["constraints"] == ["Keep cases fixed"]
    assert state["protocol"] == {**payload["protocol"], "noise_cases": None}
    assert state["budget"] == {"runs_used": 0, "runs_max": 5}
    assert state["baseline"] == {"runs": []} and state["runs"] == {}


def test_plan_only_checks_and_constraints_reach_the_run_and_gate_done(tmp_path, monkeypatch, capsys):
    """Without --check the plan's own checks are the run's checks, and they still decide done."""
    from stepbook import cli
    from stepbook.cli import run as cli_run

    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")
    repo = git_repo(tmp_path)
    monkeypatch.chdir(repo)
    log = tmp_path / "log"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"checks": {"output": "test -f missing.txt"}, "constraints": ["Keep README unchanged"]}))
    prompts = []

    def respond(system, user):
        prompts.append(user)
        state = json.loads(user.split("```json\n", 1)[1].split("\n```", 1)[0])
        assert state["checks"] == {"output": "test -f missing.txt"}
        assert state["checks_status"] == {"output": "unknown"}
        assert state["constraints"] == ["Keep README unchanged"]
        return Envelope("the plan's check decides this", {}, "done")

    monkeypatch.setattr(cli_run, "_select_run_executor", lambda *a: (ScriptedExecutor(respond), "command"))
    argv = [
        "run",
        str(ROOT / "stepbook/templates/checklist"),
        "--plan",
        str(plan),
        "--workdir",
        str(repo),
        "--goal",
        "Create and verify out.txt",
        "--steps",
        "1",
        "--quiet",
        "--log-dir",
        str(log),
    ]
    assert cli.main(argv) == 1  # the check failed, so done was refused
    assert len(prompts) == 1
    assert classify_run(log)["status"] == "failed_checks"
    assert json.loads(capsys.readouterr().out)["score"] == 0.0
    assert json.loads((log / "state.json").read_text())["checks"] == {"output": "test -f missing.txt"}


@pytest.mark.parametrize("resume", [False, True])
def test_plan_merge_and_cli_precedence(cli_pair, tmp_path, monkeypatch, resume):
    cli, run = cli_pair
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "goal": 99,
                "constraints": ["plan rule"],
                "checks": {"old": "false"},
                "facts": {"eval": "python eval.py", "case_set": "2; sha256:abc"},
                "baseline": {"notes": None},
                "budget": {"runs_max": 8},
                "current": {"phase": "baseline"},
            }
        )
    )
    args = args_for(
        cli,
        plan,
        "--workdir",
        str(tmp_path),
        "--goal",
        "CLI goal",
        "--constraint",
        "fixed cases",
        "--check",
        "ok=true",
        "--runs-max",
        "3",
    )
    args.resume = resume
    skill = run._load_run_skill(args)
    monkeypatch.setattr(run.ShellEnv, "preflight", lambda self: None)
    env = run._build_shell_environment(args, skill)
    assert env is not None
    state = skill.initial_state
    assert state["goal"] == "CLI goal" and state["constraints"] == ["fixed cases"]
    assert state["checks"] == {"ok": "true"} and state["checks_status"] == {"ok": "unknown"}
    assert state["baseline"] == {"runs": []}
    assert state["facts"]["eval"] == "python eval.py"
    assert state["protocol"]["baseline_runs"] == 2
    assert state["budget"] == {"runs_used": 0, "runs_max": 3}
    with pytest.raises(PatchError, match="protected"):
        Validator(skill.schema, skill.protected_paths).apply(state, {"budget": {"runs_max": 9}})


@pytest.mark.parametrize("resume", [False, True])
def test_plan_adapter_cli_fields_win(cli_pair, tmp_path, monkeypatch, resume):
    cli, run = cli_pair
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"goal": "plan", "budget": {"runs_max": 8}, "user_requirements": [{"text": "plan rule", "done": False}]}
        )
    )
    args = args_for(
        cli,
        plan,
        "--adapter",
        "unused",
        "--goal",
        "CLI",
        "--constraint",
        "CLI rule",
        "--runs-max",
        "3",
        skill="eval-loop-adapter",
    )
    monkeypatch.setattr(run, "AdapterEvalEnv", lambda **kw: kw)
    args.resume = resume
    skill = run._load_run_skill(args)
    env, _ = run._build_adapter_environment(args, skill, False)
    assert env["runs_max"] == 3
    assert skill.initial_state["budget"]["runs_max"] == 3
    assert skill.initial_state["protocol"]["max_runs"] == 3
    assert skill.initial_state["goal"] == "CLI"
    assert [r["text"] for r in skill.initial_state["user_requirements"]] == ["plan rule", "CLI rule"]


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
        '{"hypotheses": {"h1": {"kind": "config", "rationale": "try it", "origin": "user"}}}',
        '{"hypotheses": {"h1": {"kind": "config", "rationale": "try it", "origin": "user", "status": "todo", "unknown": 1}}}',
    ],
)
@pytest.mark.parametrize("resume", [False, True])
def test_invalid_plan_refused_before_anything_built(cli_pair, tmp_path, monkeypatch, capsys, payload, resume):
    cli, run = cli_pair
    plan = tmp_path / "plan\nname.json"
    if payload == "directory":
        plan.mkdir()
    elif payload == "bad_utf8":
        plan.write_bytes(b"\xff")
    elif payload != "missing":
        plan.write_text(payload)
    args = args_for(cli, plan, "--workdir", str(tmp_path), "--goal", "goal", "--log-dir", str(tmp_path / "run"))

    args.resume = resume

    def forbidden(*a, **kw):
        pytest.fail("invalid plan must be refused before construction")

    for name in ("_build_run_environment", "_select_run_executor", "_prepare_run_directory", "_build_runtime"):
        monkeypatch.setattr(run, name, forbidden)
    assert run.cmd_run(args) == 2
    captured = capsys.readouterr()
    assert len(captured.err.splitlines()) == 1 and "--plan" in captured.err
    assert repr(str(plan)) in captured.err
    if resume:
        assert "run was started with it" in captured.err
    assert captured.out == "" and not (tmp_path / "run").exists()


@pytest.mark.parametrize("environment", ["shell", "adapter"])
def test_plan_all_environments_and_resume_rereads_file(cli_pair, tmp_path, environment):
    cli, run = cli_pair
    plan = tmp_path / "plan.json"
    plan.write_text('{"facts": {"eval": "ready"}}')
    if environment == "adapter":
        options = ["--adapter", "unused"]
    elif environment == "shell" and cli.__name__ == "stepbook.cli":
        options = []  # the product has no --env: shell is the environment without --adapter
    else:
        options = ["--env", environment]
    args = args_for(
        cli, plan, *options, "--goal", "goal", skill="eval-loop-adapter" if environment == "adapter" else "eval-loop"
    )
    assert run._load_run_skill(args).initial_state["facts"] == {"eval": "ready"}
    args.resume = True
    assert run._load_run_skill(args).initial_state["facts"] == {"eval": "ready"}
    plan.write_text('{"facts": {"eval": "reread"}}')
    assert run._load_run_skill(args).initial_state["facts"] == {"eval": "reread"}


def test_plan_rfc_merge_objects_replace_scalars_and_remove_nested_nulls(cli_pair, tmp_path):
    cli, run = cli_pair
    folder = tmp_path / "skill"
    folder.mkdir()
    # Reuse instructions without creating a new Markdown fixture.
    (folder / "SKILL.md").symlink_to(ROOT / "stepbook/templates/eval-loop/SKILL.md")
    (folder / "schema.json").write_text('{"type": "object"}')
    (folder / "initial_state.json").write_text(
        json.dumps(
            {
                "scalar": 1,
                "list": [1, 2],
                "object": {"keep": 1, "delete": 2},
                "replace": {"old": True},
            }
        )
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "scalar": {"drop": None, "nested": {"drop": None, "keep": 2}},
                "list": [3],
                "object": {"delete": None},
                "replace": "new",
                "absent": {"drop": None},
            }
        )
    )
    args = args_for(cli, plan)
    args.skill_dir = str(folder)
    assert run._load_run_skill(args).initial_state == {
        "scalar": {"nested": {"keep": 2}},
        "list": [3],
        "object": {"keep": 1},
        "replace": "new",
        "absent": {},
    }
