"""Seeded checklists execute through both CLIs without a model or a real project checkout."""

import importlib
import json
from pathlib import Path

import pytest

from stepbook import Envelope, ScriptedExecutor
from tests.probe_workflows._helpers import events, git, git_repo, steps

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["stepbook"])
@pytest.mark.parametrize("check_passes", [True, False])
def test_seeded_checklist_works_first_item_and_finishes_only_after_check(
    tmp_path, monkeypatch, capsys, module, check_passes
):
    cli = importlib.import_module(module + ".cli")
    cli_run = importlib.import_module(module + (".cli.run" if module == "stepbook" else ".cli_run"))
    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")
    repo = git_repo(tmp_path)
    monkeypatch.chdir(repo)
    log = tmp_path / "log"
    plan = {
        "checklist": {
            "create": {"text": "Create out.txt", "status": "todo", "done_when": "cat out.txt prints ready"},
            "verify": {"text": "Run the operator check", "status": "todo"},
        },
        "facts": {"output": {"path": "out.txt", "expected": "ready"}, "inputs": ["README"], "count": 1},
    }
    plan_file = repo / ".stepbook/plan.json"
    plan_file.parent.mkdir()
    plan_file.write_text(json.dumps(plan))
    command = "echo ready > out.txt && cat out.txt"
    check = 'test "$(cat out.txt)" = ready' if check_passes else "test -f missing.txt"
    prompts = []

    def respond(system, user):
        prompts.append(user)
        state = json.loads(user.split("```json\n", 1)[1].split("\n```", 1)[0])
        assert state["facts"] == plan["facts"]
        assert state["checks"] == {"output": check}
        assert state["constraints"] == ["Keep README unchanged"]
        assert set(state["checklist"]) == {"create", "verify"}
        if len(prompts) == 1:
            assert state["checklist"] == plan["checklist"]
            assert state["checks_status"] == {"output": "unknown"}
            return Envelope("Work on the first seeded item.", {"checklist": {"create": {"status": "doing"}}}, command)
        if len(prompts) == 2:
            assert state["checklist"]["create"]["status"] == "doing"
            assert "[exit 0]\nready" in user
            return Envelope(
                "Observed the required output; run the operator check.",
                {
                    "checklist": {
                        "create": {"status": "done", "evidence": "cat out.txt exited 0 and printed ready"},
                        "verify": {"status": "doing"},
                    }
                },
                "check",
            )
        assert len(prompts) == 3
        assert state["checklist"]["create"]["status"] == "done"
        status = "pass" if check_passes else "fail"
        assert f"output: {status}" in user
        return Envelope(
            "Finish only with a passing operator check.",
            {
                "checks_status": {"output": status},
                "checklist": {
                    "verify": {"status": "done" if check_passes else "blocked", "evidence": f"output: {status}"}
                },
            },
            "done" if check_passes else "stop operator check failed",
        )

    monkeypatch.setattr(cli_run, "_select_run_executor", lambda *a: (ScriptedExecutor(respond), "command"))
    result = cli.main(
        [
            "run",
            str(ROOT / "stepbook/templates/checklist"),
            # the research harness names its environment; the product has only --adapter or shell
            *([] if module == "stepbook" else ["--env", "shell"]),
            "--workdir",
            str(repo),
            "--plan",
            str(plan_file),
            "--goal",
            "Create and verify out.txt",
            "--check",
            f"output={check}",
            "--constraint",
            "Keep README unchanged",
            "--steps",
            "3",
            "--quiet",
            "--log-dir",
            str(log),
        ]
    )
    assert result == (0 if check_passes else 3)  # Failed check leads the agent to stop, rather than say done.
    report = json.loads(capsys.readouterr().out)
    assert report["score"] == (1.0 if check_passes else 0.0)
    assert report["rejected_patches"] == 0
    assert len(prompts) == 3
    state = json.loads((log / "state.json").read_text())
    assert state["checklist"]["create"] == {
        **plan["checklist"]["create"],
        "status": "done",
        "evidence": "cat out.txt exited 0 and printed ready",
    }
    assert state["checklist"]["verify"]["status"] == ("done" if check_passes else "blocked")
    assert state["checks_status"] == {"output": "pass" if check_passes else "fail"}
    assert [step["action"] for step in steps(log)] == [
        command,
        "check",
        "done" if check_passes else "stop operator check failed",
    ]
    assert not [event for event in events(log) if event["kind"] == "patch_rejected"]
    assert (repo / "out.txt").read_text() == "ready\n"
    assert (repo / "README").read_text() == "x\n"
    assert ".stepbook/plan.json" not in git(repo, "ls-files")
