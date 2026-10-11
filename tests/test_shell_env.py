from pathlib import Path

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv

ROOT = Path(__file__).resolve().parent.parent


def test_shell_env_runs_commands_checks_and_refuses_dangerous_ones(tmp_path):
    env = ShellEnv(
        require_sandbox=False,
        workdir=tmp_path,
        goal="create hello.txt containing hi",
        checks={"file": "grep -q hi hello.txt"},
    )
    obs = env.reset()
    assert "Goal: create hello.txt" in obs and "file: grep -q hi hello.txt" in obs
    assert "Actions: a shell command line, 'check' or 'check <id>', 'done', or 'stop <reason>'." in obs
    r = env.step("echo hi > hello.txt")
    assert r.info["rc"] == 0
    assert r.info == {"ok": True, "rc": 0} and r.observation == "$ echo hi > hello.txt\n[exit 0] (no output)"
    r = env.step("check")
    assert r.info["checks"] == {"file": "pass"}
    assert r.observation == "Checks:\n  file: pass   (grep -q hi hello.txt)"
    r = env.step("rm -rf /")
    assert r.info.get("refused") and "REFUSED" in r.observation
    assert r.info == {"ok": False, "refused": True}
    assert r.observation == (
        "REFUSED by the environment: matches the denylist pattern "
        "'recursive-rm-root-or-home'. Choose a different command."
    )
    r = env.step("cat /etc/passwd")
    assert r.info.get("refused")
    assert r.info == {"ok": False, "refused": True}
    assert r.observation == (
        "REFUSED by the environment: path '/etc/passwd' is outside the working directory. Choose a different command."
    )
    assert not (tmp_path / "etc").exists()
    r = env.step("done")
    assert r.done and r.info["ok"] and env.score() == 1.0
    assert r.info == {"ok": True, "checks": {"file": "pass"}, "terminal": True}
    assert r.observation == "Checks:\n  file: pass   (grep -q hi hello.txt)\nAll checks pass. Goal met."
    assert [c["cmd"] for c in env.commands] == ["echo hi > hello.txt"] and env.t == 5


def test_shell_env_done_refused_when_checks_fail(tmp_path):
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="x", checks={"never": "false"})
    env.reset()
    r = env.step("done")
    assert r.done and not r.info["ok"] and env.score() == 0.0
    assert r.info == {"ok": False, "checks": {"never": "fail"}, "terminal": True}
    assert r.observation == (
        "Checks:\n  never: fail   (false)\nNot all checks pass; 'done' was refused and the episode ended as incomplete."
    )
    assert env.last_checks == {"never": "fail"}


def test_checklist_skill_runs_end_to_end_with_scripted_agent(tmp_path):
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    skill.initial_state.update(
        {"goal": "make a file", "checks": {"f": "test -f out.txt"}, "checks_status": {"f": "unknown"}}
    )
    steps = iter(
        [
            Envelope(
                "plan", {"checklist": {"c1": {"text": "create out.txt", "status": "doing"}}}, "echo done > out.txt"
            ),
            Envelope("verify", {"checklist": {"c1": {"status": "done", "evidence": "exit 0"}}}, "check"),
            Envelope("finish", {"checks_status": {"f": "pass"}}, "done"),
        ]
    )
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="make a file", checks={"f": "test -f out.txt"})
    res = Runtime(skill, ScriptedExecutor(lambda s, u: next(steps)), env, log_dir=tmp_path / "log").run(10)
    assert res.score == 1.0 and res.rejected_patches == 0 and res.final_state["checklist"]["c1"]["status"] == "done"


def test_shell_env_checkpoints_on_a_fresh_branch(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"], cwd=tmp_path, check=True
    )
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="edit", checks={})
    env.reset()
    assert env.branch and env.branch.startswith("stepbook/")
    r = env.step("echo 2 >> a.txt")
    assert "[checkpoint" in r.observation
    log = subprocess.run(["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "stepbook step 1" in log and "stepbook: start" in log


def test_shell_env_refuses_outside_sandbox_by_default(tmp_path, monkeypatch):
    import pytest

    monkeypatch.delenv("STEPBOOK_SANDBOX", raising=False)
    monkeypatch.setattr(ShellEnv, "in_sandbox", staticmethod(lambda: False))
    with pytest.raises(RuntimeError, match="refuses to run outside a container"):
        ShellEnv(workdir=tmp_path, goal="x").reset()
    monkeypatch.setenv("STEPBOOK_SANDBOX", "1")
    monkeypatch.setattr(ShellEnv, "in_sandbox", staticmethod(lambda: True))
    assert "Goal" in ShellEnv(workdir=tmp_path, goal="x", checkpoint=False).reset()
