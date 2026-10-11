"""Local helpers remain scripted; product routes cannot dispatch them."""

import dataclasses
import json

import pytest

from stepbook.cli import _build_parser as product_parser
from stepbook.envs.eval_loop import AdapterEvalEnv as ProductAdapterEnv
from stepbook.envs.shell import ShellEnv as ProductShellEnv


@pytest.mark.parametrize("options", [["--mine"], ["--code-repo", "."], ["--doer-command", "unused"]])
def test_product_parser_rejects_helper_flags(options):
    with pytest.raises(SystemExit) as error:
        product_parser().parse_args(["run", "unused", *options])
    assert error.value.code == 2


@pytest.mark.parametrize("argv", [["step", "unused", "--obs", "ready"], ["explain", "unused", "why"]])
def test_product_parser_rejects_local_commands(argv):
    with pytest.raises(SystemExit) as error:
        product_parser().parse_args(argv)
    assert error.value.code == 2


def test_product_constructors_and_exports_have_no_helpers():
    from stepbook import cli
    from stepbook.cli import inspection as cli_inspect
    from stepbook.cli import skill as cli_skill
    from stepbook.reporting import usage

    assert not {"miner_executor", "code_executor", "investigator_executor", "code_repo"} & {
        field.name for field in dataclasses.fields(ProductAdapterEnv)
    }
    assert not {"doer_command", "doer_timeout_s"} & {field.name for field in dataclasses.fields(ProductShellEnv)}
    for module in (cli, cli_inspect, cli_skill):
        assert not {"cmd_step", "cmd_explain"} & vars(module).keys()
    assert not hasattr(usage, "controller_tool_logs")


@pytest.mark.parametrize("verb", ["delegate", "DELEGATE touch escaped", "delegate\ttouch escaped"])
def test_product_delegate_never_falls_through_to_shell(tmp_path, monkeypatch, verb):
    env = ProductShellEnv(tmp_path, "g", checkpoint=False, require_sandbox=False)
    monkeypatch.setattr(env, "_run", lambda *_: pytest.fail("removed helper reached the shell"))
    result = env.step(verb)
    assert result.info == {"ok": False, "refused": True} and not result.done
    assert (
        result.observation == "REFUSED by the environment: delegate is not available; do the work with shell commands."
    )
    assert not list(tmp_path.iterdir())


def test_product_keeps_adapter_investigation_and_rolls_back_code_request(fake_adapter):
    adapter = fake_adapter(
        {
            "investigate h1": "Investigation of h1: redundant",
            "decide h2 revert": "h2 revert: restored",
        }
    )
    adapter.answer("apply h2", "edit", exit_code=3)  # applied, and a code change is required
    env = ProductAdapterEnv(adapter=adapter.command)
    assert env.step("investigate h1").observation == "Investigation of h1: redundant"
    result = env.step("apply h2")
    assert (
        result.info
        == {
            "ok": False,
            "adapter_exit": 3,
            "applied": None,
            "code_change_required": True,
        }
        and env._applied is None
    )
    assert "no code executor is configured; reverted" in result.observation
    frontier = json.loads(env.context().split("\n", 1)[1])
    assert not {"mine_enabled", "code_changes_enabled"} & frontier.keys()
    assert "completion" in frontier and "evidence_cwd" in frontier
