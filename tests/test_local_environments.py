"""Offline checks for unsupported CLI options and removed reexports."""

import pytest


@pytest.mark.parametrize(
    "flags",
    [
        ["--seed", "1"],
        ["--noise", "2"],
        ["--drift", "0.2"],
        ["--repo", "unused"],
        ["--test-cmd", "unused"],
        ["--test-timeout", "3"],
        ["--no-stream"],
    ],
)
def test_product_rejects_local_options_before_dispatch(monkeypatch, flags):
    from stepbook import cli

    def forbidden(*args, **kwargs):
        pytest.fail("parser must reject before preflight or dispatch")

    monkeypatch.setattr(cli, "preflight_command", forbidden)
    monkeypatch.setattr(cli, "cmd_run", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "unused", "--executor", "command", "--command", "offline-unused", *flags])
    assert error.value.code == 2


def test_old_reexports_are_removed():
    from stepbook.envs import eval_loop

    assert not hasattr(eval_loop, "Hypothesis")
    assert not hasattr(eval_loop, "SimulatedEvalEnv")
