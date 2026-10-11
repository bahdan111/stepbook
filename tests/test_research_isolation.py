"""Local experiments cannot leak into product imports, parsers, or fresh journals."""

from pathlib import Path

import pytest

from stepbook import Runtime, ScriptedExecutor, Skill
from stepbook.cli import _build_parser

ROOT = Path(__file__).resolve().parents[1]
MOVED_FLAGS = [
    ["--transcript-baseline"],
    ["--lessons-from", "state.json"],
    ["--no-schema"],
    ["--on-exhausted", "skip"],
    ["--repeat-warning", "2"],
    ["--max-source-bytes", "12"],
]


@pytest.mark.parametrize("options", MOVED_FLAGS + [["--measurement-repeats", "2"]])
def test_product_rejects_experiment_flags_before_dispatch(options, capsys):
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(["run", "unused", "--executor", "command", "--command", "unused", *options])
    assert error.value.code == 2
    assert "unrecognized arguments: " + " ".join(options) in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("repeat_limit", 2), ("on_exhausted", "skip"), ("skip_action", "stop")])
def test_product_constructor_rejects_local_failure_tuning(option, value):
    with pytest.raises(ValueError, match=option):
        Runtime(Skill("pin", "inspect", {}, {}), ScriptedExecutor(lambda s, u: None), object(), **{option: value})
