"""Root plugin assets, standalone hooks, and release separation."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_sources_are_present():
    for path in (
        "hooks/hooks.json",
        "hooks/reinject_state.py",
        "hooks/inbox_relay.py",
        ".claude-plugin/plugin.json",
        ".codex-plugin/plugin.json",
    ):
        assert (ROOT / path).is_file()
    for name in ("build_plugin.py", "build_plugin.sh"):
        assert not (ROOT / "scripts" / name).exists()
    assert (ROOT / "skills/stepbook/SKILL.md").is_file()


def test_public_release_package_and_shell_syntax():
    assert 'packages = ["stepbook"]' in (ROOT / "pyproject.toml").read_text()
    for relative in ("scripts/bump-version.sh",):
        result = subprocess.run(["bash", "-n", str(ROOT / relative)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("hook", ["inbox_relay", "reinject_state"])
def test_hook_loads_pointer_with_only_stdlib(tmp_path, hook):
    source = ROOT / "hooks" / (hook + ".py")
    code = """import runpy, sys
namespace = runpy.run_path(sys.argv[1])
pointer = namespace["load_run_pointer"]()
assert pointer is not None and callable(pointer.classify_run)
assert not any(name == "stepbook" or name.startswith("stepbook.") for name in sys.modules)
assert "claude_agent_sdk" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(source)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
