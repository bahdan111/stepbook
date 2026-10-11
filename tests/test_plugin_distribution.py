"""The repository root is the installable Claude Code and Codex plugin."""

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.conftest import package_version

ROOT = Path(__file__).resolve().parents[1]


def test_plugin_and_package_versions_agree():
    version = package_version()
    for path in (".claude-plugin/plugin.json", ".codex-plugin/plugin.json"):
        assert json.loads((ROOT / path).read_text())["version"] == version


def test_the_onboarding_skill_pins_this_package_version():
    """``uvx stepbook@<version>`` in the skill must name the version this repository builds."""
    skill = ROOT / "skills/stepbook/SKILL.md"
    if not skill.is_file():
        pytest.skip("onboarding skill is absent from this export")
    version = package_version()
    pinned = re.findall(r"stepbook@([\w.+-]+)", skill.read_text())
    assert all(pin == version for pin in pinned), pinned


def test_wheel_contains_complete_builtin_templates(tmp_path):
    pytest.importorskip("hatchling", reason="wheel regression requires the hatchling build dependency")
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        assert len(names) == len(set(names)), "wheel contains duplicate members"
        for template in ("eval-loop-adapter", "checklist", "eval-loop"):
            for filename in ("SKILL.md", "schema.json", "initial_state.json"):
                path = f"stepbook/templates/{template}/{filename}"
                assert wheel.read(path) == (ROOT / path).read_bytes(), path
        metadata = wheel.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
        assert not re.search(r'(src|srcset)="docs/assets/', metadata), "PyPI cannot serve relative pictures"


def test_root_has_no_unintended_plugin_components():
    component_names = {
        ".claude-plugin",
        ".codex-plugin",
        ".agents",
        "commands",
        "agents",
        "hooks",
        "skills",
        "bin",
        "monitors",
        ".mcp.json",
        "settings.json",
        ".app.json",
        ".lsp.json",
        "output-styles",
    }
    # "bin" stays in the list above: the plugin runs stepbook from PyPI, so no launcher may come back.
    assert {p.name for p in ROOT.iterdir()} & component_names == {
        ".claude-plugin",
        ".codex-plugin",
        ".agents",
        "hooks",
        "skills",
    }


def test_only_onboarding_and_private_skills_are_discoverable():
    for folder in (ROOT / "skills").iterdir():
        if folder.is_dir():
            assert folder.name == "stepbook" or (folder / "PRIVATE").is_file(), folder
            if (folder / "PRIVATE").is_file():
                assert not (folder / "SKILL.md").read_text().startswith("---\n"), folder
    for path in (ROOT / "skills").glob("*/SKILL.md"):
        if (path.parent / "PRIVATE").is_file():
            continue
        text = path.read_text()
        assert text.startswith("---\n"), path
        frontmatter, delimiter, _ = text[4:].partition("\n---\n")
        assert delimiter, path
        assert re.search(r"(?m)^name:\s*\S+", frontmatter), path
    for path in (ROOT / "stepbook/templates").glob("*/SKILL.md"):
        assert not path.read_text().startswith("---\n"), path


def test_marketplaces_install_the_root_without_versions():
    claude = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    codex = json.loads((ROOT / ".agents/plugins/marketplace.json").read_text())
    # Codex caches plugins under <catalog>/<plugin>/; equal names made agents misread that path.
    assert claude["name"] == codex["name"] == "stepbook-plugins"
    for catalog, source in ((claude, "./"), (codex, {"source": "local", "path": "./"})):
        assert len(catalog["plugins"]) == 1
        entry = catalog["plugins"][0]
        assert entry["name"] == "stepbook" and entry["source"] == source
        assert "version" not in entry
