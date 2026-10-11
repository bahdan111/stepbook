"""Built-in names resolve offline in checkouts and installed wheels."""

import argparse
import shutil
from pathlib import Path

import pytest

from stepbook import cli
from stepbook.cli import skill as cli_skill
from stepbook.cli.run import _load_run_skill
from stepbook.contracts.errors import MissingSkillFolder

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", cli_skill.BUILTIN_SKILLS)
def test_builtin_name_validates_and_loads_for_run(name, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["validate", name]) == 0
    args = argparse.Namespace(skill_dir=name)
    assert _load_run_skill(args).name == name


@pytest.mark.parametrize("name", cli_skill.BUILTIN_SKILLS)
def test_installed_template(name, monkeypatch, tmp_path):
    package = tmp_path / "installed" / "stepbook"
    shutil.copytree(ROOT / "stepbook/templates" / name, package / "templates" / name)
    monkeypatch.setattr(cli_skill, "__file__", str(package / "cli" / "skill.py"))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["validate", name]) == 0
    assert _load_run_skill(argparse.Namespace(skill_dir=name)).name == name


def test_existing_path_precedes_builtin_and_templates_ignore_checkout(monkeypatch, tmp_path):
    package = tmp_path / "checkout" / "stepbook"
    checkout = package.parent / "skills" / "checklist"
    shutil.copytree(ROOT / "stepbook/templates/checklist", checkout)
    shutil.copytree(ROOT / "stepbook/templates/checklist", package / "templates/checklist")
    monkeypatch.setattr(cli_skill, "__file__", str(package / "cli" / "skill.py"))
    monkeypatch.chdir(tmp_path)
    assert cli_skill.builtin_skill_path("checklist") == package / "templates/checklist"
    assert cli.main(["validate", "checklist"]) == 0
    # An existing local skill wins even when its folder has a built-in name.
    shutil.copytree(ROOT / "stepbook/templates/eval-loop-adapter", tmp_path / "checklist")
    assert (
        cli_skill.load_cli_skill("checklist").initial_state
        == cli_skill.load_cli_skill(str(ROOT / "stepbook/templates/eval-loop-adapter")).initial_state
    )


@pytest.mark.parametrize("name", ["missing", "./checklist", "nested/checklist"])
def test_fallback_only_accepts_bare_builtin_names(name, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(MissingSkillFolder):
        cli_skill.load_cli_skill(name)


def test_existing_file_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checklist").write_text("not a skill directory")
    with pytest.raises(MissingSkillFolder):
        cli_skill.load_cli_skill("checklist")


def test_version_reads_distribution_metadata(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "version", lambda name: calls.append(name) or "1.2.3")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--version"])
    assert stopped.value.code == 0
    assert calls == ["stepbook"]
    output = capsys.readouterr()
    assert output.out == "1.2.3\n" and output.err == ""
