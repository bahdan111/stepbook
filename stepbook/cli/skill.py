"""Validate an existing skill or draft a new skill folder."""

import argparse
import sys
from pathlib import Path

from jsonschema.exceptions import SchemaError

from ..contracts.errors import ExecutorError, MissingSkillFolder
from ..executors.claude import ClaudeCLIExecutor
from ..executors.command import CommandExecutor
from ..state.skill import Skill
from ..state.validator import PatchError, Validator
from .author import author_skill, author_skill_from
from .plan import _configure_shell_state, _merge_run_plan, _shell_checks, validate_plan
from .run_args import executor_command, executor_option_problem

BUILTIN_SKILLS = ("eval-loop-adapter", "checklist", "eval-loop")


def builtin_skill_path(name: str) -> Path:
    """Resolve built-in task skills from the installed package."""
    package = Path(__file__).resolve().parents[1]
    return package / "templates" / name


def cmd_validate(args: argparse.Namespace) -> int:
    skill = load_cli_skill(args.skill_dir)
    if getattr(args, "plan", None):
        try:
            keys = _merge_run_plan(args, skill)
            validate_plan(
                args,
                skill,
                lambda candidate: _configure_shell_state(args, candidate, _shell_checks(args), only_supplied=True),
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        v = Validator(skill.schema, skill.protected_paths, skill.max_deletions)
        try:
            v.apply(skill.initial_state, {})
        except PatchError as e:
            print("initial_state does not satisfy schema:", e, file=sys.stderr)
            return 1
    print(
        f"ok: skill {skill.name!r}, {len(skill.instructions)} chars of instructions, "
        f"protected={list(skill.protected_paths)}, max_deletions={skill.max_deletions}"
    )
    if getattr(args, "plan", None):
        print(f"plan valid: top-level keys: {', '.join(keys) or '(none)'}")
    return 0


def load_cli_skill(folder: str, *, include_schema: bool | None = None) -> Skill:
    """Report missing skill folders and files through the CLI refusal path."""
    path = Path(folder)
    if not path.exists() and folder in BUILTIN_SKILLS:
        path = builtin_skill_path(folder)
    if not path.is_dir():
        raise MissingSkillFolder(f"skill folder {folder!r} does not exist; choose an existing skill folder")
    try:
        return Skill.load(path, include_schema=include_schema)
    except FileNotFoundError as error:
        raise MissingSkillFolder(
            f"skill folder {folder!r} has no {Path(error.filename).name}; "
            "a task skill folder holds SKILL.md, schema.json and initial_state.json"
        ) from error


def cmd_author(args: argparse.Namespace) -> int:
    """Draft with the selected product executor; refuse invalid or existing targets."""
    if problem := executor_option_problem(args):
        print(problem, file=sys.stderr)
        return 2
    try:
        if args.description is None and args.from_skill is None:
            raise ValueError("one of DESCRIPTION or --from is needed")
        target = Path(args.dir)
        executor = (
            ClaudeCLIExecutor(model=args.model)
            if args.executor == "cli"
            else CommandExecutor(command=executor_command(args))
        )
        if args.from_skill is not None:
            out, source = author_skill_from(Path(args.from_skill), target, executor, notes=args.description)
        else:
            out = author_skill(args.description, target, executor)
    except (OSError, ValueError, PatchError, SchemaError, ExecutorError) as error:
        reason = " ".join(str(error).splitlines())
        print(f"author refused: {reason}", file=sys.stderr)
        return 2
    if args.from_skill is not None:
        print(
            f"drafted skill at {out} from {source}. Your SKILL.md text is unchanged at the top of {out}/SKILL.md; "
            f"review the added section and schema.json, then: stepbook validate {out}"
        )
    else:
        print(f"drafted skill at {out}. Review SKILL.md and schema.json, then: stepbook validate {out}")
    return 0
