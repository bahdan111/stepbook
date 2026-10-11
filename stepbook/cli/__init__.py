"""Product CLI: run, validate, scaffold and inspect durable execution state."""

from __future__ import annotations

import argparse
import os
import sys
import weakref
from importlib.metadata import version

from ..contracts.errors import (
    AutomaticMeasurementRepeats,
    CompletionPolicyChanged,
    ExistingRun,
    LegacyAdapterJournal,
    MissingExecutorBinary,
    MissingSkillFolder,
    UnreceiptedAction,
)
from ..executors.auth import SubscriptionError, preflight_command
from ..runtime.logging_setup import trace_to_stdout
from ..state.skill import LockPathError

# Public commands remain importable here; private run helpers live in cli.run.
from .adapter import ADAPTER_TEMPLATE as ADAPTER_TEMPLATE
from .adapter import GENERIC_EVAL_TEMPLATE as GENERIC_EVAL_TEMPLATE
from .adapter import cmd_adapter_check as cmd_adapter_check
from .adapter import cmd_adapter_generic as cmd_adapter_generic
from .adapter import cmd_init as cmd_init
from .controls import add_control_parsers
from .inspection import cmd_recover as cmd_recover
from .inspection import cmd_report as cmd_report
from .inspection import cmd_status as cmd_status
from .inspection import write_report_md as write_report_md
from .resume import expand_recorded_resume, is_short_resume
from .run import cmd_run as cmd_run
from .run_args import _add_run_completion_arguments as _add_run_completion_arguments
from .run_args import _add_run_environment_arguments as _add_run_environment_arguments
from .run_args import _add_run_executor_arguments as _add_run_executor_arguments
from .run_args import _add_run_limit_arguments as _add_run_limit_arguments
from .run_args import _add_run_source_arguments as _add_run_source_arguments
from .run_args import add_executor_arguments
from .run_args import infer_run_options as infer_run_options
from .run_args import nonnegative_int as nonnegative_int
from .run_args import positive_float as positive_float
from .run_args import positive_int as positive_int
from .skill import cmd_author as cmd_author
from .skill import cmd_validate as cmd_validate


def _add_skill_parsers(sub) -> None:
    """validate: the subcommands that act on a skill folder directly."""
    p = sub.add_parser(
        "validate",
        help="check a skill folder, and a launch plan for it, without running anything",
        allow_abbrev=False,
    )
    p.add_argument("skill_dir")
    p.add_argument("--plan", metavar="FILE", help="validate an initial-state JSON merge patch without launching")
    p.add_argument("--goal", help="with --plan: override the goal as run would")
    p.add_argument("--constraint", action="append", help="with --plan: replace constraints (repeatable)")
    p.add_argument("--check", action="append", help="with --plan: replace checks with 'id=<command>' (repeatable)")
    p.set_defaults(fn=cmd_validate)


def _add_author_parser(sub) -> None:
    """Draft a skill through the product executor interface."""
    p = sub.add_parser(
        "author", help="draft and validate a skill folder from a procedure or an existing skill", allow_abbrev=False
    )
    p.add_argument("dir")
    p.add_argument(
        "description",
        nargs="?",
        help="the procedure in plain words; with --from, optional notes for drafting the state",
    )
    p.add_argument(
        "--from",
        dest="from_skill",
        help="an existing skill's SKILL.md or its folder: keep its text as the instructions and draft only the state "
        "schema, the initial state and a section on running it step by step",
    )
    add_executor_arguments(p)
    p.set_defaults(fn=cmd_author)


RUN_HELP_GROUPS = (
    ("Task", ("skill_dir", "--plan", "--goal", "--check", "--constraint")),
    ("Where", ("--workdir", "--adapter", "--allow-outside")),
    ("Limits", ("--steps", "--runs-max")),
    ("Agent", ("--executor", "--model", "--command")),
    (
        "Run",
        (
            "--log-dir",
            "--resume",
            "--reopen",
            "--cmd-timeout",
            "--no-checkpoint",
            "--allow-dirty",
            "--unsandboxed-i-accept-the-risk",
        ),
    ),
)


_HIDDEN_HELP: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # option -> the help --help hides


class _HelpAllAction(argparse.Action):
    """Restore the help text of the options --help hides, then print the full help."""

    def __call__(self, parser, namespace, values, option_string=None):
        for action in parser._actions:
            if action in _HIDDEN_HELP:
                action.help = _HIDDEN_HELP[action]
        parser.print_help()
        parser.exit()


def _group_run_help(p: argparse.ArgumentParser) -> None:
    """Show the routinely typed options in plain groups and keep the rest out of ``run --help``.

    The options stay registered and accepted; only their help moves, which argparse expresses through
    the groups a parser already owns.
    """
    defaults = list(p._action_groups)
    actions = {name: a for a in p._actions for name in (a.option_strings or [a.dest])}
    placed = []
    for title, names in RUN_HELP_GROUPS:
        group = p.add_argument_group(title)
        for name in names:
            group._group_actions.append(actions[name])
            placed.append(actions[name])
    rest = p.add_argument_group("More (see --help-all)")
    for action in p._actions:
        if action in placed or action.dest in ("help", "help_all"):
            continue
        rest._group_actions.append(action)
        _HIDDEN_HELP[action] = action.help  # an option without a help text is hidden the same way
        action.help = argparse.SUPPRESS
        placed.append(action)
    for group in defaults:
        group._group_actions = [a for a in group._group_actions if a not in placed]


def _add_loop_parsers(sub) -> None:
    """run and init: start a loop, or scaffold the files one needs."""
    p = sub.add_parser(
        "run",
        help="run a skill against an adapter or a built-in environment",
        allow_abbrev=False,
        epilog=(
            "Exit codes: 0 goal met (done/finish with all configured checks passing, or normal adapter completion); "
            "1 checks failed; 2 invalid invocation or launch/resume refused; "
            "3 incomplete (agent stop, step horizon reached, abort or required completion missing), or paused (resumable)."
        ),
    )
    p.add_argument("--help-all", action=_HelpAllAction, nargs=0, help="show every option, including the rare ones")
    _add_run_source_arguments(p)
    _add_run_environment_arguments(p)
    _add_run_limit_arguments(p)
    _add_run_completion_arguments(p)
    _add_run_executor_arguments(p)
    _group_run_help(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser(
        "init",
        help="scaffold an eval loop and adapter starter for your own measurable workflow",
        allow_abbrev=False,
    )
    p.add_argument("dir")
    p.add_argument(
        "--config-only",
        action="store_true",
        help="write no-code eval.json for an existing evaluator that emits JSON or CSV",
    )
    p.set_defaults(fn=cmd_init)


def _add_adapter_parsers(sub) -> None:
    """adapter, with its own check and generic subcommands."""
    p = sub.add_parser("adapter", help="adapter tools", allow_abbrev=False)
    sp = p.add_subparsers(dest="sub", required=True)
    c = sp.add_parser("check", allow_abbrev=False)
    c.add_argument("adapter")
    c.add_argument("--run", action="store_true", help="also perform one real eval run")
    c.add_argument(
        "--evidence-root",
        action="append",
        default=[],
        help="check proposed read roots too; generic adapter requires its config and fresh result directory",
    )
    c.set_defaults(fn=cmd_adapter_check)
    c = sp.add_parser("generic", help="run the bundled no-code JSON/CSV eval adapter", allow_abbrev=False)
    c.add_argument("eval_config")
    c.add_argument("adapter_args", nargs=argparse.REMAINDER)
    c.set_defaults(fn=cmd_adapter_generic)


def _add_inspect_parsers(sub) -> None:
    """status, recover and report: read a run directory after or during the run."""
    p = sub.add_parser(
        "status",
        help="what is running or resumable (from .stepbook/ACTIVE, then LAST_RUN, or --run-dir)",
        allow_abbrev=False,
    )
    p.add_argument(
        "--run-dir", "--log-dir", dest="run_dir", help="inspect this run directory (same value as run --log-dir)"
    )
    p.add_argument("run_dir_positional", nargs="?", metavar="run_dir", help="run directory to inspect")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser(
        "recover",
        help="inspect uncertain actions; explicitly reconcile a verified external receipt",
        allow_abbrev=False,
    )
    p.add_argument("run_dir")
    p.add_argument("--action-id")
    p.add_argument("--receipt")
    p.add_argument("--reason", help="operator explanation of how the actual external outcome was verified")
    p.set_defaults(fn=cmd_recover)
    p = sub.add_parser(
        "report",
        help="print a fresh run report to stdout; does not rewrite report.md",
        description="Print a fresh run report to stdout; does not rewrite the saved report.md.",
        allow_abbrev=False,
    )
    p.add_argument("run_dir")
    p.add_argument("--last", type=int, default=30)
    p.set_defaults(fn=cmd_report)


class _VersionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(version("stepbook"))
        parser.exit()


def _build_parser() -> argparse.ArgumentParser:
    """The stepbook parser, its subcommands registered in the order they appear in --help."""
    # No abbreviated options anywhere: an acknowledgement like --unsandboxed-i-accept-the-risk must be typed in
    # full, here and in a recorded launch a resume replays.
    ap = argparse.ArgumentParser(prog="stepbook", allow_abbrev=False)
    ap.add_argument("--version", action=_VersionAction, nargs=0, help="show the installed version and exit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_skill_parsers(sub)
    _add_author_parser(sub)
    _add_loop_parsers(sub)
    _add_adapter_parsers(sub)
    _add_inspect_parsers(sub)
    add_control_parsers(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)  # recorded per run for an exact resume command
    args = ap.parse_args(raw_argv)
    args.raw_argv = raw_argv
    if args.cmd == "run" and is_short_resume(args):
        try:  # the recorded launch, re-read as an ordinary invocation in the directory it was launched from
            expanded, launch_cwd = expand_recorded_resume(args, raw_argv)
        except ValueError as refusal:
            print(str(refusal), file=sys.stderr)
            return 2
        os.chdir(launch_cwd)
        args = ap.parse_args(expanded)
        args.raw_argv = raw_argv
    if args.cmd in ("run", "author"):
        infer_run_options(args)  # before the subscription preflight, which only attests --executor cli
    try:
        preflight_command(args)
    except SubscriptionError as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        with trace_to_stdout(hasattr(args, "quiet") and not args.quiet):  # only the verb that defines --quiet
            return args.fn(args)
    except (
        ExistingRun,
        UnreceiptedAction,
        CompletionPolicyChanged,
        AutomaticMeasurementRepeats,
        LegacyAdapterJournal,
        MissingSkillFolder,
        MissingExecutorBinary,
        LockPathError,
    ) as refusal:
        message = str(refusal)
        if isinstance(refusal, AutomaticMeasurementRepeats):
            message = message.rstrip(".") + "; start a new run"
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
