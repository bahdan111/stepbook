"""Argument registration for the ``run`` subcommand: its five parser helpers and their argparse types."""

from __future__ import annotations

import argparse
import math
import shlex
import sys


def _add_run_source_arguments(p):
    """Register skill, adapter evidence, and resume arguments in their existing order."""
    p.add_argument("skill_dir", nargs="?", help="the skill folder to run; a short resume reads it from the record")
    p.add_argument("--plan", metavar="FILE", help="initial-state JSON patch; reread on resume")
    p.add_argument("--adapter", help="eval adapter command, e.g. 'python my_adapter.py' (kept outside this repo)")
    p.add_argument(
        "--evidence-root",
        action="append",
        default=[],
        help="allow read actions within this file/directory (repeatable); PDF/XLSX require documents extra",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        metavar="RUN_DIR",
        help="continue RUN_DIR (default: the active, else the last run) from the launch recorded in it; "
        "name a skill folder instead to supply the whole launch again",
    )
    p.add_argument(
        "--state-max-chars",
        type=positive_int,
        default=None,
        help="age the state RENDERING when its compact JSON exceeds N characters: the oldest entries of the largest "
        "fields are shown as one line each (last 6 stay whole) and the prompt points at the full state.json. "
        "Default: no cap.",
    )
    p.add_argument(
        "--reopen",
        default=None,
        help="an operator message that lifts a completed run's done and continues it in the same journal, e.g. 'the pause is over, continue with the remaining hypotheses'",
    )


def _add_run_environment_arguments(p):
    """Register built-in environment arguments without changing defaults."""
    p.add_argument("--workdir", help="the working directory (use a disposable checkout or a container)")
    p.add_argument("--goal", help="what to accomplish, in plain words; the plan or the skill may supply it instead")
    p.add_argument("--check", action="append", default=[], help="'id=<command>' that must exit 0 for done (repeatable)")
    p.add_argument("--constraint", action="append", default=[], help="a protected human rule (repeatable)")
    p.add_argument("--cmd-timeout", type=float, default=7200.0, help="shell mode: seconds per command (default: 2h)")
    p.add_argument("--allow-outside", action="store_true", help="shell mode: allow paths outside the working directory")
    p.add_argument("--allow-dirty", action="store_true", help="shell mode: include existing changes in checkpoints")
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="shell mode: do not create a branch and commit after each changing step",
    )
    p.add_argument("--unsandboxed-i-accept-the-risk", action="store_true", help="run outside a container/VM anyway")


def _add_run_limit_arguments(p):
    """Register controller and adapter timeouts."""
    p.add_argument(
        "--controller-timeout",
        type=positive_float,
        default=900,
        help="seconds per controller/helper query; timeout preserves journal for recovery",
    )
    p.add_argument("--adapter-timeout", type=positive_float, default=7200, help="seconds per adapter operation")


def _add_run_completion_arguments(p):
    """Register horizons and completion policy."""
    p.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help="step limit, counting total steps including earlier invocations (default: until done, stop or pause)",
    )
    p.add_argument(
        "--runs-max",
        type=nonnegative_int,
        default=None,
        help="how many evaluations the run may spend; enforced by the host only for adapter loops, "
        "otherwise shown to the agent when the skill has budget.runs_max (default: no limit)",
    )
    p.add_argument(
        "--require-completion",
        action="store_true",
        help="Require a host-saved final evidence artifact via finish instead of plain stop (adapter loops)",
    )


def _add_run_executor_arguments(p):
    """Register controller selection and logging arguments."""
    add_executor_arguments(p)
    p.add_argument("--log-dir", help="where to keep this run (default: .stepbook/<skill>-<timestamp>)")
    p.add_argument("--quiet", action="store_true")


def infer_run_options(args) -> None:
    """Fill in what a run invocation already implies, so the longer old spellings stay valid.

    ``--reopen`` continues an existing journal, and a ``--command`` names the agent to call; neither
    needs a second flag to say so. Explicit ``--resume`` and ``--executor`` keep their meaning.
    """
    normalize_resume_target(args)
    if getattr(args, "reopen", None):
        args.resume = True
    if getattr(args, "command", None) and getattr(args, "executor", None) == "cli":
        args.executor = "command"


def normalize_resume_target(args) -> None:
    """``--resume RUN_DIR`` names the run to continue, which is what ``--log-dir`` says the long way."""
    if isinstance(getattr(args, "resume", False), str):
        args.log_dir = getattr(args, "log_dir", None) or args.resume
        args.resume = True


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_int(value):
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def add_executor_arguments(p):
    """Product executor selection shared by run and author."""
    p.add_argument("--model", default="opus", help="the model for --executor cli; a --command carries its own")
    p.add_argument(
        "--executor",
        choices=["cli", "command", "codex", "opencode"],
        default="cli",
        help="cli: installed Claude CLI; codex/opencode: built-in step drivers; command: a caller-configured agent command",
    )
    p.add_argument(
        "--command",
        help="any agent command, run once per step: it reads the prompt on stdin and prints the step's JSON, "
        "e.g. 'python agent.py'",
    )
    p.add_argument(
        "--allow-api-billing",
        action="store_true",
        help="allow a model login that bills an API key instead of a subscription (same as "
        "STEPBOOK_ALLOW_API_BILLING=1); off by default",
    )


def executor_command(args) -> str:
    """Use the current interpreter for built-in drivers and leave custom commands intact."""
    if args.executor == "command":
        return args.command
    argv = [sys.executable, "-m", f"stepbook.executors.{args.executor}", "step"]
    explicit_model = any(arg == "--model" or arg.startswith("--model=") for arg in getattr(args, "raw_argv", []))
    if args.model != "opus" or explicit_model:
        argv += ["--model", args.model]
    # The step driver must allow the same query time the host does; author has no controller timeout of its own.
    if timeout := getattr(args, "controller_timeout", None):
        argv += ["--timeout", f"{timeout:g}"]
    # CommandExecutor substitutes only its placeholders, including when a model/path contains braces.
    return shlex.join(argv).replace("{", "{{").replace("}", "}}") + " --schema {schema_file}"


def executor_option_problem(args) -> str | None:
    """The command itself selects its model; no ignored CLI model option."""
    if args.executor in ("codex", "opencode") and args.command is not None:
        return f"--command is not supported by --executor {args.executor}; use --executor command"
    if args.executor != "command":
        return None
    if args.model != "opus" or "--model" in getattr(args, "raw_argv", []):
        return "--model is not supported by --executor command; put the model in the command string"
    if not args.command:
        return "--executor command needs --command"
    return None
