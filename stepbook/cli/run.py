"""The ``run`` subcommand: its arguments, the environment and controller it builds, and the run report."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from ..contracts.errors import AutomaticMeasurementRepeats, ExistingRun, MissingExecutorBinary
from ..contracts.journal import MalformedJournal
from ..envs.eval_loop import AdapterEvalEnv
from ..envs.shell import ShellEnv
from ..executors import command as executor_module
from ..executors.claude import ClaudeCLIExecutor
from ..reporting.decision_audit import against_verdict, decision_audit
from ..run_pointer import (
    LAST_RUN,
    RESUMABLE,
    classify_run,
    clear_active,
    mark_failed_launch,
    pointer_dir,
    read_pointer,
    write_launch_pointers,
    write_launch_record,
)
from ..runtime.loop import Runtime
from ..state.skill import Skill, skill_content_hash
from ..storage.events import EventLog
from .inspection import reopen_hint, write_report_md
from .plan import _configure_shell_launch_state, _merge_run_plan, _shell_checks, validate_plan
from .plan import _configure_shell_state as _configure_shell_state  # the research harness calls it through this module
from .run_args import executor_command, executor_option_problem, infer_run_options
from .skill import load_cli_skill


def _program_tokens() -> list[str]:
    """How this process was really started, for a resume command that works outside PATH."""
    exe = sys.argv[0] if sys.argv else ""
    if not exe or exe.endswith(("cli.py", "__main__.py")):
        return [sys.executable, "-m", "stepbook.cli"]
    return [str(Path(exe).resolve()) if Path(exe).exists() else exe]


def _normalize_run_args(args, require_completion):
    """Infer what the invocation implies, then validate run-only options."""
    infer_run_options(args)
    if not args.skill_dir:
        # The skill folder is optional only for a resume, which reads it from the run's launch record.
        print("stepbook run needs a skill folder, or --resume [RUN_DIR] to continue a recorded run", file=sys.stderr)
        return 2
    if problem := executor_option_problem(args):
        print(problem, file=sys.stderr)
        return 2
    if require_completion and not args.adapter:
        print("--require-completion requires --adapter", file=sys.stderr)
        return 2


def _load_run_skill(args):
    """Load the state skill with its ordinary schema instructions."""
    skill = load_cli_skill(args.skill_dir, include_schema=True)
    _merge_run_plan(args, skill)
    _validate_run_plan(args, skill)
    return skill


def _validate_run_plan(args, skill):
    """Validate the final launch state before constructing an environment or executor."""

    def configure(candidate):
        if args.adapter:
            _configure_adapter_state(args, candidate)
        else:
            _configure_shell_launch_state(args, candidate, _shell_checks(args))

    validate_plan(args, skill, configure)


def _print_shell_cap(args, skill):
    """Expose the effective shell allowance without changing adapter output."""
    if args.adapter:
        return
    budget = skill.initial_state.get("budget") or {}
    if "runs_max" not in budget:
        if args.runs_max is not None:
            print("runs_max: unavailable (skill has no budget.runs_max)", file=sys.stderr)
        return
    cap = budget["runs_max"]
    print(
        f"runs_max: {cap if cap is not None else 'unlimited'} "
        "(recorded for the agent; not host-enforced in shell mode)",
        file=sys.stderr,
    )


def _print_completed_resume(args, log_dir):
    """Explain the runtime's completed-journal no-op without changing replay."""
    if not args.resume or (getattr(args, "reopen", None) or "").strip():
        return
    info = classify_run(log_dir)
    if info["status"] == "completed":
        print(f"Completed run: resume is a no-op. {reopen_hint(info)}", file=sys.stderr)
    elif "stop_reason" in info:
        print(
            f"Run stopped by the agent: {info['stop_reason']}. Resume is a no-op. {reopen_hint(info)}", file=sys.stderr
        )


def _configure_adapter_state(args, skill):
    """Align the protected state with adapter budgets and operator requirements."""
    # the state's protected budget must agree with the environment's
    st = dict(skill.initial_state)
    st["budget"] = {**(st.get("budget") or {}), "runs_max": args.runs_max}
    st["protocol"] = {**(st.get("protocol") or {}), "max_runs": args.runs_max}
    if args.goal:
        st["goal"] = args.goal
    if args.constraint:
        st["user_requirements"] = list(st.get("user_requirements") or []) + [
            {"text": value, "done": False} for value in args.constraint
        ]
    skill.initial_state = st


def _build_adapter_environment(args, skill, require_completion):
    """Construct and preflight the adapter before configuring its state and helpers."""
    roots = tuple(str(Path(p).resolve()) for p in (getattr(args, "evidence_root", None) or []))
    if require_completion and not roots and not args.resume:
        print("--require-completion needs --evidence-root for source capture on a new run", file=sys.stderr)
        return None, roots
    if any(not Path(p).exists() for p in roots):
        print("--evidence-root must identify an existing file or directory", file=sys.stderr)
        return None, roots
    env = AdapterEvalEnv(
        adapter=args.adapter,
        runs_max=args.runs_max,
        resuming=args.resume,
        require_completion=require_completion,
        evidence_roots=roots,
        timeout_s=args.adapter_timeout,
        runs_per_hypothesis=skill.initial_state.get("protocol", {}).get("runs_per_hypothesis", 2),
        baseline_runs=skill.initial_state.get("protocol", {}).get("baseline_runs"),
    )
    if roots:
        preflight = env._call(["describe"])
        if not preflight.ok:  # the verb's exit code, not a recognisable first word
            print(preflight.text, file=sys.stderr)
            return None, roots  # no model, runtime journal or reset before a valid evidence-scoped setup
    _configure_adapter_state(args, skill)
    return env, roots


def _build_shell_environment(args: argparse.Namespace, skill: Skill) -> ShellEnv | None:
    """Validate shell checks and construct the environment from the merged initial state."""
    # A flag wins when it is given; otherwise the plan's (or the skill's) own value is the launch value.
    goal = args.goal if args.goal is not None else skill.initial_state.get("goal")
    if not args.workdir or not goal:
        print("shell mode needs --workdir and --goal", file=sys.stderr)
        return None
    try:
        checks = _shell_checks(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return None
    _configure_shell_launch_state(args, skill, checks)
    state = skill.initial_state  # the launch state now holds the goal, checks and constraints the run uses
    env = ShellEnv(
        workdir=Path(args.workdir),
        goal=goal,
        checks=dict(state.get("checks") or {}),
        constraints=list(state.get("constraints") or []),
        horizon=args.steps,
        cmd_timeout_s=args.cmd_timeout,
        allow_outside=args.allow_outside,
        checkpoint=not args.no_checkpoint,
        require_sandbox=not args.unsandboxed_i_accept_the_risk,
        log_dir=Path(args.log_dir) if getattr(args, "log_dir", None) else None,
        allow_dirty=getattr(args, "allow_dirty", False),
        resuming=args.resume,
    )
    env.preflight()
    return env


def _build_run_environment(args, skill, require_completion):
    """Select the environment the invocation implies and retain adapter evidence roots for the controller."""
    roots = ()
    if args.adapter:
        env, roots = _build_adapter_environment(args, skill, require_completion)
    else:
        env = _build_shell_environment(args, skill)  # shell is the environment without --adapter
    if env is None:
        return None, roots
    return env, roots


def _select_run_executor(args, env, roots):
    """Select the controller, preserving its model and CWD contract."""
    executor_kind = args.executor
    if executor_kind in ("command", "codex", "opencode"):
        cwd = None if args.adapter else args.workdir
        ex = executor_module.CommandExecutor(
            command=executor_command(args),
            timeout_s=args.controller_timeout,
            cwd=Path(cwd).resolve() if cwd else None,
        )
    else:
        ex = ClaudeCLIExecutor(
            model=args.model,
            timeout_s=args.controller_timeout,
            cwd=str(args.workdir) if getattr(args, "workdir", None) else None,
        )
    return ex, executor_kind


def _prepare_run_directory(args, skill, env):
    """Resolve the directory and refuse journal reuse before writing launch artifacts."""
    if args.resume and not args.log_dir:
        # A paused run must be discoverable from a fresh chat without remembering its directory.
        last = read_pointer(pointer_dir(), LAST_RUN)
        info = classify_run(last) if last else None
        if info is None or info["status"] not in RESUMABLE:
            why = "no .stepbook/LAST_RUN pointer" if info is None else f"{info['run_dir']} is {info['status']}"
            print(f"--resume needs --log-dir pointing at the interrupted run ({why})", file=sys.stderr)
            return None
        print(f"resuming {info['run_dir']} ({info['status']}; {info['steps']} steps recorded)", file=sys.stderr)
        args.log_dir = info["run_dir"]
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else Path(".stepbook") / f"{skill.name}-{__import__('time').strftime('%Y%m%d-%H%M%S')}"
    )
    if hasattr(env, "log_dir"):
        env.log_dir = log_dir
    if args.resume and classify_run(log_dir)["status"] == "failed_to_start":
        raise ValueError("This run failed to start; start a new run instead of resuming it.")
    journal = log_dir / "events.jsonl"
    if not args.resume and journal.exists() and EventLog(journal).read():
        raise ExistingRun(f"{journal} already has events; use --resume (resume=True) or a new --log-dir")
    return log_dir


def _preflight_executor(ex, executor_kind=None):
    """Resolve the executable with the same working directory as the controller subprocess."""
    if executor_kind in ("codex", "opencode"):
        configured = os.environ.get(f"{executor_kind.upper()}_BIN", executor_kind)
        tokens = shlex.split(configured) if executor_kind == "codex" else [configured]
        binary = tokens[0] if tokens else configured
    elif isinstance(ex, executor_module.CommandExecutor):
        command = ex.command.format(system_prompt_file="prompt", schema_file="schema")
        tokens = shlex.split(command)
        if not tokens:
            raise ValueError("--command needs an executable")
        binary = tokens[0]
    else:
        binary = getattr(ex, "claude_bin", None)
    if binary is None:
        return
    cwd = Path(getattr(ex, "cwd", None) or ".")
    if not cwd.is_dir():
        raise ValueError(f"controller working directory does not exist: {cwd}")
    candidate = str((cwd / binary).resolve()) if "/" in binary else binary
    search_path = os.pathsep.join(str((cwd / entry).resolve()) for entry in os.get_exec_path())
    if shutil.which(candidate, path=search_path) is None:
        raise MissingExecutorBinary(f"executor {binary!r} was not found; install it and make it available on PATH")


def _build_runtime(args, skill, ex, env, log_dir):
    """Construct the product runtime with fixed failure protections."""
    return Runtime(
        skill,
        ex,
        env,
        log_dir=log_dir,
        verbose=not args.quiet,
        state_max_chars=getattr(args, "state_max_chars", None),
    )


def _run_exit_status(log_dir, unsuccessful, *, pause_code=3):
    """Apply product journal outcome precedence to the reporter's failure rule."""
    status = classify_run(log_dir)["status"]
    if status == "paused":
        return pause_code
    if status == "failed_checks":
        return 1
    return 0 if status == "completed" and not unsuccessful else 3


def _report_run(rt, res, env, log_dir, executor_kind, require_completion):
    """Write run artifacts, print the summary and distinguish paused and unsuccessful runs."""
    out = res.summary()
    out["executor"] = executor_kind
    out["log_dir"] = str(log_dir)
    completion_events = rt.log.read()
    completions = [
        e
        for e in completion_events
        if e.get("kind") == "step" and e.get("done") and (e.get("info") or {}).get("completion_id")
    ]
    if completions:
        out["completion_artifact"] = completions[-1]["info"]["artifact_path"]
        allowed = [
            e
            for e in completion_events
            if e.get("kind") == "local_completion_allowed_after_budget" and e.get("step") == completions[-1].get("step")
        ]
        out["completion_after_spending_limit"] = bool(allowed)
        if allowed:
            out["completion_budget_excess_usd"] = max(0, allowed[-1]["cost_usd"] - allowed[-1]["max_total_usd"])
    if require_completion:
        out["completion_required"] = True
        out["completion_saved"] = bool(completions)
    try:
        rp = write_report_md(Path(log_dir), env, res)
        out["report"] = str(rp)
    except Exception as exc:  # the report must never mask the run's result
        out["report_error"] = str(exc)
    try:
        out["decisions_against_verdict"] = len(against_verdict(decision_audit(completion_events)))
    except Exception as exc:
        out["decision_audit_error"] = str(exc)
    if res.final_state is not None:
        (log_dir / "state.json").write_text(json.dumps(res.final_state, indent=2))
        out["state_file"] = str(log_dir / "state.json")
    print(json.dumps(out, indent=2))
    return _run_exit_status(log_dir, getattr(res, "aborted", False) or (require_completion and not completions))


def _run_runtime(args, rt):
    """Run with the required replay policy and preserve the malformed-journal refusal."""
    try:
        # Replay re-executes journaled actions to rebuild a deterministic environment. Both product environments
        # are live systems (adapters, docker, evals, files): their journals already hold every result, so resume
        # must never re-run commands.
        res = (
            rt.run(args.steps, resume=args.resume, replay_env=False, reopen=getattr(args, "reopen", None))
            if isinstance(rt, Runtime)
            else rt.run(args.steps)
        )
    except MalformedJournal as refusal:
        # Match the sentence and exit status recover gives for the same journal.
        print(str(refusal), file=sys.stderr)
        return None
    return res


def cmd_run(args: argparse.Namespace) -> int:
    """Run a skill to completion against an environment: --adapter <cmd> (eval-loop-adapter style), else shell mode."""
    require_completion = getattr(args, "require_completion", False)
    if _normalize_run_args(args, require_completion) == 2:
        return 2
    log_dir = None
    created = False
    published = False
    try:
        skill = _load_run_skill(args)
        env, roots = _build_run_environment(args, skill, require_completion)
        if env is None:
            return 2
        ex, executor_kind = _select_run_executor(args, env, roots)
        if ex is None:
            return 2
        _preflight_executor(ex, executor_kind)
        log_dir = _prepare_run_directory(args, skill, env)
        if log_dir is None:
            return 2
        try:
            log_dir.mkdir(parents=True)
        except FileExistsError:
            pass
        else:
            created = True
        write_launch_record(
            log_dir,
            getattr(args, "raw_argv", None) or [],
            resume=args.resume,
            program=_program_tokens(),
            skill_hash=skill_content_hash(args.skill_dir),
        )
        published = True
        write_launch_pointers(log_dir)  # ACTIVE while alive, LAST_RUN durably; the plugin hooks read both
        _print_shell_cap(args, skill)
        _print_completed_resume(args, log_dir)
        rt = _build_runtime(args, skill, ex, env, log_dir)
        res = _run_runtime(args, rt)
    except AutomaticMeasurementRepeats:
        raise  # The CLI adds the remedy while the runtime retains its historical refusal text.
    except Exception as exc:
        if log_dir is not None and classify_run(log_dir)["steps"]:
            raise
        if created and log_dir is not None and not args.resume:
            mark_failed_launch(log_dir, exc)
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        if published:
            clear_active()
    return 2 if res is None else _report_run(rt, res, env, log_dir, executor_kind, require_completion)
