"""Algorithm 1 from the paper: explicit, validated execution state.

for t in 0..T:
    O_t = latest observation
    prompt = (P, Σ_t, O_t)
    R_t, ΔΣ_t, a_t = LLM(prompt)
    validate(ΔΣ_t)            # invalid -> the error becomes the observation, retry
    Σ_{t+1} = Σ_t ⊕ ΔΣ_t
    execute(a_t)

The loop lives here; prompts in rendering.py, aging in aging.py, retry in retry.py, the run
contracts in contracts.py, the skill folder in skill.py.
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from itertools import count
from pathlib import Path
from typing import Any, Unpack, cast

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ..contracts.errors import AutomaticMeasurementRepeats, CompletionPolicyChanged, ExistingRun
from ..contracts.journal import replayable_steps
from ..contracts.runtime import Environment, RunResult
from ..contracts.runtime import StepResult as StepResult
from ..executors.base import Envelope, StepExecutor
from ..reporting.usage import controller_receipts, envelope_usage, usage_summary

# ``X as X`` keeps a moved name importable and monkeypatchable: ruff and vulture both honour the explicit re-export.
from ..state.aging import PROTECTED_REFERENCE_FIELDS as PROTECTED_REFERENCE_FIELDS
from ..state.aging import STALE_OBSERVATION_MINUTES as STALE_OBSERVATION_MINUTES
from ..state.aging import abbreviate_state as abbreviate_state
from ..state.aging import stale_observation_note
from ..state.merge import merge
from ..state.patch_schema import state_output_schema
from ..state.response_validation import StateCheck, report_output_schema, response_problems, translate_report_action
from ..state.skill import Skill
from ..state.skill import render_schema_section as render_schema_section
from ..state.validator import PatchError, Validator
from ..storage.events import EventLog
from ..storage.recovery import exclusive_run, execute_action, recover_actions
from .controls import (
    ControlStop,
    check_control,
    operator_note,
    read_inbox,
    withheld_action_note,
)
from .logging_setup import trace_step
from .observations import ObservationShaper
from .rendering import DECISION_SUMMARY as DECISION_SUMMARY
from .rendering import STATE_OUTPUT_SCHEMA, render_patch_repair, render_state_prompt
from .rendering import render_recent_steps as render_recent_steps
from .retry import call_with_retry
from .settings import RuntimeKwargs, RuntimeSettings

_ON_EXHAUSTED = "abort"
_SKIP_ACTION = "Wait"
_REPEAT_LIMIT = 3


class _RunProgress(BaseModel):
    """Mutable run accounting and the proposal shared by the step phases."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    recent: list[tuple[int, str, str]] = Field(default_factory=list)
    start: float  # Wall-clock start; steps also supplies the resumed loop offset.
    steps: int = 0
    calls: int = 0
    rejected: int = 0
    assisted: int = 0
    assisted_ok: int = 0
    prompt_tokens: list[int] = Field(default_factory=list)
    output_tokens: list[int] = Field(default_factory=list)
    cost: float = 0.0
    rendered: list[int] = Field(default_factory=list)
    model_ids: set[str] = Field(default_factory=set)
    aborted: bool = False
    episode_done: bool = False
    env_out: Envelope | None = None
    new_state: dict[str, Any] | None = None
    changed: list[str] | None = None
    attempt: int = 0
    rendered_prompt_chars: int = 0
    attempt_observation: str = ""  # Observation for the current attempt, including any repair feedback.
    shape_info: dict[str, Any] = Field(default_factory=dict)


class _Replay(BaseModel):
    """What one resume rebuilds from the journal, shared by the replay phases.

    ``SkipValidation`` as in the settings models, though it guards less here than it looks: pydantic validates
    the constructor and not assignment, and the journal's own values -- patches, results, step numbers -- reach
    this carrier by assignment. Dropping it would refuse or coerce nothing the journal wrote; the one thing it
    would newly refuse is an ``observation`` the environment's ``reset()`` did not return as a string.
    """

    state: SkipValidation[dict[str, Any]]
    observation: SkipValidation[str]
    totals: SkipValidation[dict[str, Any]]
    recent: SkipValidation[list[tuple[int, str, str]]] = Field(default_factory=list)


class Runtime:
    """stepbook runtime."""

    def __init__(
        self,
        skill: Skill,
        executor: StepExecutor,
        env: Environment,
        *,
        settings: RuntimeSettings | None = None,
        **overrides: Unpack[RuntimeKwargs],
    ):
        """Resolve product settings; failure protections have fixed defaults."""
        config = RuntimeSettings.resolve(settings, overrides)
        self.on_exhausted = _ON_EXHAUSTED
        self.skip_action = _SKIP_ACTION
        self.repeat_limit = _REPEAT_LIMIT
        self._repeats: dict[tuple[str, str], int] = {}
        self.recent_steps = config.recent_steps
        self.skill = skill
        self.output_schema = state_output_schema(STATE_OUTPUT_SCHEMA, skill.schema)
        self.report_schema = getattr(env, "response_report_schema", None)
        self.output_schema = report_output_schema(self.output_schema, self.report_schema)
        self.executor = executor
        self.env = env
        self.validator = Validator(
            skill.schema, skill.protected_paths, skill.max_deletions, write_once_paths=skill.write_once_paths
        )
        log_dir = config.log_dir
        self.log = EventLog(Path(log_dir) / "events.jsonl") if log_dir else None
        self.shaper = config.shaper or ObservationShaper(archive_dir=Path(log_dir) / "obs" if log_dir else None)
        self.state_max_chars = config.state_max_chars
        self._state_cap_warned = False
        self._max_steps: int | None = None  # the horizon run() is given; see _render
        self.max_retries = config.max_retries
        self.verbose = config.verbose

    def _journal_completed(self) -> bool:
        """A journal counts as completed only if its last `done` step comes AFTER the last `reopened` event."""
        last_done = last_reopened = None
        log = cast(EventLog, self.log)  # only reached from run() under `self.log and ...`
        for i, e in enumerate(log.read()):
            if e.get("kind") == "step" and e.get("done") and not e.get("horizon_stop"):
                last_done = i  # a step marked done by the environment's step guard is not completion (horizon_stop)
            elif e.get("kind") == "reopened":
                last_reopened = i
        return last_done is not None and (last_reopened is None or last_done > last_reopened)

    def _render(self, state, observation, recent, t):
        state_path = str(self.log.path.parent / "state.json") if self.log else None
        prompt = render_state_prompt(
            state,
            observation,
            recent,
            state_max_chars=self.state_max_chars,
            state_path=state_path,
            standing_notes=self._standing_notes(),
            step=t,
            max_steps=self._max_steps,
            on_state_size=partial(self._warn_state_overrun, t=t),
        )

        if self.report_schema is None:
            return prompt
        return (
            prompt.removesuffix("3. action: the exact command you want to execute")
            + "3. report: a JSON object in the SAME response, not an escaped string or standalone command. "
            "Return only that response object; put the decision summary in reasoning."
        )

    def _warn_state_overrun(self, actual: int, t: int) -> None:
        """Warn outside the prompt, once per run invocation (including a resumed invocation).

        The renderer measures compact JSON after aging, including omission markers and JSON escaping. The cap
        applies to that state block, not the full prompt or the unaged state saved on disk.
        """
        if not self.state_max_chars or self._state_cap_warned:
            return
        if actual > self.state_max_chars:
            print(
                f"Warning: step {t} rendered state is {actual} characters, "
                f"exceeding --state-max-chars {self.state_max_chars}; "
                "state aging is best-effort and retained content can exceed the cap. "
                "Further state-cap warnings are suppressed for this run.",
                file=sys.stderr,
            )
            self._state_cap_warned = True

    def _standing_notes(self) -> list[str]:
        """Replay operator notes so requirements survive schema changes and resumes."""
        if not self.log:
            return list(getattr(self, "_notes_memory", []))
        notes: list[str] = []
        for e in self.log.read():
            if e.get("kind") == "operator_messages":
                notes += [m for m in (e.get("messages") or []) if str(m).strip().lower() != "pause"]
            elif e.get("kind") == "reopened" and e.get("message"):
                notes.append(str(e["message"]))
        seen, out = set(), []
        for n in notes:  # the same line re-sent twice is one standing note
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out[-8:]  # the most recent eight; older ones stay in the journal

    def _record_requirements(self, state: dict[str, Any], lines: list[str]) -> dict[str, Any]:
        """Persist operator requirements so they outlive the observation that delivered them."""
        if not lines or "user_requirements" not in self.skill.schema.get("properties", {}):
            return state
        reqs = list(state.get("user_requirements") or []) + [{"text": line, "done": False} for line in lines]
        state = {**state, "user_requirements": reqs}
        if self.log:
            (self.log.path.parent / "state.json").write_text(json.dumps(state, indent=2))
        return state

    def _drain_inbox(self, state: dict[str, Any], observation: str, t: int) -> tuple[dict[str, Any], str, bool]:
        """Operator channel: every non-empty line in <log_dir>/inbox.txt is consumed once, appended verbatim to the
        protected ``user_requirements`` field (if the schema has one) and prepended to the next observation. A line
        that is exactly ``pause`` stops the run cleanly (resumable), which is the third value; the pause is a flag
        of its own rather than a word the caller would have to find in the observation it also renders. The model
        never writes this field."""
        lines = read_inbox(self.log, t)
        if not lines:
            return state, observation, False
        state = self._record_requirements(state, [line for line in lines if line.lower() != "pause"])
        if any(line.lower() == "pause" for line in lines):
            return state, observation, True
        note = operator_note(lines)
        return state, note + "\n\n" + observation, False

    def _replay(
        self, state: dict[str, Any], replay_env: bool = True
    ) -> tuple[dict[str, Any], str, int, list[tuple[int, str, str]], dict[str, Any]]:
        """Rebuild state and recent-steps digest from the event log. With ``replay_env`` the logged actions are
        re-executed in a deterministic environment; without it (real external systems that keep their own
        state, e.g. an eval adapter) the environment is only reset and the last logged result becomes the
        observation."""
        events = self.log.read() if self.log else []
        steps = replayable_steps(events, self.log.path if self.log else None)
        replay = _Replay(state=state, observation=self.env.reset(), totals=self._replay_totals(events, steps))
        self._replay_steps(replay, steps, replay_env)
        self._replay_operator_messages(replay, events)
        if not replay_env and hasattr(self.env, "restore_from_events"):
            self.env.restore_from_events(events)
            # The adapter distinguishes reserved attempts from local budget refusals.
            # Do not replace its reconstructed counter with a raw count of run labels.
            replay.totals["runs"] = getattr(self.env, "runs_used", replay.totals["runs"])
        if self.log:
            self.log.append("resumed", from_step=len(steps), env_status=getattr(self.env, "status", lambda: None)())
        note = withheld_action_note(events)
        if note:
            replay.observation = note + "\n\n" + replay.observation
        replay.observation = stale_observation_note(steps) + replay.observation
        return replay.state, replay.observation, len(steps), replay.recent, replay.totals

    @staticmethod
    def _replay_totals(events: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Sum the journal's controller receipts into the accounting the run loop resumes with."""
        model_calls = controller_receipts(events)
        return {
            "prompt": [e.get("input_tokens", e.get("prompt_tokens", 0)) or 0 for e in model_calls],
            "output": [e.get("output_tokens") or 0 for e in model_calls],
            "rejected": sum(1 for e in events if e.get("kind") == "patch_rejected"),
            # Known subtotal; _finish_run derives completeness from the full journal.
            "cost": sum(float(e.get("cost_usd") or 0) for e in model_calls),
            "runs": sum(1 for e in steps if str(e.get("action", "")).lower().startswith("run ")),
        }

    def _replay_steps(self, replay: _Replay, steps: list[dict[str, Any]], replay_env: bool) -> None:
        """Re-apply every logged patch, and either re-execute the action or take its logged result."""
        for e in steps:
            replay.state = merge(replay.state, e.get("patch") or {})
            if replay_env:
                result = self.env.step(e["action"])
                replay.observation = result.observation
            else:
                replay.observation = e["result"]
            replay.recent = (replay.recent + [(e["step"], e["action"], replay.observation)])[-self.recent_steps :]

    def _replay_operator_messages(self, replay: _Replay, events: list[dict[str, Any]]) -> None:
        """Restore logged operator messages as requirements the replayed state may have lost."""
        ops = [m for e in events if e.get("kind") == "operator_messages" for m in e.get("messages", [])]
        if ops and "user_requirements" in self.skill.schema.get("properties", {}):
            have = {r.get("text") for r in (replay.state.get("user_requirements") or [])}
            replay.state = {
                **replay.state,
                "user_requirements": list(replay.state.get("user_requirements") or [])
                + [{"text": m, "done": False} for m in ops if m.lower() != "pause" and m not in have],
            }

    def _call_controller(self, progress: _RunProgress, t: int):
        """Render the attempt and call the controller with retry boundary checks."""
        state = progress.state
        system_prompt = self.skill.instructions
        user_prompt = self._render(state, progress.attempt_observation, progress.recent, t)
        rendered_prompt_chars = len(system_prompt) + len(user_prompt)
        env_out = call_with_retry(
            self.executor,
            system_prompt,
            user_prompt,
            self.output_schema,
            output_check=lambda output: response_problems(
                output,
                state_check=StateCheck(self.validator, state),
                action_check=getattr(self.env, "action_format_problems", None),
                report_schema=self.report_schema,
            ),
            log=self.log,
            step=t,
            before_call=lambda: check_control(self.log, self.env),
        )
        env_out = translate_report_action(self.env, env_out)
        progress.env_out, progress.rendered_prompt_chars = env_out, rendered_prompt_chars

    def _prepare_journal(self, max_steps, resume, replay_env):
        """Check journal ownership and recover pending actions before recording the run."""
        if self.log and self.log.path.exists() and not resume and self.log.read():
            raise ExistingRun(f"{self.log.path} already has events; use --resume (resume=True) or a new --log-dir")
        if resume:
            if self.log is not None:
                starts = [e for e in self.log.read() if e.get("kind") == "run_start"]
                if any(e.get("measurement_repeats", 1) > 1 for e in starts):
                    raise AutomaticMeasurementRepeats(
                        "This journal ran with automatic measurement repeats, which no longer exist; "
                        "it can be read but not resumed."
                    )
                required = getattr(self.env, "require_completion", False)
                if any(e.get("require_completion", False) != required for e in starts) or (not starts and required):
                    raise CompletionPolicyChanged("completion policy changed; resume with the original policy")
            recover_actions(self.log)  # before any external reset or model call
            if self.log is not None:
                # Refuse a step the replay could not read BEFORE this attempt is recorded, so a refused
                # resume leaves the journal exactly as it found it, as the recovery refusal above does.
                replayable_steps(self.log.read(), self.log.path)
            if hasattr(self.env, "resuming"):
                self.env.resuming = not replay_env
        if self.log:
            self.log.append(
                "run_start",
                resume=resume,
                max_steps=max_steps,
                skill=self.skill.name,
                require_completion=getattr(self.env, "require_completion", False),
            )

    def _reopen_journal(self, progress: _RunProgress, resume, reopen):
        """Apply an operator reopening message after replaying the journal."""
        completed_on_resume = resume and self.log and self._journal_completed()
        if completed_on_resume and reopen and reopen.strip():
            self._log(
                "reopened",
                from_step=progress.steps,
                message=reopen.strip(),
                authority="explicit operator message; the previous done stands in the journal",
            )
            progress.observation = (
                f"OPERATOR MESSAGE (run reopened after it was marked done): {reopen.strip()}\n\n" + progress.observation
            )
            progress.state = self._record_requirements(progress.state, [reopen.strip()])
            completed_on_resume = False
        elif reopen and reopen.strip() and self.log:
            self._log("operator_messages", messages=[reopen.strip()])
            progress.observation = f"OPERATOR MESSAGE: {reopen.strip()}\n\n" + progress.observation
            progress.state = self._record_requirements(progress.state, [reopen.strip()])
        progress.episode_done = bool(completed_on_resume)

    def _render_step(self, progress: _RunProgress, t: int):
        """Add environment and policy notes, shape the observation, and measure the prompt."""
        context = getattr(self.env, "context", None)
        if context is not None:
            progress.observation = context() + "\n\n" + progress.observation
        progress.attempt_observation, progress.shape_info = self.shaper.shape(progress.observation, t)
        progress.rendered.append(
            len(self.skill.instructions)
            + len(self._render(progress.state, progress.attempt_observation, progress.recent, t))
        )

    def _apply_patch(self, state, env_out):
        """Reject empty actions and validate the patch before dispatch controls run."""
        if not (isinstance(env_out.action, str) and env_out.action.strip()):
            # Reject empty commands so they cannot waste environment turns.
            raise PatchError(["action must be a nonempty command string"])
        new_state, changed = self.validator.apply(state, env_out.state_patch)
        return new_state, changed

    def _reject_patch(self, t, attempt, shaped, err, env_out):
        """Journal the rejected proposal and render its bounded repair observation."""
        self._log(
            "patch_rejected",
            step=t,
            attempt=attempt,
            problems=err.problems,
            patch=env_out.state_patch,
            reasoning=env_out.reasoning,
            action=env_out.action,
        )
        # A fresh-state retry otherwise loses the investigation it just paid for.
        # Keep a bounded, explicitly uncommitted proposal for repair; the full
        # response remains in the journal. It is not an executed action or proof.
        attempt_obs = render_patch_repair(shaped, err, env_out)
        return attempt_obs

    def _dispatch_action(self, progress: _RunProgress, t: int):
        """Save validated state and execute the action through the intent, receipt, and commit journal."""
        if self.log:
            (self.log.path.parent / "state.json").write_text(json.dumps(progress.state, indent=2))
        env_out = cast(Envelope, progress.env_out)  # the propose phase either set it or aborted the step
        result = execute_action(
            self.log,
            self.env,
            step=t,
            observation=progress.observation,
            shaped=progress.shape_info,
            reasoning=env_out.reasoning,
            patch=env_out.state_patch,
            changed=progress.changed,
            action=env_out.action,
            prompt_tokens=env_out.input_tokens,
            output_tokens=env_out.output_tokens,
            api_calls=env_out.api_calls,
            usage=env_out.usage,
            usage_unknown=bool(env_out.raw.get("usage_unknown")),
            tool_calls=env_out.tool_calls,
            tool_log=env_out.tool_log,
            cost_usd=env_out.cost_usd,
            rendered_prompt_chars=progress.rendered_prompt_chars,
            state_chars=len(json.dumps(progress.state)),
        )
        return result

    def _track_repeated_failures(self, t, env_out, result):
        """Warn or pause after repeated identical failures without changing the committed receipt."""
        # Repeated identical failures need a host-side limit to stop dependency loops.
        sig = (env_out.action.strip(), result.observation)
        # The environment says whether its step failed; the observation is display, and is never scanned for it.
        failed = result.info.get("ok") is False
        if failed:
            self._repeats[sig] = self._repeats.get(sig, 0) + 1
        else:
            self._repeats.pop(sig, None)
        n_rep = self._repeats.get(sig, 0)
        if self.repeat_limit and n_rep >= self.repeat_limit * 2:
            self._log(
                "paused",
                step=t,
                action=sig[0],
                result=sig[1],
                count=n_rep,
                reason="Repeated failure without progress after a change-of-approach warning; checkpoint saved for review",
            )
            return True
        if self.repeat_limit and n_rep >= self.repeat_limit:
            result.observation += (
                f"\n\nSTOP REPEATING: this exact action ({str(sig[0])[:120]!r}) has produced this exact "
                f"result ({str(sig[1])[:120]!r}) {n_rep} times. Repeating it or reverting more files in the "
                f"hope that it passes is not progress. Take a different KIND of action next: read the "
                f"failing output or traceback in full; find what changed since the last passing check "
                f"(diff or bisect your own commits); test one hypothesis about the cause; or record a "
                f"blocker with the evidence and move to another item. The run pauses for review after "
                f"{self.repeat_limit * 2} identical repeats."
            )
        return False

    def _after_step_controls(self, t):
        """Check environment pause requests after completion handling."""
        if getattr(self.env, "pause_requested", False):
            self._log("paused", step=t, reason="operator pause received during a long environment call")
            return True
        return False

    def _finish_run(self, progress: _RunProgress):
        """Build the result from committed controller receipts after horizon-stop handling."""
        score = self.env.score()
        first_attempt = None
        if hasattr(self.env, "horizon") and self.env.horizon:
            first_attempt = max(0.0, score - progress.assisted_ok / self.env.horizon)
        cost_complete = False
        if self.log:
            events = self.log.read()
            cost_complete = usage_summary(events)["cost_complete"]
            receipts = controller_receipts(events)
            progress.prompt_tokens = [e.get("input_tokens", e.get("prompt_tokens", 0)) or 0 for e in receipts]
            progress.output_tokens = [e.get("output_tokens") or 0 for e in receipts]
            progress.cost = sum(e.get("cost_usd") or 0 for e in receipts)
            progress.calls = len(receipts)
            progress.model_ids.update(m for e in receipts for m in (e.get("model_ids") or []))
        return RunResult(
            "stepbook",
            progress.steps,
            progress.calls,
            progress.rejected,
            score,
            progress.prompt_tokens,
            progress.output_tokens,
            progress.cost,
            time.time() - progress.start,
            final_state=progress.state,
            aborted=progress.aborted,
            assisted_steps=progress.assisted,
            first_attempt_score=first_attempt,
            rendered_prompt_chars=progress.rendered,
            model_ids=sorted(progress.model_ids),
            cost_complete=cost_complete,
        )

    def _before_action_controls(self, progress: _RunProgress, t: int):
        """Check controls before action dispatch; draining a pause may persist operator requirements."""
        env_out = cast(Envelope, progress.env_out)  # the propose phase either set it or aborted the step
        try:
            check_control(self.log, self.env)
        except ControlStop as stop:
            self._log(
                "action_withheld",
                step=t,
                reason=str(stop),
                action=env_out.action,
                patch=env_out.state_patch,
                reasoning=env_out.reasoning,
            )
            if str(stop) == "pause":
                progress.state, progress.observation, _ = self._drain_inbox(progress.state, progress.observation, t)
                self._log("paused", step=t, reason="operator pause before environment action")
            progress.aborted = True
            return

    def _select_response(self, progress: _RunProgress, t: int):
        """Check controls before requesting the next controller response."""
        check_control(self.log, self.env)
        self._call_controller(progress, t)

    def _stop_at_controller(self, progress: _RunProgress, t: int, stop: ControlStop):
        """Record a withheld response and drain a pause received at the controller boundary."""
        if progress.env_out is not None:
            self._log(
                "action_withheld",
                step=t,
                reason=str(stop),
                action=progress.env_out.action,
                patch=progress.env_out.state_patch,
                reasoning=progress.env_out.reasoning,
            )
        if str(stop) == "pause":
            progress.state, progress.observation, _ = self._drain_inbox(progress.state, progress.observation, t)
            self._log(
                "paused",
                step=t,
                reason="operator pause at controller boundary; proposed action not dispatched",
            )

    def _controller_attempt(self, progress: _RunProgress, t: int):
        """Account for the response and handle stops before its proposed action may proceed."""
        progress.env_out, progress.rendered_prompt_chars = None, 0
        try:
            self._select_response(progress, t)
            if progress.env_out is None:
                progress.aborted = True
                return
            progress.calls += 1
            progress.prompt_tokens.append(progress.env_out.input_tokens)
            progress.output_tokens.append(progress.env_out.output_tokens)
            progress.cost += progress.env_out.cost_usd
            self._log("model_call", step=t, attempt=progress.attempt, **envelope_usage(progress.env_out))
            progress.model_ids.update(progress.env_out.raw.get("model_ids") or [])
            check_control(self.log, self.env)
        except ControlStop as stop:
            self._stop_at_controller(progress, t, stop)
            progress.aborted = True

    def _propose_step(self, progress: _RunProgress, t: int):
        """Retry rejected patches and apply the configured exhaustion policy before dispatch."""
        progress.new_state, progress.changed = None, None
        base_shaped_observation = progress.attempt_observation
        for attempt in range(self.max_retries + 1):
            progress.attempt = attempt
            self._controller_attempt(progress, t)
            if progress.aborted:
                break
            try:
                progress.new_state, progress.changed = self._apply_patch(progress.state, progress.env_out)
                break
            except PatchError as err:
                progress.rejected += 1
                progress.attempt_observation = self._reject_patch(
                    t, attempt, base_shaped_observation, err, progress.env_out
                )
        else:
            if self.on_exhausted == "abort":
                self._log("abort", step=t, reason="max retries exceeded")
                progress.aborted = True
                return
            self._log("skip", step=t, reason="max retries exceeded; state unchanged", action=self.skip_action)
            progress.new_state, progress.changed = progress.state, []
            progress.env_out = Envelope(
                reasoning="(step skipped after repeated invalid patches)", state_patch={}, action=self.skip_action
            )
        progress.attempt = attempt

    def _commit_step(self, progress: _RunProgress, t: int):
        """Dispatch validated work, update step accounting, and check failure, completion, and budget stops."""
        self._before_action_controls(progress, t)
        if progress.aborted:
            return
        progress.state = cast(dict[str, Any], progress.new_state)  # set by the propose phase unless it aborted
        result = self._dispatch_action(progress, t)
        env_out = cast(Envelope, progress.env_out)  # same invariant as the state above
        progress.steps += 1
        if progress.attempt > 0:
            progress.assisted += 1
            progress.assisted_ok += 1 if result.info.get("ok") else 0
        if self.recent_steps:
            progress.recent = (progress.recent + [(t, env_out.action, result.observation)])[-self.recent_steps :]
        trace_step(
            self.verbose,
            t,
            env_out.action,
            result.observation,
            None if env_out.raw.get("usage_unknown") else env_out.input_tokens,
        )
        if self._track_repeated_failures(t, env_out, result):
            progress.aborted = True
            return
        progress.observation = result.observation
        if result.done:
            progress.episode_done = True
            return
        if self._after_step_controls(t):
            progress.aborted = True

    def _resume_or_start(self, max_steps, resume, replay_env, reopen) -> _RunProgress:
        """Recover and replay an existing journal or reset a new environment, then apply reopening notes."""
        progress = _RunProgress(start=time.time())
        progress.state = dict(self.skill.initial_state)
        self._prepare_journal(max_steps, resume, replay_env)
        if resume and self.log and self.log.path.exists():
            progress.state, progress.observation, progress.steps, progress.recent, totals = self._replay(
                progress.state, replay_env=replay_env
            )
            progress.prompt_tokens, progress.output_tokens = list(totals["prompt"]), list(totals["output"])
            progress.rejected = totals["rejected"]
            progress.cost = totals["cost"]
            progress.calls = len(progress.prompt_tokens)
            if hasattr(self.env, "runs_used") and not replay_env:
                self.env.runs_used = totals["runs"]  # adapter runs already paid for
        else:
            progress.observation = self.env.reset()
        self._reopen_journal(progress, resume, reopen)
        return progress

    @exclusive_run
    def run(
        self, max_steps: int | None, resume: bool = False, replay_env: bool = True, reopen: str | None = None
    ) -> RunResult:
        """``reopen``: an explicit operator message that lifts a completed journal's ``done`` on resume. The message
        is journaled as an operator event and shown to the agent as the first thing in its next observation, so a
        loop that closed itself (for example after a pause it read as "stop") can be continued in the same journal,
        same state and same branch instead of being re-seeded by hand."""
        self._max_steps = max_steps  # the horizon the prompt reports; see _render
        self._state_cap_warned = False
        progress = self._resume_or_start(max_steps, resume, replay_env, reopen)
        if resume and not progress.episode_done and max_steps is not None and progress.steps >= max_steps:
            print(
                f"Step horizon already reached: {progress.steps} steps already done; --steps counts total steps "
                f"including earlier invocations. Continue with --steps {progress.steps + 1} (or higher).",
                file=sys.stderr,
            )
        for t in (
            ()
            if progress.episode_done
            else count(progress.steps)
            if max_steps is None
            else range(progress.steps, max_steps)
        ):
            progress.state, progress.observation, paused = self._drain_inbox(progress.state, progress.observation, t)
            if paused:
                self._log("paused", step=t)
                progress.aborted = True
                break
            self._render_step(progress, t)
            self._propose_step(progress, t)
            if progress.aborted:
                break
            self._commit_step(progress, t)
            if progress.aborted or progress.episode_done:
                break
        if (
            max_steps is not None
            and progress.steps >= max_steps
            and not progress.aborted
            and self.log
            # A bounded pass that ends at its limit is a stop, by the count or by the environment's step
            # guard, whose done sets episode_done without completing the episode; only a genuine done is a
            # completion, and only that case needs the journal read.
            and not (progress.episode_done and self._journal_completed())
        ):
            self.log.append("max_steps_reached", steps=progress.steps, max_steps=max_steps)
        return self._finish_run(progress)

    def _log(self, kind: str, **data: Any) -> None:
        if self.log:
            self.log.append(kind, **data)
