"""The adapter-backed environment for the eval-loop-adapter skill.

``AdapterEvalEnv`` (real) shells out to an adapter executable using the eval-adapter command protocol.
Adapters are discovered from a path, never bundled; private adapters stay private.

Every adapter call answers with an exit code and its stdout (``AdapterReply``): whether an effect happened is
read from the code, never from the adapter's sentences, and each step records what it changed in
``StepResult.info`` (``ok``, ``adapter_exit`` and the ledger facts), so later code does not re-read text either.
Replay (``restore_from_events``) rebuilds the ledger from those facts; a run recorded before they existed is
not resumable here, and only read-only reports replay it, through ``stepbook/reporting/legacy_journal.py``.

The environment lives here: its state, ``reset``/``step`` and the per-action handlers, the receipt ledgers,
``restore_from_events`` and ``measurement_frontier``. How a verdict is computed is in eval_verdict.py; how the
adapter subprocess is called is in eval_adapter.py; the research, evidence and
completion actions are in eval_research.py; the action vocabulary is in
eval_report.py. ``AdapterEvalEnv`` inherits
the mixins, so every call site keeps its ``self.`` spelling.
"""

from __future__ import annotations

import json

# ``as subprocess`` keeps the ``eval_loop.subprocess`` monkeypatch target here; eval_adapter.py's dispatch reads
# the same module object.
import subprocess as subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ..contracts.errors import LegacyAdapterJournal
from ..contracts.eval import AdapterReply, EvalReceipt
from ..contracts.runtime import StepResult
from ..reporting import legacy_journal
from .eval_adapter import AdapterGlueMixin

# ``X as X`` keeps the names this module used to define importable from here after the split; ruff and vulture
# both honour the explicit re-export, and existing tests, benches and adapters keep working.
from .eval_report import ACTION_RE as ACTION_RE
from .eval_research import ResearchMixin
from .eval_verdict import VerdictMixin

LEGACY_JOURNAL_REFUSAL = (
    "this run was recorded before the structured adapter journal, so its steps do not carry the facts a "
    "resume needs and it cannot be resumed; read it with 'stepbook report' and start a new run instead."
)


@dataclass
class AdapterEvalEnv(VerdictMixin, AdapterGlueMixin, ResearchMixin):
    """Real eval loop: ``apply``/``decide`` are delegated to the adapter too, so the harness never learns
    anything about the system under test.

    """

    adapter: str
    runs_max: int | None = None
    runs_used: int = 0
    stopped: bool = False
    investigations: list[dict] = field(default_factory=list)
    supports_investigate: bool = True  # try the adapter's own `investigate` verb when there is no investigator
    _runs: dict[str, list[dict]] = field(
        default_factory=dict
    )  # hypothesis id ("baseline" for none) -> parsed run payloads
    _run_history: list[tuple[str, dict]] = field(default_factory=list)  # citations survive comparison-phase reset
    _applied: str | None = None
    _reference_h: str = "baseline"
    _run_dirs: dict[str, str] = field(default_factory=dict)  # run_id / config_label -> artifacts.run_dir
    evidence_roots: tuple[str, ...] = ()  # explicit read authority; never expanded by model output
    max_source_bytes: int | None = None
    evidence_cwd: str | None = None  # relative locators use the same base as native file tools
    require_completion: bool = False  # opt-in final evidence artifact; ordinary stop remains default

    def context(self) -> str:
        """Small authoritative frontier, re-surfaced even when the model forgot to patch state.

        Identifiers and budgets belong to the harness. Rediscovering them from
        a transcript or inventing them is both expensive and unreliable.
        """
        findings: list[dict[str, str]] = []
        if self.log_dir and (self.log_dir / "research").exists():
            paths = sorted(
                (self.log_dir / "research").glob("finding-*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
            )
            for path in paths[:2]:
                r = self._research().recall(path.stem)
                findings.append({"id": r["id"], "summary": r["summary"][:150], "next_action": r["next_action"][:150]})
        frontier = {
            "applied": self._applied,
            "reference": self._reference_h,
            "runs_used": self.runs_used,
            "runs_remaining": max(0, self.runs_max - self.runs_used) if self.runs_max is not None else None,
            "evidence_cwd": str(Path(self.evidence_cwd or ".").resolve()),
            "latest_run_dir": self._run_dirs.get("latest"),
            "recent_findings": findings,
            "measurements": self.measurement_frontier(),
            "completion": {
                "available": self.log_dir is not None,
                "required": self.require_completion,
                "local_finish_after_known_query_overshoot": True,
            },
        }
        return "HARNESS FACTS (authoritative ids, available actions and remaining budget):\n" + json.dumps(frontier)

    @staticmethod
    def _exact_text(value):
        """An identifier is used whole or not at all: never truncate one into a different, plausible identifier."""
        return value if isinstance(value, str) and bool(value.strip()) else None

    def _recent_receipt_rows(self, receipts: list[dict]) -> list[dict]:
        """The last two receipts of one hypothesis, with the identifiers that could not be shown named."""
        recent: list[dict] = []
        for p in receipts[-2:]:
            metric = p.get("primary_metric") or {}
            row = {
                "run_id": self._exact_text(p.get("run_id")),
                "label": self._exact_text(p.get("config_label")),
                "valid": self._valid_run(p),
                "primary": metric.get("value") if self._valid_run(p) else None,
            }
            row["omitted_identifiers"] = [
                field
                for field, key in (("run_id", "run_id"), ("label", "config_label"))
                if p.get(key) is not None and row[field] is None
            ]
            recent.append(row)
        return recent

    def _receipt_aggregate(self, valid: list[dict]) -> tuple[str | None, dict | None]:
        """Aggregate the valid receipts of one configuration, or say why they cannot be aggregated."""
        issue = "no valid receipts" if not valid else self._comparison_mismatch(valid)
        keys = [self._exact_text(p.get("comparison_key")) for p in valid]
        configs = [self._exact_text((p.get("provenance") or {}).get("config_sha256")) for p in valid]
        names = [self._exact_text((p.get("primary_metric") or {}).get("name")) for p in valid]
        if valid and (not all(keys) or not all(configs) or not all(names)):
            issue = "missing or oversized comparison/configuration/metric identity"
        elif len(set(configs)) > 1:
            issue = "configuration changed within measurement group"
        aggregate = None
        if valid and not issue:
            values = [p["primary_metric"]["value"] for p in valid]
            aggregate = {
                "metric": names[0],
                "direction": self._exact_text(valid[0]["primary_metric"].get("direction", "higher")),
                "mean": sum(v / len(values) for v in values),
                "min": min(values),
                "max": max(values),
                "spread": max(values) - min(values),
                "runs": len(values),
                "comparison_key": keys[0],
                "config_sha256": configs[0],
            }
        return issue, aggregate

    def _summarize_receipts(self, hid) -> dict:
        """Counts, the two most recent rows and the aggregate for one hypothesis's receipts."""
        receipts = self._runs.get(hid, [])
        valid = [p for p in receipts if self._valid_run(p)]
        recent = self._recent_receipt_rows(receipts)
        issue, aggregate = self._receipt_aggregate(valid)
        return {
            "hypothesis": self._exact_text(hid),
            "receipt_count": len(receipts),
            "valid_count": len(valid),
            "invalid_count": len(receipts) - len(valid),
            "recent": recent,
            "older_receipts_not_shown": max(0, len(receipts) - len(recent)),
            "aggregate": aggregate,
            "aggregate_unavailable_reason": issue,
        }

    def measurement_frontier(self) -> dict:
        """Bounded receipt-derived facts, never reconstructed from model-written state.

        Retain the full ledger; this projection is not a keep/revert verdict. Missing
        identity is unknown, not permission to average unrelated measurements.
        """
        reference = self._summarize_receipts(self._reference_h)
        applied = self._applied  # bound once so the guarded comparison below is the same string, not None
        candidate = self._summarize_receipts(applied) if applied and applied != self._reference_h else None
        comparison_issue = (
            "no active candidate"
            if candidate is None
            else (reference["aggregate_unavailable_reason"] or candidate["aggregate_unavailable_reason"])
        )
        if candidate is not None and comparison_issue is None:
            comparison_issue = self._comparison_mismatch(
                [
                    p
                    for hid in (self._reference_h, cast(str, applied))
                    for p in self._runs.get(hid, [])
                    if self._valid_run(p)
                ]
            )
        spreads = [
            x["aggregate"]["spread"]
            for x in (reference, candidate)
            if x and x.get("aggregate") and x["aggregate"].get("runs", 0) >= 2
        ]
        return {
            "reference": reference,
            "candidate": candidate,
            "comparison_issue": comparison_issue,
            "noise_floor": {
                "observed_spread": max(spreads) if spreads else None,
                "note": "largest max-min of the primary metric across repeated runs of one configuration; two runs "
                "within this distance are not distinguishable",
            }
            if spreads
            else {
                "observed_spread": None,
                "note": "no configuration has two valid runs yet; repeated measurements establish the noise floor",
            },
            "scope": "Host receipts, not model aliases. Invalid runs excluded from aggregates. Full receipts remain in the journal; summaries are not decision authorization.",
        }

    resuming: bool = False

    def reset(self) -> str:
        # runs_used is deliberately NOT reset: an adapter run costs money whether or not this process restarts.
        if not self.resuming:
            reset = self._call(["reset"])  # fresh episode: never silently discard an applied experiment
            if not reset.ok:
                raise RuntimeError(reset.text)
        described = self._call(["describe"])
        if not described.ok:
            raise RuntimeError(described.text)
        out = described.stdout
        research_note = (
            (
                f"\nEvidence roots: {json.dumps(self.evidence_roots)}. Use read/record/recall/propose for "
                "open-ended investigation. No predefined hypotheses are required.\n"
                f"Durable research index (last 20): {json.dumps(self._research().recall())}"
            )
            if self.log_dir
            else ""
        )
        if self.log_dir:
            research_note += (
                "\nfinish <JSON finding> saves final evidence and ends the loop without a native Write tool. "
                "Fields: summary, evidence (allowed source locators/IDs), uncertainty, next_action, "
                "run_ids (original receipt IDs). Decide any applied candidate first. "
                + (
                    "Completion artifact required; plain stop is refused."
                    if self.require_completion
                    else "Optional; plain stop remains available."
                )
            )
        return (
            f"Eval loop ready.\n{out}\nBudget: {self.runs_max} eval runs ({self.runs_used} already used)."
            + research_note
        )

    def step(self, action: str) -> StepResult:
        """Parse the action and dispatch its ordered effects to the matching handler."""
        m = ACTION_RE.match(action or "")
        if not m:
            return StepResult(f"Error: unrecognised action {action!r}.", info={"ok": False})
        verb, rest = m.group(1).lower(), m.group(2).split()
        if verb == "finish":
            return self._finish(m.group(2))
        if verb in ("read", "record", "recall", "propose"):
            return self._research_action(verb, m.group(2))
        if verb == "stop":
            return self._step_stop()
        if verb == "add":
            return self._step_add()
        if verb == "run":
            return self._step_run(rest)
        if verb == "apply":
            return self._step_apply(rest)
        if verb == "decide":
            return self._step_decide(rest)
        if verb == "mine":
            return StepResult(
                "Error: failure mining is not enabled for this run (no miner executor).", info={"ok": False}
            )
        if verb == "investigate":
            return self._investigate(rest[0] if rest else "", " ".join(rest[1:]))
        if verb == "phase":
            return self._step_phase()
        reply = self._call([verb, *rest])
        return StepResult(reply.text, info=self._adapter_info(reply))

    @staticmethod
    def _adapter_info(reply: AdapterReply, **facts) -> dict:
        """What one adapter-route step records: whether it worked, the verb's exit code and the ledger facts
        it changed. Later code reads these instead of the observation text."""
        return {"ok": reply.ok, "adapter_exit": reply.exit_code, **facts}

    def _step_stop(self) -> StepResult:
        """Stop unless the run requires a completion artifact."""
        if self.require_completion:
            return StepResult(
                "Error: a completion artifact is required; use finish with evidence and original run_ids.",
                info={"ok": False},
            )
        self.stopped = True
        return StepResult("Stopped.", done=True, info={"ok": True})

    def _step_add(self) -> StepResult:
        """Acknowledge a state-only hypothesis without registering it with the adapter."""
        # Bookkeeping only: the hypothesis lives in the agent's state. Adapters expose their own list.
        return StepResult(
            "Noted: hypothesis recorded in state. Only hypotheses listed by the adapter can be "
            "applied; new ideas are recorded for the final report.",
            info={"ok": True},
        )

    def _step_run(self, rest: list[str]) -> StepResult:
        """Reserve an attempt, call the adapter, then record receipts and append verdict text."""
        if self.runs_max is not None and self.runs_used >= self.runs_max:
            # The only run step that reserves no attempt, and it says so: everything past this line is spent.
            return StepResult(
                "Error: run budget exhausted.",
                info={"ok": False, "evaluation_not_started": True, "attempt_spent": False},
            )
        self.runs_used += 1
        rest = rest[:1]  # label only
        label = rest[0] if rest else None
        reply = self._call(["run", *rest])
        if not reply.ok:  # a refusal (its reason on stdout) or an adapter failure; the attempt is still spent
            return StepResult(
                reply.text, info=self._adapter_info(reply, run_label=label, **self._declared_unstarted(reply))
            )
        try:
            payload = json.loads(reply.stdout)
            # The ledger keeps the receipt as the adapter sent it; the model is the harness's typed read.
            receipt = EvalReceipt.model_validate(payload)
        except json.JSONDecodeError:
            return StepResult(
                "Error: the adapter exited 0 for run but printed no JSON receipt: " + reply.stdout[:500],
                info=self._adapter_info(reply, run_label=label, ok=False),
            )
        self._runs.setdefault(self._applied or "baseline", []).append(payload)
        self._run_history.append((self._applied or "baseline", payload))
        rd = receipt.run_dir
        if rd:
            for key in (receipt.run_id, receipt.config_label, label):
                if key:
                    self._run_dirs[key] = rd
            self._run_dirs["latest"] = rd
        # The observation keeps the protocol's wording: it is what the agent reads. Nothing decides from it.
        out = "Eval finished:\n" + reply.stdout
        verdict, facts = self._verdict_result(self._applied) if self._applied else (None, None)
        if verdict:
            out += "\n" + verdict + (self._alt_note(self._applied) if self._applied else "")
        return StepResult(
            out,
            info=self._adapter_info(
                reply,
                run_label=label,
                run_id=receipt.run_id,
                run_valid=self._valid_run(payload),
                runs_used=self.runs_used,
                # The receipt as the adapter sent it, so replay rebuilds the same ledger, and the computed
                # verdict as values, so the decision audit never reads the sentence above.
                receipt=payload,
                **({"verdict": facts} if facts else {}),
            ),
        )

    @staticmethod
    def _declared_unstarted(reply: AdapterReply) -> dict:
        """A refused ``run`` may answer with a JSON document declaring that it started no evaluation and spent
        nothing (the bundled adapter's preflight refusals do). Usage accounting reads the recorded fact; an
        adapter whose refusal is prose declares nothing, and its evaluation cost stays unknown."""
        try:
            declaration = json.loads(reply.stdout)
        except ValueError:
            return {}
        declared = (
            isinstance(declaration, dict)
            and declaration.get("execution_started") is False
            and declaration.get("cost") == {"tokens": 0, "usd": 0}
        )
        return {"evaluation_not_started": True} if declared else {}

    def _step_apply(self, rest: list[str]) -> StepResult:
        """Apply the hypothesis and handle any requested code change after the adapter reply."""
        if not rest:
            return StepResult("Error: apply needs a hypothesis id.", info={"ok": False})
        reply = self._call(["apply", *rest])
        if not (reply.ok or reply.code_change_required):
            return StepResult(reply.text, info=self._adapter_info(reply))
        self._applied = rest[0]  # the adapter snapshotted and applied it, code change pending or not
        if reply.code_change_required:
            return self._apply_code_change(rest[0], reply)
        return StepResult(reply.stdout, info=self._adapter_info(reply, applied=self._applied))

    def _apply_code_change(self, hid: str, reply: AdapterReply) -> StepResult:
        """Roll back an adapter request that needs an unavailable coding helper."""
        reversal = self._call(["decide", hid, "revert"])
        if reversal.ok:
            self._applied = None
            return StepResult(
                "Error: this hypothesis needs a code change and no code executor is configured; reverted.",
                info=self._adapter_info(reply, ok=False, applied=None, code_change_required=True),
            )
        # Still applied, which is what ``info["applied"]`` says and what replay restores.
        return StepResult(
            f"Applied {hid} (code change requested): {reply.stdout}"
            " Error: no code executor; automatic rollback failed: " + reversal.text,
            info=self._adapter_info(reply, ok=False, applied=hid, code_change_required=True),
        )

    def _step_decide(self, rest: list[str]) -> StepResult:
        """Validate a decision, call the adapter, then update the applied candidate and reference."""
        if not rest:
            return StepResult("Error: decide needs a hypothesis id.", info={"ok": False})
        if len(rest) != 2 or rest[1].lower() not in ("keep", "revert", "skip"):
            return StepResult("Error: usage decide <id> keep|revert|skip.", info={"ok": False})
        rest[1] = rest[1].lower()
        if rest[1] == "skip" and not any(i["h"] == rest[0] for i in self.investigations):
            return StepResult(
                f"Error: {rest[0]} cannot be skipped without an investigation. Run "
                f"'investigate {rest[0]} <what to check>' first, or measure it.",
                info={"ok": False},
            )
        if rest[1] == "keep":
            if rest[0] != self._applied:
                return StepResult("Error: only the applied hypothesis can be kept.", info={"ok": False})
            current = [p for p in self._runs.get(rest[0], []) if self._valid_run(p)]
            reference = [p for p in self._runs.get(self._reference_h, []) if self._valid_run(p)]
            if len(current) < self.runs_per_hypothesis or not reference:
                return StepResult(
                    "Error: keep requires a valid reference and the configured number of valid candidate runs.",
                    info={"ok": False},
                )
            mismatch = self._comparison_mismatch(current + reference)
            if mismatch:
                return StepResult("Error: " + mismatch + "; establish a new comparison phase.", info={"ok": False})
        reply = self._call(["decide", *rest])
        if not reply.ok:
            return StepResult(reply.text, info=self._adapter_info(reply))
        if rest[0] == self._applied:
            if rest[1] != "revert":
                self._reference_h = rest[0]
            self._applied = None
        return StepResult(
            reply.stdout,
            info=self._adapter_info(
                reply, decision=rest[1], hypothesis=rest[0], applied=self._applied, reference=self._reference_h
            ),
        )

    def _step_phase(self) -> StepResult:
        """Advance the adapter phase and clear comparison state only after confirmed advancement."""
        if self._applied:
            return StepResult("Error: decide the applied hypothesis before changing phase.", info={"ok": False})
        reply = self._call(["phase"])
        if not reply.ok:
            return StepResult(reply.text, info=self._adapter_info(reply))
        # Where the phase stands is the adapter's bookkeeping, read from its `status` JSON, not from its sentence.
        status = self._adapter_status() or {}
        complete = status.get("phases_complete") is True
        if not complete:
            # a new stage measures its own baseline: the previous stage's runs are not a valid reference
            self._runs.clear()
            self._reference_h = "baseline"
            self._run_dirs.clear()
        return StepResult(
            reply.stdout,
            done=complete,
            info=self._adapter_info(reply, phase=status.get("phase"), phases_complete=complete),
        )

    @staticmethod
    def _valid_run(payload: dict) -> bool:
        try:
            cases = payload.get("cases") or {}
            case_errors = sum(v == "error" for v in cases.values()) / len(cases) if cases else 0
            return (
                payload.get("valid", True) is not False
                and float(payload.get("error_rate", 0) or 0) <= 0.10
                and case_errors <= 0.10
                and type(payload.get("primary_metric", {}).get("value")) in (int, float)
                and __import__("math").isfinite(payload["primary_metric"]["value"])
            )
        except (TypeError, ValueError, AttributeError, OverflowError):
            return False

    @staticmethod
    def _comparison_mismatch(runs: list[dict]) -> str | None:
        keys = {p.get("comparison_key") for p in runs}
        if len(keys) > 1:
            return "comparison identity changed or is missing on one side"
        metrics = {
            (p.get("primary_metric", {}).get("name"), p.get("primary_metric", {}).get("direction", "higher"))
            for p in runs
        }
        if len(metrics) > 1:
            return "primary metric name/direction changed"
        case_sets = {tuple(sorted(p["cases"])) for p in runs if p.get("cases")}
        if len(case_sets) > 1:
            return "evaluated case set changed"
        return None

    pause_requested: bool = False
    inbox_poll_s: float = 30.0

    # ---- resume: rebuild the environment's ledger from the event log ---------------------------------------------
    def _clear_ledger(self) -> None:
        """Reconstruct, do not append: repeated inspection/resume must not duplicate paid attempts or evidence."""
        self.runs_used = 0
        self._runs.clear()
        self._run_history.clear()
        self._applied = None
        self._reference_h = "baseline"
        self._run_dirs.clear()
        self.investigations = [i for i in self.investigations if i.get("source") != "event_replay"]

    def _restore_receipt(self, rest: list[str], payload: dict) -> None:
        """Re-file one logged run receipt into the run ledger and the run-directory index."""
        receipt = EvalReceipt.model_validate(payload)
        self._runs.setdefault(self._applied or "baseline", []).append(payload)
        self._run_history.append((self._applied or "baseline", payload))
        rd = receipt.run_dir
        if rd:
            for key in (receipt.run_id, receipt.config_label, rest[0] if rest else None):
                if key:
                    self._run_dirs[key] = rd
            self._run_dirs["latest"] = rd

    # The one fact replay needs from each verb that can change the ledger: a step which recorded it is
    # replayed from its facts, and a step which did not is a run journaled before they existed.
    _LEDGER_FACTS = {
        "run": "receipt",
        "apply": "applied",
        "decide": "applied",
        "phase": "phases_complete",
        "investigate": "investigation",
    }

    @classmethod
    def _structured_step(cls, verb: str, info: dict) -> bool:
        """Whether a logged step carries the fact replay needs, rather than only the agent's observation.

        A refusal changed nothing and records that it failed; a verb replay never reads changes nothing
        either. Both are replayed as they are, so a journal is refused only for what it actually lost.
        """
        fact = cls._LEDGER_FACTS.get(verb)
        return fact is None or fact in info or info.get("ok") is False

    def _restore_step(self, verb: str, rest: list[str], info: dict) -> None:
        """Apply one logged step's recorded effect on the ledger, in the same order the live step had.

        Replay reads what the step journaled about itself, never the sentences the adapter or this
        environment wrote for the agent: a step that changed the applied candidate or the reference recorded
        both, a run recorded its receipt, a phase recorded whether phases are complete, and an investigation
        recorded its hypothesis however the adapter worded its finding.
        """
        # Count every reserved attempt, including adapter failures/refusals. Otherwise restart refunds
        # potentially paid failed runs. Only the local exhausted-budget refusal reserved none, and it said so.
        if verb == "run" and info.get("attempt_spent") is not False:
            self.runs_used += 1
        if "applied" in info:
            self._applied = info["applied"]
        if "reference" in info:
            self._reference_h = info["reference"]
        if isinstance(info.get("receipt"), dict):
            self._restore_receipt(rest, info["receipt"])
        elif info.get("phases_complete") is False:
            # a new stage measures its own baseline: the previous stage's runs are not a valid reference
            self._runs.clear()
            self._reference_h = "baseline"
            self._run_dirs.clear()
        elif info.get("investigation"):
            # The finding itself is in the journaled observation; no decision reads it, only its hypothesis.
            self.investigations.append({"h": str(info["investigation"]), "finding": "", "source": "event_replay"})

    def _adapter_status(self) -> dict | None:
        """The adapter's own bookkeeping, as the JSON its `status` verb prints; None when it has none to give."""
        try:
            reply = self._call(["status"])
            status = json.loads(reply.stdout) if reply.ok else None
        except (OSError, ValueError):  # unreadable adapter, or stdout that is not JSON
            return None
        return status if isinstance(status, dict) else None

    def _check_adapter_agreement(self) -> None:
        """Split-brain check against the adapter's own bookkeeping (adapters without `status` are skipped)."""
        adapter_status = self._adapter_status()
        if adapter_status is not None and adapter_status.get("applied") != self._applied:
            raise RuntimeError(
                f"resume mismatch: the event log says applied={self._applied!r} but the adapter says "
                f"applied={adapter_status.get('applied')!r}. Decide or reset in the adapter, then resume."
            )

    def _restore_logged_step(self, e: dict, *, refuse_legacy: bool) -> None:
        """Replay one journaled step from its recorded facts, refusing a resume of a pre-structured run."""
        m = ACTION_RE.match(str(e.get("action", "")))
        if not m:
            return
        verb, rest, info = m.group(1).lower(), m.group(2).split(), e.get("info") or {}
        if self._structured_step(verb, info):
            self._restore_step(verb, rest, info)
        elif refuse_legacy:
            raise LegacyAdapterJournal(LEGACY_JOURNAL_REFUSAL)
        else:
            legacy_journal.restore_step(self, verb, rest, str(e.get("result", "")))

    def restore_from_events(self, events: list[dict], *, check_adapter: bool = True) -> None:
        """Rebuild `_applied`, `_runs`, `_reference_h` and `_run_dirs` from what each logged step recorded it
        did, so they match the adapter after a crash. Called by Runtime._replay when the environment is
        external, and by the read-only reports, which also accept a run journaled before those records."""
        self._clear_ledger()
        for e in events:
            if e.get("kind") == "step":
                self._restore_logged_step(e, refuse_legacy=check_adapter)
        if not check_adapter:
            return  # read-only reports reconstruct the ledger without executing adapter commands
        self._check_adapter_agreement()

    def status(self) -> dict:
        """The environment's view, for split-brain checks against the adapter's own `status` verb."""
        return {
            "applied": self._applied,
            "reference": self._reference_h,
            "runs_used": self.runs_used,
            "hypotheses_with_runs": sorted(self._runs),
        }

    # ---- verdict configuration; the computation is in eval_verdict.py ------------------------------------------
    runs_per_hypothesis: int = 2
    baseline_runs: int | None = None  # protocol's requested reference repeats; named in verdicts when unmet
    metric_tolerances: dict = field(default_factory=dict)  # name -> absolute tolerance for deterministic metrics
    default_rel_tolerance: float = 0.02  # relative move a deterministic metric needs to count

    alternative_groups: dict = field(default_factory=dict)  # group name -> [hypothesis ids], keep at most one

    def _investigate(self, hid: str, question: str) -> StepResult:
        """Investigate read-only so redundant or impossible hypotheses do not waste eval runs."""
        if not hid:
            return StepResult("Error: investigate needs a hypothesis id, e.g. 'investigate h1'.", info={"ok": False})
        reply = self._call(["investigate", hid]) if self.supports_investigate else None
        if reply is not None and reply.ok:
            self.investigations.append({"h": hid, "finding": reply.stdout, "source": "adapter"})
            return StepResult(reply.stdout, info=self._adapter_info(reply, investigation=hid))
        if reply is not None and reply.refused:  # the adapter has the verb and declined; its reason is on stdout
            return StepResult(reply.text, info=self._adapter_info(reply))
        return StepResult(
            "Error: investigation is not available for this run (no investigator and the adapter "
            "has no 'investigate' verb). Decide from measurement instead.",
            info={"ok": False, "adapter_exit": reply.exit_code if reply else None},
        )

    def score(self) -> float:
        return 0.0  # real evals have no oracle; the report carries the numbers

    timeout_s: float = 7200.0
    log_dir: Path | None = None
