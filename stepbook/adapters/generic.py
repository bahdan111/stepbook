#!/usr/bin/env python3
"""Config-only eval adapter: point the eval loop at ANY eval command that writes a results file, no Python.

    python -m stepbook.adapters.generic eval.json describe|apply <id>|run <label>|decide <id> keep|revert|reset|status|register <json>

Each verb answers with an exit code, not with a recognisable sentence: 0 done, 1 refused (the reason is on
stdout), 3 for `apply` on a code hypothesis (applied, and the instructions for the coding step are on stdout).
`run` prints one JSON receipt and nothing else. Any other exit code is this adapter failing.

eval.json:
{
  "name": "my RAG eval",
  "run_command": "python eval.py --out {out}",          # {out} is replaced by a fresh results path
  "results_format": "json",                              # "json" (list of rows or {"cases": [...]}) or "csv"
  "case_id_field": "id", "pass_field": "pass",           # pass_field may be bool, "pass"/"fail", or numeric >= pass_threshold
  "pass_threshold": 1.0,
  "metric_fields": {"latency_s": "lower"},               # optional per-row numeric fields averaged into deterministic_metrics;
                                                          #   or {"direction": "lower", "max": 1.2} to declare a constraint with a limit
  "split_field": "split", "holdout_value": "holdout",    # optional: rows with split==holdout feed holdout_primary (anti-gaming)
  "protected_files": ["eval.py", "data/golden.jsonl"],  # optional: hashed; a run is invalid if they changed
  "primary": {"name": "accuracy", "direction": "higher"},
  "config_file": "config.json",                          # JSON file the hypotheses edit (dotted keys allowed)
  "hypotheses": {"h1": {"kind": "config", "rationale": "top_k 3->5", "set": {"retrieval.top_k": 5}},
                 "h2": {"kind": "code", "rationale": "...", "code_change": "instructions"}}
}

The shapes above are models in stepbook/contracts/eval.py: eval.json, the rows of the results file and the
receipt this adapter prints are each parsed once, where they are read, instead of being re-checked field by field.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from functools import cached_property
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from stepbook.adapters.candidates import CandidatePolicy
from stepbook.contracts.eval import (
    ADAPTER_CODE_CHANGE,
    ADAPTER_OK,
    ADAPTER_REFUSED,
    AdapterConfig,
    ConfiguredHypothesis,
    CostArtifact,
    EvalReceipt,
    ReceiptArtifacts,
    ReceiptCandidatePolicy,
    ReceiptHoldout,
    ReceiptPrimaryMetric,
    ReceiptProvenance,
    ResultRow,
)


class Refused(Exception):
    """A verb declined to act. Its message is the reason the agent reads; ``main`` prints it on stdout and
    exits ``ADAPTER_REFUSED``, so no caller has to recognise a refusal by its words."""


class CodeChangeRequired(Exception):
    """``apply`` did apply the hypothesis, and it also needs a code change. Its message is the instructions;
    ``main`` prints them on stdout and exits ``ADAPTER_CODE_CHANGE``."""


_JSON_OBJECT: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])
_STRING_LIST: TypeAdapter[list[str]] = TypeAdapter(list[str])
_REGISTERED: TypeAdapter[dict[str, ConfiguredHypothesis]] = TypeAdapter(dict[str, ConfiguredHypothesis])


def _fault_text(error: ValidationError) -> str:
    """A rejected document as one sentence: where the fault is, and what is wrong there.

    pydantic's own rendering is several lines with a docs URL and the offending value echoed back; this text is
    what the model reads as the adapter's answer, so it stays a sentence.
    """
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"])
    return f"{where}: {first['msg']}" if where else first["msg"]


def _cost_fault(error: ValidationError) -> str:
    """The sentence this adapter has always written for each cost_artifact fault.

    It is journaled inside the receipt's ``cost.gaps`` and in metrics.json, so its bytes are a contract.
    """
    loc = error.errors()[0]["loc"]
    if loc[:1] != ("fields",):
        return "cost_artifact requires a path string"
    if len(loc) == 1 or loc[-1] == "[key]":
        return "cost fields must map tokens/usd/known_tokens/known_usd to dotted JSON paths"
    return "cost field paths must be nonempty strings"


class _AdapterRun(BaseModel):
    """One evaluation run's inputs, rows and receipt, shared by the phases of ``Generic.run``; never serialised."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    run_id: str
    out_dir: Path
    out_path: Path  # Results file the eval command writes; its suffix selects the reader.
    config_bytes: bytes
    repeat_identity: str | None = None
    start: float = 0.0
    proc: subprocess.CompletedProcess | None = None
    rows: list[ResultRow] = Field(default_factory=list)
    holdout_rows: list[ResultRow] = Field(default_factory=list)
    cases: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    deterministic: dict[str, float] = Field(default_factory=dict)
    directions: dict[str, str] = Field(default_factory=dict)
    limits: dict[str, dict[str, float]] = Field(default_factory=dict)
    result: EvalReceipt = Field(default_factory=EvalReceipt)  # replaced wholesale by _assemble_result


class Generic:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path.resolve()
        # The raw document: it is hashed verbatim into comparison_key and repeat_identity, so keys this adapter
        # does not read still identify the evaluation. `self._config` is the typed view of the same bytes.
        self.cfg = self._load_cfg(self.cfg_path)
        self.base = self.cfg_path.parent
        self.snap = self.base / ".stepbook-adapter"
        self.snap.mkdir(exist_ok=True)
        self.applied_f = self.snap / "applied.json"
        self.decided_f = self.snap / "decided.json"
        self.registered_f = self.snap / "registered.json"
        self.policy_f = self.snap / "candidate-policy.before.json"
        self.runs_dir = self.snap / "runs"
        self.runs_dir.mkdir(exist_ok=True)

    @staticmethod
    def _load_cfg(path: Path) -> dict:
        return json.loads(path.read_text())

    @cached_property
    def _config(self) -> AdapterConfig:
        """eval.json as typed values, parsed on first use so `status` and `decide` keep answering on a
        configuration whose shape the preflight would reject."""
        return AdapterConfig.model_validate(self.cfg)

    # ---- hypotheses -----------------------------------------------------------------------------------------
    def hyps(self) -> dict[str, ConfiguredHypothesis]:
        h = dict(self._config.hypotheses)
        if self.registered_f.exists():
            try:
                h.update(_REGISTERED.validate_json(self.registered_f.read_text()))
            except ValidationError as exc:
                # A file written before `register` validated proposals: refuse readably, never a traceback.
                raise ValueError(
                    f"registered hypotheses are unusable ({_fault_text(exc)}); delete {self.registered_f} to start over"
                ) from exc
        return h

    def describe(self) -> str:
        self._check_configuration()
        config = self._config
        lines = [f"  {k} ({v.kind}): {v.rationale}" for k, v in self.hyps().items()]
        return (
            f"System: {config.name}. Command: {config.run_command}. Primary metric: "
            f"{config.primary.name} ({config.primary.direction} is better); deterministic metrics: "
            f"{list(config.metric_fields) or 'none'}. Queue ALL hypotheses below before the baseline.\n"
            f"Relative paths resolve against {self.base}; config target: {self._cfg_file()}.\n"
            f"Candidate policy: {'configured (schema/literal checks, not a semantic judge)' if config.candidate_policy else 'not configured'}.\n"
            "Config/prompt 'set' values replace the entire existing value literally. There is no += append syntax. "
            "To extend a prompt, read its current value and submit the complete resulting text.\n"
            "Available hypotheses:\n" + "\n".join(lines)
        )

    # ---- config editing -------------------------------------------------------------------------------------
    def _check_configuration(self, *, check_hypotheses: bool = True) -> None:
        """Resolve real inputs before any paid eval or snapshot mutation, refusing the verb if they are unusable.

        A missing protected file is not a stable fingerprint. In particular, paths
        in a nested eval.json are not relative to the shell's project directory.
        ``check_hypotheses=False`` is for ``reset``, whose job is to delete the file that would fail the check.
        """
        try:
            config = self._config  # every shape error in eval.json surfaces here, as one message
            if check_hypotheses:
                self.hyps()  # and every shape error in the registered proposals, before a verb reads them
            if not config.config_file:
                raise ValueError("config_file must name the actual application JSON config")
            target = self._cfg_file()
            if not target.is_file():
                raise ValueError(f"config_file does not exist: {target}")
            self._application_config()
            missing = [str((self.base / p).resolve()) for p in config.protected_files if not (self.base / p).is_file()]
            if missing:
                raise ValueError("declared protected files do not exist: " + ", ".join(missing[:4]))
            if not config.run_command.strip():
                raise ValueError("run_command must be a nonempty command")
            CandidatePolicy(config.candidate_policy, self.base)
            declared = os.environ.get("STEPBOOK_EVIDENCE_ROOTS")
            if declared:
                try:
                    values = _STRING_LIST.validate_json(declared)
                except ValidationError as exc:
                    raise ValueError("invalid STEPBOOK_EVIDENCE_ROOTS") from exc
                roots = [Path(p).resolve() for p in values]
                required = [target, self.runs_dir.resolve()]
                unavailable = [str(p) for p in required if not any(p == r or p.is_relative_to(r) for r in roots)]
                if unavailable:
                    raise ValueError(
                        "evidence roots exclude required config/eval-artifact paths: "
                        + ", ".join(unavailable)
                        + "; add only user-authorized --evidence-root entries (never held-out answers)"
                    )
        except ValidationError as exc:
            fault = _fault_text(exc)
        except (OSError, ValueError, TypeError) as exc:
            fault = str(exc)
        else:
            return
        raise Refused(
            f"Error: adapter configuration: {fault}. Relative paths resolve against {self.base}, "
            "not the launch directory."
        )

    def _cfg_file(self) -> Path:
        return (self.base / self._config.config_file).resolve()

    def _application_config(self) -> dict[str, Any]:
        """The application's own JSON config, the object the hypotheses edit.

        Parsed before it is shaped, so a syntax error keeps the json module's message (it names the position)
        and only a parsed non-object is reported as a shape fault.
        """
        try:
            return _JSON_OBJECT.validate_python(json.loads(self._cfg_file().read_text()))
        except ValidationError as exc:
            raise ValueError("application config must be a JSON object") from exc

    def _changed_config(self, changes: dict) -> dict:
        data = self._application_config()
        for dotted, value in changes.items():
            cur = data
            parts = dotted.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value
        return data

    def _check_candidate(self, data: dict) -> None:
        """Refuse a candidate configuration the declared policy rejects, or a policy that moved since reset."""
        try:
            policy = CandidatePolicy(self._config.candidate_policy, self.base)
            if self.policy_f.exists():
                reference = json.loads(self.policy_f.read_text())
                if reference != {"spec": self._config.candidate_policy, "files": policy.fingerprints}:
                    raise Refused(
                        "Error: candidate policy inputs changed since reset; do not weaken a running "
                        "experiment's policy"
                    )
            if error := policy.check(data):
                raise Refused("Error: candidate policy: " + error)
        except (OSError, ValueError, TypeError) as exc:
            raise Refused(f"Error: candidate policy unavailable: {exc}") from exc

    def apply(self, hid: str) -> str:
        self._check_configuration()
        h = self.hyps()
        if hid not in h:
            raise Refused(f"Error: unknown hypothesis {hid}. Known: {', '.join(h)}.")
        if self.applied_f.exists():
            raise Refused(f"Error: {json.loads(self.applied_f.read_text())['h']} is still applied; decide it first.")
        hypothesis = h[hid]
        candidate = self._changed_config(hypothesis.set or {})
        self._check_candidate(candidate)
        shutil.copy(self._cfg_file(), self.snap / "config.before.json")
        self.applied_f.write_text(json.dumps({"h": hid}))
        if hypothesis.kind == "code":
            # Applied, and the instructions are the whole answer: the exit code says a code change is pending.
            raise CodeChangeRequired(hypothesis.code_change or "")
        self._cfg_file().write_text(json.dumps(candidate, indent=2))
        return f"Applied {hid} ({hypothesis.kind}): {hypothesis.rationale}. Config now {json.dumps(hypothesis.set)}."

    def decide(self, hid: str, what: str) -> str:
        if not self.applied_f.exists():
            raise Refused("Error: nothing is applied.")
        cur = json.loads(self.applied_f.read_text())["h"]
        if cur != hid:
            raise Refused(f"Error: {hid} is not the applied hypothesis ({cur}).")
        if what == "keep":
            try:
                candidate = self._application_config()
            except (OSError, ValueError, TypeError) as exc:
                raise Refused(f"Error: candidate policy unavailable: {exc}") from exc
            self._check_candidate(candidate)
        if what == "revert":
            shutil.copy(self.snap / "config.before.json", self._cfg_file())
        dec = json.loads(self.decided_f.read_text()) if self.decided_f.exists() else {}
        dec[hid] = what
        self.decided_f.write_text(json.dumps(dec))
        self.applied_f.unlink()
        return f"{hid} {what}: " + ("kept in configuration." if what == "keep" else "config restored.")

    # ---- runs ------------------------------------------------------------------------------------------------
    def run(self, label: str) -> str:
        try:
            self._check_configuration()
            self._check_candidate(self._application_config())
        except Refused as refusal:
            raise Refused(self._refused_eval(str(refusal))) from refusal
        progress = self._start_run(label)
        self._evaluate(progress)
        self._collect_rows(progress)
        self._score_cases(progress)
        self._average_metrics(progress)
        self._assemble_result(progress)
        self._add_provenance(progress)
        self._add_holdout(progress)
        self._invalidate_tampered(progress)
        return self._finish_run(progress)

    def _start_run(self, label: str) -> _AdapterRun:
        """Mint the run id, create its own output directory and snapshot the configuration being evaluated."""
        run_id = (
            time.strftime("%Y%m%d-%H%M%S")
            + "-"
            + "".join(ch for ch in label if ch.isalnum() or ch in "-_")[:30]
            + "-"
            + uuid.uuid4().hex[:8]
        )
        out_dir = self.runs_dir / run_id
        out_dir.mkdir()
        config_bytes = self._cfg_file().read_bytes()
        repeat_identity = self._repeat_identity(config_bytes)
        (out_dir / "config.json").write_bytes(config_bytes)
        out_path = out_dir / ("results.csv" if self._config.results_format == "csv" else "results.json")
        return _AdapterRun(
            label=label,
            run_id=run_id,
            out_dir=out_dir,
            out_path=out_path,
            config_bytes=config_bytes,
            repeat_identity=repeat_identity,
        )

    def _evaluate(self, progress: _AdapterRun) -> None:
        """Run the configured eval command against this run's fresh results path and keep both its streams."""
        cmd = self._config.run_command.replace("{out}", str(progress.out_path))
        # The user's evaluator inherits this process's environment, STEPBOOK_EVIDENCE_ROOTS and the
        # API-billing choice included; the subscription guard covers only the harness's own model calls.
        child_env = os.environ.copy()
        progress.start = time.time()
        progress.proc = subprocess.run(
            cmd,
            shell=True,
            cwd=self.base,
            capture_output=True,
            text=True,
            timeout=self._config.timeout_s,
            env=child_env,
        )
        (progress.out_dir / "stdout.txt").write_text(progress.proc.stdout)
        (progress.out_dir / "stderr.txt").write_text(progress.proc.stderr)

    def _collect_rows(self, progress: _AdapterRun) -> None:
        """Read the results file and separate the declared holdout split from the rows that are scored."""
        rows = self._read_rows(progress.out_path) if progress.out_path.exists() else []
        split_f, hold_v = self._config.split_field, str(self._config.holdout_value)
        progress.holdout_rows = [r for r in rows if split_f and r.in_split(split_f, hold_v)]
        progress.rows = [r for r in rows if not r.in_split(split_f, hold_v)] if split_f else rows

    def _score_cases(self, progress: _AdapterRun) -> None:
        """Read each scored row's pass field as pass, fail or error, keyed by case id."""
        cases: dict[str, str] = {}
        errors: list[str] = []
        case_field, pass_field = self._config.case_id_field, self._config.pass_field
        for i, r in enumerate(progress.rows):
            key = r.case_id(case_field, i)
            cases[key] = r.verdict(pass_field, self._config.pass_threshold)
            if cases[key] == "error":
                errors.append(key)
        progress.cases, progress.errors = cases, errors

    def _average_metrics(self, progress: _AdapterRun) -> None:
        """Average the declared per-row metric fields and record each one's direction and any limit."""
        for fld, spec in self._config.metric_fields.items():
            # "latency_s": "lower"  or  "latency_s": {"direction": "lower", "max": 1.2}  (a constraint, not an objective)
            vals = [value for value in (r.metric(fld) for r in progress.rows) if value is not None]
            if vals:
                progress.deterministic[fld] = round(sum(vals) / len(vals), 6)
                progress.directions[fld] = spec.direction
                if spec.limits:
                    progress.limits[fld] = spec.limits

    def _assemble_result(self, progress: _AdapterRun) -> None:
        """Build the receipt: primary metric, averaged metrics, validity, cost and the hypothesis under test."""
        # run() calls _evaluate first, and it always assigns proc
        proc = cast(subprocess.CompletedProcess[str], progress.proc)
        n = max(1, len(progress.cases))
        primary = self._config.primary
        applied = json.loads(self.applied_f.read_text())["h"] if self.applied_f.exists() else None
        dec = json.loads(self.decided_f.read_text()) if self.decided_f.exists() else {}
        progress.result = EvalReceipt(
            run_id=progress.run_id,
            config_label=progress.label,
            cases=progress.cases,
            primary_metric=ReceiptPrimaryMetric(
                name=primary.name,
                value=round(sum(v == "pass" for v in progress.cases.values()) / n, 4),
                direction=primary.direction,
            ),
            deterministic_metrics=progress.deterministic,
            metric_directions=progress.directions,
            metric_limits=progress.limits,
            errors=progress.errors,
            error_rate=round(len(progress.errors) / n, 3),
            valid=proc.returncode == 0 and bool(progress.rows) and len(progress.errors) / n <= 0.10,
            cost=self._read_cost(progress.out_path),
            duration_s=round(time.time() - progress.start, 1),
            artifacts=ReceiptArtifacts(run_dir=str(progress.out_dir)),
            applied=applied,
            undecided_hypotheses=[h for h in self.hyps() if h not in dec and h != applied],
        )

    def _add_provenance(self, progress: _AdapterRun) -> None:
        """Key the run for comparison and record what its provenance covers and what it cannot."""
        comparison = {
            "case_ids": sorted(progress.cases),
            "protected_files": self._digest(),
            "evaluation": {k: v for k, v in self.cfg.items() if k not in ("hypotheses", "config_file", "name")},
        }
        result = progress.result
        result.comparison_key = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()
        result.provenance = ReceiptProvenance(
            config_sha256=hashlib.sha256(progress.config_bytes).hexdigest(),
            repeat_identity=progress.repeat_identity if progress.repeat_identity == self._repeat_identity() else None,
            protected_files=comparison["protected_files"],
            candidate_policy=ReceiptCandidatePolicy(
                configured=bool(self._config.candidate_policy),
                checked_before_execution=True,
                spec_sha256=hashlib.sha256(
                    json.dumps(self._config.candidate_policy, sort_keys=True).encode()
                ).hexdigest(),
            ),
            coverage="Only declared protected files and evaluator configuration; external services are not fingerprinted",
        )
        result.primary_deterministic = self._config.primary_deterministic

    def _add_holdout(self, progress: _AdapterRun) -> None:
        """Score the held-back split on its own, so tuning against the visible cases is visible too."""
        if not progress.holdout_rows:
            return
        pass_field, threshold = self._config.pass_field, self._config.pass_threshold
        passed = sum(r.holdout_passed(pass_field, threshold) for r in progress.holdout_rows)
        progress.result.holdout_primary = ReceiptHoldout(
            name=self._config.primary.name,
            value=round(passed / len(progress.holdout_rows), 4),
            n=len(progress.holdout_rows),
        )

    def _invalidate_tampered(self, progress: _AdapterRun) -> None:
        """Invalidate a run whose protected eval files changed, and keep a failed command's stderr tail."""
        # run() calls _evaluate first, and it always assigns proc
        proc = cast(subprocess.CompletedProcess[str], progress.proc)
        tampered = self._protected_changed()
        if tampered:
            progress.result.valid = False
            progress.result.invalid_reason = f"protected eval files changed: {tampered}"
        if proc.returncode != 0:
            progress.result.command_error = proc.stderr[-500:]

    def _finish_run(self, progress: _AdapterRun) -> str:
        """Write the run's metrics file and answer with the receipt alone: exit 0 says the run happened."""
        document = progress.result.as_document()
        (progress.out_dir / "metrics.json").write_text(json.dumps(document, indent=1))
        return json.dumps(document)

    @staticmethod
    def _refused_eval(reason: str) -> str:
        """``run`` answers with exactly one JSON document whether it ran or refused before starting, so the
        host reads the declaration below -- and never a sentence -- to know this refusal spent nothing."""
        # This path is before any evaluation process or output directory exists.
        return json.dumps(
            {
                "reason": reason,
                "execution_started": False,
                "cost": {"tokens": 0, "usd": 0},
                "coverage": "Deterministic adapter preflight refusal; no evaluation command invoked",
            }
        )

    def _read_cost(self, out_path: Path) -> dict:
        """Read declared evaluator totals, never infer that an absent bill is zero.

        Restrict the artifact to this fresh run directory so a fixed/stale usage
        file cannot silently be charged to multiple evaluations. Includes failed
        evals; completeness of generation + judge coverage is the author's contract.
        """
        cost: dict[str, Any] = {"tokens": None, "usd": None}
        declared = self._config.cost_artifact
        if declared is None:
            return {**cost, "gaps": ["No evaluation cost artifact configured"]}
        try:
            spec = CostArtifact.model_validate(declared)
            path = Path(spec.path.replace("{out}", str(out_path)).replace("{out_dir}", str(out_path.parent)))
            if not path.is_absolute():
                path = self.base / path
            path = path.resolve()
            if not path.is_relative_to(out_path.parent.resolve()):
                raise ValueError("cost artifact must be inside the fresh run directory")
            with path.open("rb") as handle:
                raw = handle.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("cost artifact exceeds 1 MiB; emit a small aggregate usage file")
            data = json.loads(raw)
            gaps: list[str] = []
            for name, dotted in spec.fields.items():
                value = data
                for part in dotted.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                if value is None:
                    continue
                valid = (
                    (type(value) is int and value >= 0)
                    if name.endswith("tokens")
                    else (type(value) in (int, float) and math.isfinite(value) and value >= 0)
                )
                if valid:
                    cost[name] = value
                else:
                    gaps.append(f"Invalid {name}: expected a nonnegative finite number (integer for tokens)")
            for total, known in (("tokens", "known_tokens"), ("usd", "known_usd")):
                if cost.get(total) is not None and cost.get(known, 0) > cost[total]:
                    cost[total] = None
                    gaps.append(f"{total} is smaller than its known subtotal")
            return {
                **cost,
                "gaps": gaps,
                "artifact": str(path),
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "coverage": "Evaluator-declared generation and judge totals; not an independently verified invoice",
            }
        except ValidationError as exc:
            unavailable = _cost_fault(exc)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            unavailable = str(exc)
        return {"tokens": None, "usd": None, "gaps": [f"Evaluation cost artifact unavailable: {unavailable}"]}

    def _read_rows(self, path: Path) -> list[ResultRow]:
        if path.suffix == ".csv":
            with path.open(newline="") as fh:
                return ResultRow.from_csv_rows(csv.DictReader(fh))
        return ResultRow.from_results_document(json.loads(path.read_text()))

    # ---- protected eval files (the eval itself must not be edited to make numbers go up) ------------------------
    def _digest(self) -> dict:
        out: dict[str, str] = {}
        for rel in self._config.protected_files:
            p = self.base / rel
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        out.update(CandidatePolicy(self._config.candidate_policy, self.base).fingerprints)
        return out

    def _protected_changed(self) -> list[str]:
        if not self._config.protected_files and not self._config.candidate_policy:
            return []
        ref_f = self.snap / "protected.json"
        now = self._digest()
        if not ref_f.exists():
            ref_f.write_text(json.dumps(now))
            return []
        ref = json.loads(ref_f.read_text())
        return [k for k, v in now.items() if ref.get(k) != v]

    # ---- bookkeeping -------------------------------------------------------------------------------------------
    def reset(self) -> str:
        # Skip the hypothesis check: a poisoned registered.json is cured by the unlink below, not by refusing.
        self._check_configuration(check_hypotheses=False)
        if self.applied_f.exists():
            raise Refused("Error: an experiment is still applied; resume it or explicitly decide it before resetting.")
        for f in (self.applied_f, self.decided_f, self.registered_f):
            f.unlink(missing_ok=True)
        self.policy_f.write_text(
            json.dumps(
                {
                    "spec": self._config.candidate_policy,
                    "files": CandidatePolicy(self._config.candidate_policy, self.base).fingerprints,
                }
            )
        )
        if self._config.protected_files or self._config.candidate_policy:
            (self.snap / "protected.json").write_text(json.dumps(self._digest()))  # baseline hashes at episode start
        return "reset"

    def _repeat_identity(self, config_bytes=None):
        """Identity for unchanged repeats; only declared files/config are covered."""
        try:
            contents = self._cfg_file().read_bytes() if config_bytes is None else config_bytes
            value = {
                "version": 1,
                "configuration": hashlib.sha256(contents).hexdigest(),
                "evaluation": self.cfg,
                "protected_files": self._digest(),
            }
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        except (OSError, ValueError, TypeError):
            return None

    def _status_hypotheses(self) -> list[dict[str, str]]:
        """Every hypothesis ``apply`` can be asked for, as values; ``describe`` keeps the prose.

        Like the rest of ``status``, this keeps answering on inputs the preflight would reject: an unusable
        eval.json or registered.json is refused, with its reason, by ``describe``, ``apply`` and ``run``, so a
        resume's split-brain check never has to do without a parseable status.
        """
        try:
            return [{"id": hid, "kind": h.kind} for hid, h in self.hyps().items()]
        except (OSError, ValueError):
            return []

    def status(self) -> str:
        """The adapter's own bookkeeping, as values the environment reads instead of parsing sentences.

        ``phases_complete`` is false and ``phase`` null because this adapter has no ``phase`` verb: one
        eval.json is one stage.
        """
        return json.dumps(
            {
                "applied": json.loads(self.applied_f.read_text())["h"] if self.applied_f.exists() else None,
                "repeat_identity": self._repeat_identity(),
                "decided": json.loads(self.decided_f.read_text()) if self.decided_f.exists() else {},
                "hypotheses": self._status_hypotheses(),
                "phase": None,
                "phases_complete": False,
            }
        )

    def register(self, payload: str) -> str:
        self._check_configuration()
        try:
            h = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise Refused("Error: register needs a JSON object.") from exc
        if not isinstance(h, dict) or not {"id", "kind", "rationale"} <= set(h):
            raise Refused("Error: register needs id, kind, rationale.")
        if not isinstance(h["id"], str) or not h["id"]:
            raise Refused("Error: hypothesis id must be a non-empty string.")
        if h["id"] in self.hyps():
            raise Refused(f"Error: hypothesis {h['id']} already exists; use a new id for a new experiment.")
        if h["kind"] in ("config", "prompt"):
            if not isinstance(h.get("set"), dict) or not h["set"]:
                raise Refused(f"Error: {h['id']} needs a non-empty 'set' object.")
            data = self._application_config()
            for dotted in h["set"]:
                cur = data
                for p in dotted.split("."):
                    if not isinstance(cur, dict) or p not in cur:
                        raise Refused(f"Error: {h['id']} sets unknown config key {dotted!r}.")
                    cur = cur[p]
            self._check_candidate(self._changed_config(h["set"]))
        elif h["kind"] == "code":
            if not h.get("code_change"):
                raise Refused(f"Error: {h['id']} needs 'code_change' instructions.")
        else:
            raise Refused(f"Error: unknown kind {h['kind']!r}.")
        entry = {k: h[k] for k in ("kind", "rationale", "set", "code_change") if k in h}
        try:
            ConfiguredHypothesis.model_validate(entry)  # what `hyps` will read back, checked before it is stored
        except ValidationError as exc:
            raise Refused(f"Error: {h['id']} is not a usable hypothesis: {_fault_text(exc)}.") from exc
        reg = json.loads(self.registered_f.read_text()) if self.registered_f.exists() else {}
        reg[h["id"]] = entry
        self.registered_f.write_text(json.dumps(reg, indent=1))
        return f"Registered {h['id']} ({h['kind']}): {h['rationale']}"


def main(argv: list[str]) -> int:
    """Dispatch one verb and answer with its exit code: 0 done, 1 refused, 3 applied-needs-a-code-change,
    2 for a command this adapter does not implement."""
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    g = Generic(Path(argv[0]))
    cmd, args = argv[1], argv[2:]
    try:
        if cmd == "describe":
            print(g.describe())
        elif cmd == "apply" and len(args) == 1:
            print(g.apply(args[0]))
        elif cmd == "run":
            print(g.run(args[0] if args else "run"))
        elif cmd == "decide" and len(args) == 2 and args[1] in ("keep", "revert"):
            print(g.decide(args[0], args[1]))
        elif cmd == "reset":
            print(g.reset())
        elif cmd == "status":
            print(g.status())
        elif cmd == "register" and len(args) == 1:
            print(g.register(args[0]))
        else:
            print(f"Error: bad adapter command {argv[1:]}", file=sys.stderr)
            return 2
    except Refused as refusal:
        print(refusal)  # the reason, on stdout, for the agent to read
        return ADAPTER_REFUSED
    except CodeChangeRequired as request:
        print(request)  # the instructions for the coding step, on stdout
        return ADAPTER_CODE_CHANGE
    return ADAPTER_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
