"""Verdict computation for the real eval loop: how a candidate's receipts decide keep, revert or inconclusive.

``VerdictMixin`` is inherited by ``AdapterEvalEnv`` (``stepbook/envs/eval_loop.py``), so every phase keeps its
``self._phase(v)`` call site and every observation byte. ``_verdict_result`` filters the valid receipts, fills
one ``_VerdictComparison`` through the comparison phases, renders the text for the agent and returns the
decided verdict as values for the step's ``info``.

Decision precedence: a decisive deterministic regression, then the paired sign test (p < .05), then a
declared-deterministic primary difference, then
|z| >= 2, and otherwise the flat fallback in ``_choose_flat_verdict`` (deterministic improvement, then a cost
drop of 10% or more, then inconclusive, then revert). ``_check_holdout_divergence`` can still downgrade a keep.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field


class _VerdictComparison(BaseModel):
    """Transient comparison facts and decision shared by the private verdict phases."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current: list[dict] = Field(default_factory=list)
    reference: list[dict] = Field(default_factory=list)
    invalid_note: str = ""
    provisional: bool = False
    primary_sign: float = 1.0
    primary_delta: float = 0.0
    current_deterministic: dict = Field(default_factory=dict)
    reference_deterministic: dict = Field(default_factory=dict)
    metric_directions: dict = Field(default_factory=dict)
    metric_tolerances: dict = Field(default_factory=dict)
    regressions: list = Field(default_factory=list)
    improvements: list = Field(default_factory=list)
    decisive_regression: bool = False
    ambiguous: bool = False
    deterministic_note: str = ""
    limit_note: str = ""
    fixed: int | None = None
    broke: int | None = None
    pair_p: float | None = None
    z: float | None = None
    exact_primary: bool = False
    cost_delta: float | None = None
    verdict: str | None = None
    reason: str | None = None


class VerdictMixin:
    """The verdict phases of ``AdapterEvalEnv``, which supplies the receipts, the reference and the tolerances."""

    if TYPE_CHECKING:
        # Supplied by ``AdapterEvalEnv``; declarations for the type checker only, never executed, so the
        # environment's dataclass fields and their order are unchanged.
        _reference_h: str
        _runs: dict[str, list[dict]]
        alternative_groups: dict
        baseline_runs: int | None
        default_rel_tolerance: float
        metric_tolerances: dict
        runs_per_hypothesis: int

        @staticmethod
        def _valid_run(payload: dict) -> bool: ...

        @staticmethod
        def _comparison_mismatch(runs: list[dict]) -> str | None: ...

    def _alt_note(self, hid: str) -> str:
        """Name the first mutually exclusive alternative group containing this hypothesis."""
        for g, members in (self.alternative_groups or {}).items():
            if hid in members:
                return (
                    f" NOTE: {hid} is one of the mutually exclusive alternatives {members} (group '{g}'): they "
                    f"are competing designs for the same slot, not increments. Keep AT MOST one; measure each "
                    f"against the same reference, then keep the best and revert the rest."
                )
        return ""

    def _verdict(self, hid: str) -> str | None:
        """The verdict sentence shown to the agent; the journal records ``_verdict_result``'s facts too."""
        return self._verdict_result(hid)[0]

    def _verdict_result(self, hid: str) -> tuple[str | None, dict | None]:
        """Compare valid candidate receipts with the reference and render the decision.

        Returns the sentence for the agent and, once a verdict was decided, the same verdict as values for the
        step's ``info``, so no reader has to parse the sentence back (``stepbook/reporting/decision_audit.py``).
        A comparison that could not be made has no verdict to record.
        """
        v = self._valid_verdict_runs(hid)
        if not v.current:
            return f"Computed verdict for {hid}: no valid run yet{v.invalid_note}. Run again.", None
        if not v.reference:
            return None, None
        mismatch = self._comparison_mismatch(v.current + v.reference)
        if mismatch:
            return f"Computed verdict for {hid}: incomparable ({mismatch}); establish a new comparison phase.", None
        v.provisional = len(v.current) < self.runs_per_hypothesis
        self._compare_primary_means(v)
        self._compare_deterministic_metrics(v)
        self._compare_paired_cases(v)
        self._primary_standard_error(v)
        self._compare_cost(v)
        self._choose_verdict(v)
        self._check_holdout_divergence(v)
        rendered = self._render_verdict(hid, v)  # finishes v.reason, which the recorded facts carry too
        return rendered, self._verdict_facts(hid, v)

    def _verdict_facts(self, hid: str, v: _VerdictComparison) -> dict:
        """The decided verdict as values: what it is about, what it says, and what it was computed from."""
        return {
            "hypothesis": hid,
            "verdict": v.verdict,
            "provisional": v.provisional,
            "reference": self._reference_h,
            "runs": {"current": len(v.current), "reference": len(v.reference)},
            "reason": v.reason,
        }

    def _valid_verdict_runs(self, hid: str) -> _VerdictComparison:
        """Filter candidate and reference receipts and retain the invalid-run note."""
        v = _VerdictComparison()
        valid = self._valid_run
        cur_all = self._runs.get(hid) or []
        v.current = [p for p in cur_all if valid(p)]
        v.reference = [p for p in (self._runs.get(self._reference_h) or self._runs.get("baseline") or []) if valid(p)]
        v.invalid_note = (
            f" ({len(cur_all) - len(v.current)} invalid run(s) excluded; rerun to replace)"
            if len(v.current) < len(cur_all)
            else ""
        )
        return v

    @staticmethod
    def _primary_value(p: dict) -> float:
        """Read the primary measurement as a float."""
        return float(p["primary_metric"]["value"])

    def _compare_primary_means(self, v: _VerdictComparison) -> None:
        """Compute the signed candidate-minus-reference mean in the metric's direction."""
        direction = (v.current[0].get("primary_metric") or {}).get("direction", "higher")
        v.primary_sign = 1.0 if direction == "higher" else -1.0
        v.primary_delta = v.primary_sign * (
            sum(map(self._primary_value, v.current)) / len(v.current)
            - sum(map(self._primary_value, v.reference)) / len(v.reference)
        )

    @staticmethod
    def _mean_deterministic(runs: list[dict]) -> dict:
        """Average each deterministic metric over the runs that report it."""
        keys = set().union(*((p.get("deterministic_metrics") or {}).keys() for p in runs))
        return {
            k: sum(
                float((p.get("deterministic_metrics") or {}).get(k, 0))
                for p in runs
                if k in (p.get("deterministic_metrics") or {})
            )
            / max(1, sum(1 for p in runs if k in (p.get("deterministic_metrics") or {})))
            for k in keys
        }

    @staticmethod
    def _relative_metric_change(v: _VerdictComparison, k: str) -> float:
        """Normalize a deterministic change by the reference magnitude and direction."""
        base = abs(v.reference_deterministic[k]) or 1.0
        return (
            (v.current_deterministic[k] - v.reference_deterministic[k])
            * (-1.0 if v.metric_directions.get(k) == "lower" else 1.0)
        ) / base

    def _metric_noise_band(self, v: _VerdictComparison, k: str) -> float:
        """Use an explicit absolute tolerance or the default relative noise band."""
        t = v.metric_tolerances.get(k)
        return (t / (abs(v.reference_deterministic[k]) or 1.0)) if t is not None else self.default_rel_tolerance

    @staticmethod
    def _metric_limit_violated(value: float | None, limit: dict | None) -> bool:
        """Check the candidate metric against any declared minimum and maximum."""
        lim = limit or {}
        return value is not None and (
            ("max" in lim and value > float(lim["max"])) or ("min" in lim and value < float(lim["min"]))
        )

    def _compare_deterministic_metrics(self, v: _VerdictComparison) -> None:
        """Classify deterministic improvements, noise, regressions and constraint violations."""
        v.current_deterministic, v.reference_deterministic = (
            self._mean_deterministic(v.current),
            self._mean_deterministic(v.reference),
        )
        v.metric_directions = v.current[-1].get("metric_directions") or {}
        v.metric_tolerances = {**(v.current[-1].get("metric_tolerances") or {}), **self.metric_tolerances}
        limits = v.current[-1].get("metric_limits") or {}
        limit_violations = [
            k
            for k in v.current_deterministic
            if k in limits and self._metric_limit_violated(v.current_deterministic.get(k), limits.get(k))
        ]
        limit_ok = [
            k
            for k in v.current_deterministic
            if k in limits and not self._metric_limit_violated(v.current_deterministic.get(k), limits.get(k))
        ]
        shared = [k for k in v.current_deterministic if k in v.reference_deterministic and k not in limits]
        self._classify_deterministic_changes(v, shared)
        v.limit_note = (
            (
                "constraints: "
                + ", ".join(
                    [f"{k}={v.current_deterministic.get(k)} VIOLATES limit {limits[k]}" for k in limit_violations]
                    + [f"{k}={v.current_deterministic.get(k)} within limit {limits[k]}" for k in limit_ok]
                )
            )
            if limits
            else ""
        )
        if limit_violations:
            v.decisive_regression = True
            v.regressions = list(v.regressions) + limit_violations

    def _classify_deterministic_changes(self, v: _VerdictComparison, shared: list) -> None:
        """Classify unconstrained metric changes by their relative noise bands and competing gains."""
        # A regression decides only when clearly outside noise and materially larger than the best improvement.
        v.regressions = [k for k in shared if self._relative_metric_change(v, k) < -self._metric_noise_band(v, k)]
        v.improvements = [k for k in shared if self._relative_metric_change(v, k) > self._metric_noise_band(v, k)]
        worst = min((self._relative_metric_change(v, k) for k in v.regressions), default=0.0)
        best = max((self._relative_metric_change(v, k) for k in v.improvements), default=0.0)
        v.decisive_regression = (
            bool(v.regressions)
            and abs(worst) > 1.5 * self._metric_noise_band(v, v.regressions[0])
            and abs(worst) >= 1.5 * abs(best)
        )
        v.ambiguous = (
            bool(v.regressions or v.improvements)
            and not v.decisive_regression
            and abs(best) < 1.5 * max((self._metric_noise_band(v, k) for k in shared), default=1.0)
            and abs(worst) < 1.5 * max((self._metric_noise_band(v, k) for k in shared), default=1.0)
        )
        v.deterministic_note = (
            f"deterministic: {[f'{k} {self._relative_metric_change(v, k):+.1%}' for k in shared]}"
            if shared
            else "no deterministic metrics"
        )

    def _compare_paired_cases(self, v: _VerdictComparison) -> None:
        """Count consistently fixed and broken cases and compute the exact sign-test probability."""
        v.fixed = v.broke = None
        v.pair_p = None
        if all(p.get("cases") for p in v.current) and all(p.get("cases") for p in v.reference):
            # a case is "fixed" if it passes in every valid current run and failed in every valid reference run
            keys = set().union(*(p["cases"].keys() for p in v.current + v.reference))

            v.fixed = sum(
                1
                for k in keys
                if self._all_case_outcomes(v.current, k, "pass") and self._all_case_outcomes(v.reference, k, "fail")
            )
            v.broke = sum(
                1
                for k in keys
                if self._all_case_outcomes(v.current, k, "fail") and self._all_case_outcomes(v.reference, k, "pass")
            )
            n = v.fixed + v.broke
            if n:
                k_ = min(v.fixed, v.broke)
                v.pair_p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k_ + 1)) / 2**n)  # exact two-sided sign test

    @staticmethod
    def _all_case_outcomes(runs: list[dict], k: str, value: str) -> bool:
        """Require the same outcome for a case in every run."""
        return all(p["cases"].get(k) == value for p in runs)

    @staticmethod
    def _sample_variance(xs: list[float]) -> float | None:
        """Compute sample variance only when at least two run means exist."""
        return (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None

    def _primary_standard_error(self, v: _VerdictComparison) -> None:
        """Compute Welch standard error and a z score only when both sides support them."""
        current = list(map(self._primary_value, v.current))
        reference = list(map(self._primary_value, v.reference))
        v.exact_primary = len(current) >= 2 and len(reference) >= 2 and len(set(current)) == len(set(reference)) == 1
        if v.exact_primary:
            v.primary_delta = v.primary_sign * (current[0] - reference[0])
            return
        v_cur, v_ref = (
            self._sample_variance(list(map(self._primary_value, v.current))),
            self._sample_variance(list(map(self._primary_value, v.reference))),
        )
        se = (
            math.sqrt(v_cur / len(v.current) + v_ref / len(v.reference))
            if (v_cur is not None and v_ref is not None)
            else None
        )
        v.z = (v.primary_delta / se) if (se and se > 0) else None

    @staticmethod
    def _run_cost(p: dict) -> tuple[str, float] | None:
        """Prefer reported USD cost, falling back to tokens."""
        c = p.get("cost") or {}
        if c.get("usd") is not None:
            return ("usd", float(c["usd"]))
        if c.get("tokens") is not None:
            return ("tokens", float(c["tokens"]))
        return None

    def _mean_cost(self, runs: list[dict]) -> tuple[str, float] | None:
        """Average available costs only when they share one unit."""
        cs = [c for c in (self._run_cost(p) for p in runs) if c]
        if not cs or len({c[0] for c in cs}) != 1:
            return None
        return cs[0][0], sum(c[1] for c in cs) / len(cs)

    def _compare_cost(self, v: _VerdictComparison) -> None:
        """Compare mean costs only in matching units with a nonzero reference."""
        cc, cr = self._mean_cost(v.current), self._mean_cost(v.reference)
        v.cost_delta = (cc[1] - cr[1]) / cr[1] if (cc and cr and cc[0] == cr[0] and cr[1]) else None

    def _choose_verdict(self, v: _VerdictComparison) -> None:
        """Prioritize decisive regressions, paired cases, deterministic primary changes and significance."""
        deterministic_eval = all(p.get("primary_deterministic") is True for p in v.current + v.reference)
        v.verdict = v.reason = None
        if v.decisive_regression:
            v.verdict, v.reason = (
                "revert",
                f"decisive deterministic regression in {v.regressions} ({v.deterministic_note})",
            )
        elif v.ambiguous:
            v.verdict, v.reason = None, f"deterministic moves all inside noise ({v.deterministic_note})"
        if v.verdict is not None:
            pass
        elif v.pair_p is not None and v.pair_p < 0.05:
            # _compare_paired_cases sets fixed and broke whenever it sets pair_p.
            v.verdict = "keep" if v.fixed > v.broke else "revert"  # type: ignore[operator]
            v.reason = f"paired cases decide: {v.fixed} fixed vs {v.broke} broke (sign test p={v.pair_p:.3g})"
        elif deterministic_eval and abs(v.primary_delta) > 1e-9:
            v.verdict = "keep" if v.primary_delta > 0 else "revert"
            v.reason = (
                f"evaluator declares deterministic eval; primary {v.primary_delta:+.3f} "
                "is a measured difference on this suite"
            )
        elif v.z is not None and v.z >= 2:
            v.verdict, v.reason = (
                "keep",
                f"primary improved by {v.primary_delta:+.3f} "
                f"(z={v.z:.1f}{'; few runs, weak' if len(v.current) + len(v.reference) < 6 else ''})",
            )
        elif v.z is not None and v.z <= -2:
            v.verdict, v.reason = (
                "revert",
                f"primary worsened by {v.primary_delta:+.3f} "
                f"(z={v.z:.1f}{'; few runs, weak' if len(v.current) + len(v.reference) < 6 else ''})",
            )
        else:
            self._choose_flat_verdict(v)
            if v.exact_primary and deterministic_eval and v.verdict == "inconclusive":
                v.verdict = "no change"

    @staticmethod
    def _explain_zero_variance(v: _VerdictComparison) -> None:
        """Clarify the final zero-variance basis without relabeling holdout downgrades."""
        if v.exact_primary and v.verdict in ("inconclusive", "no change"):
            deterministic_eval = all(p.get("primary_deterministic") is True for p in v.current + v.reference)
            explanation = "both arms have zero variance; " + (
                "evaluator declares deterministic eval; no primary change beyond tolerance"
                if deterministic_eval
                else "an evaluator which is deterministic can declare primary_deterministic in the adapter's result "
                "or the generic adapter's config to get a decisive verdict"
            )
            v.reason = cast(str, v.reason).replace("SE unavailable: fewer than 2 runs on a side", explanation)

    def _choose_flat_verdict(self, v: _VerdictComparison) -> None:
        """Resolve an insignificant primary change using deterministic gains, cost and uncertainty."""
        ref_note = (
            f"; the reference has {len(v.reference)} valid run(s) "
            f"but the protocol asks for {self.baseline_runs}: repeat the "
            f"baseline before deciding if the run allowance permits"
            if self.baseline_runs and len(v.reference) < self.baseline_runs
            else ""
        )
        flat_why = (
            f"primary {v.primary_delta:+.3f}"
            + (f" (z={v.z:.1f})" if v.z is not None else f" (SE unavailable: fewer than 2 runs on a side{ref_note})")
            + (
                f"; paired {v.fixed} fixed / {v.broke} broke (p={v.pair_p:.2f})"
                if v.pair_p is not None
                else "; no per-case outcomes"
            )
        )
        if v.improvements:
            v.verdict, v.reason = (
                "keep",
                f"{flat_why}; deterministic improvement in {v.improvements} ({v.deterministic_note})",
            )
        elif v.cost_delta is not None and v.cost_delta <= -0.10:
            v.verdict, v.reason = "keep", f"{flat_why}; cost down {v.cost_delta:+.0%}"
        elif v.primary_delta > 0 and not v.regressions:
            # A positive primary move below significance is not evidence of harm.
            v.verdict, v.reason = (
                "inconclusive",
                (
                    f"{flat_why}; the primary moved the right way ({v.primary_delta:+.3f}) but not "
                    f"decisively and nothing regressed. Absence of significance is not "
                    f"evidence of harm: gather more evidence or restore the reference as inconclusive"
                ),
            )
        elif v.ambiguous or (not v.improvements and not v.regressions and v.pair_p is None and v.z is None):
            v.verdict, v.reason = (
                "inconclusive",
                (
                    f"{flat_why}; nothing is outside noise, so the measurements do not "
                    f"support either decision. Decide on mechanism or gather more runs"
                ),
            )
        else:
            v.verdict, v.reason = (
                "revert",
                (f"{flat_why}; no deterministic or cost improvement, and the primary did not move the right way"),
            )

    def _check_holdout_divergence(self, v: _VerdictComparison) -> None:
        """Downgrade a visible-primary keep when the held-out primary does not improve."""
        hold_cur = [
            float(p["holdout_primary"]["value"]) for p in v.current if isinstance(p.get("holdout_primary"), dict)
        ]
        hold_ref = [
            float(p["holdout_primary"]["value"]) for p in v.reference if isinstance(p.get("holdout_primary"), dict)
        ]
        if v.verdict == "keep" and hold_cur and hold_ref:
            hd = v.primary_sign * (sum(hold_cur) / len(hold_cur) - sum(hold_ref) / len(hold_ref))
            if hd <= 0 and v.primary_delta > 0:
                v.verdict = "inconclusive"
                v.reason = cast(str, v.reason) + (  # a "keep" verdict always carries its reason
                    f"; BUT held-out primary did not improve ({hd:+.3f} vs visible {v.primary_delta:+.3f}): "
                    "possible overfitting or metric gaming, investigate before keeping"
                )

    def _render_verdict(self, hid: str, v: _VerdictComparison) -> str:
        """Render the verdict, provisional warning and constraint and cost notes in protocol order."""
        self._explain_zero_variance(v)
        tag = "PROVISIONAL " if v.provisional else ""
        if v.provisional and v.verdict != "revert" or (v.provisional and v.verdict == "revert" and not v.regressions):
            verdict_text = f"**{v.verdict} (provisional: {len(v.current)} of {self.runs_per_hypothesis} runs)**"
        else:
            verdict_text = f"**{v.verdict}**"
        if v.limit_note:
            # Constraints are reported on every verdict, whichever branch decided it.
            v.reason = cast(str, v.reason) + "; " + v.limit_note  # a verdict always carries its reason
        cost_txt = "n/a" if v.cost_delta is None else f"{v.cost_delta:+.0%}"
        return (
            f"Computed verdict for {hid} after {len(v.current)} valid run(s){v.invalid_note} "
            f"vs {self._reference_h} ({len(v.reference)} run(s)): "
            f"{tag}{verdict_text} ({v.reason}; cost delta={cost_txt}). "
            + ("Run again before deciding. " if v.provisional else "")
            + "You may override with a written reason."
        )
