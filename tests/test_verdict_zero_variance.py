import pytest


def _run(value, direction="higher", deterministic=False):
    result = {"run_id": "r", "primary_metric": {"name": "acc", "value": value, "direction": direction}}
    if deterministic:
        result["primary_deterministic"] = True
    return result


def _assert_zero_variance_hint(verdict, difference):
    assert "**inconclusive**" in verdict
    assert "both arms have zero variance" in verdict
    assert f"primary {difference:+.3f}" in verdict
    for fragment in ("deterministic", "primary_deterministic", "adapter's result", "generic adapter's config"):
        assert fragment in verdict
    assert "SE unavailable" not in verdict and "z=" not in verdict


@pytest.mark.parametrize("direction", ["higher", "lower"])
@pytest.mark.parametrize("candidate", [0.8, 0.2, 0.5])
@pytest.mark.parametrize("deterministic", [False, True])
def test_zero_variance_requires_declaration(measure_verdict, direction, candidate, deterministic):
    runs = {
        "baseline": [_run(0.5, direction, deterministic)] * 3,
        "h1": [_run(candidate, direction, deterministic)] * 2,
    }
    verdict = measure_verdict(runs, "h1")
    difference = (candidate - 0.5) * (1 if direction == "higher" else -1)
    if not deterministic:
        _assert_zero_variance_hint(verdict, difference)
    else:
        expected = "no change" if candidate == 0.5 else "keep" if difference > 0 else "revert"
        assert f"**{expected}**" in verdict
        assert "deterministic eval" in verdict
        assert "SE unavailable" not in verdict and "z=" not in verdict


@pytest.mark.parametrize("deterministic", [False, True])
def test_identical_decimal_runs_do_not_fabricate_a_difference(measure_verdict, deterministic):
    verdict = measure_verdict(
        {"baseline": [_run(0.1, deterministic=deterministic)] * 3, "h1": [_run(0.1, deterministic=deterministic)] * 2},
        "h1",
    )
    assert "primary +0.000" in verdict
    assert "SE unavailable" not in verdict and "z=" not in verdict
    if deterministic:
        assert "**no change**" in verdict and "deterministic eval" in verdict
    else:
        _assert_zero_variance_hint(verdict, 0.0)


@pytest.mark.parametrize("deterministic", [False, True])
def test_three_fixed_cases_with_zero_variance(measure_verdict, deterministic):
    reference = {**_run(0.5, deterministic=deterministic), "cases": {"a": "fail", "b": "fail", "c": "fail"}}
    candidate = {**_run(0.8, deterministic=deterministic), "cases": {"a": "pass", "b": "pass", "c": "pass"}}
    verdict = measure_verdict({"baseline": [reference] * 3, "h1": [candidate] * 2}, "h1")
    if deterministic:
        assert "**keep**" in verdict and "deterministic eval" in verdict
    else:
        _assert_zero_variance_hint(verdict, 0.3)


@pytest.mark.parametrize("reference_count,candidate_count", [(1, 1), (1, 2), (2, 1)])
def test_one_run_is_still_insufficient(measure_verdict, reference_count, candidate_count):
    verdict = measure_verdict(
        {"baseline": [_run(0.5)] * reference_count, "h1": [_run(0.8)] * candidate_count},
        "h1",
        runs_per_hypothesis=1,
    )
    assert "**inconclusive**" in verdict and "SE unavailable" in verdict
    assert "z=" not in verdict and "both arms have zero variance" not in verdict


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("gain", ["cost", "recall"])
def test_constant_primary_preserves_independent_gain_and_reason(measure_verdict, deterministic, gain):
    reference = {
        **_run(0.5, deterministic=deterministic),
        "cost": {"usd": 100},
        "deterministic_metrics": {"recall": 0.5},
    }
    candidate = {
        **reference,
        **({"cost": {"usd": 60}} if gain == "cost" else {"deterministic_metrics": {"recall": 0.8}}),
    }
    verdict = measure_verdict({"baseline": [reference] * 3, "h1": [candidate] * 2}, "h1")
    reason = (
        "cost down -40%"
        if gain == "cost"
        else "deterministic improvement in ['recall'] (deterministic: ['recall +60.0%'])"
    )
    cost = "-40%" if gain == "cost" else "+0%"
    assert verdict == (
        "Computed verdict for h1 after 2 valid run(s) vs baseline (3 run(s)): **keep** "
        "(primary +0.000 (SE unavailable: fewer than 2 runs on a side); no per-case outcomes; "
        f"{reason}; cost delta={cost}). You may override with a written reason."
    )


@pytest.mark.parametrize("candidate_value", [0.2, 0.8])
def test_undeclared_constant_primary_preserves_cost_fallback(measure_verdict, candidate_value):
    reference = {**_run(0.5), "cost": {"usd": 100}}
    candidate = {**_run(candidate_value), "cost": {"usd": 60}}
    verdict = measure_verdict({"baseline": [reference] * 3, "h1": [candidate] * 2}, "h1")
    assert "**keep**" in verdict and "cost down -40%" in verdict


@pytest.mark.parametrize("deterministic", [False, True])
def test_constant_primary_preserves_flat_revert(measure_verdict, deterministic):
    reference = {**_run(0.5, deterministic=deterministic), "cases": {"a": "pass"}}
    candidate = {**_run(0.5, deterministic=deterministic), "cases": {"a": "fail"}}
    verdict = measure_verdict({"baseline": [reference] * 3, "h1": [candidate] * 2}, "h1")
    assert verdict == (
        "Computed verdict for h1 after 2 valid run(s) vs baseline (3 run(s)): **revert** "
        "(primary +0.000 (SE unavailable: fewer than 2 runs on a side); paired 0 fixed / 1 broke (p=1.00); "
        "no deterministic or cost improvement, and the primary did not move the right way; cost delta=n/a). "
        "You may override with a written reason."
    )


def test_zero_variance_explanation_preserves_inconclusive_basis(measure_verdict):
    reference = {**_run(0.5), "cases": {"a": "fail"}}
    candidate = {**_run(0.8), "cases": {"a": "pass"}}
    verdict = measure_verdict({"baseline": [reference] * 3, "h1": [candidate] * 2}, "h1", baseline_runs=4)
    _assert_zero_variance_hint(verdict, 0.3)
    assert "paired 1 fixed / 0 broke (p=1.00)" in verdict
    assert "the primary moved the right way (+0.300) but not decisively and nothing regressed" in verdict
    assert "the reference has 3 valid run(s) but the protocol asks for 4" in verdict


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("gain", ["usd", "tokens", "recall"])
def test_holdout_downgrade_explains_final_zero_variance(measure_verdict, deterministic, gain):
    reference = {**_run(0.5, deterministic=deterministic), "holdout_primary": {"value": 0.5}}
    candidate = {**_run(0.75, deterministic=deterministic), "holdout_primary": {"value": 0.5}}
    if gain == "recall":
        reference["deterministic_metrics"] = {"recall": 0.5}
        candidate["deterministic_metrics"] = {"recall": 0.8}
    else:
        reference["cost"] = {gain: 100}
        candidate["cost"] = {gain: 60}
    verdict = measure_verdict({"baseline": [reference] * 3, "h1": [candidate] * 2}, "h1")
    assert "**inconclusive**" in verdict and "no change" not in verdict
    assert "BUT held-out primary did not improve (+0.000 vs visible +0.250)" in verdict
    assert "possible overfitting or metric gaming, investigate before keeping" in verdict
    if deterministic:
        assert "evaluator declares deterministic eval; primary +0.250 is a measured difference" in verdict
    else:
        _assert_zero_variance_hint(verdict, 0.25)
        assert "deterministic improvement in ['recall']" in verdict if gain == "recall" else "cost down -40%" in verdict
