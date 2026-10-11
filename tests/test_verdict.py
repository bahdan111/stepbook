def _payload(prim, cases, det, cost=100.0, direction="higher"):
    return {
        "run_id": "r",
        "primary_metric": {"name": "acc", "value": prim, "direction": direction},
        "cases": cases,
        "deterministic_metrics": det,
        "cost": {"usd": cost},
    }


def test_verdict_keeps_clear_improvement_and_reverts_deterministic_regression(measure_verdict):
    runs = {}
    options = dict()
    base_cases = {f"c{i}": ("pass" if i < 50 else "fail") for i in range(100)}
    runs["baseline"] = [_payload(0.50, base_cases, {"recall": 0.8}), _payload(0.51, base_cases, {"recall": 0.8})]
    better = {f"c{i}": ("pass" if i < 70 else "fail") for i in range(100)}
    runs["h1"] = [_payload(0.70, better, {"recall": 0.8}), _payload(0.71, better, {"recall": 0.8})]
    v = measure_verdict(runs, "h1", **options)
    assert "**keep**" in v and "20 fixed vs 0 broke" in v
    runs["h2"] = [_payload(0.60, better, {"recall": 0.7})]
    v = measure_verdict(runs, "h2", **options)
    assert "**revert**" in v and "deterministic regression" in v


def test_verdict_flat_costlier_is_reverted_and_cheaper_kept(measure_verdict):
    runs = {}
    options = dict()
    cases = {f"c{i}": "pass" for i in range(100)}
    runs["baseline"] = [
        _payload(0.50, cases, {"recall": 0.8}, cost=100),
        _payload(0.50, cases, {"recall": 0.8}, cost=100),
    ]
    runs["h1"] = [
        _payload(0.505, cases, {"recall": 0.8}, cost=150),
        _payload(0.495, cases, {"recall": 0.8}, cost=150),
    ]
    assert "**revert**" in measure_verdict(runs, "h1", **options)
    runs["h2"] = [
        _payload(0.505, cases, {"recall": 0.8}, cost=60),
        _payload(0.495, cases, {"recall": 0.8}, cost=60),
    ]
    v = measure_verdict(runs, "h2", **options)
    assert "**keep**" in v and "cost down" in v


def test_verdict_respects_lower_is_better(measure_verdict):
    runs = {}
    options = dict()
    cases = {f"c{i}": "pass" for i in range(100)}
    runs["baseline"] = [_payload(10.0, cases, {}, direction="lower"), _payload(10.2, cases, {}, direction="lower")]
    runs["h1"] = [_payload(2.0, cases, {}, direction="lower"), _payload(2.1, cases, {}, direction="lower")]
    assert "**keep**" in measure_verdict(runs, "h1", **options)


def test_verdict_excludes_invalid_runs(measure_verdict):
    runs = {}
    options = dict()
    cases = {f"c{i}": "pass" for i in range(100)}
    runs["baseline"] = [_payload(0.50, cases, {"recall": 0.8}), _payload(0.52, cases, {"recall": 0.8})]
    bad = _payload(0.03, cases, {"recall": 0.8})
    bad["valid"] = False
    bad["error_rate"] = 0.9
    runs["h1"] = [bad]
    assert "no valid run yet" in measure_verdict(runs, "h1", **options)
    runs["h1"].append(_payload(0.51, cases, {"recall": 0.8}))
    v = measure_verdict(runs, "h1", **options)
    assert "1 valid run(s) (1 invalid" in v


def test_paired_cases_decide_even_when_run_means_are_noisy(measure_verdict):
    runs = {}
    options = dict(runs_per_hypothesis=2)
    base = {f"c{i}": ("pass" if i < 150 else "fail") for i in range(300)}
    better = dict(base)
    better.update({f"c{i}": "pass" for i in range(150, 165)})  # 15 fixed, 0 broke
    runs["baseline"] = [_payload(0.50, base, {}), _payload(0.50, base, {})]
    runs["h1"] = [_payload(0.55, better, {}), _payload(0.55, better, {})]
    v = measure_verdict(runs, "h1", **options)
    assert "**keep**" in v and "15 fixed vs 0 broke" in v and "sign test" in v


def test_no_cases_means_no_binomial_floor_and_provisional_with_one_run(measure_verdict):
    runs = {}
    options = dict(runs_per_hypothesis=2)
    runs["baseline"] = [_payload(0.50, {}, {})]
    runs["h1"] = [_payload(0.80, {}, {})]
    v = measure_verdict(runs, "h1", **options)
    assert "PROVISIONAL" in v and "SE unavailable" in v and "z=" not in v


def test_lower_is_better_latency_does_not_get_fake_significance(measure_verdict):
    runs = {}
    options = dict(runs_per_hypothesis=1)
    runs["baseline"] = [_payload(10.0, {}, {}, direction="lower")]
    runs["h1"] = [_payload(10.5, {}, {}, direction="lower")]
    v = measure_verdict(runs, "h1", **options)
    # one run each side: no standard error exists, so nothing is decisive and no z may be fabricated
    assert "z=" not in v and "**inconclusive**" in v


def test_deterministic_tolerance_and_cost_units(measure_verdict):
    runs = {}
    options = dict(runs_per_hypothesis=1, metric_tolerances={"recall": 0.01})
    cases = {f"c{i}": "pass" for i in range(30)}
    runs["baseline"] = [_payload(0.5, cases, {"recall": 0.8000}, cost=100)]
    runs["h1"] = [_payload(0.5, cases, {"recall": 0.8001}, cost=100)]
    assert "deterministic improvement" not in measure_verdict(
        runs, "h1", **options
    )  # within tolerance -> not an improvement
    ref = _payload(0.5, cases, {"recall": 0.8})
    ref["cost"] = {"tokens": 1000}
    runs["baseline"] = [ref]
    runs["h2"] = [_payload(0.5, cases, {"recall": 0.8}, cost=1.0)]
    assert "cost delta=n/a" in measure_verdict(runs, "h2", **options)  # usd vs tokens are not compared


def test_deterministic_eval_requires_explicit_declaration(measure_verdict):
    runs = {}
    options = dict(runs_per_hypothesis=2)
    runs["baseline"] = [_payload(0.50, {}, {}), _payload(0.50, {}, {})]
    runs["h1"] = [_payload(0.80, {}, {}), _payload(0.80, {}, {})]
    assert "**inconclusive**" in measure_verdict(runs, "h1", **options)
    for rows in runs.values():
        for p in rows:
            p["primary_deterministic"] = True
    v = measure_verdict(runs, "h1", **options)
    assert "**keep**" in v and "deterministic eval" in v


def test_holdout_divergence_downgrades_a_keep(measure_verdict):
    runs = {}
    options = dict(runs_per_hypothesis=2)
    cases = {f"c{i}": "pass" for i in range(100)}
    base = [_payload(0.50, cases, {}), _payload(0.50, cases, {})]
    for p in base:
        p["holdout_primary"] = {"value": 0.50}
        p["primary_deterministic"] = True
    better = [_payload(0.80, cases, {}), _payload(0.80, cases, {})]
    for p in better:
        p["holdout_primary"] = {"value": 0.49}
        p["primary_deterministic"] = True
    runs["baseline"], runs["h1"] = base, better
    v = measure_verdict(runs, "h1", **options)
    assert "inconclusive" in v and "metric gaming" in v


def test_a_small_dip_in_one_metric_does_not_veto_a_large_gain_in_another(measure_verdict):
    """A small metric dip must not veto a larger gain in another metric."""
    runs = {}
    options = dict(runs_per_hypothesis=2)
    cases_ref = {f"c{i}": ("pass" if i < 6 else "fail") for i in range(10)}
    cases_cur = {f"c{i}": ("pass" if i < 8 else "fail") for i in range(10)}
    runs["h2"] = [
        _payload(0.60, cases_ref, {"recall": 0.80, "mrr": 0.20}),
        _payload(0.65, cases_ref, {"recall": 0.80, "mrr": 0.20}),
    ]
    options["reference"] = "h2"
    runs["h3"] = [
        _payload(0.80, cases_cur, {"recall": 0.795, "mrr": 0.60}),
        _payload(0.85, cases_cur, {"recall": 0.795, "mrr": 0.60}),
    ]
    v = measure_verdict(runs, "h3", **options)
    assert "**keep**" in v, v
    assert "revert" not in v.split("**")[1] if "**" in v else True


def test_a_real_regression_with_nothing_improving_still_reverts(measure_verdict):
    runs = {}
    options = dict(runs_per_hypothesis=2)
    cases = {f"c{i}": ("pass" if i < 5 else "fail") for i in range(10)}
    runs["baseline"] = [
        _payload(0.51, cases, {"recall": 0.80, "mrr": 0.50}),
        _payload(0.50, cases, {"recall": 0.80, "mrr": 0.50}),
    ]
    runs["h6"] = [
        _payload(0.49, cases, {"recall": 0.70, "mrr": 0.45}),
        _payload(0.48, cases, {"recall": 0.70, "mrr": 0.45}),
    ]
    v = measure_verdict(runs, "h6", **options)
    assert "**revert**" in v and "deterministic regression" in v


def test_moves_inside_noise_are_inconclusive_not_a_revert(measure_verdict):
    """Metric changes within the noise band must be inconclusive rather than force a revert."""
    runs = {}
    options = dict(runs_per_hypothesis=2)
    ref = {f"c{i}": ("pass" if i < 8 else "fail") for i in range(10)}
    cur = {f"c{i}": ("pass" if i < 8 else "fail") for i in range(10)}
    runs["h3"] = [
        _payload(0.50, ref, {"recall": 0.50, "mrr": 0.50}),
        _payload(0.60, ref, {"recall": 0.50, "mrr": 0.50}),
    ]
    options["reference"] = "h3"
    runs["h4"] = [
        _payload(0.50, cur, {"recall": 0.505, "mrr": 0.489}),
        _payload(0.55, cur, {"recall": 0.505, "mrr": 0.489}),
    ]
    v = measure_verdict(runs, "h4", **options)
    assert "**inconclusive**" in v and "nothing is outside noise" in v, v


def test_a_positive_but_not_significant_primary_is_inconclusive_not_a_revert(measure_verdict):
    """A positive primary change without statistical significance must be inconclusive rather than a revert."""
    runs = {}
    options = dict(runs_per_hypothesis=2)
    ref = {f"c{i}": ("pass" if i < 8 else "fail") for i in range(10)}
    cur = {f"c{i}": ("pass" if i < 9 else "fail") for i in range(10)}
    runs["h13"] = [
        _payload(0.60, ref, {"recall": 0.50, "mrr": 0.50}),
        _payload(0.60, ref, {"recall": 0.50, "mrr": 0.50}),
    ]
    options["reference"] = "h13"
    runs["h15"] = [
        _payload(0.61, cur, {"recall": 0.50, "mrr": 0.50}),
        _payload(0.65, cur, {"recall": 0.50, "mrr": 0.50}),
    ]
    v = measure_verdict(runs, "h15", **options)
    assert "**revert**" not in v and "not evidence of harm" in v, v


def test_alternative_group_is_announced_with_the_verdict(measure_verdict):
    runs = {
        "baseline": [_payload(0.5, {}, {})],
        "h11": [_payload(0.8, {}, {})],
        "h13": [_payload(0.8, {}, {})],
    }
    options = {"alternative_groups": {"retriever": ["h11", "h12"]}, "runs_per_hypothesis": 1}
    verdict, note = measure_verdict(runs, "h11", **options).split(" NOTE: ")
    assert "**inconclusive**" in verdict
    assert note == (
        "h11 is one of the mutually exclusive alternatives ['h11', 'h12'] (group 'retriever'): they "
        "are competing designs for the same slot, not increments. Keep AT MOST one; measure each "
        "against the same reference, then keep the best and revert the rest."
    )
    text = measure_verdict(runs, "h13", **options)
    assert "**inconclusive**" in text and " NOTE:" not in text
