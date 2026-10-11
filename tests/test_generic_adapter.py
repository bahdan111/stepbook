import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

FAKE_EVAL = """#!/usr/bin/env python3
import json, sys
cfg = json.load(open("config.json"))
out = sys.argv[sys.argv.index("--out") + 1]
k = cfg["retrieval"]["top_k"]
rows = [{"id": f"c{i}", "pass": (i % 5 != 0) if k >= 5 else (i % 3 != 0), "latency_s": 0.5 + 0.1 * k} for i in range(30)]
json.dump(rows, open(out, "w"))
"""


def _setup(tmp_path: Path) -> Path:
    (tmp_path / "eval.py").write_text(FAKE_EVAL)
    (tmp_path / "config.json").write_text(json.dumps({"retrieval": {"top_k": 3}, "prompt": "basic"}))
    cfg = {
        "name": "fake",
        "run_command": f"{sys.executable} eval.py --out {{out}}",
        "results_format": "json",
        "case_id_field": "id",
        "pass_field": "pass",
        "metric_fields": {"latency_s": "lower"},
        "primary": {"name": "accuracy", "direction": "higher"},
        "config_file": "config.json",
        "hypotheses": {"h1": {"kind": "config", "rationale": "top_k 3->5", "set": {"retrieval.top_k": 5}}},
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(cfg))
    return p


def _adapter(cfg: Path, *args: str, expect: int = 0) -> str:
    """One verb, with its contract exit code asserted: 0 done, 1 refused, 3 applied-needs-a-code-change."""
    done = subprocess.run(
        [sys.executable, "-m", "stepbook.adapters.generic", str(cfg), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == expect, (done.returncode, done.stdout, done.stderr)
    return done.stdout.strip()


def test_generic_adapter_full_cycle(tmp_path):
    cfg = _setup(tmp_path)
    assert "h1 (config)" in _adapter(cfg, "describe")
    base = json.loads(_adapter(cfg, "run", "baseline"))  # the receipt alone, no header line
    assert (
        base["primary_metric"]["value"] == 0.6667
        and base["deterministic_metrics"]["latency_s"] == 0.8
        and base["valid"]
    )
    assert _adapter(cfg, "apply", "h1").startswith("Applied h1")
    assert json.loads((tmp_path / "config.json").read_text())["retrieval"]["top_k"] == 5
    h1 = json.loads(_adapter(cfg, "run", "h1"))
    assert h1["primary_metric"]["value"] == 0.8 and h1["applied"] == "h1" and h1["undecided_hypotheses"] == []
    assert "restored" in _adapter(cfg, "decide", "h1", "revert")
    assert json.loads((tmp_path / "config.json").read_text())["retrieval"]["top_k"] == 3
    assert json.loads(_adapter(cfg, "status"))["decided"] == {"h1": "revert"}
    assert "Registered m1" in _adapter(
        cfg, "register", json.dumps({"id": "m1", "kind": "config", "rationale": "x", "set": {"prompt": "tables"}})
    )
    assert "unknown config key" in _adapter(
        cfg, "register", json.dumps({"id": "m2", "kind": "config", "rationale": "x", "set": {"nope": 1}}), expect=1
    )


def test_generic_adapter_passes_adapter_check(tmp_path, capsys):
    cfg = _setup(tmp_path)
    from stepbook.cli import main

    assert main(["adapter", "check", f"{sys.executable} -m stepbook.adapters.generic {cfg}", "--run"]) == 0
    assert capsys.readouterr().out == "adapter OK\n"
    # The check really drove the adapter: it applied h1, reverted it, and ran one eval.
    assert json.loads((tmp_path / "config.json").read_text()) == {"retrieval": {"top_k": 3}, "prompt": "basic"}
    runs = sorted(p.name for p in (tmp_path / ".stepbook-adapter" / "runs").iterdir())
    assert len(runs) == 1 and re.fullmatch(r"\d{8}-\d{6}-adapter-check-[0-9a-f]{8}", runs[0])
    assert sorted(p.name for p in (tmp_path / ".stepbook-adapter" / "runs" / runs[0]).iterdir()) == [
        "config.json",
        "metrics.json",
        "results.json",
        "stderr.txt",
        "stdout.txt",
    ]
    (tmp_path / "config.json").unlink()
    assert main(["adapter", "check", f"{sys.executable} -m stepbook.adapters.generic {cfg}", "--run"]) == 1
    assert capsys.readouterr().out.startswith("adapter problems:\n- describe failed (exit 1): Error: ")


GUARD_PROBE_ADAPTER = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

Path(sys.argv[1]).write_text(repr(os.environ.get("STEPBOOK_ALLOW_API_BILLING")))
if sys.argv[2] == "status":
    print(json.dumps({"applied": None, "decided": {}, "hypotheses": []}))
else:
    print("System: guard probe. Metrics: none.")
"""


@pytest.mark.parametrize("allowed", [False, True])
def test_the_api_billing_choice_reaches_the_adapter_and_its_evaluator(tmp_path, monkeypatch, allowed):
    """The guard covers the harness's own model calls; a user's adapter and evaluator inherit the choice."""
    from stepbook.cli import main

    if allowed:
        monkeypatch.setenv("STEPBOOK_ALLOW_API_BILLING", "1")
    else:
        monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
    expected = repr("1" if allowed else None)
    cfg = _setup(tmp_path)  # the evaluator the generic adapter spawns, which inherits the parent environment
    (tmp_path / "eval.py").write_text(
        FAKE_EVAL + "\nimport os\n"
        'open(out + ".guard.txt", "w").write(repr(os.environ.get("STEPBOOK_ALLOW_API_BILLING")))\n'
    )
    _adapter(cfg, "run", "guard")
    runs = tmp_path / ".stepbook-adapter" / "runs"
    assert [p.read_text() for p in runs.glob("*/results.json.guard.txt")] == [expected]
    probe, record = tmp_path / "probe.py", tmp_path / "guard.txt"  # the adapter `adapter check` itself spawns
    probe.write_text(GUARD_PROBE_ADAPTER)
    assert main(["adapter", "check", f"{sys.executable} {probe} {record}"]) == 0
    assert record.read_text() == expected


PROSE_DESCRIBE_ADAPTER = '''#!/usr/bin/env python3
"""An adapter whose describe is prose for the agent: the ids live in status, where the harness reads them."""
import json, os, sys

applied_f, cmd, args = sys.argv[1], sys.argv[2], sys.argv[3:]
applied = open(applied_f).read() if os.path.exists(applied_f) else None
if cmd == "describe":
    print("We score 40 recorded support questions. Accuracy is noisy; repeat a measurement before trusting it.")
elif cmd == "status":
    print(json.dumps({"applied": applied, "decided": {}, __HYPOTHESES__ "phases_complete": False}))
elif cmd == "apply" and not applied:
    open(applied_f, "w").write(args[0])
    print("context widened to 8 passages")
elif cmd == "decide" and args[1] == "revert" and applied == args[0]:
    os.remove(applied_f)
    print("previous configuration restored")
else:
    print(f"Error: cannot {cmd} now")
    sys.exit(1)
'''


def test_adapter_check_reads_the_ids_from_status_and_never_parses_describe(tmp_path, capsys):
    """An adapter that lists nothing in its describe text still checks out, because `adapter check` applies
    the first id `status` publishes; an adapter that publishes no list is refused by name."""
    from stepbook.cli import main

    applied_f = tmp_path / "applied.txt"
    listed = tmp_path / "listed.py"
    listed.write_text(
        PROSE_DESCRIBE_ADAPTER.replace("__HYPOTHESES__", '"hypotheses": [{"id": "wider-context", "kind": "config"}],')
    )
    assert main(["adapter", "check", f"{sys.executable} {listed} {applied_f}"]) == 0
    assert capsys.readouterr().out == "adapter OK\n"
    assert not applied_f.exists(), "the check reverts the candidate it applied"

    silent = tmp_path / "silent.py"
    silent.write_text(PROSE_DESCRIBE_ADAPTER.replace("__HYPOTHESES__", ""))
    assert main(["adapter", "check", f"{sys.executable} {silent} {applied_f}"]) == 1
    assert "status JSON must carry 'hypotheses'" in capsys.readouterr().out
    assert not applied_f.exists(), "a refused check applies nothing"


def test_empty_queue_and_immutable_dynamic_registration(tmp_path):
    cfg_path = _setup(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["hypotheses"] = {}
    cfg_path.write_text(json.dumps(cfg))
    from stepbook.cli import main

    assert main(["adapter", "check", f"{sys.executable} -m stepbook.adapters.generic {cfg_path}"]) == 0
    h = {"id": "next1", "kind": "prompt", "rationale": "Test explicit units", "set": {"prompt": "include units"}}
    assert _adapter(cfg_path, "register", json.dumps(h)).startswith("Registered")
    assert "next1" in _adapter(cfg_path, "describe")  # a fresh subprocess sees the proposal
    h["set"]["prompt"] = "different"
    assert "already exists" in _adapter(cfg_path, "register", json.dumps(h), expect=1)
    assert _adapter(cfg_path, "apply", "next1").startswith("Applied")
    assert "Error" in _adapter(cfg_path, "reset", expect=1)
    assert json.loads(_adapter(cfg_path, "status"))["applied"] == "next1"
    assert "restored" in _adapter(cfg_path, "decide", "next1", "revert")


def test_generic_run_records_candidate_and_comparison_identity(tmp_path):
    cfg = _setup(tmp_path)
    a = json.loads(_adapter(cfg, "run", "same-label"))
    _adapter(cfg, "apply", "h1")
    b = json.loads(_adapter(cfg, "run", "same-label"))
    assert a["run_id"] != b["run_id"]
    assert a["provenance"]["repeat_identity"] != b["provenance"]["repeat_identity"]
    assert json.loads(_adapter(cfg, "status"))["repeat_identity"] == b["provenance"]["repeat_identity"]
    assert a["comparison_key"] == b["comparison_key"]
    assert a["provenance"]["config_sha256"] != b["provenance"]["config_sha256"]
    assert json.loads((Path(a["artifacts"]["run_dir"]) / "config.json").read_text())["retrieval"]["top_k"] == 3


def test_generic_adapter_is_available_through_the_cli(tmp_path):
    cfg = _setup(tmp_path)
    out = subprocess.run(
        [sys.executable, "-m", "stepbook.cli", "adapter", "generic", str(cfg), "describe"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    ).stdout
    assert "h1 (config)" in out
    from stepbook.cli import main

    assert main(["adapter", "check", f"{sys.executable} -m stepbook.cli adapter generic {cfg}", "--run"]) == 0


def test_generic_adapter_holdout_and_protected_files(tmp_path):
    cfg_path = _setup(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["split_field"] = "split"
    cfg["holdout_value"] = "holdout"
    cfg["protected_files"] = ["eval.py"]
    cfg_path.write_text(json.dumps(cfg))
    (tmp_path / "eval.py").write_text(
        FAKE_EVAL.replace(
            '"latency_s": 0.5 + 0.1 * k}', '"latency_s": 0.5 + 0.1 * k, "split": "holdout" if i >= 20 else "dev"}'
        )
    )
    _adapter(cfg_path, "reset")
    r = json.loads(_adapter(cfg_path, "run", "b"))
    assert len(r["cases"]) == 20 and r["holdout_primary"]["n"] == 10 and r["valid"]
    # Exactly the 20 dev cases, with their verdicts; the 10 holdout cases are scored apart.
    assert r["cases"] == {f"c{i}": "fail" if i % 3 == 0 else "pass" for i in range(20)}
    assert r["primary_metric"] == {"name": "accuracy", "value": 0.65, "direction": "higher"}
    assert r["holdout_primary"] == {"name": "accuracy", "value": 0.7, "n": 10}
    protected = r["provenance"]["protected_files"]
    digest = hashlib.sha256((tmp_path / "eval.py").read_bytes()).hexdigest()
    assert protected == {"eval.py": digest}
    (tmp_path / "eval.py").write_text("# tampered\n" + (tmp_path / "eval.py").read_text())
    r2 = json.loads(_adapter(cfg_path, "run", "b2"))
    assert r2["valid"] is False and "protected eval files changed" in r2["invalid_reason"]
    assert r2["invalid_reason"] == "protected eval files changed: ['eval.py']"
    assert r2["provenance"]["protected_files"]["eval.py"] != digest


def _cost_eval(tmp_path, payload, exit_code=0):
    cfg_path = _setup(tmp_path)
    script = (
        FAKE_EVAL + "\njson.dump(" + repr(payload) + ', open(out + ".usage.json", "w"))\n' + f"sys.exit({exit_code})\n"
    )
    (tmp_path / "eval.py").write_text(script)
    cfg = json.loads(cfg_path.read_text())
    cfg["cost_artifact"] = {
        "path": "{out}.usage.json",
        "fields": {k: "cost." + k for k in ("tokens", "usd", "known_tokens", "known_usd")},
    }
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


@pytest.mark.parametrize("exit_code", [0, 1])
def test_generic_cost_file_covers_success_and_failed_eval(tmp_path, exit_code):
    cfg = _cost_eval(tmp_path, {"cost": {"tokens": 75, "usd": 0.125}}, exit_code)
    result = json.loads(_adapter(cfg, "run", "b"))
    assert result["valid"] == (exit_code == 0)
    assert result["cost"]["tokens"] == 75 and result["cost"]["usd"] == 0.125
    assert len(result["cost"]["artifact_sha256"]) == 64


def test_generic_partial_costs_flow_into_report(tmp_path):
    from stepbook.reporting.usage import usage_summary

    cfg = _cost_eval(tmp_path, {"cost": {"tokens": None, "known_tokens": 50, "usd": None, "known_usd": 0.1}})
    receipt = _adapter(cfg, "run", "partial")
    usage = usage_summary(
        [
            {"kind": "model_call", "step": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
            # The environment, not the adapter, writes the journaled observation around the receipt.
            {"kind": "step", "step": 0, "action": "run partial", "result": "Eval finished:\n" + receipt},
        ]
    )
    assert usage["known_total_tokens"] == 50 and usage["known_total_usd"] == 0.1
    assert not usage["tokens_complete"] and not usage["cost_complete"]


@pytest.mark.parametrize("value", [-1, True, "10", 1.5])
def test_generic_rejects_invalid_token_totals(tmp_path, value):
    cfg = _cost_eval(tmp_path, {"cost": {"tokens": value, "usd": 0.1}})
    result = json.loads(_adapter(cfg, "run", "b"))
    assert result["cost"]["tokens"] is None and result["cost"]["gaps"]
    assert result["cost"]["usd"] == 0.1


@pytest.mark.parametrize("path", ["stale-usage.json", "{out}.missing.json", "{out}"])
def test_generic_missing_or_outside_cost_file_is_not_free(tmp_path, path):
    cfg_path = _cost_eval(tmp_path, {"cost": {"tokens": 50, "usd": 0.1}})
    cfg = json.loads(cfg_path.read_text())
    cfg["cost_artifact"]["path"] = path
    cfg_path.write_text(json.dumps(cfg))
    (tmp_path / "stale-usage.json").write_text('{"cost":{"tokens":50,"usd":0.1}}')
    cost = json.loads(_adapter(cfg_path, "run", "b"))["cost"]
    assert cost["tokens"] is None and cost["usd"] is None


@pytest.mark.parametrize("usd", [float("nan"), float("inf"), -1, True, "0.1", 10**400])
def test_generic_nonfinite_or_invalid_usd_cannot_enter_budget(tmp_path, usd):
    from stepbook.adapters.generic import Generic

    cfg = _cost_eval(tmp_path, {})
    out = tmp_path / "fresh-run" / "results.json"
    out.parent.mkdir()
    Path(str(out) + ".usage.json").write_text(json.dumps({"cost": {"tokens": 10, "usd": usd}}))
    cost = Generic(cfg)._read_cost(out)
    assert cost["usd"] is None and cost["gaps"]


def test_generic_cost_file_rejects_symlink_escape_and_oversized_artifact(tmp_path):
    from stepbook.adapters.generic import Generic

    cfg = _cost_eval(tmp_path, {})
    out = tmp_path / "fresh-run" / "results.json"
    out.parent.mkdir()
    usage = Path(str(out) + ".usage.json")
    external = tmp_path / "stale.json"
    external.write_text('{"cost":{"usd":0,"tokens":0}}')
    usage.symlink_to(external)
    assert Generic(cfg)._read_cost(out)["usd"] is None
    usage.unlink()
    usage.write_text(" " * (1024 * 1024 + 1))
    assert "exceeds" in Generic(cfg)._read_cost(out)["gaps"][0]


def test_zero_spend_on_a_declared_cost_artifact_is_noted_without_failing_the_check(tmp_path, capsys):
    """A declared bill of exactly zero is suspicious and printed, but the check still exits 0."""
    from stepbook.cli import main

    for name in ("zero", "spent"):
        (tmp_path / name).mkdir()
    zero = _cost_eval(tmp_path / "zero", {"cost": {"tokens": 0, "usd": 0}})
    assert main(["adapter", "check", f"{sys.executable} -m stepbook.adapters.generic {zero}", "--run"]) == 0
    noted = capsys.readouterr().out
    assert noted.startswith("note: suspicious result: ") and "totals tokens=0, usd=0." in noted
    assert noted.endswith("adapter OK\n")
    spent = _cost_eval(tmp_path / "spent", {"cost": {"tokens": 75, "usd": 0.125}})
    assert main(["adapter", "check", f"{sys.executable} -m stepbook.adapters.generic {spent}", "--run"]) == 0
    # A real bill is silent, and so is an evaluator that declared no cost artifact at all
    # (test_generic_adapter_passes_adapter_check pins that same "adapter OK\n").
    assert capsys.readouterr().out == "adapter OK\n"


@pytest.mark.parametrize("bad_field", ["config_file", "protected_files"])
def test_nested_adapter_paths_fail_preflight_before_any_eval_or_apply(tmp_path, bad_field):
    from stepbook.adapters.generic import Generic, Refused
    from stepbook.cli import main

    cfg_path = _setup(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    nested = tmp_path / "loop"
    nested.mkdir()
    cfg["config_file"] = "../config.json"
    cfg["protected_files"] = ["../eval.py"]
    cfg[bad_field] = "config.json" if bad_field == "config_file" else ["eval.py"]
    nested_cfg = nested / "eval.json"
    nested_cfg.write_text(json.dumps(cfg))
    adapter = Generic(nested_cfg)
    with pytest.raises(Refused, match="Relative paths resolve against"):
        adapter.describe()
    # A refused run answers with one JSON declaration, so the host reads that it spent nothing.
    with pytest.raises(Refused, match='"execution_started": false'):
        adapter.run("must-not-spend")
    with pytest.raises(Refused, match="^Error: "):
        adapter.apply("h1")
    assert not list(adapter.runs_dir.iterdir()) and not adapter.applied_f.exists()
    assert main(["adapter", "check", f"{sys.executable} -m stepbook.adapters.generic {nested_cfg}"]) == 1


def test_missing_protected_input_does_not_reset_existing_bookkeeping(tmp_path):
    from stepbook.adapters.generic import Generic, Refused

    cfg_path = _setup(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["protected_files"] = ["missing-gold.json"]
    cfg_path.write_text(json.dumps(cfg))
    adapter = Generic(cfg_path)
    adapter.registered_f.write_text('{"prior":{}}')
    with pytest.raises(Refused, match="^Error: "):
        adapter.reset()
    assert adapter.registered_f.read_text() == '{"prior":{}}'


def test_evidence_preflight_checks_config_and_artifacts_without_expanding_roots(tmp_path, monkeypatch):
    from stepbook.cli import main

    cfg = _setup(tmp_path)
    dev = tmp_path / "dev"
    dev.mkdir()
    adapter = f"{sys.executable} -m stepbook.adapters.generic {cfg}"
    monkeypatch.setattr("stepbook.cli.run.ClaudeCLIExecutor", lambda **_: pytest.fail("must fail before executor"))
    root = Path(__file__).resolve().parents[1]
    log = tmp_path / "run"
    assert (
        main(
            [
                "run",
                str(root / "stepbook/templates/eval-loop-adapter"),
                "--adapter",
                adapter,
                "--evidence-root",
                str(dev),
                "--log-dir",
                str(log),
            ]
        )
        == 2
    )
    assert not (log / "events.jsonl").exists()
    assert main(["adapter", "check", adapter, "--evidence-root", str(dev)]) == 1
    # Explicit, narrow config/result roots work; no need to expose the whole repo or heldout.
    assert (
        main(
            [
                "adapter",
                "check",
                adapter,
                "--evidence-root",
                str(tmp_path / "config.json"),
                "--evidence-root",
                str(tmp_path / ".stepbook-adapter/runs"),
            ]
        )
        == 0
    )


def test_null_primary_metric_describes_the_default_objective(tmp_path):
    """A null primary metric uses the default objective consistently across all read sites."""
    cfg_path = _setup(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["primary"] = None
    cfg_path.write_text(json.dumps(cfg))
    assert "Primary metric: pass_rate (higher is better)" in _adapter(cfg_path, "describe")


def test_status_without_a_config_file_answers_like_a_missing_config(tmp_path):
    """An eval.json without `config_file` gives the same status as a missing config file."""
    cfg_path = _setup(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    del cfg["config_file"]
    cfg_path.write_text(json.dumps(cfg))
    assert json.loads(_adapter(cfg_path, "status")) == {
        "applied": None,
        "repeat_identity": None,
        "decided": {},
        "hypotheses": [{"id": "h1", "kind": "config"}],  # what apply accepts, as values, not describe's prose
        "phase": None,
        "phases_complete": False,
    }


def test_reset_cures_an_unreadable_registered_file(tmp_path):
    """`reset` is the in-band cure for a registered.json written before `register` validated proposals: every
    other verb refuses on it, and `reset` deletes it instead of refusing with the same sentence."""
    from stepbook.adapters.generic import Generic

    cfg_path = _setup(tmp_path)
    registered_f = Generic(cfg_path).registered_f
    registered_f.parent.mkdir(parents=True, exist_ok=True)
    registered_f.write_text('{"m1": {"kind": 17}}')
    assert "registered hypotheses are unusable" in _adapter(cfg_path, "describe", expect=1)
    assert _adapter(cfg_path, "reset") == "reset"
    assert not registered_f.exists()
    assert "h1 (config)" in _adapter(cfg_path, "describe")
