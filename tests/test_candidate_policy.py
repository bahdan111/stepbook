import json

import pytest

from stepbook.adapters.candidates import CandidatePolicy
from stepbook.adapters.generic import Generic, Refused


def setup(tmp_path):
    (tmp_path / "config.json").write_text('{"prompt":"Extract facts","limit":1}')
    (tmp_path / "literals.json").write_text('["417", "case-private-id"]')
    cfg = {
        "config_file": "config.json",
        "run_command": "must-not-be-executed",
        "candidate_policy": {
            "forbidden_literal_files": {"prompt": ["literals.json"]},
            "config_schema": {
                "type": "object",
                "required": ["prompt", "limit"],
                "additionalProperties": False,
                "properties": {"prompt": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
            },
        },
        "hypotheses": {
            "bad": {"kind": "prompt", "set": {"prompt": "Example: 417"}},
            "good": {"kind": "prompt", "set": {"prompt": "Extract values without labels"}},
        },
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(cfg))
    adapter = Generic(path)
    assert adapter.reset() == "reset"
    return adapter


def test_candidate_rejected_before_registration_snapshot_or_paid_execution(tmp_path, monkeypatch):
    adapter = setup(tmp_path)
    before = adapter._cfg_file().read_bytes()
    h = {"id": "new", "kind": "prompt", "rationale": "x", "set": {"prompt": "４１７"}}
    # Each verb refuses: the reason reaches the agent on stdout, and the process exits 1.
    with pytest.raises(Refused, match="candidate policy"):
        adapter.register(json.dumps(h))
    assert "new" not in adapter.hyps()
    with pytest.raises(Refused, match="candidate policy"):
        adapter.apply("bad")
    assert not adapter.applied_f.exists() and not (adapter.snap / "config.before.json").exists()
    assert adapter._cfg_file().read_bytes() == before
    adapter._cfg_file().write_text('{"prompt":"417","limit":1}')
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("paid eval must not run"))
    with pytest.raises(Refused, match="candidate policy"):
        adapter.run("bad")
    assert not list(adapter.runs_dir.iterdir())


def test_keep_rechecks_current_candidate_and_revert_is_always_available(tmp_path):
    adapter = setup(tmp_path)
    original = adapter._cfg_file().read_bytes()
    assert adapter.apply("good").startswith("Applied")
    adapter._cfg_file().write_text('{"prompt":"case-private-id","limit":1}')
    with pytest.raises(Refused, match="candidate policy"):
        adapter.decide("good", "keep")
    assert adapter.applied_f.exists()
    assert "restored" in adapter.decide("good", "revert")
    assert adapter._cfg_file().read_bytes() == original


def test_policy_files_and_policy_spec_cannot_be_weakened_mid_run(tmp_path):
    adapter = setup(tmp_path)
    assert "literals.json" in adapter._digest()
    (tmp_path / "literals.json").write_text("[]")
    with pytest.raises(Refused, match="changed since reset"):
        adapter.apply("good")
    adapter.cfg["candidate_policy"] = {}
    with pytest.raises(Refused, match="changed since reset"):
        adapter.apply("good")


def test_invalid_policy_fails_closed_without_reference_fetching(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.socket", lambda *a, **k: pytest.fail("a policy must never open a connection"))
    for spec, error, message in (
        (
            {"unknown": True},
            ValueError,
            "candidate_policy accepts config_schema, forbidden_literals and forbidden_literal_files",
        ),
        ({"config_schema": {"type": "invalid"}}, ValueError, "candidate config_schema is not a valid JSON schema"),
        (
            {"config_schema": {"$ref": "https://example.com/schema"}},
            ValueError,
            "candidate config_schema must be self-contained; references are not supported",
        ),
        ({"forbidden_literals": {"prompt": [""]}}, ValueError, "forbidden_literals must be a list of nonempty strings"),
        ({"forbidden_literal_files": {"prompt": ["missing.json"]}}, FileNotFoundError, None),
    ):
        with pytest.raises((ValueError, OSError)) as refused:
            CandidatePolicy(spec, tmp_path)
        assert type(refused.value) is error
        if message is not None:
            assert str(refused.value) == message
        else:
            assert str(refused.value).endswith("missing.json'") and "No such file or directory" in str(refused.value)
    policy = CandidatePolicy({"forbidden_literals": {"prompt": ["secret value"]}}, tmp_path)
    assert policy.check({"prompt": "SECRET\n VALUE"})
    assert policy.check({"other": "allowed"})
    assert policy.check({"prompt": 12})
    assert policy.check({"prompt": "generic wording"}) is None
    assert policy.check({"prompt": "SECRET\n VALUE"}) == (
        "forbidden literal in prompt; remove copied examples/identifiers, not the policy"
    )
    assert policy.check({"other": "allowed"}) == "required checked field prompt is missing"
    assert policy.check({"prompt": 12}) == "checked field prompt must be a string"


def test_refused_eval_has_known_zero_cost_but_still_consumes_an_attempt_on_resume(tmp_path, fake_adapter):
    from stepbook.envs.eval_loop import AdapterEvalEnv
    from stepbook.reporting.usage import usage_summary

    adapter = setup(tmp_path)
    adapter._cfg_file().write_text('{"prompt":"417","limit":1}')
    with pytest.raises(Refused) as refusal:
        adapter.run("refused")
    scripted = fake_adapter()
    # The refusal's own stdout: one JSON declaration that nothing started, answered with exit 1.
    scripted.answer("run refused", str(refusal.value), exit_code=1)
    env = AdapterEvalEnv(adapter=scripted.command, runs_max=2)
    events = []
    for step in range(3):
        reply = env.step("run refused")
        events.extend(
            [
                {"kind": "model_call", "step": step, "input_tokens": 3, "output_tokens": 1, "cost_usd": 0.01},
                {
                    "kind": "step",
                    "step": step,
                    "action": "run refused",
                    "result": reply.observation,
                    "info": reply.info,
                },
            ]
        )
    assert env.runs_used == 2
    # First two deterministic adapter refusals have known zero eval charge, recorded as facts: the adapter
    # declared it started nothing, and only the third (local) refusal says it reserved no attempt.
    first, last = events[1]["info"], events[5]["info"]
    assert first == {"ok": False, "adapter_exit": 1, "run_label": "refused", "evaluation_not_started": True}
    assert last == {"ok": False, "evaluation_not_started": True, "attempt_spent": False}
    usage = usage_summary(events)
    assert usage["cost_complete"] and usage["tokens_complete"]
    assert usage["components"]["evaluation"]["known_usd"] == 0
    resumed = AdapterEvalEnv(adapter="unused", runs_max=2)
    resumed.restore_from_events(events, check_adapter=False)
    assert resumed.runs_used == env.runs_used


def test_unknown_failed_eval_attempt_not_refunded_by_resume():
    from stepbook.envs.eval_loop import AdapterEvalEnv

    env = AdapterEvalEnv(adapter="unused")
    env.restore_from_events(
        [{"kind": "step", "action": "run a", "result": "Adapter error: timed out"}], check_adapter=False
    )
    assert env.runs_used == 1
