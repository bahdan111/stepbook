import json
from types import SimpleNamespace

from stepbook.cli import main
from stepbook.cli.inspection import write_report_md
from stepbook.envs import eval_loop
from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.storage.events import EventLog


def test_report_uses_ledger_mean_and_latest_terminal_event(tmp_path, replay_receipts):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("paused")
    log.append("step", action="stop", done=True, result="Stopped")
    (tmp_path / "state.json").write_text(
        json.dumps({"hypotheses": {"h1": {"status": "kept"}}, "reference_metrics": {"primary": 0.5627}})
    )
    env = AdapterEvalEnv(adapter="unused")
    runs = {
        "h1": [
            {
                "run_id": str(i),
                "primary_metric": {"name": "accuracy", "value": value},
                "cases": {"c": "pass"},
                "valid": True,
            }
            for i, value in enumerate((0.5, 0.625))
        ]
    }
    replay_receipts(env, runs, reference="h1")
    path = write_report_md(tmp_path, env, SimpleNamespace(cost_usd=0, total_tokens=0))
    text = path.read_text()
    assert "measured mean primary=0.5625" in text
    assert "ended: stop action / episode done" in text and "ended: paused" not in text
    replay_receipts(env, runs, reference="h1", applied="h2")
    text = write_report_md(tmp_path, env, SimpleNamespace(cost_usd=0, total_tokens=0)).read_text()
    assert "Incomplete: candidate h2 remains applied and undecided" in text


def test_cli_report_recomputes_mean_without_calling_adapter(tmp_path, monkeypatch, capsys):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("step", step=0, action="apply h1", result="Applied h1: config changed")
    for i, value in enumerate((0.5, 0.625), 1):
        log.append(
            "step",
            step=i,
            action=f"run r{i}",
            result="Eval finished:\n"
            + json.dumps({"primary_metric": {"name": "accuracy", "value": value}, "cases": {"c": "pass"}}),
        )
    log.append("step", step=3, action="decide h1 keep", result="h1 keep: kept in configuration")
    (tmp_path / "state.json").write_text('{"hypotheses":{},"reference_metrics":{"primary":0.5627}}')

    def forbidden(argv, *args, **kwargs):
        raise AssertionError(f"report must not execute the adapter: {argv!r}")

    monkeypatch.setattr(eval_loop.subprocess, "Popen", forbidden)
    monkeypatch.setattr(eval_loop.subprocess, "run", forbidden)
    assert main(["report", str(tmp_path)]) == 0
    assert "measured mean primary=0.5625" in capsys.readouterr().out


def _priced_receipt(log, step, models):
    """One controller receipt whose provider usage prices each model separately."""
    log.append(
        "model_call",
        step=step,
        input_tokens=sum(
            r["inputTokens"] + r["cacheReadInputTokens"] + r["cacheCreationInputTokens"] for r in models.values()
        ),
        output_tokens=sum(r["outputTokens"] for r in models.values()),
        cost_usd=sum(r["costUSD"] for r in models.values()),
        token_scope="all_reported_models",
        provider_model_usage=models,
        model_ids=sorted(models),
    )


OPUS = {"inputTokens": 1, "outputTokens": 2, "cacheReadInputTokens": 3, "cacheCreationInputTokens": 4, "costUSD": 0.18}
HAIKU = {"inputTokens": 5, "outputTokens": 6, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0, "costUSD": 0.02}


def _blended_receipt(log, step, model_ids):
    """A controller receipt with a known bill that names its models but carries no per-model usage."""
    log.append("model_call", step=step, input_tokens=10, output_tokens=2, cost_usd=0.1, model_ids=model_ids)
