"""Step horizons, max_steps_reached + resume, and cost accounting with cached tokens."""

import pytest

from stepbook import Envelope, Runtime
from stepbook.reporting.usage import usage_summary
from stepbook.run_pointer import classify_run

from ._helpers import ROOT, Script, checklist, cmd, events, kinds, shell_env, steps


def _run(tmp_path, script, n, resume=False, **rt_kw):
    env = shell_env(tmp_path)
    return Runtime(checklist(), script.executor(), env, log_dir=tmp_path / "log", **rt_kw).run(
        n, resume=resume, replay_env=False
    )


def test_steps_one_then_resume_with_two_continues_from_the_journal(tmp_path):
    res = _run(tmp_path, Script(cmd("echo one")), 1)
    assert res.steps == 1 and kinds(tmp_path / "log")[-1] == "max_steps_reached"
    assert classify_run(tmp_path / "log")["status"] == "stopped"
    again = Script(cmd("echo two"))
    res2 = _run(tmp_path, again, 2, resume=True)
    assert again.calls == 1 and res2.steps == 2 and [e["step"] for e in steps(tmp_path / "log")] == [0, 1]


def test_cli_rejects_steps_zero(tmp_path, capsys):
    from stepbook.cli import main

    run_dir = tmp_path / "run"  # the default log dir is cwd-relative, so name one we can watch
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                str(ROOT / "stepbook/templates" / "checklist"),
                "--workdir",
                str(tmp_path),
                "--log-dir",
                str(run_dir),
                "--goal",
                "g",
                "--steps",
                "0",
                "--quiet",
                "--no-checkpoint",
                "--unsandboxed-i-accept-the-risk",
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.splitlines()[-1].endswith("error: argument --steps: must be positive"), err
    assert not run_dir.exists() and list(tmp_path.iterdir()) == [], "a rejected --steps must not start a run"


def test_runtime_zero_steps_makes_no_call_and_logs_max_steps_reached(tmp_path):
    script = Script(exhausted=None)
    res = _run(tmp_path, script, 0)
    assert script.calls == 0 and res.steps == 0 and kinds(tmp_path / "log")[-1] == "max_steps_reached"


def test_no_max_steps_event_when_the_episode_completes_on_the_last_allowed_step(tmp_path):
    res = _run(tmp_path, Script(cmd("echo one"), cmd("done")), 2)
    assert res.steps == 2 and "max_steps_reached" not in kinds(tmp_path / "log")


def test_cached_tokens_and_cost_are_accounted_once_across_a_resume(tmp_path):
    def paid(action):
        return Envelope(
            "r",
            {},
            action,
            input_tokens=1000,
            output_tokens=20,
            cost_usd=0.01,
            usage={"input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 20},
        )

    res = _run(tmp_path, Script(paid("echo one")), 1)
    assert res.cost_usd == pytest.approx(0.01) and res.prompt_tokens == [1000] and res.llm_calls == 1
    res2 = _run(tmp_path, Script(paid("echo two"), paid("done")), 5, resume=True)
    assert res2.cost_usd == pytest.approx(0.03) and res2.llm_calls == 3 and res2.prompt_tokens == [1000, 1000, 1000]
    summary = usage_summary(events(tmp_path / "log"))
    assert (
        summary["known_total_usd"] == pytest.approx(0.03)
        and summary["cost_complete"]
        and summary["known_total_tokens"] == 3060
    )
    calls = [e for e in events(tmp_path / "log") if e.get("kind") == "model_call"]
    assert all(e["provider_usage"]["cache_read_input_tokens"] == 900 for e in calls)


def test_expensive_actions_dispatch_and_resume_without_dollar_stop(tmp_path):
    def paid(action):
        return Envelope("r", {}, action, input_tokens=10, output_tokens=1, cost_usd=100)

    res = _run(tmp_path, Script(paid("echo run >> marks.txt"), paid("echo run >> marks.txt")), 2)
    assert not res.aborted and res.steps == 2 and (tmp_path / "marks.txt").read_text().count("run") == 2
    assert "action_withheld" not in kinds(tmp_path / "log") and "budget_stop" not in kinds(tmp_path / "log")
    again = Script(paid("done"))
    res2 = _run(tmp_path, again, 5, resume=True)
    assert again.calls == 1 and not res2.aborted and res2.steps == 3 and res2.cost_usd == 300
