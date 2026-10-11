"""Offline regressions for dead launchers, typed locks and incomplete cost displays."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from stepbook import run_pointer
from stepbook.cli.inspection import cmd_report, cmd_status, write_report_md
from stepbook.run_pointer import classify_run, write_launch_pointers, write_launch_record
from stepbook.state.merge import diff
from stepbook.state.validator import PatchError, Validator


def test_pointer_writer_accepts_a_run_directory_not_yet_created(tmp_path):
    run = tmp_path / "logs" / "new"
    write_launch_pointers(run, cwd=tmp_path)
    assert json.loads((run / "launcher.json").read_text())["pid"] == os.getpid()
    assert (tmp_path / ".stepbook" / "ACTIVE").read_text() == str(run.resolve())


@pytest.mark.parametrize("with_events", [False, True])
def test_killed_launcher_is_interrupted_and_status_prints_resume(tmp_path, monkeypatch, capsys, with_events):
    run = tmp_path / "run"
    source = """
import sys
import time
from pathlib import Path
from stepbook.run_pointer import write_launch_record, write_launch_pointers
run, project = map(Path, sys.argv[1:])
write_launch_record(run, ['run', 'stepbook/templates/checklist'], cwd=project)
write_launch_pointers(run, cwd=project)
print('ready', flush=True)
time.sleep(20)
"""
    child = subprocess.Popen([sys.executable, "-c", source, str(run), str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        if with_events:
            (run / "events.jsonl").write_text('{"kind":"run_start"}\n')
        assert classify_run(run, active=True)["status"] == "running"
        for name in ("ACTIVE", "LAST_RUN"):
            assert (tmp_path / ".stepbook" / name).read_text() == str(run.resolve())
        child.kill()
        assert child.wait(timeout=5) == -signal.SIGKILL
        monkeypatch.chdir(tmp_path)
        assert cmd_status(argparse.Namespace(run_dir=None)) == 0
        output = capsys.readouterr().out
        assert "status: interrupted" in output
        launch = json.loads((run / "launch.json").read_text())
        assert launch["program"] == ["stepbook"]  # the record keeps how the launcher was really invoked
        assert f"resume: stepbook run --resume {run.resolve()}" in output
        owner = json.loads((run / "launcher.json").read_text())
        assert owner["pid"] == child.pid and "start_time" in owner
        assert owner["start_time"] is None or isinstance(owner["start_time"], str)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdout.close()


def test_reused_pid_and_resume_refresh_launcher_identity(tmp_path, monkeypatch):
    # Process inspection can be denied by the test sandbox; PID death above is tested on a real child.
    monkeypatch.setattr(
        run_pointer.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="S Thu Sep 24 08:00:00 2026\n"),
    )
    write_launch_record(tmp_path, ["run", "stepbook/templates/checklist"])
    write_launch_pointers(tmp_path, cwd=tmp_path)
    owner_path = tmp_path / "launcher.json"
    owner = {"pid": os.getpid(), "start_time": "a previous process start"}
    owner_path.write_text(json.dumps(owner))
    assert classify_run(tmp_path, active=True)["status"] == "interrupted"
    original = (tmp_path / "launch.json").read_text()
    write_launch_pointers(tmp_path, cwd=tmp_path)
    refreshed = json.loads(owner_path.read_text())
    assert refreshed["pid"] == os.getpid() and refreshed["start_time"] != owner["start_time"]
    assert classify_run(tmp_path, active=True)["status"] == "running"
    assert (tmp_path / "launch.json").read_text() == original


def test_zombie_launcher_is_interrupted_before_parent_reaps_it(tmp_path, monkeypatch):
    (tmp_path / "launcher.json").write_text(json.dumps({"pid": os.getpid(), "start_time": "Thu Sep 24 08:00:00 2026"}))
    monkeypatch.setattr(
        run_pointer.subprocess,
        "run",
        lambda argv, **k: SimpleNamespace(
            returncode=0, stdout=("Z " if "stat=" in argv else "") + "Thu Sep 24 08:00:00 2026\n"
        ),
    )
    assert classify_run(tmp_path, active=True)["status"] == "interrupted"


@pytest.mark.parametrize("launch_zone,inspect_zone", [("UTC0", "EET-2"), ("EET-2", "UTC0")])
def test_launcher_identity_is_stable_across_caller_timezones(tmp_path, monkeypatch, launch_zone, inspect_zone):
    calls = []

    def ps_local_start(argv, **kwargs):
        env = kwargs["env"]
        calls.append((env.get("TZ"), env.get("LC_ALL")))
        with monkeypatch.context() as context:
            context.setenv("TZ", env["TZ"])
            time.tzset()
            token = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime(1758700800))
        return SimpleNamespace(returncode=0, stdout=f"S {token}\n")

    try:
        with monkeypatch.context() as context:
            context.setattr(run_pointer.subprocess, "run", ps_local_start)
            context.setenv("LC_ALL", "caller-locale")
            context.setenv("TZ", launch_zone)
            write_launch_pointers(tmp_path, cwd=tmp_path)
            context.setenv("TZ", inspect_zone)
            assert classify_run(tmp_path, active=True)["status"] == "running"
            assert calls == [("UTC0", "C"), ("UTC0", "C")]
            assert os.environ["TZ"] == inspect_zone
    finally:
        time.tzset()


@pytest.mark.parametrize(
    "recorded,current,status",
    [
        ("old", "new", "interrupted"),
        ("old", "old", "running"),
        ("old", None, "running"),
        (None, "new", "running"),
        ("old", "", "interrupted"),
    ],
)
def test_permission_denied_still_checks_launcher_identity(tmp_path, monkeypatch, recorded, current, status):
    (tmp_path / "launcher.json").write_text(json.dumps({"pid": os.getpid(), "start_time": recorded}))
    inspected = []

    def denied(pid, sig):
        raise PermissionError("different owner")

    def inspect(pid):
        inspected.append(pid)
        return current

    monkeypatch.setattr(run_pointer.os, "kill", denied)
    monkeypatch.setattr(run_pointer, "_process_start_time", inspect)
    assert classify_run(tmp_path, active=True)["status"] == status
    assert inspected == [os.getpid()]


@pytest.mark.parametrize("failure", [PermissionError("denied"), subprocess.TimeoutExpired("ps", 2)])
def test_unavailable_start_time_inspection_preserves_live_pid(tmp_path, monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(run_pointer.subprocess, "run", unavailable)
    write_launch_pointers(tmp_path, cwd=tmp_path)
    assert json.loads((tmp_path / "launcher.json").read_text()) == {"pid": os.getpid(), "start_time": None}
    assert classify_run(tmp_path, active=True)["status"] == "running"


@pytest.mark.parametrize(
    "metadata",
    [
        "{",
        "null",
        "[]",
        '{"pid":true}',
        '{"pid":-1}',
        '{"pid":"bad"}',
        json.dumps({"pid": 10**50, "start_time": "old"}),
    ],
)
def test_invalid_launcher_metadata_keeps_legacy_behavior(tmp_path, metadata):
    (tmp_path / "launcher.json").write_text(metadata)
    assert classify_run(tmp_path, active=True)["status"] == "running"


@pytest.mark.parametrize(
    "event,status",
    [
        ({"kind": "paused"}, "paused"),
        ({"kind": "budget_stop"}, "stopped"),
        ({"kind": "step", "done": True}, "completed"),
    ],
)
def test_dead_launcher_does_not_override_terminal_journal(tmp_path, monkeypatch, event, status):
    monkeypatch.setattr(run_pointer, "_launcher_dead", lambda _: True)
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n")
    assert classify_run(tmp_path, active=True)["status"] == status


@pytest.mark.parametrize("terminal,status", [(None, "running"), ("paused", "paused"), ("budget_stop", "stopped")])
def test_legacy_pointer_classification_is_unchanged(tmp_path, terminal, status):
    if terminal:
        (tmp_path / "events.jsonl").write_text(json.dumps({"kind": terminal}) + "\n")
    assert classify_run(tmp_path, active=True)["status"] == status


@pytest.mark.parametrize("mode", ["protected", "write_once"])
@pytest.mark.parametrize("replacement", [True, 1.0])
@pytest.mark.parametrize("shape", ["scalar", "object", "list", "list_leaf"])
def test_locked_values_reject_json_type_changes(mode, replacement, shape):
    before, after, path = {"locked": 1}, {"locked": replacement}, "locked"
    if shape == "object":
        before, after = {"locked": {"value": 1}}, {"locked": {"value": replacement}}
    elif shape in ("list", "list_leaf"):
        before, after = {"locked": [{"value": 1}]}, {"locked": [{"value": replacement}]}
        if shape == "list_leaf":
            path = "locked.0.value"
    validator = Validator({"type": "object"}, **{f"{mode}_paths": (path,)})
    snapshot = json.dumps(before)
    with pytest.raises(PatchError, match="protected|write-once"):
        validator.apply(before, after)
    assert json.dumps(before) == snapshot


@pytest.mark.parametrize("mode", ["protected", "write_once"])
def test_lock_noops_and_unlocked_python_comparison_stay_unchanged(mode):
    validator = Validator({"type": "object"}, **{f"{mode}_paths": ("locked",)})
    state = {"locked": {"b": [1, True, 1.0], "a": 0}, "free": 1}
    patch = {"locked": {"a": 0, "b": [1, True, 1.0]}, "free": True}
    updated, changed = validator.apply(state, patch)
    assert updated["free"] is True and changed == []
    assert diff({"v": [1]}, {"v": [1.0]}) == []


@pytest.mark.parametrize("surface", ["status", "report", "markdown"])
@pytest.mark.parametrize(
    "costs,expected",
    [
        ([None], "unknown"),
        ([0.25, None], "$0.2500 (partial)"),
        ([0, None], "$0.0000 (partial)"),
        ([0], "$0.0000"),
        ([0.25], "$0.2500"),
    ],
    ids=["unknown", "partial", "partial-zero", "complete-zero", "complete"],
)
def test_cost_display_qualifies_incomplete_totals(tmp_path, monkeypatch, capsys, surface, costs, expected):
    events = [
        {"kind": "model_call", "step": i, "cost_usd": cost, "input_tokens": 1, "output_tokens": 1}
        for i, cost in enumerate(costs)
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    monkeypatch.chdir(tmp_path)
    if surface == "markdown":
        result = SimpleNamespace(cost_usd=sum(c or 0 for c in costs), total_tokens=2 * len(costs))
        output = write_report_md(tmp_path, SimpleNamespace(), result).read_text()
        assert f"agent cost: {expected.replace('0.2500', '0.25').replace('0.0000', '0.00')} |" in output
    else:
        command = cmd_status if surface == "status" else cmd_report
        assert command(argparse.Namespace(run_dir=str(tmp_path), last=5)) == 0
        output = capsys.readouterr().out
    assert f"subtotal: {expected};" in output


@pytest.mark.parametrize(
    "costs,expected",
    [
        ([{}], "unknown"),
        ([{"usd": 0.25}, {}], "$0.25 (partial)"),
        ([{"known_usd": 0}], "$0.00 (partial)"),
        ([{"usd": 0}], "$0.00"),
    ],
)
def test_markdown_evaluation_subtotal_also_marks_unknown_cost(tmp_path, costs, expected):
    env = SimpleNamespace(_runs={"h1": [{"cost": cost} for cost in costs]}, _valid_run=lambda _: False)
    result = SimpleNamespace(cost_usd=0, total_tokens=0)
    output = write_report_md(tmp_path, env, result).read_text()
    assert f"eval cost (usd, where reported): {expected}\n" in output


@pytest.mark.parametrize("surface", ["status", "report", "markdown"])
@pytest.mark.parametrize("known_usd", [None, 0, 0.25])
@pytest.mark.parametrize("durable_receipt", [False, True])
def test_mixed_component_partial_evaluation_subtotal(
    tmp_path, monkeypatch, capsys, surface, known_usd, durable_receipt
):
    cost = {"usd": None, "known_usd": known_usd, "tokens": 2}
    result_text = "Eval finished:\n" + json.dumps({"cost": cost})
    events = [{"kind": "model_call", "step": 0, "cost_usd": None, "input_tokens": 1, "output_tokens": 1}]
    if durable_receipt:
        events += [
            {"kind": "action_started", "action_id": "a1", "record": {"step": 0, "action": "run h1"}},
            {"kind": "action_result", "action_id": "a1", "result": result_text},
        ]
    else:
        events.append({"kind": "step", "step": 0, "action": "run h1", "result": result_text})
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    monkeypatch.chdir(tmp_path)
    if surface == "markdown":
        env = SimpleNamespace(_runs={"h1": [{"cost": cost}]}, _valid_run=lambda _: False)
        output = write_report_md(tmp_path, env, SimpleNamespace(cost_usd=0, total_tokens=2)).read_text()
        eval_cost = "unknown" if known_usd is None else f"${known_usd:.2f} (partial)"
        assert f"eval cost (usd, where reported): {eval_cost}\n" in output
        assert "agent cost: unknown |" in output
    else:
        command = cmd_status if surface == "status" else cmd_report
        assert command(argparse.Namespace(run_dir=str(tmp_path), last=5)) == 0
        output = capsys.readouterr().out
    expected = "unknown" if known_usd is None else f"${known_usd:.4f} (partial)"
    assert f"subtotal: {expected};" in output
