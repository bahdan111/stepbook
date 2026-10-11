import json
from pathlib import Path

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.cli import main
from stepbook.contracts import errors as errors_module
from stepbook.contracts.runtime import StepResult
from stepbook.executors.claude import ClaudeCLIExecutor
from stepbook.storage.events import EventLog


class OfflineEnv:
    require_completion = False

    def reset(self):
        return "ready"

    def step(self, action):
        return StepResult("ok", done=action == "done")

    def score(self):
        return 0

    def restore_from_events(self, events):
        pass


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill = Skill("offline", "work", {"type": "object"}, {"goal": "g"}, protected_paths=("goal",))
    env = OfflineEnv()
    monkeypatch.setattr("stepbook.cli.run._load_run_skill", lambda args: skill)
    monkeypatch.setattr("stepbook.cli.run._build_run_environment", lambda *args: (env, ()))
    executor = ScriptedExecutor(lambda *args: Envelope("r", {}, "done"))
    monkeypatch.setattr("stepbook.cli.run._select_run_executor", lambda *args: (executor, "command"))
    run = tmp_path / "run"
    args = ["run", "offline", "--steps", "1", "--log-dir", str(run), "--quiet"]
    return args, run, env, executor


@pytest.mark.parametrize("outcome,code", [("done", 0), ("horizon", 3), ("abort", 3), ("pause", 3)])
def test_run_exit_distinguishes_outcomes(launch, capsys, outcome, code):
    args, run, env, executor = launch
    if outcome == "pause":
        run.mkdir()
        (run / "inbox.txt").write_text("pause\n")
    elif outcome == "abort":
        executor.fn = lambda *args: Envelope("r", {"goal": "changed"}, "done")
    elif outcome == "horizon":
        executor.fn = lambda *args: Envelope("r", {}, "work")
    assert main(args) == code
    summary = json.loads(capsys.readouterr().out)
    assert summary["aborted"] == (outcome in ("abort", "pause"))


@pytest.mark.parametrize("verb", ["status", "report"])
def test_missing_run_is_one_sentence_and_exit_two(tmp_path, monkeypatch, capsys, verb):
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing"
    assert main([verb, *(["--run-dir"] if verb == "status" else []), str(missing)]) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err == f"no run at {missing}\n"
    assert not missing.exists()


def test_run_help_explains_exit_codes(capsys):
    with pytest.raises(SystemExit) as result:
        main(["run", "--help"])
    assert result.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for fragment in ("0 goal met", "1 checks failed", "2 invalid", "3 incomplete", "horizon", "paused"):
        assert fragment in text


HIDDEN_RUN_OPTIONS = ("--evidence-root", "--adapter-timeout", "--quiet")


def test_run_help_groups_routine_options_and_help_all_lists_the_rest(capsys):
    with pytest.raises(SystemExit) as result:
        main(["run", "--help"])
    assert result.value.code == 0
    short = capsys.readouterr().out
    for group in ("Task:", "Where:", "Limits:", "Agent:", "Run:"):
        assert group in short
    for option in (
        "--plan",
        "--goal",
        "--check",
        "--constraint",
        "--workdir",
        "--adapter",
        "--allow-outside",
        "--steps",
        "--runs-max",
        "--cmd-timeout",
        "--no-checkpoint",
        "--allow-dirty",
    ):
        assert option in short
    for option in HIDDEN_RUN_OPTIONS:
        assert option not in short
    with pytest.raises(SystemExit) as result:
        main(["run", "--help-all"])
    assert result.value.code == 0
    full = capsys.readouterr().out
    for option in (*HIDDEN_RUN_OPTIONS, "--goal", "--steps", "--state-max-chars", "--require-completion"):
        assert option in full
    assert "--evidence-cwd" not in full


@pytest.mark.parametrize(
    "option,value", [("--evidence-cwd", "."), ("--env", "shell"), ("--response-contract", "report_object")]
)
def test_removed_run_options_are_refused(capsys, option, value):
    # Relative evidence paths resolve against the launch directory, shell is the environment without --adapter,
    # and the one response contract is the action command string; none of these options exists any more.
    with pytest.raises(SystemExit) as result:
        main(["run", "checklist", option, value])
    assert result.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv,option",
    [
        (["run", "checklist", "--unsandboxed"], "--unsandboxed"),
        (["run", "checklist", "--allow"], "--allow"),
        (["status", "--run"], "--run"),
        (["say", "hi", "--log"], "--log"),
    ],
)
def test_abbreviated_options_are_refused(capsys, argv, option):
    """An option is typed in full: no prefix stands in for --unsandboxed-i-accept-the-risk or any other."""
    with pytest.raises(SystemExit) as result:
        main(argv)
    assert result.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err


def test_a_launch_recorded_with_an_abbreviation_cannot_be_resumed(tmp_path, monkeypatch, capsys):
    """A recorded abbreviation is not a launch this CLI could have written, and a resume never replays one.

    Nothing can record it any more (the launch would have been refused at exit 2), and the expanded argv is
    re-parsed as an ordinary invocation, so an abbreviation in an existing record is refused too, instead of
    handing the resume an acknowledgement nobody typed in full.
    """
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "launch.json").write_text(
        json.dumps({"program": ["stepbook"], "argv": ["run", "checklist", "--unsandboxed"], "cwd": str(tmp_path)})
    )
    with pytest.raises(SystemExit) as result:
        main(["run", "--resume", str(run)])
    assert result.value.code == 2
    assert "unrecognized arguments: --unsandboxed" in capsys.readouterr().err


def test_hidden_run_options_are_still_accepted():
    from stepbook.cli import _build_parser

    args = _build_parser().parse_args(["run", "skill", "--evidence-root", ".", "--adapter-timeout", "5", "--quiet"])
    assert (args.evidence_root, args.adapter_timeout, args.quiet) == (["."], 5.0, True)


def test_pointerless_status_names_an_accepted_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["status"]) == 2
    assert capsys.readouterr().err == "no run found; start a run or supply --run-dir\n"
    assert main(["status", "--run-dir", "missing"]) == 2
    assert capsys.readouterr().err == "no run at missing\n"
    with pytest.raises(SystemExit) as caught:
        main(["status", "--help"])
    assert caught.value.code == 0
    assert "--run-dir" in capsys.readouterr().out


def test_resume_of_a_pre_structured_adapter_journal_is_refused_in_one_sentence(launch, capsys, monkeypatch):
    """An adapter run journaled before the structured facts is refused by name, and commits no step."""
    from stepbook.envs.eval_loop import AdapterEvalEnv

    args, run, env, executor = launch
    adapter_env = AdapterEvalEnv(adapter="true", log_dir=run, resuming=True)
    monkeypatch.setattr("stepbook.cli.run._build_run_environment", lambda *a: (adapter_env, ()))
    log = EventLog(run / "events.jsonl")
    log.append("run_start", measurement_repeats=1, require_completion=False)
    log.append("step", step=0, action="run baseline", result="Eval finished:\n{}", done=False)
    before = log.path.read_bytes()
    executor.fn = lambda *args: pytest.fail("the refusal must precede the controller")
    assert main([*args, "--resume"]) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and len(printed.err.splitlines()) == 1
    assert "recorded before the structured adapter journal" in printed.err and "Traceback" not in printed.err
    assert log.path.read_bytes().startswith(before)
    assert [e["kind"] for e in log.read()] == ["run_start", "step", "run_start"], "only the resume marker is added"


@pytest.mark.parametrize(
    "refusal,exception_name,problem,remedy",
    [
        ("existing", "ExistingRun", "already has events", "--resume"),
        ("pending", "UnreceiptedAction", "unresolved action", "Reconcile"),
        ("policy", "CompletionPolicyChanged", "completion policy changed", "original policy"),
        ("repeats", "AutomaticMeasurementRepeats", "automatic measurement repeats", "new run"),
    ],
)
def test_resume_refusal_is_named_and_cli_leaves_journal_untouched(
    launch, capsys, refusal, exception_name, problem, remedy
):
    args, run, env, executor = launch
    log = EventLog(run / "events.jsonl")
    log.append("run_start", measurement_repeats=2 if refusal == "repeats" else 1, require_completion=False)
    if refusal != "existing":
        args += ["--resume"]
    if refusal == "pending":
        log.append("action_started", action_id="a1", record={"step": 0, "action": "run baseline"})
    if refusal == "policy":
        env.require_completion = True
    before = log.path.read_bytes()
    executor.fn = lambda *args: pytest.fail("refusal must precede controller")
    env.reset = lambda: pytest.fail("refusal must precede reset")
    runtime = Runtime(Skill("offline", "work", {"type": "object"}, {}), executor, env, log_dir=run)
    with pytest.raises(getattr(errors_module, exception_name)) as caught:
        runtime.run(1, resume=refusal != "existing", replay_env=False)
    assert type(caught.value).__name__ == exception_name
    assert main(args) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and len(printed.err.splitlines()) == (2 if refusal == "pending" else 1)
    assert problem in printed.err and remedy in printed.err and "Traceback" not in printed.err
    if refusal != "repeats":
        assert printed.err == f"{caught.value}\n"
    assert log.path.read_bytes() == before


@pytest.mark.parametrize("verb", ["run", "validate"])
def test_missing_skill_is_one_sentence(tmp_path, monkeypatch, capsys, verb):
    monkeypatch.chdir(tmp_path)
    assert main([verb, "absent-skill"]) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and len(printed.err.splitlines()) == 1
    assert "absent-skill" in printed.err and "existing skill folder" in printed.err


@pytest.mark.parametrize("verb", ["validate", "run"])
@pytest.mark.parametrize("missing", ["schema.json", "SKILL.md"])
def test_missing_skill_file_is_one_sentence(tmp_path, monkeypatch, capsys, verb, missing):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "skill"
    folder.mkdir()
    if missing == "schema.json":
        (folder / "SKILL.md").write_text("# hi\n")
    else:
        (folder / "schema.json").write_text('{"type": "object"}')
    assert main([verb, str(folder)]) == 2
    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err == (
        f"skill folder {str(folder)!r} has no {missing}; "
        "a task skill folder holds SKILL.md, schema.json and initial_state.json\n"
    )
    assert "Traceback" not in printed.out + printed.err


def test_missing_executor_binary_is_one_sentence(launch, monkeypatch, capsys):
    args, run, env, executor = launch
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("stepbook.cli.run._select_run_executor", lambda *args: (ClaudeCLIExecutor(), "cli"))
    assert main(args) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and len(printed.err.splitlines()) == 1
    assert printed.err == "executor 'claude' was not found; install it and make it available on PATH\n"


@pytest.mark.parametrize("source", ["executor", "environment"])
def test_unexpected_run_failure_still_raises(launch, source):
    args, run, env, executor = launch
    args[args.index("--steps") + 1] = "2"
    executor.fn = lambda *args: Envelope("r", {}, "work")
    first_call = executor.fn if source == "executor" else env.step
    calls = 0
    failure = RuntimeError("unexpected failure")

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first_call(*args)
        assert sum(e["kind"] == "step" for e in EventLog(run / "events.jsonl").read()) == 1
        raise failure

    if source == "executor":
        executor.fn = fail
    else:
        env.step = fail
    with pytest.raises(RuntimeError, match="unexpected failure") as caught:
        main(args)
    assert caught.value is failure
    assert calls == 2
    assert "failed_to_start" not in json.loads((run / "launch.json").read_text())
    assert (run.parent / ".stepbook/LAST_RUN").exists()
    assert not (run.parent / ".stepbook/ACTIVE").exists()


def test_missing_executor_workdir_is_not_mislabeled(tmp_path, monkeypatch):
    executor = ClaudeCLIExecutor(cwd=str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError) as caught:
        executor._run([str(Path.cwd() / "also-absent")])
    assert type(caught.value) is FileNotFoundError


def test_a_default_run_refuses_api_key_billing_in_one_sentence(tmp_path, monkeypatch, capsys):
    """Subscription-only is the default: an API-key environment is refused before any subprocess runs."""
    from stepbook.executors import auth

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STEPBOOK_ALLOW_API_BILLING", raising=False)
    for name in auth.OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setattr(auth.shutil, "which", lambda _name: pytest.fail("refuse before resolving a binary"))
    monkeypatch.setattr(auth.subprocess, "run", lambda *a, **k: pytest.fail("refuse before any subprocess"))
    run = tmp_path / "run"
    assert main(["run", "checklist", "--workdir", str(tmp_path), "--goal", "offline", "--log-dir", str(run)]) == 2
    assert capsys.readouterr().err.splitlines() == [
        "Subscription-only call refused: API/provider override present (ANTHROPIC_API_KEY) would bill an API key; "
        "log in with your subscription, or pass --allow-api-billing. No model call or fallback; values are not logged."
    ]
    assert not run.exists()
