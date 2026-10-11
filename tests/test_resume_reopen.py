"""A completed journal stays completed on plain resume; an explicit operator message reopens it in place."""

from pathlib import Path

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv
from stepbook.storage.events import EventLog

ROOT = Path(__file__).resolve().parent.parent


def _finished_run(tmp_path):
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    log = tmp_path / "log"
    steps = iter(
        [Envelope("a", {"current": "one"}, "echo run >> marks.txt"), Envelope("b", {"current": "closing"}, "done")]
    )
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    res = Runtime(skill, ScriptedExecutor(lambda s, u: next(steps)), env, log_dir=log).run(10)
    assert res.steps == 2 and (tmp_path / "marks.txt").read_text().count("run") == 1
    return skill, log


def test_plain_resume_of_a_completed_run_does_nothing(tmp_path):
    skill, log = _finished_run(tmp_path)
    called = []
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    Runtime(skill, ScriptedExecutor(lambda s, u: called.append(u) or Envelope("x", {}, "done")), env, log_dir=log).run(
        10, resume=True, replay_env=False
    )
    assert called == [] and (tmp_path / "marks.txt").read_text().count("run") == 1


def test_reopen_message_continues_the_same_journal_and_is_shown_first(tmp_path):
    skill, log = _finished_run(tmp_path)
    seen = []
    steps = iter([Envelope("c", {"current": "continuing"}, "echo run >> marks.txt"), Envelope("d", {}, "done")])
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    res = Runtime(skill, ScriptedExecutor(lambda s, u: seen.append(u) or next(steps)), env, log_dir=log).run(
        10, resume=True, replay_env=False, reopen="the pause is over, continue with the remaining work"
    )
    assert seen and "OPERATOR MESSAGE (run reopened" in seen[0] and "continue with the remaining work" in seen[0]
    assert (tmp_path / "marks.txt").read_text().count("run") == 2, "exactly one new command ran; nothing was replayed"
    events = EventLog(log / "events.jsonl").read()
    kinds = [e.get("kind") for e in events]
    assert "reopened" in kinds and kinds.index("reopened") > kinds.index("resumed")
    reopened = next(e for e in events if e.get("kind") == "reopened")
    assert reopened["from_step"] == 2 and "authority" in reopened
    assert res.final_state["current"] == "continuing" and sum(1 for e in events if e.get("kind") == "step") == 4


def test_plain_resume_after_reopen_and_pause_continues(tmp_path):
    """reopen -> continue -> (process ends without a new done) -> plain resume must NOT treat the old done as completion."""
    skill, log = _finished_run(tmp_path)
    steps = iter([Envelope("c", {"current": "continuing"}, "echo run >> marks.txt")])
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    Runtime(skill, ScriptedExecutor(lambda s, u: next(steps)), env, log_dir=log).run(
        3, resume=True, replay_env=False, reopen="continue"
    )
    assert (tmp_path / "marks.txt").read_text().count("run") == 2
    # horizon stopped the reopened run after one step; now a plain resume (no reopen message) must keep going
    steps2 = iter([Envelope("d", {"current": "still going"}, "echo run >> marks.txt"), Envelope("e", {}, "done")])
    seen = []
    env2 = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    Runtime(skill, ScriptedExecutor(lambda s, u: seen.append(u) or next(steps2)), env2, log_dir=log).run(
        10, resume=True, replay_env=False
    )
    assert seen, "plain resume after a reopen ran nothing"
    assert (tmp_path / "marks.txt").read_text().count("run") == 3
    # and now the journal ends with a fresh done after the reopen: a further plain resume does nothing
    called = []
    env3 = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    Runtime(skill, ScriptedExecutor(lambda s, u: called.append(u) or Envelope("x", {}, "done")), env3, log_dir=log).run(
        10, resume=True, replay_env=False
    )
    assert called == []


def test_resumed_observation_is_stamped_when_old(tmp_path):
    """A resume long after the last step tells the model the observation is old."""
    import json as _json
    import time

    from stepbook.runtime.loop import stale_observation_note

    steps = [{"step": 7, "ts": time.time() - 2 * 3600, "action": "x"}]
    note = stale_observation_note(steps)
    assert "RESUMED after a pause" in note and "step 7" in note and "minutes ago" in note
    assert stale_observation_note([{"step": 1, "ts": time.time() - 60}]) == "", "recent observations get no note"
    assert stale_observation_note([]) == ""
    # end to end: a run paused two hours ago (journal timestamps aged) resumes with the note first in the prompt
    skill, log = _finished_run(tmp_path)
    path = log / "events.jsonl"
    aged = []
    for line in path.read_text().splitlines():
        e = _json.loads(line)
        e["ts"] = e.get("ts", time.time()) - 3 * 3600
        aged.append(_json.dumps(e))
    path.write_text("\n".join(aged) + "\n")
    seen = []
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    Runtime(skill, ScriptedExecutor(lambda s, u: seen.append(u) or Envelope("c", {}, "done")), env, log_dir=log).run(
        10, resume=True, replay_env=False, reopen="continue"
    )
    assert (
        seen
        and "RESUMED after a pause" in seen[0]
        and seen[0].index("OPERATOR MESSAGE") < seen[0].index("RESUMED after a pause")
    )


def test_horizon_stop_is_not_completion_and_resumes_with_a_higher_guard(tmp_path):
    """A horizon stop remains resumable when the limit increases."""
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    log = tmp_path / "log"
    steps = iter(
        [
            Envelope("a", {"current": "one"}, "echo run >> marks.txt"),
            Envelope("b", {"current": "two"}, "echo run >> marks.txt"),
        ]
    )
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False, horizon=2)
    Runtime(skill, ScriptedExecutor(lambda s, u: next(steps)), env, log_dir=log).run(2)
    assert (tmp_path / "marks.txt").read_text().count("run") == 2
    more = iter([Envelope("c", {"current": "three"}, "echo run >> marks.txt"), Envelope("d", {}, "done")])
    seen = []
    env2 = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False, horizon=10)
    Runtime(skill, ScriptedExecutor(lambda s, u: seen.append(u) or next(more)), env2, log_dir=log).run(
        10, resume=True, replay_env=False
    )
    assert seen, "resume after a horizon stop ran nothing"
    assert (tmp_path / "marks.txt").read_text().count("run") == 3


def test_reopen_message_persists_in_user_requirements(tmp_path):
    """The reopen message is recorded in the protected user_requirements field like an inbox line,
    so it survives every later render."""
    import json

    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    log = tmp_path / "log"
    env = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False)
    first = iter([Envelope("a", {"current": "one"}, "echo one"), Envelope("b", {}, "done")])
    Runtime(skill, ScriptedExecutor(lambda s, u: next(first)), env, log_dir=log).run(5)
    seen = []
    again = iter(
        [Envelope("c", {"current": "two"}, "echo two"), Envelope("d", {}, "echo three"), Envelope("e", {}, "done")]
    )
    env2 = ShellEnv(require_sandbox=False, workdir=tmp_path, goal="g", checks={}, checkpoint=False, resuming=True)
    Runtime(skill, ScriptedExecutor(lambda s, u: seen.append(u) or next(again)), env2, log_dir=log).run(
        10, resume=True, replay_env=False, reopen="hypothesis h2 is still at todo; decide it, then finish"
    )
    state = json.loads((log / "state.json").read_text())
    assert any(r["text"].startswith("hypothesis h2") for r in state["user_requirements"])
    assert all("hypothesis h2 is still at todo" in u for u in seen), (
        "the message must be in every later prompt, not only the first"
    )
