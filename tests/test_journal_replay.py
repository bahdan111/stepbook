"""Stored-journal replay lock; regenerate explicitly with REGENERATE_REPLAY_FIXTURE=1."""

import json
import os
import shutil
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import UUID

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill
from stepbook.envs.shell import ShellEnv
from stepbook.storage.events import EventLog

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "replay"
OPERATOR_NOTE = "Keep the inspection evidence when updating the checklist."


def _generate_fixture():
    """Run a fixed offline script with synthetic paths, timestamps and identifiers."""
    if os.environ.get("REGENERATE_REPLAY_FIXTURE") != "1":
        raise RuntimeError("Set REGENERATE_REPLAY_FIXTURE=1 to replace the replay fixture")
    with TemporaryDirectory(prefix="journal-replay-", dir=ROOT) as directory:
        workdir = Path(directory)
        log = workdir / "log"
        log.mkdir()
        (log / "inbox.txt").write_text(OPERATOR_NOTE + "\n", encoding="utf-8")
        script = iter(
            [
                Envelope(
                    "Start inspection",
                    {
                        "checklist": {"inspect": {"text": "Inspect the sample", "status": "todo"}},
                        "notes": {"temporary": "remove after inspection", "keep": "durable note"},
                    },
                    "echo alpha >> replay-marker.txt",
                ),
                Envelope(
                    "Record evidence",
                    {
                        "checklist": {"inspect": {"status": "doing", "evidence": "sample inspected"}},
                        "facts": {"sample": "alpha"},
                    },
                    "echo beta",
                ),
                Envelope("Verify the check", {"checks_status": {"ready": "unknown"}}, "check ready"),
                Envelope(
                    "Finish inspection",
                    {
                        "checklist": {"inspect": {"status": "done"}},
                        "checks_status": {"ready": "pass"},
                        "notes": {"temporary": None},
                    },
                    "echo gamma",
                ),
                Envelope(
                    "Retain a pending item",
                    {
                        "checklist": {"review": {"text": "Review the evidence", "status": "todo"}},
                        "current": "await operator review",
                    },
                    "echo delta",
                ),
                Envelope("Request a resumable pause", {"facts": {"review": "pending"}}, "echo pause > log/inbox.txt"),
            ]
        )
        env = ShellEnv(
            workdir=workdir, goal="Inspect a sample", checks={"ready": "true"}, require_sandbox=False, checkpoint=False
        )
        reset = env.reset
        identifiers = count(1)
        # Normalize the observation before rendering so prompt lengths and token estimates are stable too.
        with (
            patch.object(env, "reset", side_effect=lambda: reset().replace(str(workdir), "/tmp/example")),
            patch("time.time", return_value=946771200.0),  # 2000-01-02 00:00:00 UTC
            patch("uuid.uuid4", side_effect=lambda: UUID(int=next(identifiers))),
        ):
            result = Runtime(
                Skill.load(ROOT / "stepbook/templates" / "checklist"),
                ScriptedExecutor(lambda s, u: next(script)),
                env,
                log_dir=log,
            ).run(8)
        events = EventLog(log / "events.jsonl").read()
        assert result.steps == 6 and result.rejected_patches == 0 and result.aborted
        assert events[-1]["kind"] == "paused"
        assert not any(e.get("done") for e in events if e["kind"] == "step")
        FIXTURE.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log / "events.jsonl", FIXTURE / "events.jsonl")
        (FIXTURE / "expected_state.json").write_text(json.dumps(result.final_state, indent=2) + "\n", encoding="utf-8")


class RecordingShellEnv(ShellEnv):
    """Record every dispatched action, including built-ins absent from commands."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return super().step(action)


def _resume_fixture(tmp_path, monkeypatch):
    log = tmp_path / "log"
    log.mkdir()
    shutil.copyfile(FIXTURE / "events.jsonl", log / "events.jsonl")
    stored = EventLog(log / "events.jsonl").read()
    env = RecordingShellEnv(
        workdir=tmp_path,
        goal="Inspect a sample",
        checks={"ready": "true"},
        require_sandbox=False,
        checkpoint=False,
        resuming=True,
    )
    seen = []

    def finish(system_prompt, user_prompt):
        seen.append((user_prompt, env.t, dict(env.last_checks), list(env.commands)))
        return Envelope("Close the resumed run", {"current": "closed after resume"}, "done")

    # Runtime.__init__ and Runtime.run expose no injectable clock; retain the
    # scoped time patch until such a public seam exists.
    # Make the observation stale even immediately after regeneration, without sleeping
    # or changing the stored journal. Neither timestamp nor minutes-ago text is compared.
    with monkeypatch.context() as clock:
        clock.setattr("stepbook.runtime.loop.time.time", lambda: max(e["ts"] for e in stored) + 3600)
        result = Runtime(
            Skill.load(ROOT / "stepbook/templates" / "checklist"), ScriptedExecutor(finish), env, log_dir=log
        ).run(8, resume=True, replay_env=False)
    assert env.actions == ["done"], "resume must dispatch only the new done, never any journal action"
    assert len(seen) == 1, "a paused journal must dispatch exactly one new done step"
    return seen[0], result, EventLog(log / "events.jsonl").read()[len(stored) :]


def test_stored_journal_reconstructs_state_and_continues_steps(tmp_path, monkeypatch):
    (prompt, shell_steps, checks, commands), result, appended = _resume_fixture(tmp_path, monkeypatch)
    state_json = prompt.split("Skill Execution State:\n```json\n", 1)[1].split("\n```", 1)[0]
    expected = json.loads((FIXTURE / "expected_state.json").read_text(encoding="utf-8"))
    assert json.loads(state_json) == expected
    assert result.final_state == {**expected, "current": "closed after resume"}
    assert shell_steps == 6 and result.steps == 7
    assert checks == {"ready": "pass"}
    assert commands == [
        {"t": 0, "cmd": "echo alpha >> replay-marker.txt", "rc": 0},
        {"t": 1, "cmd": "echo beta", "rc": 0},
        {"t": 3, "cmd": "echo gamma", "rc": 0},
        {"t": 4, "cmd": "echo delta", "rc": 0},
        {"t": 5, "cmd": "echo pause > log/inbox.txt", "rc": 0},
    ]
    assert [e["from_step"] for e in appended if e["kind"] == "resumed"] == [6]
    assert [(e["step"], e["action"], e["done"]) for e in appended if e["kind"] == "step"] == [(6, "done", True)]
    assert not (tmp_path / "replay-marker.txt").exists(), "resume must not execute stored shell actions"
    assert "RESUMED after a pause" in prompt
    standing = prompt.split("Skill Execution State:", 1)[0]
    assert OPERATOR_NOTE in standing


def test_stored_journal_restores_recent_steps_in_order(tmp_path, monkeypatch):
    (prompt, _, _, _), _, _ = _resume_fixture(tmp_path, monkeypatch)
    digest = prompt.split("<<<RECENT\n", 1)[1].split("\nRECENT>>>", 1)[0]
    assert digest.splitlines() == [
        "  [1] echo beta -> $ echo beta",
        "  [2] check ready -> Checks:",
        "  [3] echo gamma -> $ echo gamma",
        "  [4] echo delta -> $ echo delta",
        "  [5] echo pause > log/inbox.txt -> $ echo pause > log/inbox.txt",
    ]
    assert "RESUMED after a pause" in prompt


if __name__ == "__main__":
    _generate_fixture()
