import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from stepbook.envs.eval_loop import AdapterEvalEnv
from stepbook.runtime.controls import read_inbox
from stepbook.storage.events import EventLog


@pytest.mark.parametrize("pause", [False, True])
def test_adapter_step_delivers_non_utf8_inbox_line(tmp_path, monkeypatch, pause):
    inbox = tmp_path / "inbox.txt"
    process = Mock(returncode=0)

    def communicate(timeout):
        if not inbox.exists():
            inbox.write_bytes(b"keep \xff small\n" + (b"pause\n" if pause else b""))
            raise subprocess.TimeoutExpired("fake-adapter", timeout)
        return "Applied h1", ""

    process.communicate.side_effect = communicate
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    env = AdapterEvalEnv(adapter="fake-adapter", log_dir=tmp_path, inbox_poll_s=0.01)

    result = env.step("apply h1")

    assert result.observation == "Applied h1"
    assert env.pause_requested is pause
    log = EventLog(tmp_path / "events.jsonl")
    assert read_inbox(log, 1) == ["keep \ufffd small"]
    assert log.read()[-1]["messages"] == ["keep \ufffd small"]


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_adapter_step_tolerates_inbox_read_error(tmp_path, monkeypatch, error):
    (tmp_path / "inbox.txt").write_text("note\n")
    process = Mock(returncode=0)
    process.communicate.side_effect = [subprocess.TimeoutExpired("fake-adapter", 0.01), ("Applied h1", "")]
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(Path, "read_text", Mock(side_effect=error("inbox unavailable")))
    env = AdapterEvalEnv(adapter="fake-adapter", log_dir=tmp_path, inbox_poll_s=0.01)

    assert env.step("apply h1").observation == "Applied h1"
