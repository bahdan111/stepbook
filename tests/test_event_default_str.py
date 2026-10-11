"""Non-JSON event payloads retain their documented string serialization."""

from pathlib import PurePosixPath
from unittest.mock import patch

from stepbook.storage.events import EventLog


def test_non_json_payload_is_stringified_in_exact_event_bytes(tmp_path):
    """Bug: EventLog drops default=str and rejects a non-JSON payload."""
    log = EventLog(tmp_path / "events.jsonl")
    with patch("time.time", return_value=1700000000.0):
        log.append("artifact", path=PurePosixPath("café/雪.txt"))
    assert log.path.read_bytes() == ('{"ts": 1700000000.0, "kind": "artifact", "path": "café/雪.txt"}\n'.encode())
    assert log.read() == [{"ts": 1700000000.0, "kind": "artifact", "path": "café/雪.txt"}]
