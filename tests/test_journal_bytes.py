"""Journal byte lock. Regenerate deliberately, then review the resulting fixture diff:

    .venv/bin/python tests/test_journal_bytes.py --regenerate

Only timestamps and the temporary directory are normalized, directly on bytes:
JSON tokens identify numeric values of keys named ts, and ISO string values of
keys ending in _at or named timestamp/ts. Other numbers and date strings survive.
Only eligible value spans become placeholders; no object re-serialization, key
sorting, whitespace cleanup, or newline insertion occurs during comparison.
The goldens normalize the path and therefore require an ASCII, space-free
temporary root (otherwise JSON escaping and shell quoting change the bytes).

UUIDs and the clock are deterministic inputs, not normalized output. The fixed
clock preserves numeric started_at/resumed_at and makes summary wall_s stable.
The scripted environment has path-independent observations so prompt lengths and
estimated token counts remain real, stable values in the golden journal.
"""

import argparse
import importlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import pytest

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill, StepResult
from stepbook.run_pointer import write_launch_record
from stepbook.storage import recovery
from stepbook.storage.events import EventLog

GOLDEN = Path(__file__).parent / "fixtures" / "journal_bytes"
JSON_TOKEN = re.compile(rb'"(?:[^"\\]|\\.)*"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|[^\s]')
ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
OBSERVATION = "Inspection: café, 雪, ✓"
FIXED_TIME = 1_700_000_000.0
GOLDEN_NAMES = ("events.jsonl", "state.json", "launch.json", "launch-resumed.json", "summary.json")


def _normalize(raw, directory):
    raw = raw.replace(str(directory.resolve()).encode("utf-8"), b"<TMP>")
    tokens = list(JSON_TOKEN.finditer(raw))
    # Match a complete JSON key token, colon, and value token. Escaped quotes
    # inside strings cannot masquerade as keys; preserve all bytes between spans.
    for index in range(len(tokens) - 1, 1, -1):
        token = tokens[index]
        if tokens[index - 1][0] != b":" or not tokens[index - 2][0].startswith(b'"'):
            continue
        key = json.loads(tokens[index - 2][0])
        value = token[0]
        if key == "ts" and value[:1] in b"-0123456789":
            replacement = b"0"
        elif (
            (key.endswith("_at") or key in {"timestamp", "ts"})
            and value.startswith(b'"')
            and ISO_TIMESTAMP.fullmatch(json.loads(value))
        ):
            replacement = b'"<ISO_TIMESTAMP>"'
        else:
            continue
        raw = raw[: token.start()] + replacement + raw[token.end() :]
    return raw


def _run_directory(root):
    resolved = str(root.resolve())
    if not resolved.isascii() or any(char.isspace() for char in resolved):
        pytest.skip(f"journal byte goldens require an ASCII, space-free temporary root; got {resolved!r}")
    directory = root / "journal-run"
    directory.mkdir()
    return directory


@pytest.fixture
def run_directory(tmp_path):
    return _run_directory(tmp_path)


class ScriptedEnv:
    def __init__(self):
        self.actions = []

    def reset(self):
        return "Inspect the sample."

    def step(self, action):
        self.actions.append(action)
        return StepResult(OBSERVATION, done=action == "done", info={"ok": True})

    def score(self):
        return 1.0


def _skill():
    return Skill(
        "journal-bytes",
        "Inspect once, record evidence, then finish.",
        {
            "type": "object",
            "properties": {
                "phase": {"type": "string"},
                "evidence": {"type": "object"},
                "items": {"type": "array"},
                "optional": {"type": "null"},
                "label": {"type": "string"},
            },
            "required": ["phase", "evidence", "items", "optional", "label"],
            "additionalProperties": False,
        },
        _state("new"),
    )


def _state(phase):
    # Deliberately nonalphabetical at both levels; state.json also locks escaping.
    return {
        "phase": phase,
        "evidence": {"zeta": {"verified": True, "count": 17}, "alpha": "00123"},
        "items": ["first", {"rank": 2}, None],
        "optional": None,
        "label": "naïve café — 日本語",
    }


def _scripted_run(directory, max_steps=2):
    directory = directory.resolve()
    log_dir = directory / "log"
    argv = [
        "run",
        "skills/runbook",
        "--env",
        "shell",
        "--workdir",
        str(directory),
        "--goal",
        "Inspect sample",
        "--no-checkpoint",
        "--log-dir",
        str(log_dir),
    ]
    responses = iter(
        [
            # These free fixture responses must report zero explicitly to keep
            # the known-zero byte contract; unpriced scripts now have incomplete cost.
            Envelope("Inspect sample", {"phase": "inspected"}, "inspect", cost_usd=0.0),
            Envelope("Close sample", {"phase": "closed"}, "done", cost_usd=0.0),
        ]
    )
    env = ScriptedEnv()
    # Each step consumes one inbox filename UUID and one action UUID.
    expected_ids = [UUID(int=i + 1) for i in range(2 * max_steps)]
    consumed_ids = []

    def next_id():
        assert len(consumed_ids) < len(expected_ids), "runtime consumed an unexpected action UUID"
        action_id = expected_ids[len(consumed_ids)]
        consumed_ids.append(action_id)
        return action_id

    # uuid.uuid4 is the only UUID seam the runtime offers (no ID factory input).
    # Reload under the source patch so even `from uuid import uuid4` binds it;
    # reload after restoration too, so no patched alias leaks into other tests.
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("uuid.uuid4", next_id)
            monkeypatch.setattr("time.time", lambda: FIXED_TIME)
            importlib.reload(recovery)
            write_launch_record(log_dir, argv, program="stepbook", cwd=directory)
            runtime = Runtime(_skill(), ScriptedExecutor(lambda s, u: next(responses)), env, log_dir=log_dir)
            result = runtime.run(max_steps)
    finally:
        importlib.reload(recovery)
    assert consumed_ids == expected_ids, "runtime must consume exactly the expected inbox and action UUIDs"
    assert result.steps == max_steps and result.rejected_patches == 0 and not result.aborted
    assert env.actions == ["inspect", "done"][:max_steps]
    return log_dir, argv, result


def _capture(directory):
    log_dir, argv, result = _scripted_run(directory)
    captured = {name: (log_dir / name).read_bytes() for name in ("events.jsonl", "state.json", "launch.json")}
    # Pin the public RunResult summary serialization.
    captured["summary.json"] = json.dumps(result.summary(), indent=2, sort_keys=False).encode("utf-8")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("time.time", lambda: FIXED_TIME)
        write_launch_record(log_dir, argv + ["--resume"], resume=True, program="stepbook", cwd=directory)
    captured["launch-resumed.json"] = (log_dir / "launch.json").read_bytes()
    return {name: _normalize(raw, directory) for name, raw in captured.items()}


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_journal_bytes_match_golden(run_directory, name):
    actual = _capture(run_directory)[name]
    assert actual == (GOLDEN / name).read_bytes(), (
        f"{name}: persisted journal bytes changed; review before regenerating"
    )


def test_event_lines_spacing_newlines_and_unescaped_unicode(run_directory):
    log_dir, _, result = _scripted_run(run_directory)
    raw = (log_dir / "events.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    assert lines and all(line.endswith(b"\n") for line in lines), "every event must end with a newline"
    events = [json.loads(line) for line in lines]  # Exactly one complete JSON object per physical line.
    assert all(isinstance(event, dict) for event in events)
    assert raw.count(b'"kind": "step"') == result.steps == 2, "shell consumers grep this exact spacing"
    assert sum(event["kind"] == "step" for event in events) == 2
    assert OBSERVATION.encode("utf-8") in raw, "EventLog must store non-ASCII observations unescaped"
    assert b"\\u00e9" not in raw and b"\\u96ea" not in raw and b"\\u2713" not in raw
    assert [event["kind"] for event in events][:2] == ["run_start", "model_call"]
    assert all(list(event)[:2] == ["ts", "kind"] for event in events), "ts and kind lead every record, in that order"
    # default=str, not a crash or a dropped record, for a value json cannot serialize.
    EventLog(log_dir / "events.jsonl").append("step", step=99, action=Path("/tmp/unserializable"))
    tail = (log_dir / "events.jsonl").read_bytes().splitlines(keepends=True)[-1]
    assert tail.endswith(b"\n") and json.loads(tail)["action"] == "/tmp/unserializable"
    assert b'"kind": "step"' in tail, "the added record keeps the spacing shell consumers grep for"


def test_normalization_preserves_non_timestamp_bytes(run_directory):
    raw = (
        b'{"ts": 12.25, "started_at": 3e2, "resumed_at": 4, "cost_usd": 12.25, '
        b'"date": "2000-01-02T09:34:56.123+00:00", "utc": "2000-01-02T09:34:56Z", '
        b'"created_at": "2000-01-02T09:34:56.123+00:00", "timestamp": "2000-01-02T09:34:56Z", '
        b'"nested": {"ts": -3e+2, "ts_string": "12.25", "digits": "00123"}, '
        b'"strings": {"ts": "12.25", "timestamp": "00123", "other_at": "00123"}, '
        b'"values": [12.25, "2000-01-02T09:34:56Z"], '
        b'"ts": "2000-01-02T09:34:56", "text": "escaped \\"ts\\": 12.25", '
        b'"path": "' + str(run_directory.resolve()).encode() + b'/log", "kind": "step"}\n'
    )
    assert _normalize(raw, run_directory) == (
        b'{"ts": 0, "started_at": 3e2, "resumed_at": 4, "cost_usd": 12.25, '
        b'"date": "2000-01-02T09:34:56.123+00:00", "utc": "2000-01-02T09:34:56Z", '
        b'"created_at": "<ISO_TIMESTAMP>", "timestamp": "<ISO_TIMESTAMP>", '
        b'"nested": {"ts": 0, "ts_string": "12.25", "digits": "00123"}, '
        b'"strings": {"ts": "12.25", "timestamp": "00123", "other_at": "00123"}, '
        b'"values": [12.25, "2000-01-02T09:34:56Z"], '
        b'"ts": "<ISO_TIMESTAMP>", "text": "escaped \\"ts\\": 12.25", '
        b'"path": "<TMP>/log", "kind": "step"}\n'
    )
    once = _normalize(raw, run_directory)
    assert _normalize(once, run_directory) == once, "normalization is idempotent; placeholders are not renormalized"
    for mutation, changed in (
        (b'"cost_usd": 12.25', b'"cost_usd": 12.26'),
        (b'"digits": "00123"', b'"digits": "00124"'),
        (b'"kind": "step"', b'"kind":"step"'),
        (b'"date": "2000-01-02T09:34:56.123+00:00"', b'"date": "2000-01-03T09:34:56.123+00:00"'),
    ):
        assert mutation in raw
        assert _normalize(raw.replace(mutation, changed), run_directory) != once, (
            f"normalization hid a real change: {mutation!r} -> {changed!r}"
        )


def test_reserialization_is_not_byte_equal_to_golden(run_directory):
    golden = (GOLDEN / "state.json").read_bytes()
    state = json.loads(golden)
    assert list(state) != sorted(state), "fixture must exercise nonalphabetical key order"
    altered = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert json.loads(altered) == state
    assert _normalize(altered, run_directory) != golden, "comparison must detect key order and whitespace changes"
    # Each writer convention state.json depends on, changed one at a time; parsed equality hides all of them.
    for description, variant in (
        ("key order", json.dumps(state, indent=2, sort_keys=True)),
        ("indentation", json.dumps(state, indent=4)),
        ("separators", json.dumps(state, indent=2, separators=(" ,", ": "))),
        ("unescaped unicode", json.dumps(state, indent=2, ensure_ascii=False)),
        ("trailing newline", json.dumps(state, indent=2) + "\n"),
    ):
        encoded = variant.encode("utf-8")
        assert json.loads(encoded) == state, description
        assert _normalize(encoded, run_directory) != golden, f"comparison must detect a change of {description}"
    assert _normalize(json.dumps(state, indent=2).encode("utf-8"), run_directory) == golden, (
        "state.json is indent=2, ensure_ascii=True, insertion order, no trailing newline"
    )


def test_unknown_extra_journal_field_still_resumes(run_directory):
    log_dir, _, _ = _scripted_run(run_directory, max_steps=1)
    path = log_dir / "events.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if b'"kind": "step"' in line)
    record = json.loads(lines[index])
    record["legacy_extra_field"] = {"version": 17, "note": "unknown to this runtime"}
    lines[index] = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(b"".join(lines))
    prefix = path.read_bytes()
    seen = []

    def finish(system, user):
        seen.append(user)
        return Envelope("Resume legacy journal", {"phase": "closed"}, "done")

    env = ScriptedEnv()
    result = Runtime(_skill(), ScriptedExecutor(finish), env, log_dir=log_dir).run(2, resume=True, replay_env=False)
    assert len(seen) == 1
    state = seen[0].split("Skill Execution State:\n```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(state) == _state("inspected")
    assert env.actions == ["done"]
    assert result.steps == 2 and result.final_state == _state("closed") and not result.aborted
    assert path.read_bytes().startswith(prefix), "resume must preserve the unknown legacy field in place"
    steps = [event for event in EventLog(path).read() if event["kind"] == "step"]
    assert [(event["step"], event["action"]) for event in steps] == [(0, "inspect"), (1, "done")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", required=True)
    parser.parse_args()
    with TemporaryDirectory(prefix="journal-bytes-") as temporary:
        captured = _capture(_run_directory(Path(temporary)))
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name, raw in captured.items():
        path = GOLDEN / name
        print(f"Wrote {path} ({path.write_bytes(raw)} bytes)")
    print(f"Regenerated {len(captured)} golden files.")
