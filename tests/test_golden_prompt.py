"""Golden-prompt byte lock.

The prompt bytes are fixed for a given skill, state, observation and recent steps.
Capture the executor boundary through Runtime and exercise protected references
directly through the renderer; compare both with stored UTF-8 bytes.

Regenerate explicitly, then review the files:
    .venv/bin/python tests/test_golden_prompt.py --regenerate
Pytest only reads the golden files; environment variables cannot enable writes.
"""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill, StepResult
from stepbook.runtime.loop import render_state_prompt

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
OBSERVATION = (
    "Working directory: /work\nGoal: make a file\n"
    "Checks that must pass (run with the 'check' action):\n  f: test -f out.txt\n"
    "Actions: a shell command line, 'check' or 'check <id>', 'done', or 'stop <reason>'."
)


class FixedEnv:
    """An environment whose observations carry no host-specific paths or timestamps."""

    def reset(self) -> str:
        self.done = False
        return OBSERVATION

    def step(self, action: str) -> StepResult:
        self.done = action.strip() == "done"
        return StepResult(f"$ {action}\n[exit 0] (no output)", done=self.done, info={"ok": True})

    def score(self) -> float:
        return 1.0 if self.done else 0.0


class RecordingExecutor:
    """Capture the complete call at the executor boundary, before delegation."""

    def __init__(self, fn):
        self.scripted = ScriptedExecutor(fn)
        self.seen = []

    def step(self, system, user, output_schema):
        self.seen.append((system, user, deepcopy(output_schema)))
        return self.scripted.step(system, user, output_schema)


def _compare(name: str, text: str) -> None:
    path = GOLDEN / name
    assert path.exists(), f"missing golden {path}; run the --regenerate command in this module's docstring"
    assert text.encode("utf-8") == path.read_bytes(), (
        f"rendered prompt differs from {path.name}; review before regenerating"
    )


def _step_zero_prompts(log_dir):
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    executor = RecordingExecutor(lambda system, user: Envelope("r", {}, "done"))
    runtime = Runtime(skill, executor, FixedEnv(), log_dir=log_dir)
    result = runtime.run(1)
    assert result.score == 1.0 and not result.aborted
    assert len(executor.seen) == 1
    system, user, schema = executor.seen[0]
    return {
        "prompt_checklist_step0.system.txt": system,
        "prompt_checklist_step0.user.txt": user,
        "prompt_checklist_step0.schema.json": json.dumps(schema, indent=2, sort_keys=True) + "\n",
        # Sorted keys above are readable but blind to insertion order; this is byte-for-byte what
        # ClaudeCLIExecutor._command puts after --json-schema (stepbook/executor.py:303).
        "prompt_checklist_step0.schema.argv.json": json.dumps(schema),
    }


def _capped_prompt():
    """The aged rendering (one-lined and omitted entries, the state pointer, standing operator notes) is part of the
    contract too: it is what every long run sees after the first hour."""
    state = {
        "goal": "improve the thing",
        "current": "run_003",
        "baseline": {"runs": ["run_000", "run_001"], "pass_mean": 1.5},
        "kept_stack": ["h_02"],
        "hypotheses": {
            f"h_{i:02d}": {"status": "reverted" if i % 3 else "kept", "evidence": "e" * 120} for i in range(12)
        },
        "runs": {f"run_{i:03d}": {"pass": 1 + i % 3, "recall": (i % 5) / 10, "note": "n" * 200} for i in range(41)},
        "facts": {f"fact_{i:03d}": f"{i}: " + "x" * (30 + i % 90) for i in range(150)},
        "lessons": [f"lesson {i}: " + "y" * 150 for i in range(20)],
    }
    # Synthetic observations use successive fractions of a toy evaluation set.
    recent = [
        (38, "./eval.sh run_038", "pass=1/10 recall=0.1"),
        (39, "./eval.sh run_039", "pass=2/10 recall=0.2"),
        (40, "git commit -am 'keep'", "[main abc1234] keep"),
    ]
    text = render_state_prompt(
        state,
        "$ ./eval.sh run_040\n[exit 0]\npass=3/10 recall=0.3",
        recent,
        state_max_chars=12000,
        state_path="/run/state.json",
        standing_notes=["operator: the example task is blocked; do not retry it"],
    )
    assert "shown as one line" in text
    assert "older entries omitted" in text
    assert "readable at /run/state.json" in text
    for field, key in (("runs", "run_003"), ("hypotheses", "h_02")):
        whole_entry = json.dumps({key: state[field][key]}, separators=(",", ":"), sort_keys=True)[1:-1]
        assert whole_entry in text
    return {"prompt_capped_with_notes.user.txt": text}


def _aged_runtime_prompt(log_dir):
    skill = Skill.load(ROOT / "stepbook/templates" / "checklist")
    facts = {f"fact_{i:03d}": f"Evidence {i}: " + "x" * 240 for i in range(100)}
    note = "operator: keep the verified evidence; do not repeat the blocked check"
    responses = iter(
        [
            Envelope("record evidence", {"facts": facts}, "record"),
            Envelope("acknowledge operator", {}, "acknowledge"),
            Envelope("finish", {}, "done"),
        ]
    )

    def scripted_step(system, user):
        response = next(responses)
        if response.action == "record":
            (log_dir / "inbox.txt").write_bytes((note + "\n").encode("utf-8"))
        return response

    executor = RecordingExecutor(scripted_step)
    runtime = Runtime(skill, executor, FixedEnv(), log_dir=log_dir, state_max_chars=6000)
    result = runtime.run(3)
    assert result.steps == 3 and result.score == 1.0 and not result.aborted
    assert len(executor.seen) == 3
    assert result.final_state["facts"] == facts
    assert len(json.dumps(result.final_state, separators=(",", ":"))) > 6000
    assert "abbreviated for size" not in executor.seen[0][1]
    text = executor.seen[-1][1].replace(str(log_dir), "<LOG_DIR>")
    assert "shown as one line" in text
    assert "older entries omitted" in text
    assert "readable at <LOG_DIR>/state.json" in text
    assert f"STANDING OPERATOR NOTES (delivered earlier in this run; still in force):\n  - {note}\n" in text
    assert note not in text.split("<<<OBSERVATION\n", 1)[1]
    assert "[0] record -> $ record" in text
    assert "[1] acknowledge -> $ acknowledge" in text
    for host_specific in ("/private/tmp", "/private/var/", "/Users/", str(log_dir), '"ts":', "timestamp"):
        assert host_specific not in text
    assert re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}:\d{2}:\d{2}\b|\b\d{10}(?:\.\d+)?\b", text) is None
    return {"prompt_checklist_aged.user.txt": text}


def test_step_zero_prompt_is_byte_identical(tmp_path):
    for name, text in _step_zero_prompts(tmp_path / "log").items():
        _compare(name, text)


def test_capped_rendering_with_standing_notes_is_byte_identical():
    for name, text in _capped_prompt().items():
        _compare(name, text)


def test_aged_runtime_prompt_is_byte_identical(tmp_path):
    for name, text in _aged_runtime_prompt(tmp_path / "log").items():
        _compare(name, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", required=True)
    parser.parse_args()
    with TemporaryDirectory(prefix="golden-prompt-") as temporary:
        directory = Path(temporary)
        prompts = {
            **_step_zero_prompts(directory / "step0"),
            **_capped_prompt(),
            **_aged_runtime_prompt(directory / "aged"),
        }
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name, text in prompts.items():
        path = GOLDEN / name
        size = path.write_bytes(text.encode("utf-8"))
        print(f"Wrote {path.relative_to(ROOT)} ({size} bytes)")
    print(f"Regenerated {len(prompts)} golden files.")
