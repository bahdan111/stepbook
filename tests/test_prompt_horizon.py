"""The prompt tells the model which step it is on and how many the run has.

One line sits between the aging note and the state block; a caller that passes no step renders
exactly what it rendered before. The runtime passes the absolute journal index, so a resume
continues the count instead of restarting it.
"""

from copy import deepcopy
from pathlib import Path

from stepbook import Envelope, Runtime, ScriptedExecutor, Skill, StepResult
from stepbook.runtime.rendering import render_state_prompt

ROOT = Path(__file__).resolve().parent.parent
OBSERVATION = "Working directory: /work\nGoal: make a file"


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


def _horizon_lines(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if line.startswith("Step ")]


def _checklist() -> Skill:
    return Skill.load(ROOT / "stepbook/templates" / "checklist")


def test_horizon_line_precedes_the_state_block():
    text = render_state_prompt({"a": 1}, "obs", step=0, max_steps=400)
    lines = text.splitlines()
    assert lines[lines.index("Skill Execution State:") - 1] == "Step 1 of 400."
    assert _horizon_lines(text) == ["Step 1 of 400."]


def test_without_a_step_the_rendering_is_unchanged():
    text = render_state_prompt({"a": 1}, "obs")
    assert _horizon_lines(text) == []
    assert text.startswith("Skill Execution State:\n")
    assert text == render_state_prompt({"a": 1}, "obs", max_steps=400)  # a horizon alone says nothing


def test_an_unbounded_run_still_names_the_step():
    text = render_state_prompt({"a": 1}, "obs", step=4, max_steps=None)
    assert _horizon_lines(text) == ["Step 5; no step limit."]


def test_runtime_counts_the_steps_of_a_bounded_run(tmp_path):
    executor = RecordingExecutor(lambda system, user: Envelope("work", {}, "keep going"))
    result = Runtime(_checklist(), executor, FixedEnv(), log_dir=tmp_path / "log").run(2)
    assert result.steps == 2 and not result.aborted
    assert [_horizon_lines(user) for _, user, _ in executor.seen] == [["Step 1 of 2."], ["Step 2 of 2."]]


def test_a_resume_continues_the_count(tmp_path):
    log = tmp_path / "log"
    Runtime(
        _checklist(), RecordingExecutor(lambda s, u: Envelope("work", {}, "keep going")), FixedEnv(), log_dir=log
    ).run(2)
    executor = RecordingExecutor(lambda system, user: Envelope("finish", {}, "done"))
    result = Runtime(_checklist(), executor, FixedEnv(), log_dir=log).run(3, resume=True)
    assert result.steps == 3 and not result.aborted
    assert [_horizon_lines(user) for _, user, _ in executor.seen] == [["Step 3 of 3."]]


def test_an_unbounded_runtime_run_says_so(tmp_path):
    executor = RecordingExecutor(lambda system, user: Envelope("finish", {}, "done"))
    result = Runtime(_checklist(), executor, FixedEnv(), log_dir=tmp_path / "log").run(None)
    assert result.steps == 1 and not result.aborted
    assert [_horizon_lines(user) for _, user, _ in executor.seen] == [["Step 1; no step limit."]]
