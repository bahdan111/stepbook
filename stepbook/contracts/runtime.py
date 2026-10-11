"""What a runtime and an environment exchange: one step's result, the environment protocol and the run result."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class StepResult:
    observation: str
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    def reset(self) -> str: ...
    def step(self, action: str) -> StepResult: ...
    def score(self) -> float: ...


@dataclass
class RunResult:
    runtime: str
    steps: int
    llm_calls: int
    rejected_patches: int
    score: float
    prompt_tokens: list[int]
    output_tokens: list[int]
    cost_usd: float
    wall_s: float
    final_state: dict[str, Any] | None = None
    aborted: bool = False
    assisted_steps: int = 0  # steps whose executed action came from a retry after a rejected patch
    first_attempt_score: float | None = None  # score if every assisted step is counted as a failure
    rendered_prompt_chars: list[int] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    cost_complete: bool = False  # Only journal accounting can establish completeness.

    @property
    def total_tokens(self) -> int:
        return sum(self.prompt_tokens) + sum(self.output_tokens)

    @property
    def avg_prompt_tokens(self) -> float:
        return statistics.fmean(self.prompt_tokens) if self.prompt_tokens else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "steps": self.steps,
            "llm_calls": self.llm_calls,
            "rejected_patches": self.rejected_patches,
            "score": round(self.score, 4),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 1),
            "max_prompt_tokens": max(self.prompt_tokens) if self.prompt_tokens else 0,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "cost_complete": self.cost_complete,
            "wall_s": round(self.wall_s, 1),
            "aborted": self.aborted,
            "assisted_steps": self.assisted_steps,
            "first_attempt_score": self.first_attempt_score,
            "rendered_prompt_tokens_est": [round(c / 3.6) for c in self.rendered_prompt_chars],
            "rendered_prompt_tokens_est_total": round(sum(self.rendered_prompt_chars) / 3.6),
            "model_ids": self.model_ids,
        }
