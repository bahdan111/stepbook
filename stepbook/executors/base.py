"""Shared executor envelopes, parsed steps and text helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


class _UnreportedCost(float):
    """Numeric zero whose origin survives dataclasses.replace without a new price."""


@dataclass
class Envelope:
    """One model response. ``state_patch`` may be None for transcript baselines."""

    reasoning: str
    state_patch: dict[str, Any] | None
    action: str
    input_tokens: int = 0  # all prompt tokens incl. cache reads/writes
    output_tokens: int = 0
    cost_usd: float | None = None
    duration_s: float = 0.0
    api_calls: int = 1  # internal model calls the executor made for this step (CLI num_turns)
    usage: dict[str, Any] = field(
        default_factory=dict
    )  # raw provider usage (input, cache_read, cache_creation, output)
    tool_calls: int = 0  # tool invocations attempted inside this step (SDK executors)
    tool_log: list[dict[str, Any]] = field(default_factory=list)  # every tool call: name + truncated input
    extra: dict[str, Any] = field(default_factory=dict)  # structured-output fields beyond reasoning/state_patch/action
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Keep numeric backing for existing runtime accumulators; normalized receipts
        # use cost_unknown to write null. Only an explicitly supplied zero is known free.
        # Recompute only our generated flag; provider/caller unknown flags stay authoritative.
        if self.raw.get("_default_cost_unknown"):
            self.raw = {k: v for k, v in self.raw.items() if k not in {"cost_unknown", "_default_cost_unknown"}}
        if self.cost_usd is None or isinstance(self.cost_usd, _UnreportedCost):
            self.cost_usd = _UnreportedCost(0.0)
            if not self.raw.get("usage_unknown") and not self.raw.get("cost_unknown"):
                self.raw = {**self.raw, "cost_unknown": True, "_default_cost_unknown": True}


class StepExecutor(Protocol):
    def step(self, system_prompt: str, user_prompt: str, output_schema: dict[str, Any]) -> Envelope: ...


def strip_nulls(text: str) -> str:
    """Replace NUL bytes so decoded text cannot trigger embedded-null errors in argv or provider rejection."""
    return text.replace("\x00", "\ufffd") if isinstance(text, str) and "\x00" in text else text


def _last_envelope_object(text: str) -> dict[str, Any] | None:
    """Return the last complete, outermost JSON envelope in ``text``.

    CLIs print progress lines first, so the first '{' is not necessarily the envelope; a state patch or a
    report object is nested and may itself contain "action", so the last decodable '{' is not the envelope
    either. Decode every '{', keep envelope-like objects, drop any nested inside another candidate, and
    return the last remaining one."""
    dec = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("action" in obj or "reasoning" in obj or "report" in obj):
            candidates.append((i, i + end, obj))
    outermost = [c for c in candidates if not any(o is not c and o[0] <= c[0] and c[1] <= o[1] for o in candidates)]
    return outermost[-1][2] if outermost else None


USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


@dataclass
class ParsedStep:
    """Provider-independent fold of one agent event stream; no USD price inferred."""

    text: str | None = None
    usage: dict[str, int] = field(default_factory=lambda: dict.fromkeys(USAGE_KEYS, 0))
    api_calls: int = 0
    tool_calls: int = 0
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    errors: list[Any] = field(default_factory=list)
    session_id: str | None = None
    agent_cost: float | None = None
