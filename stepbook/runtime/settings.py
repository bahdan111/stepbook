"""Constructor settings for the runtimes, the controller retry and the state prompt.

One model per wide signature: the defaults live in one place and a caller still names each knob.
Each model has a ``TypedDict`` twin so ``**overrides`` keeps the keyword names and types that the
callers used to get from the explicit parameters (PEP 692), and the module is a leaf so ``runtime``,
``retry`` and ``rendering`` can all reach it without importing one another.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self, TypedDict

from pydantic import BaseModel, ConfigDict, SkipValidation

from ..storage.events import EventLog
from .observations import ObservationShaper


class _Settings(BaseModel):
    """Base for the settings models: unknown keywords raise, and every field is stored as given.

    The signatures these replace never enforced their annotations, so validating now would change
    behaviour: coercion would turn an ``int`` budget into a ``float`` that the journal then records
    differently, and a duck-typed shaper or event log would be rejected. ``SkipValidation`` keeps the
    annotation as documentation without letting pydantic touch the value.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def resolve(cls, settings: Self | None, overrides: Mapping[str, Any]) -> Self:
        """The caller's settings object, its keyword spelling, or the object with keywords on top."""
        if settings is None:
            return cls(**overrides)
        if not isinstance(settings, cls):
            raise TypeError(f"settings must be a {cls.__name__}, got {type(settings).__name__}")
        if not overrides:
            return settings.model_copy()  # the caller's object stays theirs; the runtime reads a copy
        given = {name: getattr(settings, name) for name in settings.model_fields_set}
        return type(settings)(**{**given, **overrides})


class RuntimeSettings(_Settings):
    """The settings ``Runtime.__init__`` takes after ``skill``, ``executor`` and ``env``."""

    log_dir: SkipValidation[Path | str | None] = None
    max_retries: SkipValidation[int] = 3
    verbose: SkipValidation[bool] = False
    shaper: SkipValidation[ObservationShaper | None] = None
    recent_steps: SkipValidation[int] = 5
    state_max_chars: SkipValidation[int | None] = None  # None renders the whole state; see abbreviate_state


class RuntimeKwargs(TypedDict, total=False):
    """``RuntimeSettings`` spelled as keywords, for ``**overrides``."""

    log_dir: Path | str | None
    max_retries: int
    verbose: bool
    shaper: ObservationShaper | None
    recent_steps: int
    state_max_chars: int | None


class RetryPolicy(_Settings):
    """What ``call_with_retry`` needs besides the prompts: the transient budget, the boundary guard and the
    journal. A rate limit has no schedule to configure: it is recorded and re-raised."""

    attempts: SkipValidation[int] = 3
    log: SkipValidation[EventLog | None] = None
    step: SkipValidation[int] = -1
    # ``before_call()`` guards the boundary; callers bind values as lambda defaults.
    before_call: SkipValidation[Callable[..., object] | None] = None


class RetryKwargs(TypedDict, total=False):
    """``RetryPolicy`` spelled as keywords, for ``**overrides``."""

    attempts: int
    log: EventLog | None
    step: int
    before_call: Callable[..., object] | None


class PromptSettings(_Settings):
    """What ``render_state_prompt`` needs besides the state, the observation and the recent steps."""

    state_max_chars: SkipValidation[int | None] = None
    state_path: SkipValidation[str | None] = None
    standing_notes: SkipValidation[list[str] | None] = None
    step: SkipValidation[int | None] = None  # 0-based; None renders no horizon line
    max_steps: SkipValidation[int | None] = None


class PromptKwargs(TypedDict, total=False):
    """``PromptSettings`` spelled as keywords, for ``**overrides``."""

    state_max_chars: int | None
    state_path: str | None
    standing_notes: list[str] | None
    step: int | None
    max_steps: int | None
