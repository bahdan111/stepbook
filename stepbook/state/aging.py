"""Bounded state aging: abbreviate the oldest entries of oversized fields, and flag a stale observation."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, SkipValidation


def _one_line(value: Any, width: int = 96) -> str:
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), sort_keys=True)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


PROTECTED_REFERENCE_FIELDS = ("baseline", "kept_stack", "current")


def _referenced_keys(state: dict[str, Any]) -> set[str]:
    """Keep referenced entries whole so aging cannot erase evidence needed for later comparisons.

    A reference is an exact string value in one of the reference fields -- the id as the state itself wrote it,
    collected from nested lists and dict values -- so an id of any length is protected, ``h1`` included, and a
    word that merely appears inside a sentence there is not.
    """
    referenced: set[str] = set()
    pending = [state[field] for field in PROTECTED_REFERENCE_FIELDS if field in state]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            referenced.add(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return referenced


def _size(v: Any) -> int:
    """Rendered length of a value, in the compact form the prompt uses."""
    return len(json.dumps(v, separators=(",", ":"), sort_keys=True))


def _entry_size(container: Any, k0: Any, v0: Any) -> int:
    """Rendered length of one entry: key and value in a dict field, the value alone in a list."""
    return _size({k0: v0}) if isinstance(container, dict) else _size(v0)


class _Aging(BaseModel):
    """One render's shrinking copy and its incremental sizes, shared by the abbreviate and drop phases.

    ``SkipValidation`` for the same reason as the settings models: the caller's values reach the phases exactly
    as the render computed them, uncoerced and uncopied. It is load-bearing for the two numbers a caller supplies,
    and for a different value in each. Nothing checks ``keep_recent``, the count of entries kept whole, before the
    carrier, so a plain ``int`` field would coerce a string count that the phase raises ``TypeError`` on today and
    reject a fractional one that renders. A string ``max_chars`` never reaches the field, failing two lines above
    the constructor on the ``total <= max_chars`` comparison, but a fractional character limit does reach it
    through the settings model's ``SkipValidation[int | None]`` and renders, where a plain ``int`` field would
    reject it. The other five fields hold only what the function built for them a few lines above: validating
    ``rendered`` would copy the outer dict both phases read from the carrier and ``abbreviate_state`` returns, and
    ``order``'s lists, which neither phase mutates.
    """

    rendered: SkipValidation[dict[str, Any]]
    total: SkipValidation[int]
    protected: SkipValidation[set[str]]
    field_size: SkipValidation[dict[str, int]]
    order: SkipValidation[dict[str, list[Any]]]
    max_chars: SkipValidation[int]
    keep_recent: SkipValidation[int]


def abbreviate_state(state: dict[str, Any], max_chars: int, keep_recent: int = 6) -> tuple[dict[str, Any], list[str]]:
    # Shrink copied fields best-effort to preserve live state; protected and recent entries may exceed the cap.
    # Size incrementally to avoid quadratic re-rendering.
    rendered: dict[str, Any] = {
        k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in state.items()
    }
    total = _size(rendered)
    if total <= max_chars:
        return rendered, []
    aging = _Aging(
        rendered=rendered,
        total=total,
        protected=_referenced_keys(state),
        field_size={k: _size(v) for k, v in rendered.items() if isinstance(v, (dict, list))},
        # insertion order, fixed for the render
        order={k: list(v) for k, v in rendered.items() if isinstance(v, dict)},
        max_chars=max_chars,
        keep_recent=keep_recent,
    )
    notes = _abbreviate_oldest(aging)
    notes += _drop_oldest(aging)
    notes += _shorten_scalars(aging)
    return aging.rendered, notes


def _abbreviate_oldest(aging: _Aging) -> list[str]:
    """Render the oldest entries of the largest oversized fields as one line each."""
    rendered, field_size = aging.rendered, aging.field_size
    aged: dict[str, int] = {}
    cursor: dict[str, int] = {}
    while aging.total > aging.max_chars:
        candidates = [
            (field_size[k], k)
            for k, v in rendered.items()
            if isinstance(v, (dict, list)) and len(v) - aging.keep_recent > cursor.get(k, 0)
        ]
        if not candidates:
            break
        _, key = max(candidates)
        value = rendered[key]
        pos = cursor.get(key, 0)
        if isinstance(value, dict):
            k0 = aging.order[key][pos]
            v0 = value[k0]
            if k0 not in aging.protected and (not isinstance(v0, str) or len(v0) > 96):
                short = _one_line(v0)
                delta = _entry_size(value, k0, short) - _entry_size(value, k0, v0)
                value[k0] = short
                aging.total += delta
                field_size[key] += delta
                aged[key] = aged.get(key, 0) + 1
        else:
            v0 = value[pos]
            if not isinstance(v0, str) or len(v0) > 96:
                short = _one_line(v0)
                delta = _size(short) - _size(v0)
                value[pos] = short
                aging.total += delta
                field_size[key] += delta
                aged[key] = aged.get(key, 0) + 1
        cursor[key] = pos + 1
    return [f"{k}: oldest {n} entr{'y' if n == 1 else 'ies'} shown as one line" for k, n in aged.items()]


def _drop_oldest(aging: _Aging) -> list[str]:
    """Drop the oldest unprotected entries of the largest fields and leave one marker per field."""
    rendered, field_size, order, protected = aging.rendered, aging.field_size, aging.order, aging.protected
    omitted: dict[str, int] = {}
    drop_ptr: dict[str, int] = {}
    while aging.total > aging.max_chars:
        candidates = [
            (field_size[k], k)
            for k, v in rendered.items()
            if isinstance(v, (dict, list))
            and drop_ptr.get(k, 0) < (len(order[k]) if isinstance(v, dict) else len(v)) - aging.keep_recent
        ]
        if not candidates:
            break
        _, key = max(candidates)
        value = rendered[key]
        if isinstance(value, dict):
            limit = len(order[key]) - aging.keep_recent
            ptr = drop_ptr.get(key, 0)
            while ptr < limit and order[key][ptr] in protected:
                ptr += 1
            drop_ptr[key] = ptr + 1
            if ptr >= limit:
                drop_ptr[key] = limit
                if not any(k0 not in protected for k0 in order[key][:limit]):
                    field_size[key] = -1  # nothing droppable here; stop picking this field
                continue
            k0 = order[key][ptr]
            delta = _entry_size(value, k0, value[k0]) - 1  # {k:v} minus its braces, plus the separating comma
            del value[k0]
        else:
            ptr = drop_ptr.get(key, 0)
            delta = _size(value[ptr]) + 1
            drop_ptr[key] = ptr + 1
        aging.total -= delta
        field_size[key] -= delta
        omitted[key] = omitted.get(key, 0) + 1
    for key, n in omitted.items():
        marker = f"{n} older entries omitted from this rendering; full text in state.json"
        value = rendered[key]
        rendered[key] = {"_omitted": marker, **value} if isinstance(value, dict) else [marker] + value[drop_ptr[key] :]
    return [
        f"{k}: {n} older entr{'y' if n == 1 else 'ies'} omitted (full text in state.json)" for k, n in omitted.items()
    ]


def _shorten_string(value: str, target: int) -> str:
    """Keep the longest prefix fitting its JSON budget, including the exact cut marker."""
    low, high = 0, len(value) - 1
    while low < high:
        mid = (low + high + 1) // 2
        candidate = value[:mid] + _cut_marker(len(value) - mid)
        if _size(candidate) <= target:
            low = mid
        else:
            high = mid - 1
    return value[:low] + _cut_marker(len(value) - low)


def _cut_marker(cut: int) -> str:
    return f"[{cut} characters cut; full value in state.json]"


def _shorten_scalars(aging: _Aging) -> list[str]:
    """Apply the remaining budget to scalar fields after container aging."""
    total = _size(aging.rendered)
    notes = []
    candidates = sorted(
        ((_size(value), key) for key, value in aging.rendered.items() if isinstance(value, str)), reverse=True
    )
    for size, key in candidates:
        if total <= aging.max_chars:
            break
        value = aging.rendered[key]
        short = _shorten_string(value, size - (total - aging.max_chars))
        if _size(short) >= size:
            continue
        aging.rendered[key] = short
        total += _size(short) - size
        notes.append(f"{key}: string shortened (full value in state.json)")
    return notes


STALE_OBSERVATION_MINUTES = 10


def stale_observation_note(steps: list[dict[str, Any]], now: float | None = None) -> str:
    """Warn on stale observations so resumed agents do not close the loop prematurely using outdated evidence."""
    if not steps or not steps[-1].get("ts"):
        return ""
    age_min = ((now if now is not None else time.time()) - float(steps[-1]["ts"])) / 60
    if age_min < STALE_OBSERVATION_MINUTES:
        return ""
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(steps[-1]["ts"])))
    return (
        f"RESUMED after a pause: the observation below was produced by step {steps[-1].get('step')} at {when}, "
        f"about {int(age_min)} minutes ago. The environment may have changed since (operator work, restarts, "
        "resets). Re-verify anything time-sensitive in it before acting on it; operator messages about the "
        "current state take precedence over these old numbers. Values in your facts recorded before the pause are "
        "as old as this observation: re-read every one you report, and do not mix old and new readings.\n\n"
    )
