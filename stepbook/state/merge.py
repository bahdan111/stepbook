"""State merge operator (the paper's ⊕).

Semantics, fixed here because the paper leaves them implicit:
- recursive (deep) merge of objects: a nested object in the patch touches only the keys it names,
  so ``{"inventory": {"shelf_42": null}}`` deletes one shelf and never replaces ``inventory``;
- ``null`` deletes a key (RFC 7396 style); omission is always a no-op;
- non-object values (strings, numbers, lists) are replaced wholesale. Lists are therefore
  discouraged in schemas; use id-keyed objects so patches can never drop siblings by accident.
"""

from __future__ import annotations

from typing import Any


def merge(base: Any, patch: Any) -> Any:
    """Return base ⊕ patch without mutating either argument."""
    if not isinstance(patch, dict) or not isinstance(base, dict):
        return _copy(patch)
    out = {k: _copy(v) for k, v in base.items()}
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = _copy(value)
    return out


def diff(before: Any, after: Any, path: str = "", *, strict_types: bool = False) -> list[str]:
    """Paths whose value differs between two states ('a.b.c' form). Used to log real changes
    and to enforce protected paths; no-op patch keys do not show up here.
    Locks opt into recursive JSON type equality; ordinary change logging retains Python equality."""
    if isinstance(before, dict) and isinstance(after, dict):
        changed: list[str] = []
        for key in sorted(set(before) | set(after)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                changed.append(sub)
            else:
                changed.extend(diff(before[key], after[key], sub, strict_types=strict_types))
        return changed
    equal = _json_equal(before, after) if strict_types else before == after
    return [] if equal else [path or "<root>"]


def _json_equal(before: Any, after: Any) -> bool:
    """Compare JSON values without Python's bool/int/float coercion, including inside lists."""
    if type(before) is not type(after):
        return False
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(_json_equal(value, after[key]) for key, value in before.items())
    if isinstance(before, list):
        return len(before) == len(after) and all(_json_equal(a, b) for a, b in zip(before, after, strict=True))
    return bool(before == after)


def count_deletions(patch: Any) -> int:
    """Number of explicit ``null`` deletions in a patch, at any depth."""
    if not isinstance(patch, dict):
        return 0
    n = 0
    for value in patch.values():
        if value is None:
            n += 1
        elif isinstance(value, dict):
            n += count_deletions(value)
    return n


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value
