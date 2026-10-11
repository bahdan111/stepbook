"""Deterministic patch validation. The runtime owns the schema; the model only proposes.

Checks, in order:
1. the patch is a JSON object;
2. no placeholder values ("..." or "…"): a known failure where the model abbreviates instead of patching
   (Dynamic Cheatsheet, arXiv 2504.07952);
3. no more than ``max_deletions`` explicit deletions in one patch (mass-deletion guard, aimed at the
   paper's dominant small-model error: accidental key loss);
4. protected paths are not changed;
5. the *resulting* state validates against the JSON Schema (types, enums, required keys);
6. any schema node carrying ``"x-budget": <chars>`` keeps its serialized size under that budget,
   when explicitly selected by a custom schema. Shipped skills use concise-memory guidance, not size caps.

On failure a ``PatchError`` carries a message meant to be fed back to the model as the next
observation. Rejected output may consume additional model calls; do not assume retries are free.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from .merge import count_deletions, diff, merge

PLACEHOLDERS = ("...", "…")
"""The two values that mean "the rest is unchanged" and would overwrite the data they stand for.

Only these exact values, stripped. Words a model might abbreviate with -- "same", "no change" -- are also
ordinary values a state field can really hold (a status, a diff summary), and a patch carrying one is applied.
"""


class PatchError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems

    def as_observation(self) -> str:
        bullets = "\n".join(f"- {p}" for p in self.problems)
        return (
            "STATE PATCH REJECTED. The state was NOT changed and no action was executed. "
            "Fix the patch and answer again for the same observation.\n" + bullets
        )


@dataclass
class Validator:
    schema: dict[str, Any]
    protected_paths: tuple[str, ...] = ()
    max_deletions: int = 20
    write_once_paths: tuple[str, ...] = ()  # may be set while missing/empty, then behave as protected
    forbid_placeholders: bool = True
    _validator: Draft202012Validator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        Draft202012Validator.check_schema(self.schema)
        self._validator = Draft202012Validator(self.schema)

    def apply(self, state: dict[str, Any], patch: Any) -> tuple[dict[str, Any], list[str]]:
        """Validate and apply. Returns (new_state, changed_paths) or raises PatchError."""
        problems: list[str] = []
        if not isinstance(patch, dict):
            raise PatchError([f"state_patch must be a JSON object, got {type(patch).__name__}"])
        if self.forbid_placeholders:
            problems += [
                f"placeholder value at {p!r}; write the real value or omit the key" for p in _placeholders(patch)
            ]
        deletions = count_deletions(patch)
        if deletions > self.max_deletions:
            problems.append(
                f"{deletions} deletions in one patch exceeds the limit of {self.max_deletions}; "
                "delete fewer keys per step or check you are not rewriting the whole state"
            )
        if problems:
            raise PatchError(problems)

        new_state = merge(state, patch)
        changed = diff(state, new_state)
        locked_changes = diff(state, new_state, strict_types=True)
        problems.extend(self._direct_locked_problems(state, locked_changes))
        problems.extend(self._locked_problems(state, new_state, locked_changes))
        for err in sorted(self._validator.iter_errors(new_state), key=lambda e: list(e.path)):
            where = ".".join(str(p) for p in err.path) or "<root>"
            message = err.message
            if err.validator == "maxLength" and isinstance(err.instance, str):
                size, limit = len(err.instance), err.validator_value
                message = (
                    f"string has {size} characters (Unicode code points); maxLength is {limit}, "
                    f"excess {size - limit}. Shorten this field while preserving supported facts; "
                    "do not invent evidence or remove a needed update. No text was trimmed or applied."
                )
            problems.append(f"schema violation at {where!r}: {message}")
        problems += _budget_violations(self.schema, new_state)
        if problems:
            raise PatchError(problems)
        return new_state, changed

    def _direct_locked_problems(self, state: dict[str, Any], changed: list[str]) -> list[str]:
        """Retain direct changed-path checks, including literal dots and repeated diagnostics."""
        problems: list[str] = []
        for path in changed:
            for protected in self.protected_paths:
                if path == protected or path.startswith(protected + "."):
                    problems.append(f"{path!r} is protected and cannot be changed by the model")
            for wo in self.write_once_paths:
                if path == wo or path.startswith(wo + "."):
                    cur: Any = state
                    for part in wo.split("."):
                        cur = cur.get(part) if isinstance(cur, dict) else None
                    if cur not in (None, {}, [], ""):
                        problems.append(f"{path!r} is write-once and already set; it cannot be changed")
        return problems

    def _locked_problems(self, state: dict[str, Any], new_state: dict[str, Any], changed: list[str]) -> list[str]:
        """Add initialized-value checks for ancestor edits, including whole-list replacements."""
        changes: list[tuple[str, str]] = []
        for paths, write_once, message in (
            (self.protected_paths, False, "is protected and cannot be changed by the model"),
            (self.write_once_paths, True, "is write-once and already set; it cannot be changed"),
        ):
            for path in paths:
                if not any(path.startswith(ancestor + ".") for ancestor in changed):
                    continue
                before = _path_value(state, path)
                if before is _MISSING or (write_once and before in (None, {}, [], "")):
                    continue
                changes.extend(
                    (changed_path, f"{changed_path!r} {message}")
                    for changed_path in diff(before, _path_value(new_state, path), path, strict_types=True)
                )
        return [message for _, message in sorted(changes, key=lambda item: item[0])]


def _budget_violations(schema: Any, value: Any, path: str = "") -> list[str]:
    """Walk schema and state together; report subtrees whose JSON size exceeds their x-budget."""
    out: list[str] = []
    if not isinstance(schema, dict):
        return out
    budget = schema.get("x-budget")
    if isinstance(budget, int):
        size = len(_json.dumps(value, separators=(",", ":")))
        if size > budget:
            out.append(
                f"{path or '<root>'!r} is {size} chars, over its budget of {budget}; "
                "compress it, drop resolved items, or move detail to notes and delete it here"
            )
    if isinstance(value, dict):
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties")
        for k, v in value.items():
            sub = props.get(k, extra if isinstance(extra, dict) else None)
            if sub is not None:
                out += _budget_violations(sub, v, f"{path}.{k}" if path else str(k))
    return out


def _placeholders(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            found += _placeholders(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, str) and value.strip() in PLACEHOLDERS:
        found.append(path or "<root>")
    return found


_MISSING = object()


def _path_value(state: dict[str, Any], path: str) -> Any:
    """Read a locked value while distinguishing absent keys from explicit nulls."""
    value: Any = state
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value
