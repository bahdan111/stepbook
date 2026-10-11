"""A skill folder: instructions, the state schema and the compact rendering of that schema for the model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts.skill import JsonSchemaNode, StepbookMeta


class LockPathError(ValueError):
    """A declared lock route violates the supported load-time path policy."""


def skill_content_hash(folder: Path | str) -> str:
    """Hash the names and bytes of exactly the files Skill.load reads, including optional-file absence."""
    digest = hashlib.sha256()
    for name in ("SKILL.md", "schema.json", "initial_state.json"):
        path = Path(folder) / name
        content = path.read_bytes() if path.exists() else None
        digest.update(name.encode() + b"\0")
        digest.update(b"missing\0" if content is None else str(len(content)).encode() + b"\0" + content)
    return digest.hexdigest()


@dataclass
class Skill:
    """A skill is a folder: instructions (P), a JSON Schema for Σ, an initial state and protected paths."""

    name: str
    instructions: str
    schema: dict[str, Any]
    initial_state: dict[str, Any]
    protected_paths: tuple[str, ...] = ()
    write_once_paths: tuple[str, ...] = ()
    max_deletions: int = 20

    @classmethod
    def load(cls, folder: Path | str, include_schema: bool | None = None) -> Skill:
        """Load a skill folder. By default the state schema is appended to the instructions (the paper's P
        includes the domain schema); ``"x-stepbook": {"include_schema": false}`` or the argument turn it
        off. Without it, models invent field names and every step costs a validator round-trip."""
        folder = Path(folder)
        instructions = (folder / "SKILL.md").read_text(encoding="utf-8")
        schema = json.loads((folder / "schema.json").read_text(encoding="utf-8"))
        init_path = folder / "initial_state.json"
        initial = json.loads(init_path.read_text(encoding="utf-8")) if init_path.exists() else {}
        meta = StepbookMeta.read(schema)
        for path in (*meta.protected, *meta.write_once):
            _check_lock_path(path)
            if _dotted_state_path(initial, path) or _dotted_schema_path(schema, path, path):
                raise LockPathError(f"lock path {path!r}: dotted keys cannot be locked; paths are dot-separated")
        if include_schema is None:
            include_schema = meta.include_schema
        if include_schema:
            instructions = (
                instructions.rstrip() + "\n\n" + render_schema_section(schema, meta.protected, meta.write_once)
            )
        return cls(
            name=folder.name,
            instructions=instructions,
            schema=schema,
            initial_state=initial,
            protected_paths=meta.protected,
            write_once_paths=meta.write_once,
            max_deletions=meta.max_deletions,
        )


def _check_lock_path(path: str) -> None:
    for part in path.split("."):
        if not part:
            raise LockPathError(f"lock path {path!r}: empty segments are unsupported")
        try:
            index = int(part)
        except ValueError:
            continue
        if part != str(index):
            raise LockPathError(f"lock path {path!r}: numeric segments must be canonical (for example, '0' or '1')")


def _dotted_state_path(value: Any, path: str) -> bool:
    """Detect literal keys that would be mistaken for one or more path segments."""
    if not path:
        return False
    if isinstance(value, list):
        value = {str(index): item for index, item in enumerate(value)}
    if not isinstance(value, dict):
        return False
    for key, child in value.items():
        if path == key or path.startswith(key + "."):
            if "." in key:
                return True
            if _dotted_state_path(child, path[len(key) + 1 :]):
                return True
    return False


def _dotted_schema_path(schema: Any, path: str, lock_path: str) -> bool:
    """Detect declared literal dotted names along the lock's route, not inside its value."""
    if not isinstance(schema, dict):
        return False
    unsupported = {"allOf", "anyOf", "oneOf", "$ref"}.intersection(schema)
    if unsupported:
        raise LockPathError(
            f"lock path {lock_path!r}: schema keywords {', '.join(sorted(unsupported))} are unsupported for locks"
        )
    if not path:
        return False
    properties = schema.get("properties", {})
    if any("." in key and (path == key or path.startswith(key + ".")) for key in properties):
        return True
    part, _, rest = path.partition(".")
    if _dotted_schema_path(properties.get(part), rest, lock_path):
        return True
    if part.isdecimal():
        prefix = schema.get("prefixItems", [])
        index = int(part)
        item = prefix[index] if index < len(prefix) else schema.get("items")
        return _dotted_schema_path(item, rest, lock_path)
    return False


def render_schema_section(schema: dict[str, Any], protected: tuple[str, ...], write_once: tuple[str, ...] = ()) -> str:
    """Compact, model-facing rendering of the state schema: field names, types, enums, budgets."""
    lines = ["State schema (your state_patch must keep the state valid against this; unknown keys are rejected):"]

    def walk(node: JsonSchemaNode, path: str, indent: int) -> None:
        pad = "  " * indent
        t = node.type if node.type is not None else "any"
        extras: list[str] = []
        if node.enum is not None:
            extras.append("one of " + "|".join(map(str, node.enum)))
        if node.x_budget is not None:
            extras.append(f"max {node.x_budget} chars")
        if node.max_length is not None:
            extras.append(f"max {node.max_length} chars")
        if path in protected:
            extras.append("PROTECTED, never change")
        if path in write_once:
            extras.append("WRITE-ONCE: set it once, then never change it")
        if node.description:
            extras.append(node.description)
        head = f"{pad}{path.rsplit('.', 1)[-1] if path else '<state>'}: {t}" + (
            f" ({'; '.join(extras)})" if extras else ""
        )
        lines.append(head)
        req = set(node.required)
        for k, v in node.properties.items():
            if not v.is_boolean:
                described = (("required. " if k in req else "") + (v.description or "")).strip() or None
                walk(v.model_copy(update={"description": described}), f"{path}.{k}" if path else k, indent + 1)
        ap = node.additional_properties
        if ap is not None and not ap.is_boolean:
            walk(ap, f"{path}.<id>" if path else "<id>", indent + 1)
        if node.items is not None and node.items.type == "object":
            walk(node.items, f"{path}[]", indent + 1)

    walk(JsonSchemaNode.read(schema), "", 0)
    lines.append(
        "Patch rules: merge patch, only changed keys; null deletes a key (a missing key means empty); "
        "never rewrite whole objects."
    )
    return "\n".join(lines)
