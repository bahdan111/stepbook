"""Deterministic, user-configured candidate rules; not a semantic safety judge."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _strings(values, label):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return values


class CandidatePolicy:
    """Schema plus literal-copy exclusions, configured by the operator, not the agent.

    Policy files are bounded JSON string lists, resolved against eval.json. No
    network references, executable callbacks or paid judging are involved.
    """

    def __init__(self, spec, base: Path):
        spec = {} if spec is None else spec
        if not isinstance(spec, dict) or set(spec) - {"config_schema", "forbidden_literals", "forbidden_literal_files"}:
            raise ValueError("candidate_policy accepts config_schema, forbidden_literals and forbidden_literal_files")
        schema = spec.get("config_schema", {})

        def check_refs(value):
            if isinstance(value, dict):
                if set(value) & {"$ref", "$dynamicRef", "$recursiveRef"}:
                    raise ValueError("candidate config_schema must be self-contained; references are not supported")
                for item in value.values():
                    check_refs(item)
            elif isinstance(value, list):
                for item in value:
                    check_refs(item)

        check_refs(schema)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError("candidate config_schema is not a valid JSON schema") from exc
        self.validator = Draft202012Validator(schema)
        self.rules: dict[str, list[str]] = {}
        self.fingerprints: dict[str, str] = {}
        for key in ("forbidden_literals", "forbidden_literal_files"):
            fields = spec.get(key, {})
            if not isinstance(fields, dict) or any(
                not isinstance(f, str) or not f or any(not p for p in f.split(".")) for f in fields
            ):
                raise ValueError(f"{key} must map dotted field paths to string lists")
            for field, values in fields.items():
                for value in _strings(values, key):
                    if key == "forbidden_literals":
                        self.rules.setdefault(field, []).append(normalized(value))
                    else:
                        path = (base / value).resolve()
                        with path.open("rb") as handle:
                            raw = handle.read(1024 * 1024 + 1)
                        if len(raw) > 1024 * 1024:
                            raise ValueError("candidate literal file exceeds 1 MiB")
                        literals = _strings(json.loads(raw), "candidate literal file")
                        self.fingerprints[value] = hashlib.sha256(raw).hexdigest()
                        self.rules.setdefault(field, []).extend(normalized(v) for v in literals)

    def check(self, config: dict) -> str | None:
        error = next(self.validator.iter_errors(config), None)
        if error is not None:
            location = ".".join(map(str, error.absolute_path)) or "<config>"
            # Do not echo the candidate or a potentially private denied value.
            return f"config_schema violation at {location} ({error.validator})"
        for field, literals in self.rules.items():
            value = config
            for part in field.split("."):
                if not isinstance(value, dict) or part not in value:
                    return f"required checked field {field} is missing"
                value = value[part]
            if not isinstance(value, str):
                return f"checked field {field} must be a string"
            text = normalized(value)
            if any(literal in text for literal in literals):
                return f"forbidden literal in {field}; remove copied examples/identifiers, not the policy"
        return None
