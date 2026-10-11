"""Native merge-patch shape guidance, not a replacement for host state validation.

Objects are partial; null deletes members. Complex schemas stay unconstrained rather
than incorrectly rejecting legal patches. The host validates the resulting state.
"""

from __future__ import annotations

import copy
from typing import Any

from ..contracts.skill import JsonSchemaNode


def patch_shape(schema: dict[str, Any] | bool, *, root: bool = False) -> dict[str, Any]:
    """The merge-patch shape of a state schema document. The document is parsed here, once.

    A document, never a parsed node: ``JsonSchemaNode.read`` hands a node straight back, with its remembered
    keywords rewritten from the model's own fields.
    """
    return _shape(JsonSchemaNode.read(schema), root=root)


def _shape(node: JsonSchemaNode, *, root: bool) -> dict[str, Any]:
    if _unconstrained(node):
        return {"type": "object"} if root else {}
    kinds = node.types
    result: dict[str, Any] = {"type": "object" if root else list(dict.fromkeys([*kinds, "null"]))}
    if "object" in kinds:
        result.update(_object_members(node))
    elif node.enum is not None and "array" not in kinds:
        result["enum"] = copy.deepcopy(node.enum)
        if None not in result["enum"]:
            result["enum"].append(None)
    # Complex array items/tuple constraints still belong to the host.
    return result


def _unconstrained(node: JsonSchemaNode) -> bool:
    """A schema this module leaves unnarrowed: not an object, composed, referenced, or untyped."""
    return node.is_boolean or node.composed or node.type is None


def _object_members(node: JsonSchemaNode) -> dict[str, Any]:
    """An object's patch members: every property partial, a forbidden extra only deletable."""
    # No required/minProperties/dependencies: members may already exist.
    properties = {name: _shape(member, root=False) for name, member in node.properties.items()}
    extra = node.additional_properties
    additional: Any = True
    if extra is not None and extra.rejects_everything:
        # Deleting an absent, otherwise forbidden member is a legal no-op.
        additional = {"type": "null"}
    elif extra is not None and not extra.is_boolean:
        additional = _shape(extra, root=False)
    return {"properties": properties, "additionalProperties": additional}


def state_output_schema(base: dict[str, Any], schema: dict[str, Any] | bool) -> dict[str, Any]:
    result = copy.deepcopy(base)
    description = result["properties"]["state_patch"].get("description")
    result["properties"]["state_patch"] = patch_shape(schema, root=True)
    if description:
        result["properties"]["state_patch"]["description"] = description
    return result
