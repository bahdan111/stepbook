"""The two documents a skill folder contributes to a prompt, parsed once where they are read.

``schema.json`` is the skill author's JSON Schema for the state. Two readers walk it: the renderer
that puts the state shape in front of the model (``stepbook/state/skill.py``) and the merge-patch shaper
that turns it into the executor's output schema (``stepbook/state/patch_schema.py``). Both used to ask
``isinstance`` again at every keyword; ``JsonSchemaNode`` answers those questions once, on the way in.

The node is deliberately narrow: it owns the keywords those two readers branch on, print or copy, and
nothing else. It does not re-implement the JSON Schema vocabulary, because the same document goes to
``jsonschema``'s ``Draft202012Validator.check_schema`` in ``Validator`` before the first step.
Skill checkpoint validation uses jsonschema, so an unknown type name, an unknown keyword and a keyword
this package never reads all pass through untouched (``extra="allow"``): schemas carry ``title``,
``format``, ``default`` and ``$defs``, and an array's item schema is copied out exactly as written.

``x-stepbook`` is the other document, and that one is ours: four keys, ``extra="forbid"``, because
a misspelled ``protected`` would silently unprotect the goal instead of naming the mistake.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    ValidatorFunctionWrapHandler,
    model_validator,
)

SCALAR_TYPES = frozenset({"string", "number", "integer", "boolean", "null"})
# Keywords that build a node out of other schemas, or point at one: nothing here is ours to narrow.
UNNARROWED_KEYWORDS = frozenset({"$ref", "$dynamicRef", "anyOf", "oneOf", "allOf", "not", "if", "patternProperties"})
COMPOSITION_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "not", "if"})


def _first_problem(error: ValidationError) -> str:
    """The first failure as ``where: what``.

    Pydantic's own report spans several lines, repeats the offending input and links its
    documentation; a skill author needs the keyword and the rule, on one line.
    """
    problem = error.errors()[0]
    where = ".".join(str(part) for part in problem["loc"]) or "<root>"
    return f"{where}: {problem['msg']}"


def _written_value(value: Any) -> Any:
    """Refuse explicit ``null`` keyword values; omit a keyword to express no limit."""
    if value is None:
        raise ValueError("null is not a value for this keyword; omit the keyword instead")
    return value


def _object_form_only(value: Any) -> Any:
    """``items`` may also be ``true``/``false`` or, in draft-07 style, a tuple of schemas. This module
    narrows none of those and never has: anything but an object of keywords reads as no item schema."""
    return value if isinstance(value, dict) else None


TypeNames = Annotated[StrictStr | list[StrictStr] | None, BeforeValidator(_written_value)]
Choices = Annotated[list[Any] | None, BeforeValidator(_written_value)]
Characters = Annotated[StrictInt | None, BeforeValidator(_written_value)]
Number = Annotated[StrictInt | StrictFloat | None, BeforeValidator(_written_value)]
Text = Annotated[StrictStr | None, BeforeValidator(_written_value)]
ObjectItemSchema = Annotated["JsonSchemaNode | None", BeforeValidator(_object_form_only)]
MemberSchema = Annotated["JsonSchemaNode | None", BeforeValidator(_written_value)]


class JsonSchemaNode(BaseModel):
    """One node of a skill's state schema: an object of keywords, or the boolean schema true/false."""

    model_config = ConfigDict(extra="allow")

    type: TypeNames = None
    properties: dict[str, JsonSchemaNode] = Field(default_factory=dict)
    additional_properties: MemberSchema = Field(default=None, alias="additionalProperties")
    items: ObjectItemSchema = None  # the object form only; boolean and draft-07 tuple items read as absent
    enum: Choices = None
    required: list[StrictStr] = Field(default_factory=list)
    description: StrictStr | None = None
    min_length: Characters = Field(default=None, alias="minLength")
    max_length: Characters = Field(default=None, alias="maxLength")
    pattern: Text = None
    minimum: Number = None
    maximum: Number = None
    exclusive_minimum: Number = Field(default=None, alias="exclusiveMinimum")
    exclusive_maximum: Number = Field(default=None, alias="exclusiveMaximum")
    multiple_of: Number = Field(default=None, alias="multipleOf")
    x_budget: Characters = Field(default=None, alias="x-budget")

    _keywords: dict[str, Any] = PrivateAttr(default_factory=dict)
    _boolean: bool | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _as_the_author_wrote_it(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> JsonSchemaNode:
        """A JSON Schema is an object of keywords or a boolean. Keep the source: a node copied into
        another schema keeps the author's keys in the author's order, including keywords we ignore."""
        if value is True or value is False:
            boolean = handler({})
            boolean._boolean = value
            return boolean
        node = handler(value)
        node._keywords = dict(value)
        return node

    @classmethod
    def read(cls, document: Any) -> Self:
        """Parse a state schema, reporting a keyword of the wrong shape as one line."""
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise ValueError(f"state schema: {_first_problem(error)}") from error

    @property
    def as_written(self) -> dict[str, Any]:
        """The keywords exactly as the author wrote them; empty for a boolean schema."""
        return dict(self._keywords)

    @property
    def is_boolean(self) -> bool:
        """Written as ``true``/``false`` instead of an object: legal, with no keyword to read."""
        return self._boolean is not None

    @property
    def rejects_everything(self) -> bool:
        """The boolean schema ``false``: no value is valid against it."""
        return self._boolean is False

    @property
    def types(self) -> list[str]:
        """The declared types, always as a list; empty when the author declared none."""
        match self.type:
            case None:
                return []
            case str() as single:
                return [single]
            case names:
                return list(names)

    @property
    def scalar_type(self) -> str | None:
        """The one scalar type this node declares, when it declares exactly one and it is scalar."""
        match self.type:
            case str() as name if name in SCALAR_TYPES:
                return name
            case _:
                return None

    @property
    def composed(self) -> bool:
        """Built out of other schemas, or pointing at one: there is nothing here to narrow."""
        return not UNNARROWED_KEYWORDS.isdisjoint(self._keywords)

    @property
    def self_contained(self) -> bool:
        """No reference and no composition, so the node keeps its meaning inside another schema."""
        return not any(key.startswith("$") or key in COMPOSITION_KEYWORDS for key in self._keywords)


class StepbookMeta(BaseModel):
    """The ``x-stepbook`` block of a state schema: this package's own keys, so a typo is an error."""

    model_config = ConfigDict(extra="forbid")

    protected: tuple[str, ...] = ()
    write_once: tuple[str, ...] = ()
    max_deletions: int = 20
    include_schema: StrictBool = True

    @classmethod
    def read(cls, schema: dict[str, Any]) -> Self:
        """The block of a state schema, or the defaults when the author wrote none."""
        try:
            return cls.model_validate(schema.get("x-stepbook", {}))
        except ValidationError as error:
            raise ValueError(f"x-stepbook: {_first_problem(error)}") from error
