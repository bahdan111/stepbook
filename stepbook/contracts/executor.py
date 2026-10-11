"""Data contract of an executor receipt: what a provider hands back for one paid step, parsed once.

CLI documents cross a process boundary and are parsed here instead of being re-read with ``.get`` chains:

* ``CliReceipt`` -- the JSON object ``claude -p --output-format json`` prints on stdout (``ClaudeCLIExecutor``).

Receipts describe work the provider has already been paid for, so the parse keeps what the provider sent: a
``CliReceipt`` keeps the decoded object, because ``Envelope.raw`` carries the receipt verbatim, and the token
mappings stay mappings, because ``Envelope.usage`` and ``Envelope.raw["model_usage"]`` are quoted into the
journal, whose bytes are a contract (tests/fixtures/journal_bytes).

The typing rule, applied field by field:

* where the executor guarded a field with ``isinstance``, the model reproduces that guard and a value of the
  wrong shape reads as absent (``structured_output``, whose fallback to the text result must still run);
* where the executor coerced blindly (``int(...)``, ``float(...)``, truthiness), the field is typed, and a value
  the coercion would have mangled -- or crashed on three use sites later -- is refused here, once;
* counters the executor echoes verbatim into ``Envelope.usage`` are strict, so an accepted receipt can never
  change one of its own numbers on the way into a journal.

Nothing here raises a harness error: a refused receipt surfaces as a ``ValidationError`` that the executor
renders as one sentence (``first_problem``) and never replays automatically.

A leaf module on purpose: standard library and pydantic only, no ``stepbook`` import at all, so
``stepbook/executors/claude.py`` imports it with no cycle (tests/test_import_graph.py).
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    ValidationError,
    model_validator,
)


def _object_or_empty(value: Any) -> Any:
    """``data.get("usage") or {}``: a falsy token mapping is an absent one, as every call site read it."""
    return {} if not value else value


def _kept_if_object(value: Any) -> Any:
    """The executor's own ``isinstance(out, dict)`` guard: a response object of any other shape reads as absent,
    so the fallback that parses the text result runs exactly as before instead of failing the whole receipt."""
    return value if isinstance(value, dict) else None


def _kept_if_text(value: Any) -> Any:
    """A field the executor only ever compared against a string. Nothing else can equal one, so a value of
    another shape reads as absent and compares false exactly as it did, instead of failing a paid receipt."""
    return value if isinstance(value, str) else None


def _integral_or_absent(value: Any) -> Any:
    """A status code the executor only ever compared against a number (``... == 429``). ``429.0`` equalled it
    and ``"429"`` did not, so an integral number is kept and everything else reads as absent."""
    if type(value) is int:
        return value
    if type(value) is float and value.is_integer():
        return int(value)
    return None


StrictCount = Annotated[int, Field(strict=True)]
TokenMapping = Annotated[dict[str, Any], BeforeValidator(_object_or_empty)]
ResponseObject = Annotated[dict[str, Any] | None, BeforeValidator(_kept_if_object)]
ComparedText = Annotated[str | None, BeforeValidator(_kept_if_text)]
StatusCode = Annotated[int | None, BeforeValidator(_integral_or_absent)]


def first_problem(error: ValidationError) -> str:
    """One sentence for a refused receipt: where the problem is and what it is.

    Never pydantic's multi-line report, its documentation URL, or the provider's own bytes echoed back; a
    receipt can carry a model's prose, and an error message is not a place to replay it.
    """
    problem = error.errors(include_url=False, include_input=False, include_context=False)[0]
    location = ".".join(str(part) for part in problem["loc"]) or "receipt"
    return f"{location}: {problem['msg']}"


def reports_rate_limit(document: dict[str, Any]) -> bool:
    """Whether a decoded CLI receipt says HTTP 429, read from the document itself by the same rule as
    ``CliReceipt.api_error_status`` -- for a receipt this module refused, which is the only caller.

    A refused receipt is still a receipt the provider was paid for, and one field the parse cannot read must
    not demote its rate limit to an ordinary failure: the class of the error decides how the run answers it.
    """
    return _integral_or_absent(document.get("api_error_status")) == 429


class _ReceivedDocument(BaseModel):
    """A receipt that keeps the object it was decoded from.

    ``Envelope.raw`` and ``Envelope.usage`` quote the provider's own object into the journal, key order
    included, so the parse is not allowed to become the only surviving copy: unknown keys are not read
    (``extra="ignore"`` -- reading them would mean re-emitting a dump whose key order is the model's field
    order, not the provider's), and the decoded object is kept as ``document``.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    _document: dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _keep_document(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        receipt = handler(data)
        if isinstance(data, dict):  # the boundary itself; anything else the handler has already refused
            receipt._document = data
        return receipt

    @property
    def document(self) -> dict[str, Any]:
        """The provider's object as it was decoded, which is what the envelope quotes."""
        return self._document


class CliUsage(_ReceivedDocument):
    """The ``usage`` object of a CLI receipt: the main loop's four token counters.

    Strict integers. The executor reports these counters verbatim in ``Envelope.usage`` -- a counter the CLI
    did not report stays ``null`` there rather than becoming a zero -- so a coerced value would silently change
    a journalled number; a counter that is not an integer is refused instead.
    """

    input_tokens: StrictCount | None = None
    cache_read_input_tokens: StrictCount | None = None
    cache_creation_input_tokens: StrictCount | None = None
    output_tokens: StrictCount | None = None

    @property
    def prompt_tokens(self) -> int:
        """Every prompt token the main loop reported, cache reads and cache writes included."""
        return (self.input_tokens or 0) + (self.cache_read_input_tokens or 0) + (self.cache_creation_input_tokens or 0)

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens or 0

    @property
    def reported(self) -> dict[str, int | None]:
        """The four counters, in the order the executor has always written them into ``Envelope.usage``."""
        return {
            "input_tokens": self.input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "output_tokens": self.output_tokens,
        }


UsageObject = Annotated[CliUsage, BeforeValidator(_object_or_empty)]


class CliReceipt(_ReceivedDocument):
    """The JSON object ``claude -p --output-format json`` prints for one step, successful or not.

    Only the fields the harness reads are declared; the rest of the receipt survives in ``document`` and is
    copied into ``Envelope.raw`` unchanged.
    """

    # ``None`` is "the receipt did not say", which is how a missing or null flag has always read: falsy.
    is_error: bool | None = None
    result: str | None = None
    stop_reason: ComparedText = None
    api_error_status: StatusCode = None
    num_turns: int | None = None
    total_cost_usd: float | None = None
    usage: UsageObject = Field(default_factory=CliUsage)
    model_usage: TokenMapping = Field(default_factory=dict, alias="modelUsage")
    structured_output: ResponseObject = None

    @classmethod
    def from_document(cls, document: Any) -> Self | None:
        """The receipt a decoded stdout carries, or None when that stdout is not a JSON object at all.

        The absent case is returned rather than raised because the two call sites answer it differently: a
        successful exit reports the output it could not read, a failed exit reports its exit status.
        """
        return cls.model_validate(document) if isinstance(document, dict) else None

    @property
    def api_calls(self) -> int:
        """Internal model calls this step made; a receipt without a turn count reports one."""
        return self.num_turns or 1

    @property
    def cost_usd(self) -> float:
        """What the provider billed. A receipt that reports no cost bills zero and is marked unknown."""
        return self.total_cost_usd or 0.0

    @property
    def cost_unknown(self) -> bool:
        return self.total_cost_usd is None

    @property
    def model_ids(self) -> list[str]:
        return sorted(self.model_usage)

    @property
    def rate_limited(self) -> bool:
        return self.api_error_status == 429

    @property
    def attempted_tool_call(self) -> bool:
        """True when the receipt's error is the model calling a tool instead of writing a text action.

        The provider's own stop reason decides it, never the error's prose: a paid failure that merely
        mentions a toolchain is an ordinary error, not a tool call scored as a no-op step.
        """
        return self.stop_reason == "tool_use"
