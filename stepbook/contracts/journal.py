"""Data contract of the run journal (``events.jsonl``), parsed once when a journal is read.

Every line of the journal is a JSON object this harness appended; the recovery and control boundaries
then read a handful of kinds back out of it and decide what a resume may do. Those decisions are parsed
here instead of being re-derived with ``e.get("kind") == ...`` chains and bare ``e["action_id"]`` lookups
at every use site:

* ``Journal`` -- one ``events.jsonl``, parsed in a single pass into the kinds the boundaries reason about
  (``stepbook/storage/recovery.py``, ``stepbook/runtime/controls.py``, via ``EventLog.journal()``).
* ``OperatorReceipt`` -- the JSON file an operator hands to ``stepbook recover --receipt``.

A leaf module on purpose: standard library and pydantic only, no ``stepbook`` import at all, so
``events.py``, ``recovery.py`` and ``controls.py`` can all import it with no cycle
(tests/test_import_graph.py).

Two rules shape the field types. The journal's BYTES on write are a contract
(tests/fixtures/journal_bytes): a recovered step re-emits the proposal and the receipt it recovered from
exactly as they were written, so everything that is replayed into a later line stays ``Any`` and every
mapping that is replayed keeps the key order it was read in. And the journal is an append-only archive
older and newer runtimes both write: an unknown field is kept (``extra="allow"``), never a reason to
refuse a resume. What is typed is what a resume must be able to trust -- that a receipt names the action
it belongs to, and that a proposal names an action.
"""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictStr,
    ValidationError,
    model_validator,
)


def _kind_name(value: Any) -> Any:
    """A ``kind`` that is not a string names no kind, which is how every ``kind == "step"`` test read it."""
    return value if isinstance(value, str) else None


class MalformedJournal(RuntimeError):
    """A journal record of a known kind that no writer of this harness produces.

    A named ``RuntimeError`` lets ``run`` use the recovery diagnostic and exit status,
    without catching unrelated errors of the same base class.
    """


def _malformed_event(index: int, source: str | PathLike[str] | None, error: ValidationError) -> MalformedJournal:
    """A rejected record as one sentence: which record, where the fault is, and what is wrong there.

    pydantic's own report spans several lines, echoes the offending record back and links its
    documentation; this text reaches an operator's terminal and the resume path, so it stays a sentence.
    """
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"])
    # The only root-level failure these models raise is a line that is not a JSON object.
    problem = f"{where}: {first['msg']}" if where else "line is not a JSON object"
    return MalformedJournal(f"malformed event {index}{f' in {source}' if source is not None else ''}: {problem}")


class JournalEvent(BaseModel):
    """Any line of the journal: an object whose ``kind`` says which record it is.

    ``extra="ignore"``: this model reads the kind and hands the record to the model that knows it; the
    narrowed kinds below keep every field.
    """

    model_config = ConfigDict(extra="ignore")

    kind: Annotated[str | None, BeforeValidator(_kind_name)] = None


class ActionRecord(RootModel[dict[str, Any]]):
    """The controller's proposal, as ``action_started`` recorded it.

    A mapping rather than a model with fields: ``recover_actions`` re-emits it verbatim into the
    recovered ``step`` line, so its keys keep the order they were written in and its values are never
    coerced.
    """

    @model_validator(mode="after")
    def _names_an_action(self) -> Self:
        """Without an action the proposal cannot be replayed, and the ambiguous-action message that asks
        an operator to inspect the adapter could not name what may have run."""
        if "action" not in self.root:
            raise ValueError("a started action must name an action")
        return self

    @property
    def action(self) -> Any:
        """What was proposed. Untyped: it is replayed into the journal exactly as it was written."""
        return self.root["action"]

    @property
    def step(self) -> Any:
        """The step number the proposal belongs to, if it was recorded."""
        return self.root.get("step")

    @property
    def fields(self) -> dict[str, Any]:
        """The proposal as recorded, for replay into the recovered step line."""
        return self.root


class StepEvent(JournalEvent):
    """A committed step. ``action_id`` is absent in journals whose steps were not dispatched through
    ``execute_action`` (the baseline runtimes, and hand-written test journals)."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["step"]
    action_id: StrictStr | None = None


class ReplayedStep(StepEvent):
    """A committed step as a resume replays it: the three fields ``Runtime._replay`` reads by subscript.

    Missing fields or a non-string observation must be refused with the recovery diagnostic.
    ``step`` and ``action`` are replayed verbatim and stay ``Any``; ``result`` is the observation
    the replay concatenates, so it is text.
    """

    step: Any
    action: Any
    result: StrictStr


def replayable_steps(records: Iterable[Any], source: str | PathLike[str] | None = None) -> list[Any]:
    """The journal's ``step`` records, each checked for the fields a resume reads out of it.

    Returns the records themselves, not the parsed models: the replay reads their patch, action, step and
    result and coerces none of them.
    """
    steps: list[Any] = []
    for index, record in enumerate(records):
        if record.get("kind") != "step":
            continue
        try:
            ReplayedStep.model_validate(record)
        except ValidationError as error:
            raise _malformed_event(index, source, error) from error
        steps.append(record)
    return steps


class ActionStartedEvent(JournalEvent):
    """Intent recorded before dispatch: this action may have run."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["action_started"]
    action_id: StrictStr
    record: ActionRecord


class ActionResultEvent(JournalEvent):
    """The receipt of a dispatched action. ``result``, ``info`` and ``done`` are whatever the environment
    returned and are replayed into the recovered step unchanged, so none of them is narrowed here."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["action_result"]
    action_id: StrictStr
    result: Any
    info: Any
    done: Any
    horizon_stop: Any = None


class ActionWithheldEvent(JournalEvent):
    """A proposal the host refused to dispatch. Both fields reach the model as text or JSON and were
    always read through ``str(...)``, so neither is narrowed."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["action_withheld"]
    action: Any = ""
    reason: Any = None


class Journal(BaseModel):
    """One ``events.jsonl``, parsed into the records the recovery and control boundaries read.

    ``extra="forbid"``: unlike the records it holds, this object is built here and nowhere else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int = 0
    """Records read, of every kind: a journal with none of them is not a run that can be resumed."""

    committed_action_ids: set[str | None] = Field(default_factory=set)
    """The actions a step has already committed; a step without an action_id commits nothing.

    The ids, not the step records: nothing else reads a committed step, and a long journal's steps are the
    bulk of it."""

    started: list[ActionStartedEvent] = Field(default_factory=list)
    receipts: dict[StrictStr, ActionResultEvent] = Field(default_factory=dict)
    withheld_since_last_step: list[ActionWithheldEvent] = Field(default_factory=list)
    """Proposals refused after the last committed step: what a resumed run must be told did NOT happen."""

    @classmethod
    def from_records(cls, records: Iterable[Any], source: str | PathLike[str] | None = None) -> Self:
        """Parse a journal's records in one pass; a record of a known kind that is malformed stops the read."""
        committed: set[str | None] = set()
        started: list[ActionStartedEvent] = []
        receipts: dict[str, ActionResultEvent] = {}
        withheld: list[ActionWithheldEvent] = []
        count = 0
        for index, record in enumerate(records):
            count += 1
            try:
                kind = JournalEvent.model_validate(record).kind
                if kind == "step":
                    committed.add(StepEvent.model_validate(record).action_id)
                    withheld.clear()  # a committed step ends the run of withheld proposals before it
                elif kind == "action_started":
                    started.append(ActionStartedEvent.model_validate(record))
                elif kind == "action_result":
                    receipt = ActionResultEvent.model_validate(record)
                    receipts[receipt.action_id] = receipt  # a later receipt for one action replaces the earlier
                elif kind == "action_withheld":
                    withheld.append(ActionWithheldEvent.model_validate(record))
            except ValidationError as error:
                raise _malformed_event(index, source, error) from error
        return cls(
            count=count,
            committed_action_ids=committed,
            started=started,
            receipts=receipts,
            withheld_since_last_step=withheld,
        )

    @property
    def pending(self) -> list[ActionStartedEvent]:
        """Actions that were dispatched and never committed: each may or may not have happened."""
        committed = self.committed_action_ids
        return [event for event in self.started if event.action_id not in committed]


class RecordedOutcome(BaseModel):
    """What an action produced: the observation, its structured detail, and whether the environment
    called the episode finished.

    ``extra="forbid"`` and strict scalars, because this is the shape an operator asserts by hand and it
    has always been checked exactly this way: no key beyond these three, and a ``done`` of ``1`` or
    ``"true"`` is not a boolean. A journaled ``action_result`` is deliberately not held to it -- the
    environment wrote that one, and a resume replays it rather than trusting it.
    """

    model_config = ConfigDict(extra="forbid")

    result: StrictStr
    info: dict[str, Any]
    done: StrictBool


class OperatorReceipt(RootModel[dict[str, Any]]):
    """The JSON file an operator hands to ``stepbook recover --receipt``.

    Kept as the mapping it was read as: ``reconcile_receipt`` quotes it verbatim into the journal, so the
    file's key order survives the parse.
    """

    @model_validator(mode="after")
    def _records_an_outcome(self) -> Self:
        RecordedOutcome.model_validate(self.root)
        return self

    @property
    def fields(self) -> dict[str, Any]:
        """The receipt as supplied, for the ``action_result`` line that records it."""
        return self.root
