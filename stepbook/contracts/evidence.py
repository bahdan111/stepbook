"""Data contract of the evidence store's documents, parsed once where each one arrives.

Every JSON document that reaches ``stepbook/storage/evidence.py`` from outside this process is parsed here, once,
instead of being re-checked at every use site:

* ``ReadRequest`` -- ``read {...}``: which source to open, and which one-based range of it.
* ``FindingDocument`` -- ``record {...}``: a conclusion and the evidence it rests on.
* ``CompletionDocument`` -- ``finish {...}``: the same document, plus the receipt IDs it cites.
* ``Proposal`` / ``EvidencedProposal`` -- ``propose {...}``: a hypothesis, with a saved or inline finding.
* ``StoredRecord`` -- a record read back from disk, whose body is whatever the host wrote when it was saved.

A leaf module on purpose: standard library and pydantic only, no ``stepbook`` import at all, so the store
and the eval environment can both import it with no cycle (tests/test_import_graph.py).

Three deliberate choices:

``extra="forbid"`` on the documents the agent writes. An unknown key is a misunderstanding of the action, not
extra evidence: the store's records are content-addressed and immutable, so a typo'd key would be hashed into
a record that can never be corrected. ``Proposal`` allows unknown keys, because a proposal carries the
adapter's own hypothesis fields (``set``, ``code_change``, ...) and is forwarded to it verbatim.

``strict=True`` everywhere. Every check these models replace was an ``isinstance``/``type(...) is`` test, so
lax coercion (``"2"`` read as ``2``, a tuple read as a list, ``True`` read as ``1``) would accept documents
the old code refused.

The wording of a refusal is the module's own. ``EvidenceError`` carries the sentence the store has always
written for that fault -- one line, no pydantic report, no documentation URL, no echo of the input, which a
model's own prose may have reached. ``_faults`` below reproduces the order the checks used to run in, so the
same document is refused for the same stated reason as before.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, ClassVar, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    StrictInt,
    StrictStr,
    ValidationError,
)
from pydantic_core import ErrorDetails

# A saved record's id, assigned by the host: the kind, then a prefix of the body digest. Also the pattern that
# tells a citation of a saved record from a file path in a finding's evidence.
SAVED_RECORD_ID = re.compile(r"[a-z]+-[0-9a-f]{24}")
# Native agents commonly cite ``file.py:31-33``; read that suffix as a locator when no such file exists.
CITATION_SUFFIX = re.compile(r"(.+):([1-9][0-9]*)(?:-([1-9][0-9]*))?")
_HYPOTHESIS_ID = re.compile(r"[A-Za-z0-9_-]+")

NOT_AN_OBJECT = "record needs a JSON object"
FINDING_KEYS = (
    "record needs summary, evidence (saved ids, file paths or read locators), uncertainty, next_action; "
    "optional category/run_ids/supersedes"
)
EVIDENCE_REFERENCES = "evidence must reference saved ids or source read requests"
RUN_IDS = "finish needs unique original receipt IDs (empty for no cited measurements)"
FINDING_REFERENCE = "proposal needs a saved finding id or inline finding object"
READ_REQUEST = "read needs a JSON object with path, and optional start/end/sheet"
READ_OPTION = "unknown read option; use path, start, end, sheet"
SOURCE_RANGE = "start/end must select a positive one-based range"


class EvidenceError(ValueError):
    pass


class _Fault(ValueError):
    """A validator's own final wording, recognised by the fault readers and surfaced unchanged."""


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain nonempty text")
    return value


def _hypothesis_id(value: str) -> str:
    if not _HYPOTHESIS_ID.fullmatch(value):
        raise _Fault("hypothesis id must contain letters, digits, underscores or hyphens")
    return value


def _unique(value: list[str]) -> list[str]:
    if len(set(value)) != len(value):
        raise ValueError("must be unique")
    return value


NonBlankText = Annotated[StrictStr, AfterValidator(_nonblank)]
HypothesisId = Annotated[NonBlankText, AfterValidator(_hypothesis_id)]
ReceiptIds = Annotated[list[NonBlankText], AfterValidator(_unique)]


def _written_fault(problem: ErrorDetails) -> str | None:
    """The sentence a validator wrote for itself, when this problem came from one."""
    cause = (problem.get("ctx") or {}).get("error")
    return str(cause) if isinstance(cause, _Fault) else None


class ReadRequest(BaseModel):
    """``read {...}``: one source, and the one-based range of it to capture.

    ``start``/``end`` are strict integers because the store's range check has always been ``type(x) is int``
    (a bool is not a line number). Whether a range is *usable* still depends on what the path turns out to be,
    so the ordering rules -- and the separate sentence a directory listing writes -- stay in the store.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    path: StrictStr
    start: StrictInt = 1
    end: StrictInt | None = None  # absent, or an explicit null, which each read resolves its own way
    sheet: StrictStr | None = None

    @classmethod
    def parse(cls, request: dict[str, Any] | ReadRequest) -> Self:
        """The request as written, or an already-parsed one passed straight through."""
        try:
            return cls.model_validate(request)
        except ValidationError as error:
            raise EvidenceError(_read_fault(error)) from error

    @property
    def written(self) -> frozenset[str]:
        """The options the request actually named, as opposed to the ones that defaulted."""
        return frozenset(self.model_fields_set)


def _read_fault(error: ValidationError) -> str:
    """Why a read request is refused, in the order the store used to check it: shape, options, then range."""
    problems = error.errors(include_url=False, include_input=False)
    if any(problem["loc"][:1] in ((), ("path",), ("sheet",)) for problem in problems):
        return READ_REQUEST
    if any(problem["type"] == "extra_forbidden" for problem in problems):
        return READ_OPTION
    return SOURCE_RANGE


class FindingDocument(BaseModel):
    """``record {...}``: structure only. Evidence existence, authority and capture remain in ``record()``.

    ``evidence`` entries are saved record ids, file paths, or read requests, so a request written inline is
    parsed here by the same model the ``read`` action uses. ``category``, ``run_ids`` and ``supersedes`` are
    declared so an unknown key is still refused, but stay opaque: the store has never constrained them, and
    records holding any of the three are already saved on disk.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    summary: NonBlankText
    evidence: list[StrictStr | ReadRequest] = Field(min_length=1)
    uncertainty: NonBlankText
    next_action: NonBlankText
    category: Any = None
    run_ids: Any = None
    supersedes: Any = None

    @classmethod
    def parse(cls, document: Any) -> Self:
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise EvidenceError(_finding_fault(error.errors(include_url=False, include_input=False))) from error


class CompletionDocument(FindingDocument):
    """``finish {...}``: a finding that also names every original receipt it draws a measurement from."""

    run_ids: ReceiptIds


def _inside_request(where: tuple[int | str, ...]) -> bool:
    """Is this problem located inside an evidence entry's inline read request, rather than on the finding?

    The union arm is named, not counted, so another arm beside ``ReadRequest`` would not be read as one; the
    length still matters, because the entry's own fault (``("evidence", i, "ReadRequest")``, an entry that is
    no object at all) is the finding's to answer.
    """
    return where[:1] == ("evidence",) and where[2:3] == ("ReadRequest",) and len(where) > 3


def _request_fault(located: list[tuple[ErrorDetails, tuple[int | str, ...]]]) -> str:
    """Why an evidence entry's inline read request is refused, in the order ``read`` checks it.

    The last bucket excepted: a start/end of the wrong type ends in ``EVIDENCE_REFERENCES`` where ``read``
    answers ``SOURCE_RANGE``. A well-typed but impossible range is no parse fault and never arrives here at
    all -- the read that captures the source raises it, in the same words a top-level ``read`` would.
    """
    inside = [(where[1], where[3:], problem["type"]) for problem, where in located if _inside_request(where)]
    first = [(option, kind) for entry, option, kind in inside if entry == inside[0][0]]
    if any(option[:1] in (("path",), ("sheet",)) for option, _ in first):
        return READ_REQUEST
    if any(kind == "extra_forbidden" for _, kind in first):
        return READ_OPTION
    return EVIDENCE_REFERENCES


def _finding_fault(problems: list[ErrorDetails], depth: int = 0) -> str:
    """Why a finding is refused, in the order the store used to check it.

    ``depth`` skips the location prefix of a finding written inside another document. A problem inside an
    evidence entry's inline read request is that request's fault, not the record's keys: the store reached
    those checks only at capture time, after every check here, and still answers them with ``read``'s words.
    """
    located = [(problem, problem["loc"][depth:]) for problem in problems]
    document = [(problem, where) for problem, where in located if not _inside_request(where)]
    if any(not where for _, where in document):
        return NOT_AN_OBJECT
    if any(
        problem["type"] in ("missing", "extra_forbidden") and where[:1] != ("run_ids",) for problem, where in document
    ):
        return FINDING_KEYS
    for key in ("summary", "uncertainty", "next_action"):
        if any(where[:1] == (key,) for _, where in document):
            return f"{key} must contain nonempty text"
    # An entry that is neither a string nor an object fails the union's ``str`` arm *and* its model arm; an
    # entry written as an object fails only the ``str`` arm, and speaks for itself through _request_fault.
    if any(where[:1] == ("evidence",) and where[-1] != "str" for _, where in document):
        return EVIDENCE_REFERENCES
    if any(_inside_request(where) for _, where in located) and not any(
        where[:1] == ("run_ids",) for _, where in document
    ):
        return _request_fault(located)
    return RUN_IDS


class Proposal(BaseModel):
    """``propose {...}`` without a finding: the hypothesis the adapter is asked to register.

    Unknown keys are kept (``extra="allow"``): the adapter's own hypothesis fields travel in this document,
    which is forwarded to it verbatim, and ``document`` is the object as written so neither the adapter nor a
    saved proposal record sees a re-serialised copy.
    """

    model_config = ConfigDict(extra="allow", strict=True)

    NAMED: ClassVar[tuple[str, ...]] = ("id", "kind", "rationale")

    id: HypothesisId
    kind: NonBlankText
    rationale: NonBlankText

    _document: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def parse(cls, raw: str | bytes) -> Self:
        """Parse the action's JSON argument. Format only: not registration, semantics or evidence."""
        document = json.loads(raw)
        try:
            parsed = cls.model_validate(document)
        except ValidationError as error:
            raise EvidenceError(_proposal_fault(error, cls.NAMED)) from error
        parsed._document = document
        return parsed

    @property
    def document(self) -> dict[str, Any]:
        """The proposal as written, which is what the adapter and the proposal record receive."""
        return self._document


class EvidencedProposal(Proposal):
    """``propose {...}``: the form the eval loop registers, which must stand on a finding."""

    NAMED: ClassVar[tuple[str, ...]] = ("id", "kind", "rationale", "prediction", "test")

    prediction: NonBlankText
    test: NonBlankText
    finding: FindingDocument | NonBlankText

    @property
    def inline_finding(self) -> FindingDocument | None:
        """The finding written inline, or None when the proposal cites one already saved."""
        return self.finding if isinstance(self.finding, FindingDocument) else None


def _proposal_fault(error: ValidationError, named: tuple[str, ...]) -> str:
    """Why a proposal is refused, in the order the store used to check it: the named fields, the id, the finding."""
    problems = error.errors(include_url=False, include_input=False)
    if any(not problem["loc"] or (problem["loc"][0] in named and not _written_fault(problem)) for problem in problems):
        return "propose needs nonempty " + ", ".join(named)
    for problem in problems:
        written = _written_fault(problem)
        if written:
            return written
    inline = [
        problem
        for problem in problems
        if problem["loc"][:2] == ("finding", FindingDocument.__name__) and len(problem["loc"]) > 2
    ]
    return _finding_fault(inline, depth=2) if inline else FINDING_REFERENCE


class StoredRecord(RootModel[dict[str, Any]]):
    """A record read back from the store: the host-assigned ``id`` and ``kind``, over a body of any shape.

    The body is not re-interpreted. Records of every kind this harness has ever written are on disk and stay
    readable, and the bytes are the record's identity: ``recall`` returns the object as stored, key order
    included, and only checks that it still hashes to the id it is filed under.
    """

    model_config = ConfigDict(strict=True)

    @classmethod
    def parse(cls, document: Any) -> Self:
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise EvidenceError("research record is not an object") from error

    @property
    def document(self) -> dict[str, Any]:
        return self.root

    @property
    def declared_id(self) -> Any:
        return self.root.get("id")

    @property
    def kind(self) -> str:
        return str(self.root.get("kind"))

    def canonical_body(self) -> str:
        """The record without its id, in the encoding the id is a digest of."""
        return json.dumps({k: v for k, v in self.root.items() if k != "id"}, sort_keys=True, ensure_ascii=False)
