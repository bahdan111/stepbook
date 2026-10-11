"""Data contract of the eval-adapter protocol, parsed once at each boundary.

Three JSON documents cross a process boundary in the eval loop, and each is parsed here instead of being
re-checked at every use site:

* ``AdapterConfig`` -- the ``eval.json`` the bundled adapter is pointed at (``stepbook/adapters/generic.py``).
* ``ResultRow`` -- one row of the results file the evaluated command writes (JSON or CSV). Its column names are
  configured, not fixed, so a row stays a mapping and answers typed questions about the configured columns.
* ``EvalReceipt`` -- the JSON object an adapter prints for ``run <label>``. The bundled adapter builds one;
  ``stepbook/envs/eval_loop.py`` reads whatever an arbitrary adapter printed.

Every verb also answers with an exit code -- ``ADAPTER_OK``, ``ADAPTER_REFUSED``, ``ADAPTER_CODE_CHANGE``,
anything else being an adapter failure -- and ``AdapterReply`` is one call's outcome as values, so no caller
decides anything by reading the sentences an adapter writes.

A leaf module on purpose: standard library and pydantic only, no ``stepbook`` import at all, so the adapter
package and the environment package can both import it with no cycle (tests/test_import_graph.py).

Receipt leniency is deliberate. An adapter is not ours: a receipt field of the wrong shape reads as absent,
which is exactly what every call site did by hand (``(payload.get("artifacts") or {}).get("run_dir")``,
``isinstance(identity, str)``). Rejecting the whole receipt instead would turn a sloppy third-party adapter
into a crashed run. Unknown receipt keys are kept (``extra="allow"``) because adapters carry their own
evidence and the journal quotes the receipt as received. Configuration models also allow unknown keys: the
raw ``eval.json`` is hashed verbatim into ``comparison_key`` and ``repeat_identity``, so a key this adapter
does not read still identifies the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    field_validator,
    model_validator,
)

# ---- the verbs' exit codes: whether the effect happened is a number, never a sentence --------------------

ADAPTER_OK = 0  # the verb did what it was asked; `run` printed its receipt, `status` its JSON
ADAPTER_REFUSED = 1  # the verb declined; stdout is the reason, and it is shown to the agent unread
# A refused ``run`` may answer with one JSON document instead of prose, declaring that it started no
# evaluation and spent nothing (``execution_started: false``, zero ``cost``); the bundled adapter's preflight
# refusals do, and usage accounting counts the zero only because the document declares it.
ADAPTER_CODE_CHANGE = 3  # `apply` only: applied, and this hypothesis needs a code change (stdout = instructions)
ADAPTER_EXIT_CODES = (ADAPTER_OK, ADAPTER_REFUSED, ADAPTER_CODE_CHANGE)


@dataclass(frozen=True)
class AdapterReply:
    """One adapter invocation's outcome.

    The exit code is the only thing a caller branches on. ``stdout`` is the adapter's own words: it is
    displayed, journaled and (for ``run`` and ``status``) parsed as JSON, but never scanned for tokens.
    ``error`` is the host's own sentence about a call that failed, or refused without saying why.
    """

    exit_code: int
    stdout: str = ""
    error: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == ADAPTER_OK

    @property
    def refused(self) -> bool:
        return self.exit_code == ADAPTER_REFUSED

    @property
    def code_change_required(self) -> bool:
        return self.exit_code == ADAPTER_CODE_CHANGE

    @property
    def failed(self) -> bool:
        """Neither done nor refused: a timeout, a crash, or an exit code the contract does not define."""
        return self.exit_code not in ADAPTER_EXIT_CODES

    @property
    def text(self) -> str:
        """What the agent reads: the adapter's own answer, or the host's sentence when there is none."""
        return self.stdout if self.stdout and not self.failed else self.error


def _exactly_true(value: Any) -> bool:
    """A declared claim: only a literal ``true`` makes it, which is how the adapter always read the key."""
    return value is True


NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
DeclaredFlag = Annotated[bool, BeforeValidator(_exactly_true)]
CostFieldName = Literal["tokens", "usd", "known_tokens", "known_usd"]
COST_FIELDS: tuple[CostFieldName, ...] = ("tokens", "usd", "known_tokens", "known_usd")


def _default_cost_fields() -> dict[CostFieldName, str]:
    """Each declared total is read from the key of its own name unless the configuration says otherwise."""
    fields: dict[CostFieldName, str] = {name: name for name in COST_FIELDS}
    return fields


# ---- eval.json: the bundled adapter's configuration ------------------------------------------------------


class PrimaryMetric(BaseModel):
    """The objective the loop optimises. Adapters name and orient it; the harness only compares like with like."""

    model_config = ConfigDict(extra="allow")

    name: str = "pass_rate"
    direction: str = "higher"


class MetricField(BaseModel):
    """A per-row numeric column averaged into ``deterministic_metrics``, with an optional declared limit.

    ``"latency_s": "lower"`` is shorthand for ``{"direction": "lower"}``; ``max``/``min`` turn the metric into a
    constraint rather than an objective.
    """

    model_config = ConfigDict(extra="allow")

    direction: str = "lower"
    max: float | None = None
    min: float | None = None

    @model_validator(mode="after")
    def _declared_limits_are_numbers(self) -> Self:
        """Refuse an explicitly null metric limit before running the evaluation command."""
        for name in ("max", "min"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"metric limit {name} must be a number")
        return self

    @property
    def limits(self) -> dict[str, float]:
        """Declared limits in the receipt's key order; a metric without limits contributes no entry."""
        return {name: value for name, value in (("max", self.max), ("min", self.min)) if value is not None}


class ConfiguredHypothesis(BaseModel):
    """One experiment the adapter can apply: a config edit (``set``) or an instruction for a coding step."""

    model_config = ConfigDict(extra="allow")

    kind: str = "config"
    rationale: str = ""
    set: dict[str, Any] | None = None
    code_change: str | None = None

    @model_validator(mode="after")
    def _code_hypotheses_say_what_to_change(self) -> Self:
        """The same rule ``register`` applies to a proposed code hypothesis, applied to eval.json too: without
        instructions, applying it would ask a coding step to implement nothing."""
        if self.kind == "code" and not self.code_change:
            raise ValueError("a code hypothesis needs 'code_change' instructions")
        return self


class CostArtifact(BaseModel):
    """Where the evaluator declares what a run cost, and which dotted paths in it hold the totals."""

    model_config = ConfigDict(extra="allow")

    path: str
    fields: dict[CostFieldName, NonEmptyStr] = Field(default_factory=_default_cost_fields)


class AdapterConfig(BaseModel):
    """``eval.json``: everything the bundled adapter needs to run someone else's evaluation.

    ``run_command`` and ``config_file`` are typed but not required here, because the adapter answers a missing
    or empty one with its own guidance instead of a validation error, and because ``status``/``decide`` still
    work on a configuration the preflight would reject.
    """

    model_config = ConfigDict(extra="allow")

    name: str = "eval"
    run_command: str = ""
    results_format: str | None = None
    case_id_field: str = "id"
    pass_field: str = "pass"
    pass_threshold: float = 1.0
    metric_fields: dict[str, MetricField] = Field(default_factory=dict)
    split_field: str | None = None
    holdout_value: str | int | float | bool | None = "holdout"
    protected_files: list[NonEmptyStr] = Field(default_factory=list)
    primary: PrimaryMetric = Field(default_factory=PrimaryMetric)
    config_file: str = ""
    hypotheses: dict[str, ConfiguredHypothesis] = Field(default_factory=dict)
    # Opaque sub-documents: CandidatePolicy validates its own spec, and a malformed cost_artifact must degrade
    # to a cost gap on a finished run, not refuse the evaluation. Both are hashed into the receipt verbatim.
    candidate_policy: Any = None
    cost_artifact: Any = None
    timeout_s: float = 7200.0
    primary_deterministic: DeclaredFlag = False

    @field_validator("primary", "hypotheses", mode="before")
    @classmethod
    def _null_reads_as_absent(cls, value: Any) -> Any:
        """An explicit ``null`` means "not configured", as the adapter's ``or {}`` always read it."""
        return {} if value is None else value

    @field_validator("metric_fields", mode="before")
    @classmethod
    def _expand_direction_shorthand(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {name: {"direction": spec} if isinstance(spec, str) else spec for name, spec in value.items()}
        return value


# ---- the results file the evaluated command writes -------------------------------------------------------


class ResultRow(RootModel[dict[str, Any]]):
    """One row of the results file. Column names come from ``eval.json``, so the row is a mapping that answers
    questions about the configured columns; interpreting one untyped cell happens here and nowhere else."""

    @classmethod
    def from_results_document(cls, document: Any) -> list[Self]:
        """Rows of a JSON results file: a sequence, or an object with ``cases``/``results``/``rows``. Entries
        that are not objects are skipped; a document that is not iterable is still the evaluator's error."""
        if isinstance(document, dict):
            document = document.get("cases") or document.get("results") or document.get("rows") or []
        return [cls(row) for row in document if isinstance(row, dict)]

    @classmethod
    def from_csv_rows(cls, rows: Any) -> list[Self]:
        """Rows of a CSV results file. ``csv.DictReader`` files surplus cells under the ``None`` restkey; no
        configured column name can be ``None``, so that entry is dropped rather than rejected."""
        return [cls({name: cell for name, cell in row.items() if name is not None}) for row in rows]

    def case_id(self, field: str, index: int) -> str:
        """The row's case identifier; rows without one are identified by position."""
        return str(self.root.get(field, index))

    def in_split(self, field: str, value: str) -> bool:
        """Whether the row's split column reads as the declared holdout value."""
        return str(self.root.get(field)) == value

    def metric(self, field: str) -> float | None:
        """The row's value for a declared metric column; an absent or empty cell is not a measurement."""
        value = self.root.get(field)
        return None if value in (None, "") else float(value)

    def verdict(self, field: str, threshold: float) -> Literal["pass", "fail", "error"]:
        """Pass, fail or error for a scored row. The pass column may be a boolean, a number or text, and a CSV
        file makes every cell text, so a numeric string is compared against the threshold too."""
        value = self.root.get(field)
        if value is None or value == "":
            return "error"
        if isinstance(value, bool):
            passed = value
        elif isinstance(value, (int, float)):
            passed = float(value) >= threshold
        else:
            text = str(value).strip().lower()
            passed = text in ("pass", "passed", "true", "1", "yes") or (
                text.replace(".", "", 1).isdigit() and float(text) >= threshold
            )
        return "pass" if passed else "fail"

    def holdout_passed(self, field: str, threshold: float) -> bool:
        """Holdout rows are scored pass or fail only; an unreadable value counts as a failure, never an error.
        Deliberately stricter than ``verdict`` about text: a holdout cell is not read as a number."""
        value = self.root.get(field)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) >= threshold
        return str(value).strip().lower() in ("pass", "passed", "true", "1", "yes")


# ---- the receipt one `run <label>` prints -------------------------------------------------------------------


def _kept_if(value: Any, kinds: tuple[type, ...]) -> Any:
    """A receipt field of the wrong shape reads as absent, which is what each call site checked by hand."""
    return value if isinstance(value, kinds) else None


def _only(*kinds: type) -> BeforeValidator:
    return BeforeValidator(partial(_kept_if, kinds=kinds))


ReceiptText = Annotated[str | None, _only(str)]
ReceiptNumber = Annotated[float | None, _only(int, float)]
ReceiptCount = Annotated[int | None, _only(int)]
ReceiptFlag = Annotated[bool | None, _only(bool)]
ReceiptObject = Annotated[dict[str, Any] | None, _only(dict)]
ReceiptList = Annotated[list[Any] | None, _only(list)]


class ReceiptPrimaryMetric(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: ReceiptText = None
    value: ReceiptNumber = None
    direction: ReceiptText = None


class ReceiptArtifacts(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_dir: ReceiptText = None


class ReceiptCandidatePolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    configured: ReceiptFlag = None
    checked_before_execution: ReceiptFlag = None
    spec_sha256: ReceiptText = None


class ReceiptProvenance(BaseModel):
    """What the run's identity covers. ``repeat_identity`` is what authorises an automatic repeat."""

    model_config = ConfigDict(extra="allow")

    config_sha256: ReceiptText = None
    repeat_identity: ReceiptText = None
    protected_files: ReceiptObject = None
    candidate_policy: Annotated[ReceiptCandidatePolicy | None, _only(dict, BaseModel)] = None
    coverage: ReceiptText = None


class ReceiptHoldout(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: ReceiptText = None
    value: ReceiptNumber = None
    n: ReceiptCount = None


class EvalReceipt(BaseModel):
    """The whole of a ``run <label>`` stdout, and a run's ``metrics.json``.

    Field order is the stdout and ``metrics.json`` key order: both are a contract, and ``as_document`` dumps
    only the fields the adapter actually set, so an optional key stays absent instead of appearing as null.
    The environment wraps the receipt in the ``Eval finished:`` observation for the agent and records the
    receipt itself in the step's ``info``, which is where replay and the reports read it from.
    """

    model_config = ConfigDict(extra="allow")

    run_id: ReceiptText = None
    config_label: ReceiptText = None
    cases: ReceiptObject = None
    primary_metric: Annotated[ReceiptPrimaryMetric | None, _only(dict, BaseModel)] = None
    deterministic_metrics: ReceiptObject = None
    metric_directions: ReceiptObject = None
    metric_limits: ReceiptObject = None
    errors: ReceiptList = None
    error_rate: ReceiptNumber = None
    valid: ReceiptFlag = None
    cost: ReceiptObject = None
    duration_s: ReceiptNumber = None
    artifacts: Annotated[ReceiptArtifacts | None, _only(dict, BaseModel)] = None
    applied: ReceiptText = None
    undecided_hypotheses: ReceiptList = None
    comparison_key: ReceiptText = None
    provenance: Annotated[ReceiptProvenance | None, _only(dict, BaseModel)] = None
    primary_deterministic: ReceiptFlag = None
    holdout_primary: Annotated[ReceiptHoldout | None, _only(dict, BaseModel)] = None
    invalid_reason: ReceiptText = None
    command_error: ReceiptText = None

    @property
    def run_dir(self) -> str | None:
        """The run's artifact directory, when the adapter reported one."""
        return self.artifacts.run_dir if self.artifacts else None

    @property
    def repeat_identity(self) -> str | None:
        """The adapter's identity for "the same configuration measured again"."""
        return self.provenance.repeat_identity if self.provenance else None

    def as_document(self) -> dict[str, Any]:
        """The receipt as the adapter's stdout and ``metrics.json`` carry it."""
        return self.model_dump(exclude_unset=True)
