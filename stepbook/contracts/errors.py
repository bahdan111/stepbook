"""Named executor failures and operator-correctable launch refusals."""

from __future__ import annotations


class ExecutorError(RuntimeError):
    def __init__(self, message, *, envelope=None, retryable=True):
        super().__init__(message)
        self.envelope = envelope
        self.retryable = retryable


class RateLimited(ExecutorError):
    """The provider reported HTTP 429 in its own receipt. Retrying is pointless, so the runtime records the
    failed call and stops; the run stays resumable from its event log whenever the operator returns."""


class ExistingRun(RuntimeError):
    """A new run would overwrite an existing journal."""


class UnreceiptedAction(RuntimeError):
    """An action's external outcome must be reconciled before resuming."""


class CompletionPolicyChanged(ValueError):
    """Resume must preserve the journal's completion policy."""


class AutomaticMeasurementRepeats(ValueError):
    """A legacy journal with automatic repeats cannot be resumed."""


class LegacyAdapterJournal(RuntimeError):
    """A journal written before the structured adapter journal cannot be resumed on the adapter route."""


class MissingSkillFolder(FileNotFoundError):
    """The requested skill folder does not exist."""


class MissingExecutorBinary(FileNotFoundError):
    """The configured executor executable could not be launched."""
