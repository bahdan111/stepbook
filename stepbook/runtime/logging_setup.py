"""The package logger, and the per-step trace that is its only record today.

At import the logger carries a ``NullHandler`` and propagation turned off, and nothing more. Propagation is off
because an adapter subprocess's error stream is read back as the model's observation -
``envs/eval_adapter.py::_call`` feeds its last 1500 bytes into the observation - so a record must never climb to
a handler the consumer did not point at this package. Nothing here calls ``basicConfig`` or touches the root
logger. A consumer that wants the records attaches a handler to this package's own logger, and a verbose run
then sends it the per-step trace.

The stdout handler that makes the trace visible is built when it is switched on and removed again, never at import:
the trace goes to stdout, and pytest swaps ``sys.stdout`` per test,
so the stream has to be read at call time. The formatter emits the message alone - no level, no timestamp - because
the bytes on stdout are the contract.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("stepbook")
logger.addHandler(logging.NullHandler())
# Never climb to a consumer's root handler: an adapter subprocess's stderr is read back as the model's observation.
logger.propagate = False

_installed = False  # one stdout handler per process: the CLI's, or the one a direct API caller's trace puts up


@contextmanager
def trace_to_stdout(verbose: bool) -> Iterator[None]:
    """Show the package's INFO records on the current stdout for the duration of one call, then undo it."""
    global _installed
    if not verbose or _installed:
        yield
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    _installed = True
    try:
        yield
    finally:
        _installed = False
        logger.removeHandler(handler)
        logger.setLevel(level)


def trace_step(verbose: bool, t: int, action: str, observation: str, input_tokens: int | None) -> None:
    """One per-step line, identical in both runtimes: what the model did, and what the environment answered.

    ``verbose`` stays the switch the runtimes already own, so configuring the package logger elsewhere cannot turn
    a quiet run into a talking one; a direct API caller (the bench runners) gets its own stdout handler for the
    record when the CLI has not already put one up.
    """
    if not verbose:
        return
    with trace_to_stdout(True):
        tokens = "unknown" if input_tokens is None else str(input_tokens)
        logger.info(f"[{t:3d}] {action!r} -> {observation[:80]!r}  (prompt {tokens} tok)")
