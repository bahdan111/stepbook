"""Queue operator messages through the runtime's existing inbox protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..run_pointer import ACTIVE, LAST_RUN, pointer_dir, read_pointer
from ..runtime.controls import inbox_lock


def cmd_control(args: argparse.Namespace) -> int:
    base = pointer_dir()
    run = Path(args.log_dir) if args.log_dir else read_pointer(base, ACTIVE) or read_pointer(base, LAST_RUN)
    if run is None or not run.is_dir():
        print("No run found; use --log-dir or launch a run first.", file=sys.stderr)
        return 2
    message = args.text if args.cmd == "say" else "pause"
    try:
        inbox = run / "inbox.txt"
        with inbox_lock(run), inbox.open("a+", encoding="utf-8", errors="replace") as stream:
            stream.seek(0)
            previous = stream.read()
            prefix = "\n" if previous and not previous.endswith("\n") else ""
            stream.write(prefix + message + "\n")
    except OSError as exc:
        print(f"Could not queue control for {run}: {exc}", file=sys.stderr)
        return 2
    print(f"Queued {message!r} for run {run}")
    return 0


def add_control_parsers(sub) -> None:
    for name, help_text in (("say", "queue an operator message"), ("pause", "request a pause")):
        parser = sub.add_parser(name, help=help_text, allow_abbrev=False)
        if name == "say":
            parser.add_argument("text")
        parser.add_argument("--log-dir", help="run directory (defaults to ACTIVE, then LAST_RUN)")
        parser.set_defaults(fn=cmd_control)
