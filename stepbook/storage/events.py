"""Append-only JSONL event log. Never enters the prompt; it is the audit trail and the archive
for later recall. Everything the model produced (including discarded reasoning) is kept here."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..contracts.journal import Journal


class EventLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **data: Any) -> None:
        # A killed append may leave a partial final JSON line. Preserve that exact
        # tail before truncating it; otherwise every later event would be hidden
        # behind the unreadable line. A complete final record just needs a newline.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.seek(0)
                    raw = fh.read()
                    boundary = raw.rfind(b"\n") + 1
                    tail = raw[boundary:]
                    try:
                        json.loads(tail)
                    except (ValueError, UnicodeError):
                        saved = self.path.with_name(self.path.name + f".torn-{uuid4().hex}")
                        with saved.open("xb") as backup:
                            backup.write(tail)
                            backup.flush()
                            os.fsync(backup.fileno())
                        fh.truncate(boundary)
                    else:
                        fh.seek(0, os.SEEK_END)
                        fh.write(b"\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        record = {"ts": time.time(), "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def journal(self) -> Journal:
        """The same records ``read`` returns, parsed into the events a resume reasons about."""
        return Journal.from_records(self.read(), self.path)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("rb") as fh:
            for raw in fh:
                line = raw
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeError) as err:
                    if raw.endswith(b"\n"):
                        raise RuntimeError(f"corrupt completed event in {self.path}; repair explicitly") from err
                    break  # incomplete final line; append preserves it before repair
        return out
