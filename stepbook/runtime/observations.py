"""Observation shaping: keep the prompt bounded when a tool emits a huge output.

Large observations are archived in full to ``<log_dir>/obs/<step>.txt`` and replaced in the prompt
by head + tail plus a pointer, so the model can save the pointer cheaply and read the full text
later if it turns out to matter (restorable compression).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ObservationShaper:
    # Smaller previews hide source needed for edits, and agents may ignore archive pointers.
    max_chars: int = 12000
    head_chars: int = 8000
    tail_chars: int = 3000
    archive_dir: Path | None = None

    def shape(self, observation: str, step: int) -> tuple[str, dict]:
        """Return (text_for_prompt, info). info['archived'] is the path when truncation happened."""
        if len(observation) <= self.max_chars:
            return observation, {"truncated": False}
        pointer = None
        if self.archive_dir is not None:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            path = self.archive_dir / f"{step:05d}.txt"
            path.write_text(observation, encoding="utf-8")
            pointer = str(path)
        omitted = len(observation) - self.head_chars - self.tail_chars
        marker = (
            f"\n[... {omitted} characters omitted"
            + (
                f"; the full output is archived at {pointer} - run `cat {pointer}` (or `sed -n` a range) "
                "when the omitted part matters"
                if pointer
                else ""
            )
            + " ...]\n"
        )
        shaped = observation[: self.head_chars] + marker + observation[-self.tail_chars :]
        return shaped, {"truncated": True, "archived": pointer, "original_chars": len(observation)}
