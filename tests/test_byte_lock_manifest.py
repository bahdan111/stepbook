"""Byte lock: the golden and fixture files that pin prompts and journals are
listed with their SHA-256 in tests/BYTE-LOCK.sha256. Regenerating any of them means updating the manifest in the same
change, so a byte change is visible in review instead of hidden behind a test that compares against the new file.

    shasum -a 256 -c tests/BYTE-LOCK.sha256            # check by hand
    find tests/golden tests/fixtures/journal_bytes tests/fixtures/replay -type f | sort | xargs shasum -a 256 \
        > tests/BYTE-LOCK.sha256                       # after a deliberate, reviewed regeneration
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "BYTE-LOCK.sha256"
LOCKED_DIRS = ("tests/golden", "tests/fixtures/journal_bytes", "tests/fixtures/replay")


def _manifest() -> dict[str, str]:
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            entries[rel] = digest
    return entries


def test_every_locked_file_is_listed():
    on_disk = sorted(str(p.relative_to(ROOT)) for d in LOCKED_DIRS for p in (ROOT / d).rglob("*") if p.is_file())
    assert on_disk == sorted(_manifest()), "a locked file was added or removed without updating tests/BYTE-LOCK.sha256"


def test_locked_files_match_manifest():
    changed = [
        rel for rel, digest in _manifest().items() if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest
    ]
    assert not changed, f"locked bytes changed; review, then regenerate the manifest deliberately: {changed}"
