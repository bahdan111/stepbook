"""Deterministic mutant injection for the test-repair demo. Each mutant is a (search, replace) pair on
pkg/__init__.py; injecting N of them with a seed breaks a known subset of the 30 tests."""
from __future__ import annotations

import random
from pathlib import Path

MUTANTS = [
    ("return max(lo, min(hi, x))", "return min(lo, max(hi, x))"),
    ("return sum(xs) / len(xs)", "return sum(xs) / (len(xs) + 1)"),
    ("return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2", "return s[mid] if n % 2 else (s[mid] + s[mid + 1]) / 2"),
    ("for i in range(len(xs) - k + 1)]", "for i in range(len(xs) - k)]"),
    ("if not line or line.startswith(\"#\"):", "if not line:"),
    ("return slug.strip(\"-\")", "return slug"),
    ("for i in range(0, len(xs), size)]", "for i in range(0, len(xs) - 1, size)]"),
    ("return s[lo] + (s[hi] - s[lo]) * (idx - lo)", "return s[lo]"),
    ("if amount > self.balance:", "if amount >= self.balance:"),
    ("a, b = b, a + b", "a, b = b, a + b + 1"),
    ("return cleaned == cleaned[::-1]", "return text == text[::-1]"),
    ("(900, \"CM\"), (500, \"D\")", "(900, \"CM\"), (500, \"C\")"),
]


def inject(repo: Path, n: int, seed: int) -> list[int]:
    src = repo / "pkg" / "__init__.py"
    text = src.read_text()
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(MUTANTS)), n))
    for i in chosen:
        a, b = MUTANTS[i]
        assert a in text, f"mutant {i} anchor missing"
        text = text.replace(a, b, 1)
    src.write_text(text)
    return chosen
