"""A small, deliberately plain library used as the target of the test-repair demo.
Mutants are injected into these functions; the agent must make the test suite green again."""
from __future__ import annotations

from dataclasses import dataclass


def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError("lo > hi")
    return max(lo, min(hi, x))


def mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("empty")
    return sum(xs) / len(xs)


def median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("empty")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def moving_average(xs: list[float], k: int) -> list[float]:
    if k <= 0:
        raise ValueError("k must be positive")
    return [sum(xs[i : i + k]) / k for i in range(len(xs) - k + 1)]


def parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def slugify(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def chunk(xs: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [xs[i : i + size] for i in range(0, len(xs), size)]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        raise ValueError("empty")
    if not 0 <= p <= 100:
        raise ValueError("p out of range")
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


@dataclass
class Account:
    balance: float = 0.0

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount


def fib(n: int) -> int:
    if n < 0:
        raise ValueError("negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def is_palindrome(text: str) -> bool:
    cleaned = "".join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]


def roman(n: int) -> str:
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
             (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, sym in table:
        while n >= value:
            out += sym
            n -= value
    return out
