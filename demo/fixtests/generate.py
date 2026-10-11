"""Generate a package of many DISTINCT small functions with one verified mutant each, for long-horizon
test-repair demo runs where a grep-and-sed pass cannot fix everything.

    python demo/fixtests/generate.py --modules 20 --out demo/fixtests_big --seed 0

Each module gets five functions drawn from parametrised templates with random constants, one test per
function, and one mutant per function chosen from several bug kinds. Every mutant is verified by executing
the mutated function against its test at generation time; mutants that do not break the test are discarded.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def templates(rng: random.Random, i: int) -> list[dict]:
    k = rng.randint(2, 9); c = rng.randint(3, 40); sep = rng.choice(["-", "_", ".", ":"]); lo, hi = rng.randint(-5, 5), rng.randint(10, 50)
    return [
        dict(name=f"scale_{i}", src=f"def scale_{i}(xs):\n    return [x * {k} for x in xs]\n",
             test=f"assert scale_{i}([1, 2]) == [{k}, {2*k}]",
             muts=[(f"x * {k}", f"x * {k+1}"), (f"x * {k}", f"x + {k}")]),
        dict(name=f"above_{i}", src=f"def above_{i}(xs):\n    return [x for x in xs if x > {c}]\n",
             test=f"assert above_{i}([{c-1}, {c}, {c+1}, {c+5}]) == [{c+1}, {c+5}]",
             muts=[(f"x > {c}", f"x >= {c}"), (f"x > {c}", f"x > {c+1}")]),
        dict(name=f"join_{i}", src=f"def join_{i}(parts):\n    return '{sep}'.join(p.strip() for p in parts if p.strip())\n",
             test=f"assert join_{i}([' a', '', 'b ']) == 'a{sep}b'",
             muts=[("if p.strip()", "if p"), ("p.strip() for", "p for")]),
        dict(name=f"clip_{i}", src=f"def clip_{i}(x):\n    return max({lo}, min({hi}, x))\n",
             test=f"assert clip_{i}({hi+3}) == {hi} and clip_{i}({lo-3}) == {lo} and clip_{i}({lo+1}) == {lo+1}",
             muts=[(f"max({lo}, min({hi}, x))", f"min({lo}, max({hi}, x))"), (f"min({hi}, x)", f"min({hi+1}, x)")]),
        dict(name=f"window_{i}", src=f"def window_{i}(xs):\n    return [xs[j:j + {k}] for j in range(len(xs) - {k} + 1)]\n",
             test=f"assert window_{i}(list(range({k+2}))) == [list(range(j, j + {k})) for j in range(3)]",
             muts=[(f"range(len(xs) - {k} + 1)", f"range(len(xs) - {k})"), (f"xs[j:j + {k}]", f"xs[j:j + {k-1}]")]),
        dict(name=f"count_ge_{i}", src=f"def count_ge_{i}(xs):\n    return sum(1 for x in xs if x >= {c})\n",
             test=f"assert count_ge_{i}([{c-1}, {c}, {c+1}]) == 2",
             muts=[(f"x >= {c}", f"x > {c}"), ("sum(1 for", "sum(0 for")]),
        dict(name=f"pad_{i}", src=f"def pad_{i}(s):\n    return s.rjust({k}, '{sep}')\n",
             test=f"assert pad_{i}('ab') == '{sep * (k-2)}ab'",
             muts=[("rjust", "ljust"), (f"rjust({k},", f"rjust({k+1},")]),
        dict(name=f"safe_div_{i}", src=f"def safe_div_{i}(a, b):\n    if b == 0:\n        return {c}\n    return a / b\n",
             test=f"assert safe_div_{i}(6, 3) == 2 and safe_div_{i}(1, 0) == {c}",
             muts=[(f"return {c}", f"return {c+1}"), ("return a / b", "return b / a")]),
        dict(name=f"dedupe_{i}", src=f"def dedupe_{i}(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
             test=f"assert dedupe_{i}([3, 1, 3, 2, 1]) == [3, 1, 2]",
             muts=[("if x not in seen:", "if x in seen:"), ("            seen.add(x)\n", "")]),
        dict(name=f"pct_{i}", src=f"def pct_{i}(part, whole):\n    if whole == 0:\n        raise ValueError('whole')\n    return round(100 * part / whole, {k % 3 + 1})\n",
             test=f"assert pct_{i}(1, 3) == round(100 / 3, {k % 3 + 1})",
             muts=[(f"round(100 * part / whole, {k % 3 + 1})", f"round(100 * part / whole, {k % 3 + 2})"), ("100 * part / whole", "100 * whole / part")]),
    ]


def verified_mutant(spec: dict, rng: random.Random) -> tuple[str, str] | None:
    muts = list(spec["muts"]); rng.shuffle(muts)
    muts.sort(key=lambda m: 0 if any(ch.isdigit() for ch in m[0]) else 1)  # prefer constant-bearing (unique) mutants
    for search, replace in muts:
        if search not in spec["src"]:
            continue
        mutated = spec["src"].replace(search, replace, 1)
        ns: dict = {}
        try:
            exec(mutated, ns); exec(spec["test"], ns)
        except AssertionError:
            return search, replace
        except Exception:
            return search, replace  # raising where the test expected a value also counts as broken
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", type=int, default=20)
    ap.add_argument("--out", default="demo/fixtests_big")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "pkg").mkdir(parents=True, exist_ok=True); (out / "tests").mkdir(parents=True, exist_ok=True)
    (out / "pkg" / "__init__.py").write_text(""); (out / "tests" / "__init__.py").write_text("")
    mutants = []
    for m in range(args.modules):
        mod = f"m{m:02d}"
        src_lines, test_lines = ['"""Generated module."""', ""], [f"from pkg.{mod} import *", ""]
        pool = templates(rng, m * 10)
        rng.shuffle(pool)
        for j, spec in enumerate(pool[:5]):
            spec = dict(spec)
            # re-instantiate with a unique index so names never collide across modules
            spec_i = templates(random.Random(rng.random()), m * 10 + j)[[t["name"].split("_")[0] for t in pool].index(spec["name"].split("_")[0])]
            ns: dict = {}; exec(spec_i["src"], ns); exec(spec_i["test"], ns)  # pristine must pass
            mut = verified_mutant(spec_i, rng)
            assert mut, f"no breaking mutant for {spec_i['name']}"
            src_lines.append(spec_i["src"])
            test_lines.append(f"def test_{spec_i['name']}():\n    {spec_i['test']}\n")
            mutants.append({"file": f"pkg/{mod}.py", "search": mut[0], "replace": mut[1], "test": f"tests/test_{mod}.py::test_{spec_i['name']}"})
        (out / "pkg" / f"{mod}.py").write_text("\n".join(src_lines))
        (out / "tests" / f"test_{mod}.py").write_text("\n".join(test_lines))
    (out / "mutants.json").write_text(json.dumps(mutants, indent=1))
    print(f"wrote {args.modules} modules, {len(mutants)} verified mutants to {out}")


if __name__ == "__main__":
    main()
