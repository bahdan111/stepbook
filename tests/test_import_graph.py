"""The package's own import graph: acyclic, and every module loads in a fresh interpreter.

An import cycle is why modules grow imports inside functions; a module that only loads because
another one was imported first is the same defect seen from the other side.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "stepbook"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _modules() -> dict[str, Path]:
    return {_module_name(path): path for path in sorted(PACKAGE.rglob("*.py"))}


MODULES = _modules()


def _package_of(module: str) -> str:
    return module if (MODULES[module].name == "__init__.py") else module.rsplit(".", 1)[0]


def _target(module: str, known: set[str], candidate: str) -> str | None:
    """Map an imported dotted name onto the module that owns it, or None when it is not ours."""
    while candidate:
        if candidate in known and candidate != module:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _imports(module: str, known: set[str]) -> set[str]:
    """Every `stepbook.*` module this one imports, at module level or inside a function."""
    tree = ast.parse(MODULES[module].read_text(encoding="utf-8"), filename=str(MODULES[module]))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = _package_of(module)
                for _ in range(node.level - 1):
                    package = package.rpartition(".")[0]
                base = f"{package}.{base}" if base else package
            found.update(f"{base}.{alias.name}" for alias in node.names)
            if node.module and base != _package_of(module):
                found.add(base)
            # `from . import x` and `from stepbook import x` both name only x: whatever the
            # spelling, a module does not depend on the package it is itself a member of.
    return {t for t in (_target(module, known, name) for name in sorted(found)) if t}


def _graph() -> dict[str, set[str]]:
    known = set(MODULES)
    return {module: _imports(module, known) for module in sorted(known)}


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Depth-first search over sorted edges, so a reported cycle is the same on every run."""
    state: dict[str, int] = {}
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        path.append(node)
        for neighbour in sorted(graph[node]):
            if state.get(neighbour) == 1:
                return path[path.index(neighbour) :] + [neighbour]
            if not state.get(neighbour):
                cycle = visit(neighbour)
                if cycle:
                    return cycle
        path.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if not state.get(node):
            cycle = visit(node)
            if cycle:
                return cycle
    return None


def test_package_import_graph_is_acyclic():
    cycle = _find_cycle(_graph())
    assert cycle is None, "import cycle: " + " -> ".join(cycle or [])


@pytest.mark.parametrize("module", sorted(MODULES))
def test_module_imports_standalone(module):
    """A fresh interpreter importing only this module: no cycle, no implicit import order."""
    done = subprocess.run(
        [sys.executable, "-c", f"import {module}"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, f"import {module} failed:\n{done.stderr}"
