"""Architecture rules, checked on every test run.

These read the source's import statements (no code is executed), so a change
that breaks the layering fails CI instead of being found in review:

* no import cycles between top-level packages;
* ``core`` is the bottom layer and imports no other Groundwire package;
* ``domain`` builds only on ``core``;
* infrastructure (``observability``) never imports policy (``resilience``);
* nothing below the API imports the web layer (``api``, ``main``);
* only the composition root (``container``) and the app (``main``) may
  import ``container``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "groundwire"
# Wiring and entry points depend on everything by design.
ENTRY_POINTS = frozenset({"container", "main", "__main__"})


def _package_of(path: Path) -> str:
    """Top-level Groundwire package a source file belongs to."""
    relative = path.relative_to(ROOT)
    return relative.parts[0].removesuffix(".py")


def _imports(path: Path) -> Iterator[str]:
    """Top-level Groundwire packages a file imports (absolute imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "groundwire" and len(parts) > 1:
                yield parts[1]


def _graph() -> dict[str, set[str]]:
    """Package -> packages it imports (self-imports removed)."""
    graph: dict[str, set[str]] = {}
    for path in ROOT.rglob("*.py"):
        package = _package_of(path)
        graph.setdefault(package, set()).update(
            dep for dep in _imports(path) if dep != package)
    return graph


GRAPH = _graph()


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """Return one import cycle as a path (``[a, b, a]``), or ``None``."""
    visiting: list[str] = []
    done: set[str] = set()

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            return [*visiting[visiting.index(node):], node]
        if node in done:
            return None
        visiting.append(node)
        for dep in sorted(graph.get(node, ())):
            cycle = visit(dep)
            if cycle:
                return cycle
        visiting.pop()
        done.add(node)
        return None

    for start in sorted(graph):
        cycle = visit(start)
        if cycle:
            return cycle
    return None


def test_the_graph_was_actually_read() -> None:
    assert {"core", "services", "routing", "api"} <= GRAPH.keys()
    assert "core" in GRAPH["services"]


def test_there_are_no_import_cycles_between_packages() -> None:
    layered = {pkg: deps - ENTRY_POINTS for pkg, deps in GRAPH.items()
               if pkg not in ENTRY_POINTS}
    assert _find_cycle(layered) is None


@pytest.mark.parametrize(("package", "allowed"), [
    ("core", set()),
    ("domain", {"core"}),
])
def test_bottom_layers_depend_only_on_what_is_below_them(package: str,
                                                         allowed: set[str]) -> None:
    assert GRAPH[package] <= allowed, GRAPH[package] - allowed


@pytest.mark.parametrize(("package", "forbidden"), [
    ("observability", "resilience"),
    ("guardrails", "services"),
    ("providers", "routing"),
])
def test_lower_layers_never_import_higher_ones(package: str, forbidden: str) -> None:
    assert forbidden not in GRAPH[package]


def test_only_the_web_layer_imports_the_web_layer() -> None:
    offenders = {pkg for pkg, deps in GRAPH.items()
                 if deps & {"api", "main"} and pkg not in {"api", "main", "__main__"}}
    assert offenders == set()


def test_only_entry_points_import_the_composition_root() -> None:
    offenders = {pkg for pkg, deps in GRAPH.items()
                 if "container" in deps and pkg not in ENTRY_POINTS | {"api"}}
    assert offenders == set()


def test_the_cycle_finder_finds_cycles() -> None:
    assert _find_cycle({"a": {"b"}, "b": {"a"}}) == ["a", "b", "a"]
    assert _find_cycle({"a": {"b"}, "b": set()}) is None
