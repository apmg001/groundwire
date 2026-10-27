"""Layering rules, read from the source's imports on every test run.

    core ─► domain ─► statements / insights / memory ─► services ─► agent ─► api
                       adapters implement domain ports      container wires everything

* each package may import only the packages listed for it (no upward imports);
* there are no import cycles between packages;
* web, database and driver libraries stay in the layers that own them;
* the app never imports Groundwire's code: it talks to the gateway over HTTP only.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "money_agent"

ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "domain": {"core"},
    "statements": {"core", "domain"},
    "insights": {"domain"},
    "memory": {"core", "domain"},
    "services": {"core", "domain", "statements", "insights", "memory"},
    "agent": {"core", "domain", "insights", "memory", "services"},
    "adapters": {"core", "domain", "memory"},
    "api": {"core", "domain", "insights", "services", "agent", "container"},
    "container": {"core", "domain", "statements", "memory", "services", "agent", "adapters"},
    "main": {"core", "api", "container"},
}
# Third-party libraries and the only packages allowed to use them.
LIBRARY_OWNERS: dict[str, set[str]] = {
    "fastapi": {"api"},
    "starlette": {"api"},
    "psycopg": {"adapters"},
    "psycopg_pool": {"adapters"},
    "httpx": {"adapters", "memory", "container"},
    "uvicorn": {"main"},
    "groundwire": set(),
}


def _graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    internal: dict[str, set[str]] = defaultdict(set)
    external: dict[str, set[str]] = defaultdict(set)
    for path in ROOT.rglob("*.py"):
        package = path.relative_to(ROOT).parts[0].removesuffix(".py")
        if package == "__init__":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                     and node.level == 0 else [])
            for name in names:
                parts = name.split(".")
                if parts[0] == "money_agent" and len(parts) > 1 and parts[1] != package:
                    internal[package].add(parts[1])
                elif parts[0] in LIBRARY_OWNERS:
                    external[parts[0]].add(package)
    return internal, external


INTERNAL, EXTERNAL = _graph()


def test_every_package_has_a_rule() -> None:
    assert set(INTERNAL) <= set(ALLOWED), set(INTERNAL) - set(ALLOWED)


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_packages_only_import_what_they_may(package: str) -> None:
    extra = INTERNAL.get(package, set()) - ALLOWED[package]
    assert not extra, f"{package} must not import {sorted(extra)}"


def test_there_are_no_cycles() -> None:
    def reaches(start: str, target: str, seen: set[str]) -> bool:
        for dep in INTERNAL.get(start, set()):
            if dep == target or (dep not in seen and reaches(dep, target, seen | {dep})):
                return True
        return False

    cyclic = sorted(p for p in INTERNAL if reaches(p, p, {p}))
    assert cyclic == []


@pytest.mark.parametrize("library", sorted(LIBRARY_OWNERS))
def test_libraries_stay_in_their_layer(library: str) -> None:
    users = EXTERNAL.get(library, set())
    assert users <= LIBRARY_OWNERS[library], f"{library} used by {sorted(users)}"


def test_the_rules_were_actually_checked() -> None:
    assert "services" in INTERNAL["agent"] and "fastapi" in EXTERNAL
