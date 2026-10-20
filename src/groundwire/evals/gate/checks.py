"""Evaluate gates against their bounds, and suggest bounds when calibrating.

Gates **fail closed**: a gate whose benchmark crashes, whose data is missing,
or that does not report a checked metric counts as a *failure*. A quality
gate that silently passes when it could not measure anything is worse than
no gate.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groundwire.core.errors import (
    CacheUnavailableError,
    ConfigurationError,
    DatasetError,
    GuardrailUnavailableError,
)
from groundwire.evals.gate.runners import Metrics, run_qqp_gate, run_ragtruth_gate
from groundwire.evals.gate.spec import Bound, QQPGate, RagTruthGate, direction

logger = logging.getLogger(__name__)

AnyGate = RagTruthGate | QQPGate

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one metric check.

    Attributes:
        metric: Metric name.
        value: Measured value (``None`` if the benchmark did not report it).
        bound: Allowed range.
    """

    metric: str
    value: float | None
    bound: Bound

    @property
    def passed(self) -> bool:
        """True when the metric was measured and is inside its bound."""
        return self.value is not None and math.isfinite(self.value) and self.bound.allows(
            self.value)


@dataclass(slots=True)
class GateResult:
    """Outcome of one gate.

    Attributes:
        gate: The gate definition.
        metrics: Everything the benchmark measured.
        checks: One result per configured check.
        seconds: Wall time.
        error: Why the gate could not run, if it could not.
    """

    gate: AnyGate
    metrics: Metrics = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        """True only if the gate ran and every check passed."""
        return self.error is None and bool(self.checks) and all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "name": self.gate.name, "benchmark": self.gate.benchmark, "passed": self.passed,
            "seconds": round(self.seconds, 2), "error": self.error,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "checks": [{"metric": c.metric, "value": None if c.value is None else round(c.value, 4),
                        "min": c.bound.min, "max": c.bound.max, "passed": c.passed}
                       for c in self.checks],
        }


def evaluate_checks(gate: AnyGate, metrics: Metrics) -> list[CheckResult]:
    """Compare measured metrics with the gate's bounds.

    Args:
        gate: Gate definition.
        metrics: Measured metrics.

    Returns:
        One result per check, in the order they are declared.
    """
    return [CheckResult(metric=name, value=metrics.get(name), bound=bound)
            for name, bound in gate.checks.items()]


# --------------------------------------------------------------------------- #
# Running a suite
# --------------------------------------------------------------------------- #

GateRunner = Callable[[AnyGate], Awaitable[Metrics]]


@dataclass(frozen=True, slots=True)
class DataDirs:
    """Where each benchmark's data lives.

    Attributes:
        ragtruth: RAGTruth clone.
        qqp: QQP files.
    """

    ragtruth: Path = Path("data/ragtruth")
    qqp: Path = Path("data/qqp")


def default_runner(data: DataDirs) -> GateRunner:
    """Build the production runner that dispatches on benchmark type.

    Args:
        data: Data locations.

    Returns:
        An async callable gate -> metrics.
    """
    async def run(gate: AnyGate) -> Metrics:
        if isinstance(gate, RagTruthGate):
            return await run_ragtruth_gate(gate, data.ragtruth)
        return run_qqp_gate(gate, data.qqp)

    return run


async def run_gates(gates: Sequence[AnyGate], runner: GateRunner) -> list[GateResult]:
    """Run gates one after another; a failing gate never stops the others.

    Args:
        gates: Gates to run.
        runner: Benchmark runner.

    Returns:
        One result per gate.
    """
    results: list[GateResult] = []
    for gate in gates:
        logger.info("gate %s: running (%s)", gate.name, gate.benchmark)
        started = time.perf_counter()
        result = GateResult(gate=gate)
        try:
            result.metrics = await runner(gate)
            result.checks = evaluate_checks(gate, result.metrics)
        except (DatasetError, ConfigurationError, CacheUnavailableError,
                GuardrailUnavailableError) as exc:
            result.error = exc.message
        except (OSError, RuntimeError, ValueError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - started
        status = "PASS" if result.passed else "FAIL"
        logger.info("gate %s: %s in %.1fs%s", gate.name, status, result.seconds,
                    f" ({result.error})" if result.error else "")
        results.append(result)
    return results


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def suggest_bound(gate: AnyGate, metric: str, value: float, tolerance: float) -> Bound:
    """Suggest a bound ``tolerance`` away from the measured value, in the safe direction.

    Higher-is-better metrics get a floor, lower-is-better ones a ceiling.
    ``scorer_errors`` always gets ``max: 0``: any model failure is a regression.

    Args:
        gate: Gate the metric belongs to.
        metric: Metric name.
        value: Measured value.
        tolerance: Absolute slack (e.g. 0.02 = 2 points).

    Returns:
        The suggested bound, rounded outward to 3 decimals.
    """
    if metric == "scorer_errors":
        return Bound(max=0)
    # round() first: 0.70 - 0.02 is 0.67999... in binary floating point and would
    # otherwise floor to 0.679 instead of 0.68.
    if direction(gate, metric) == "higher":
        return Bound(min=max(0.0, math.floor(round((value - tolerance) * 1000, 6)) / 1000))
    return Bound(max=min(1.0, math.ceil(round((value + tolerance) * 1000, 6)) / 1000))
