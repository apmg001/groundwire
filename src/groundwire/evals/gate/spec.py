"""Quality-gate specification: which benchmarks run in CI and what they must score.

A gate file (``config/quality_gates.yaml``) groups gates into **suites**::

    suites:
      fast:                       # every push / PR, no ML dependencies
        - name: grounding-lexical
          benchmark: ragtruth
          scorer: lexical
          per_task: 150
          threshold: 0.3014
          checks:
            roc_auc: {min: 0.76}
            false_block_rate: {max: 0.45}

Each gate runs a **fixed, seeded sample** with a **fixed threshold**, so its
numbers only change when code, models or data change, never between runs.
That is what makes a small sample a reliable regression detector.

Bounds are reviewed like code: ``--calibrate`` *suggests* them, a human
commits them. A gate never loosens itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from groundwire.cache.embeddings import DEFAULT_EMBEDDING_MODEL
from groundwire.core.config import AnswerAggregation, GroundingScorerType
from groundwire.core.errors import ConfigurationError

# Metrics each benchmark can be checked on, and which direction is "better".
RAGTRUTH_METRICS: dict[str, Literal["higher", "lower"]] = {
    "precision": "higher", "recall": "higher", "f1": "higher", "accuracy": "higher",
    "roc_auc": "higher", "false_block_rate": "lower", "scorer_errors": "lower",
}
QQP_METRICS: dict[str, Literal["higher", "lower"]] = {
    "hit_rate": "higher", "precision": "higher", "roc_auc": "higher",
    "false_hit_rate": "lower",
}

RagTruthMetric = Literal["precision", "recall", "f1", "accuracy", "roc_auc",
                         "false_block_rate", "scorer_errors"]
QQPMetric = Literal["hit_rate", "precision", "roc_auc", "false_hit_rate"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Bound(_Strict):
    """Allowed range for one metric.

    Attributes:
        min: Lowest acceptable value (inclusive).
        max: Highest acceptable value (inclusive).
    """

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("a bound needs min, max, or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) is greater than max ({self.max})")
        return self

    def allows(self, value: float) -> bool:
        """Return True if ``value`` is inside the bound.

        Args:
            value: Measured value.

        Returns:
            Whether the check passes.
        """
        return ((self.min is None or value >= self.min)
                and (self.max is None or value <= self.max))

    def describe(self) -> str:
        """Human-readable bound, e.g. ``>= 0.76`` or ``0.1 .. 0.2``."""
        if self.min is not None and self.max is not None:
            return f"{self.min:g} .. {self.max:g}"
        return f">= {self.min:g}" if self.min is not None else f"<= {self.max:g}"


class RagTruthGate(_Strict):
    """Grounding-verifier gate on a fixed RAGTruth test sample.

    Attributes:
        name: Unique gate name.
        benchmark: Always ``"ragtruth"``.
        scorer: Grounding scorer to evaluate.
        model: Model id for ``nli``/``hybrid``/``token`` (``None`` = default).
        per_task: Test responses per task (seeded sample).
        seed: Sampling seed.
        threshold: Fixed decision threshold on the answer score.
        aggregation: How claim scores combine into an answer score.
        min_claim_tokens: Shortest claim that is checked.
        evidence_passages: Passages joined per claim (0 = each separately).
        checks: Metric -> allowed range.
    """

    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_.\-]*$")]
    benchmark: Literal["ragtruth"]
    scorer: GroundingScorerType = GroundingScorerType.LEXICAL
    model: str | None = None
    per_task: Annotated[int, Field(ge=1, le=10_000)] = 100
    seed: int = 13
    threshold: Annotated[float, Field(ge=0, le=1)]
    aggregation: AnswerAggregation = AnswerAggregation.LOWEST_TWO
    min_claim_tokens: Annotated[int, Field(ge=1, le=50)] = 3
    evidence_passages: Annotated[int, Field(ge=0, le=10)] = 0
    checks: Annotated[dict[RagTruthMetric, Bound], Field(min_length=1)]

    @property
    def needs_models(self) -> bool:
        """True when the gate needs the ``nli`` extra (torch + a model download)."""
        return self.scorer is not GroundingScorerType.LEXICAL


class QQPGate(_Strict):
    """Semantic-cache gate on a fixed QQP validation sample.

    Attributes:
        name: Unique gate name.
        benchmark: Always ``"qqp"``.
        model: Sentence-embedding model id.
        pairs: Validation pairs (seeded sample).
        seed: Sampling seed.
        threshold: Similarity threshold to evaluate (normally the shipped default).
        use_guard: Apply the number/placeholder guards, as the live cache does.
        checks: Metric -> allowed range.
    """

    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_.\-]*$")]
    benchmark: Literal["qqp"]
    model: str = DEFAULT_EMBEDDING_MODEL
    pairs: Annotated[int, Field(ge=10, le=500_000)] = 5000
    seed: int = 13
    threshold: Annotated[float, Field(gt=0, le=1)]
    use_guard: bool = True
    checks: Annotated[dict[QQPMetric, Bound], Field(min_length=1)]

    @property
    def needs_models(self) -> bool:
        """Always True: embeddings need the ``nli`` extra."""
        return True


Gate = Annotated[RagTruthGate | QQPGate, Field(discriminator="benchmark")]


class GateSpec(_Strict):
    """Root of the gate file.

    Attributes:
        suites: Suite name -> gates, run in order.
    """

    suites: Annotated[dict[str, Annotated[list[Gate], Field(min_length=1)]], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [g.name for gates in self.suites.values() for g in gates]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"gate names must be unique across suites: {duplicates}")
        return self

    def suite(self, name: str) -> list[RagTruthGate | QQPGate]:
        """Return a suite's gates.

        Args:
            name: Suite name.

        Returns:
            Its gates.

        Raises:
            ConfigurationError: If the suite does not exist.
        """
        if name not in self.suites:
            raise ConfigurationError(f"unknown suite '{name}'; defined: {sorted(self.suites)}")
        return list(self.suites[name])


def direction(gate: RagTruthGate | QQPGate, metric: str) -> Literal["higher", "lower"]:
    """Return whether higher or lower values of a metric are better.

    Args:
        gate: The gate the metric belongs to.
        metric: Metric name.

    Returns:
        ``"higher"`` or ``"lower"``.
    """
    table = RAGTRUTH_METRICS if isinstance(gate, RagTruthGate) else QQP_METRICS
    return table[metric]


def parse_spec(raw: Any, *, source: str = "<dict>") -> GateSpec:
    """Validate an already-parsed gate mapping.

    Args:
        raw: Parsed YAML.
        source: Label for error messages.

    Returns:
        The spec.

    Raises:
        ConfigurationError: If validation fails.
    """
    try:
        return GateSpec.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                             for e in exc.errors())
        raise ConfigurationError(f"invalid quality-gate file ({source}): {problems}") from exc


def load_spec(path: Path) -> GateSpec:
    """Load and validate a gate file.

    Args:
        path: YAML path.

    Returns:
        The spec.

    Raises:
        ConfigurationError: If the file is missing, not YAML, or invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"quality-gate file not found: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read quality-gate file {path}: {exc}") from exc
    return parse_spec(raw, source=str(path))
