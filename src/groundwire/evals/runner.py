"""Run the grounding verifier over benchmark examples.

The runner uses the *same* :class:`GroundingVerifier` the gateway uses in
production, so benchmark numbers describe the real code path (claim
splitting, passage chunking, truncation limits), not a separate re-implementation.

Each example is reduced to one **support score**: the verifier's
``answer_score``, i.e. the claim scores combined by the configured
aggregation (``min``, ``lowest_two`` or ``mean``). A response is flagged when
that score falls below a threshold, exactly as the gateway does in the
score-based aggregation modes.

Examples run one at a time so per-example latency is measured cleanly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from groundwire.core.errors import GuardrailUnavailableError
from groundwire.evals.datasets import EvalExample
from groundwire.guardrails.grounding import GroundingVerifier, Verdict

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 25


@dataclass(frozen=True, slots=True)
class ExampleResult:
    """Verifier output for one example.

    Attributes:
        id: Example id.
        task: Task name.
        is_hallucinated: Ground-truth label.
        support_score: Aggregated answer score (1.0 if no checkable claims).
        claims: Number of claims checked.
        unsupported_claim: Text of the weakest claim, for error analysis.
        latency_ms: Verification time.
        error: Scorer error message, if verification failed.
        claim_scores: Every claim's score, for offline analysis.
    """

    id: str
    task: str
    is_hallucinated: bool
    support_score: float
    claims: int
    unsupported_claim: str | None
    latency_ms: float
    error: str | None = None
    claim_scores: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view."""
        data = asdict(self)
        data["support_score"] = round(self.support_score, 4)
        data["latency_ms"] = round(self.latency_ms, 2)
        data["claim_scores"] = [round(score, 4) for score in self.claim_scores]
        return data


async def evaluate(verifier: GroundingVerifier,
                   examples: Sequence[EvalExample]) -> list[ExampleResult]:
    """Score every example with the verifier.

    Args:
        verifier: Verifier to evaluate (its policy should be fail-open, so a
            scorer error is recorded instead of aborting the run).
        examples: Examples to score.

    Returns:
        One result per example, in input order.

    Raises:
        GuardrailUnavailableError: If the verifier is configured fail-closed
            and the scorer fails.
    """
    results: list[ExampleResult] = []
    started = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        try:
            report = await verifier.verify(example.response, list(example.context),
                                          prompt=example.prompt)
        except GuardrailUnavailableError:
            logger.exception("verifier failed (fail-closed)", extra={"example": example.id})
            raise

        if report.verdict is Verdict.UNVERIFIED:
            # Unknown, so score it neutrally rather than guessing either way.
            results.append(ExampleResult(example.id, example.task, example.is_hallucinated,
                                         1.0, 0, None, report.latency_ms, report.error))
        elif report.claims:
            weakest = min(report.claims, key=lambda c: c.score)
            results.append(ExampleResult(example.id, example.task, example.is_hallucinated,
                                         report.answer_score, len(report.claims),
                                         weakest.claim, report.latency_ms,
                                         claim_scores=tuple(c.score for c in report.claims)))
        else:
            results.append(ExampleResult(example.id, example.task, example.is_hallucinated,
                                         1.0, 0, None, report.latency_ms))

        if index % PROGRESS_EVERY == 0 or index == len(examples):
            elapsed = time.perf_counter() - started
            eta = elapsed / index * (len(examples) - index)
            logger.info("progress %d/%d (%.0fs elapsed, ~%.0fs left)",
                        index, len(examples), elapsed, eta)
    return results
