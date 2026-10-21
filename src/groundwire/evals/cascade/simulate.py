"""Simulate "cheap model first, escalate when the fact-check fails" on RAGTruth.

RAGTruth answers every source prompt with six models (GPT-4, GPT-3.5,
Llama-2 7B/13B/70B, Mistral-7B), and humans labelled each answer for
hallucinations. That lets us replay a cascade offline, with no API calls:

1. take the **cheap** model's answer to a prompt,
2. score it with the grounding verifier (the gateway's own code),
3. if the score is below the threshold, **escalate**: deliver the **strong**
   model's answer to the same prompt instead (and pay for both calls).

Policies compared on the same prompts:

* ``cheap_only``: never escalate (cheapest, least accurate),
* ``strong_only``: always use the strong model (the usual default),
* ``cascade``: escalate on a failed check (what Groundwire would do),
* ``oracle``: escalate exactly the cheap answers humans marked as
  hallucinated. No real detector does this; it is the ceiling a perfect
  fact-checker would reach.

Cost model: one cheap call costs 1 unit and one strong call costs ``R`` units
(the *price ratio*). A cascade request costs 1, plus ``R`` when escalated.
Costs are reported relative to ``strong_only``. Real prices depend on tokens,
providers and date, so results are shown for several ratios rather than one
made-up price. Latency is not simulated: RAGTruth has no generation timings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from groundwire.evals.datasets import EvalExample
from groundwire.evals.runner import ExampleResult

# --------------------------------------------------------------------------- #
# Pairing cheap and strong answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CascadeCase:
    """One prompt, as the cascade sees it.

    Attributes:
        source_id: RAGTruth prompt id.
        task: Task name.
        cheap_score: Verifier answer score of the cheap model's answer.
        cheap_error: The verifier could not score the cheap answer.
        cheap_hallucinated: Human label of the cheap answer.
        strong_hallucinated: Human label of the strong model's answer.
    """

    source_id: str
    task: str
    cheap_score: float
    cheap_error: bool
    cheap_hallucinated: bool
    strong_hallucinated: bool

    def escalates(self, threshold: float) -> bool:
        """Would the cascade call the strong model?

        Unscorable answers escalate: when the check cannot run, the safe
        choice for a correctness-first policy is the stronger model.

        Args:
            threshold: Escalate when the answer score is below this.

        Returns:
            True to escalate.
        """
        return self.cheap_error or self.cheap_score < threshold


def pair_by_source(examples: Sequence[EvalExample], cheap_model: str,
                   strong_model: str) -> list[tuple[EvalExample, EvalExample]]:
    """Match each prompt's cheap-model answer with its strong-model answer.

    Args:
        examples: RAGTruth examples (all models).
        cheap_model: Model name used as the cheap tier.
        strong_model: Model name used as the strong tier.

    Returns:
        ``(cheap, strong)`` pairs, ordered by task then source id; prompts
        missing either answer are skipped.

    Raises:
        ValueError: If the two models are the same or no prompt has both.
    """
    if cheap_model == strong_model:
        raise ValueError("cheap and strong models must differ")
    by_source: dict[str, dict[str, EvalExample]] = defaultdict(dict)
    for example in examples:
        source = str(example.meta.get("source_id", ""))
        model = str(example.meta.get("model", ""))
        if source and model in (cheap_model, strong_model):
            by_source[source][model] = example
    pairs = [(answers[cheap_model], answers[strong_model])
             for _, answers in sorted(by_source.items(),
                                      key=lambda kv: (next(iter(kv[1].values())).task, kv[0]))
             if cheap_model in answers and strong_model in answers]
    if not pairs:
        raise ValueError(f"no prompt has answers from both '{cheap_model}' and '{strong_model}'")
    return pairs


def build_cases(pairs: Sequence[tuple[EvalExample, EvalExample]],
                cheap_results: Sequence[ExampleResult]) -> list[CascadeCase]:
    """Combine pairs with the verifier's scores of the cheap answers.

    Args:
        pairs: ``(cheap, strong)`` example pairs.
        cheap_results: Verifier results for the cheap examples, same order.

    Returns:
        Cascade cases.

    Raises:
        ValueError: If the lengths or ids do not line up.
    """
    if len(pairs) != len(cheap_results):
        raise ValueError(f"{len(pairs)} pairs but {len(cheap_results)} results")
    cases: list[CascadeCase] = []
    for (cheap, strong), result in zip(pairs, cheap_results, strict=True):
        if result.id != cheap.id:
            raise ValueError(f"result {result.id} does not match example {cheap.id}")
        cases.append(CascadeCase(
            source_id=str(cheap.meta.get("source_id")), task=cheap.task,
            cheap_score=result.support_score, cheap_error=result.error is not None,
            cheap_hallucinated=cheap.is_hallucinated,
            strong_hallucinated=strong.is_hallucinated))
    return cases


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """What a routing policy delivers over a set of prompts.

    Attributes:
        policy: Policy name.
        prompts: Prompts served.
        escalated: Prompts that called the strong model after the cheap one.
        strong_calls: Total strong-model calls.
        cheap_calls: Total cheap-model calls.
        hallucinated: Delivered answers that humans marked as hallucinated.
    """

    policy: str
    prompts: int
    escalated: int
    strong_calls: int
    cheap_calls: int
    hallucinated: int

    @property
    def hallucination_rate(self) -> float:
        """Fraction of delivered answers that are hallucinated."""
        return self.hallucinated / self.prompts if self.prompts else 0.0

    @property
    def escalation_rate(self) -> float:
        """Fraction of prompts that paid for both models."""
        return self.escalated / self.prompts if self.prompts else 0.0

    def relative_cost(self, price_ratio: float) -> float:
        """Cost relative to always using the strong model.

        Args:
            price_ratio: Strong-call price / cheap-call price (``R``).

        Returns:
            1.0 = same cost as ``strong_only``; 0.25 = 75% cheaper.

        Raises:
            ValueError: If the ratio is not positive.
        """
        if price_ratio <= 0:
            raise ValueError("price_ratio must be positive")
        if not self.prompts:
            return 0.0
        return (self.cheap_calls + self.strong_calls * price_ratio) / (self.prompts * price_ratio)


def _outcome(policy: str, cases: Sequence[CascadeCase], escalate: Sequence[bool], *,
             cheap_calls: bool = True) -> PolicyOutcome:
    """Tally a policy given its per-case escalation decisions.

    Args:
        policy: Policy name.
        cases: Cases.
        escalate: Whether each case uses the strong model.
        cheap_calls: Whether the cheap model is called first.

    Returns:
        The outcome.
    """
    strong = sum(escalate)
    delivered = sum(c.strong_hallucinated if e else c.cheap_hallucinated
                    for c, e in zip(cases, escalate, strict=True))
    return PolicyOutcome(policy=policy, prompts=len(cases),
                         escalated=strong if cheap_calls else 0, strong_calls=strong,
                         cheap_calls=len(cases) if cheap_calls else 0, hallucinated=delivered)


def cheap_only(cases: Sequence[CascadeCase]) -> PolicyOutcome:
    """Always deliver the cheap answer."""
    return _outcome("cheap_only", cases, [False] * len(cases))


def strong_only(cases: Sequence[CascadeCase]) -> PolicyOutcome:
    """Always call (only) the strong model."""
    return _outcome("strong_only", cases, [True] * len(cases), cheap_calls=False)


def cascade(cases: Sequence[CascadeCase], threshold: float) -> PolicyOutcome:
    """Cheap first; escalate when the verifier score is below ``threshold``.

    Args:
        cases: Cases.
        threshold: Escalation threshold on the answer score.

    Returns:
        The outcome.
    """
    return _outcome("cascade", cases, [c.escalates(threshold) for c in cases])


def oracle(cases: Sequence[CascadeCase]) -> PolicyOutcome:
    """Escalate exactly the hallucinated cheap answers (a perfect detector)."""
    return _outcome("oracle", cases, [c.cheap_hallucinated for c in cases])


@dataclass(frozen=True, slots=True)
class DetectorQuality:
    """How well escalation decisions track the cheap answers' labels.

    Attributes:
        caught: Hallucinated cheap answers that were escalated.
        missed: Hallucinated cheap answers delivered anyway.
        wasted: Good cheap answers escalated (paid twice for nothing).
        kept: Good cheap answers delivered.
    """

    caught: int
    missed: int
    wasted: int
    kept: int


def detector_quality(cases: Sequence[CascadeCase], threshold: float) -> DetectorQuality:
    """Break escalation decisions down by the cheap answer's true label.

    Args:
        cases: Cases.
        threshold: Escalation threshold.

    Returns:
        Counts of caught, missed, wasted and kept.
    """
    caught = missed = wasted = kept = 0
    for c in cases:
        up = c.escalates(threshold)
        if c.cheap_hallucinated:
            caught, missed = caught + up, missed + (not up)
        else:
            wasted, kept = wasted + up, kept + (not up)
    return DetectorQuality(caught=caught, missed=missed, wasted=wasted, kept=kept)
