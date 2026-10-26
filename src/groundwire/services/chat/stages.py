"""The chat pipeline's stages: one small class per concern.

Each stage wraps one optional collaborator (the safety guard, the cache, the
tool gate, ...) and behaves as a *null object* when that collaborator is
switched off: ``SafetyStage(None).restore(text, prepared)`` returns ``text``
unchanged. The orchestrators in :mod:`.pipeline` and :mod:`.streaming`
therefore never branch on "is this feature enabled?"; they just call the stage.

Stages are stateless per request (everything request-specific is passed in),
so one instance is shared by all requests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from groundwire.agents.answer_check import AnswerChecker, AnswerCheckReport
from groundwire.agents.gate import ToolGate
from groundwire.agents.policy import PolicyResult
from groundwire.cache.service import CacheLookup, CacheStatus, ResponseCache
from groundwire.core.config import AnswerCheckAction
from groundwire.core.errors import AnswerCheckFailedError, GroundingViolationError
from groundwire.domain.models import (
    ChatCompletionRequest,
    Completion,
    GroundingMode,
    GroundingOptions,
    ToolCall,
)
from groundwire.guardrails.grounding import GroundingReport, GroundingVerifier, Verdict
from groundwire.observability.metrics import GatewayMetrics
from groundwire.safety.guard import PreparedRequest, SafetyGuard
from groundwire.safety.pii import StreamRestorer
from groundwire.services.chat.common import grounding_options

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Safety: injection screening and PII redaction / restoration
# --------------------------------------------------------------------------- #


class SafetyStage:
    """Applies the safety guard; a pass-through when the guard is off."""

    def __init__(self, guard: SafetyGuard | None) -> None:
        """Create the stage.

        Args:
            guard: Prompt-injection and PII guard; ``None`` disables both.
        """
        self._guard = guard

    def prepare(self, request: ChatCompletionRequest) -> PreparedRequest:
        """Screen and redact a request.

        Args:
            request: Client request.

        Returns:
            The prepared request (the original when the guard is off).

        Raises:
            PromptInjectionError: Injection detected and policy is ``block``.
            GuardrailUnavailableError: The injection detector failed.
        """
        if self._guard is None:
            return PreparedRequest(request=request)
        grounding = grounding_options(request)
        return self._guard.prepare(request, grounding.context if grounding else None)

    def restore(self, text: str, prepared: PreparedRequest) -> str:
        """Put real PII values back into an answer.

        Args:
            text: Answer in redacted form.
            prepared: Safety-processed request holding the vault.

        Returns:
            The text the client should see.
        """
        return self._guard.restore(text, prepared) if self._guard else text

    def restore_calls(self, calls: list[ToolCall], prepared: PreparedRequest) -> list[ToolCall]:
        """Put real PII values back into tool-call arguments.

        Args:
            calls: Tool calls in redacted form.
            prepared: Safety-processed request holding the vault.

        Returns:
            The calls the client should see.
        """
        return self._guard.restore_tool_calls(calls, prepared) if self._guard else calls

    def stream_restorer(self, prepared: PreparedRequest) -> StreamRestorer | None:
        """A stateful restorer for one stream (placeholders can span chunks).

        Args:
            prepared: Safety-processed request holding the vault.

        Returns:
            The restorer, or ``None`` when there is nothing to restore.
        """
        return self._guard.stream_restorer(prepared) if self._guard else None


# --------------------------------------------------------------------------- #
# Cache: lookup, store, and the key for merging identical in-flight requests
# --------------------------------------------------------------------------- #


class CacheStage:
    """Response-cache access; a no-op when caching is off."""

    def __init__(self, cache: ResponseCache | None) -> None:
        """Create the stage.

        Args:
            cache: Response cache; ``None`` disables caching and coalescing.
        """
        self._cache = cache

    async def lookup(self, prepared: PreparedRequest, route: str) -> CacheLookup | None:
        """Look the redacted request up in the cache.

        Args:
            prepared: Safety-processed request.
            route: Route alias.

        Returns:
            The lookup, or ``None`` when caching is disabled.
        """
        if self._cache is None:
            return None
        return await self._cache.lookup(prepared.request, route,
                                        injection_flagged=prepared.injection_action != "none")

    def coalescing_key(self, lookup: CacheLookup | None) -> str | None:
        """The key identical in-flight requests share, if they may be merged.

        Only misses (and lookups whose store errored) are merged: a hit has
        nothing to wait for, and a bypassed request opted out of sharing.

        Args:
            lookup: The request's lookup.

        Returns:
            The exact-match key, or ``None`` when this request must call the
            provider alone.
        """
        if (self._cache is None or not self._cache.coalesce or lookup is None
                or lookup.key is None
                or lookup.status not in (CacheStatus.MISS, CacheStatus.ERROR)):
            return None
        return lookup.key.exact

    async def store(self, lookup: CacheLookup | None, completion: Completion, provider: str,
                    report: GroundingReport | None) -> None:
        """Store a provider answer unless it failed (or could not pass) grounding.

        Args:
            lookup: The request's cache lookup (``None``: caching is off).
            completion: Answer in redacted form.
            provider: Provider that produced it.
            report: Grounding report, if grounding ran.
        """
        if self._cache is None or lookup is None:
            return
        if report is not None and report.verdict is not Verdict.PASSED:
            logger.info("answer not cached: grounding verdict %s", report.verdict.value)
            return
        await self._cache.store(lookup, completion, provider)


# --------------------------------------------------------------------------- #
# Tools: policy rules and human approval for the calls a model requests
# --------------------------------------------------------------------------- #


class ToolStage:
    """Runs the tool gate; releases every call unchanged when it is off."""

    def __init__(self, gate: ToolGate | None) -> None:
        """Create the stage.

        Args:
            gate: Guardrails (policy and human approval); ``None`` disables them.
        """
        self._gate = gate

    async def enforce(self, calls: list[ToolCall], request: ChatCompletionRequest,
                      prepared: PreparedRequest) -> tuple[list[ToolCall], PolicyResult | None]:
        """Run the tool gate on the calls a model requested.

        Runs on PII-restored calls, so rules and approvals see real values (an
        email-domain rule must see the real address, not ``[EMAIL_1]``).

        Args:
            calls: Proposed calls.
            request: The client's request (offered tools, history, ``user``).
            prepared: Safety-processed request (injection verdict).

        Returns:
            ``(calls_to_release, result)``; ``result`` is ``None`` when the gate
            is off or there were no calls.

        Raises:
            ToolCallBlockedError: A call was blocked and ``on_violation`` is ``error``.
        """
        if self._gate is None:
            return calls, None
        return await self._gate.enforce(calls, request,
                                        injection_suspected=prepared.injection_action != "none")


# --------------------------------------------------------------------------- #
# Answer check: an agent's final answer against its tool results
# --------------------------------------------------------------------------- #


class AnswerCheckStage:
    """Fact-checks agents' final answers; inactive when no checker is set."""

    def __init__(self, checker: AnswerChecker | None, metrics: GatewayMetrics) -> None:
        """Create the stage.

        Args:
            checker: The answer checker; ``None`` disables the check.
            metrics: Metrics sink.
        """
        self._checker = checker
        self._metrics = metrics

    @property
    def action(self) -> AnswerCheckAction | None:
        """The configured action, or ``None`` when the check is off."""
        checker = self._checker
        return checker.action if checker is not None and checker.enabled else None

    def applies_to(self, request: ChatCompletionRequest) -> bool:
        """True if this request's answer will be checked (it has tool results).

        Args:
            request: Client request.

        Returns:
            Whether the check runs.
        """
        return self._checker is not None and self._checker.applies_to(request)

    def must_buffer(self, request: ChatCompletionRequest) -> bool:
        """True if this stream's answer may have to be changed or refused.

        ``annotate`` and ``block`` act on the complete answer, so agent
        conversations (those with tool results) are produced in full first.

        Args:
            request: Client request.

        Returns:
            Whether to use the buffered streaming path.
        """
        checker = self._checker
        return (checker is not None and checker.applies_to(request)
                and checker.action is not AnswerCheckAction.FLAG)

    def check(self, answer: str, request: ChatCompletionRequest) -> AnswerCheckReport | None:
        """Fact-check a final answer against the request's tool results.

        Args:
            answer: Final answer, PII-restored.
            request: Client request (unredacted tool results).

        Returns:
            The report, or ``None`` when the check is off or the conversation
            has no tool results.

        Raises:
            AnswerCheckFailedError: Issues found and the action is ``block``.
        """
        checker = self._checker
        if checker is None or not checker.applies_to(request):
            return None
        report = checker.check(answer, request)
        self._metrics.answer_checks.labels(verdict=report.verdict.value).inc()
        for issue in report.issues:
            self._metrics.answer_check_issues.labels(kind=issue.kind.value).inc()
        if report.failed:
            # Tool names and issue kinds only: never the values (they may be personal data).
            logger.warning("agent answer contradicts tool results", extra={
                "issues": [issue.kind.value for issue in report.issues],
                "tools": sorted({issue.tool for issue in report.issues if issue.tool}),
                "action": checker.action.value})
            if checker.action is AnswerCheckAction.BLOCK:
                raise AnswerCheckFailedError(
                    "the answer does not match the results of the tools it used",
                    details={"answer_check": report.as_dict()})
        return report


# --------------------------------------------------------------------------- #
# Grounding: the answer against the documents the client supplied
# --------------------------------------------------------------------------- #


class GroundingStage:
    """Runs grounding verification and applies the request's flag/block policy."""

    def __init__(self, verifier: GroundingVerifier, metrics: GatewayMetrics) -> None:
        """Create the stage.

        Args:
            verifier: Grounding verifier.
            metrics: Metrics sink.
        """
        self._verifier = verifier
        self._metrics = metrics

    async def check(self, answer: str, options: GroundingOptions,
                    question: str | None = None, *,
                    precomputed: GroundingReport | None = None) -> GroundingReport:
        """Verify an answer and apply the request's policy.

        Args:
            answer: Generated text (PII-restored).
            options: Grounding options from the request.
            question: The user's question, for scorers that read the full prompt.
            precomputed: A report already computed on exactly this answer and
                context (a cascade's check); the policy is still applied.

        Returns:
            The report (when the answer passes, or fails in ``flag`` mode).

        Raises:
            GroundingViolationError: Failed in ``block`` mode.
            GuardrailUnavailableError: Scorer failed and policy is fail-closed.
        """
        report = precomputed or await self._verifier.verify(answer, options.context,
                                                            question=question)
        self._metrics.grounding.labels(verdict=report.verdict.value).inc()
        self._metrics.grounding_latency.observe(report.latency_ms / 1000)
        logger.info("grounding checked", extra={
            "verdict": report.verdict.value, "supported_ratio": round(report.supported_ratio, 3),
            "claims": len(report.claims), "latency_ms": round(report.latency_ms, 1)})
        if report.verdict is Verdict.FAILED and options.mode is GroundingMode.BLOCK:
            raise GroundingViolationError(
                "answer is not sufficiently supported by the provided context",
                details={"grounding": report.as_dict()},
            )
        return report


# --------------------------------------------------------------------------- #
# Request metrics
# --------------------------------------------------------------------------- #


class RequestRecorder:
    """Records one request's outcome and latency."""

    def __init__(self, metrics: GatewayMetrics) -> None:
        """Create the recorder.

        Args:
            metrics: Metrics sink.
        """
        self._metrics = metrics

    def record(self, route: str, provider: str, status: str, *, stream: bool,
               started: float) -> None:
        """Count a finished request and observe its latency.

        Args:
            route: Route alias.
            provider: Serving provider (``"none"`` if nothing answered).
            status: ``success``, an error code, or ``client_disconnected``.
            stream: Whether the client streamed.
            started: ``perf_counter`` at request start.
        """
        stream_label = "true" if stream else "false"
        self._metrics.requests.labels(route=route, provider=provider, status=status,
                                      stream=stream_label).inc()
        self._metrics.request_latency.labels(route=route, stream=stream_label).observe(
            time.perf_counter() - started)


@dataclass(frozen=True, slots=True)
class ChatStages:
    """All stages, handed to the orchestrators as one parameter object.

    Attributes:
        safety: Injection screening and PII.
        cache: Response cache.
        tools: Tool gate.
        answer_check: Fact-check of agents' final answers.
        grounding: Grounding verification.
        recorder: Request metrics.
    """

    safety: SafetyStage
    cache: CacheStage
    tools: ToolStage
    answer_check: AnswerCheckStage
    grounding: GroundingStage
    recorder: RequestRecorder
