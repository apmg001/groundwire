"""The non-streaming pipeline: one request in, one finished response out.

Steps, in order (each is one method of :class:`CompletionPipeline`)::

    prepare ─► lookup ─┬─ hit ─────────────┐
                       └─ miss ─► dispatch ┤
                                           ▼
     restore ─► tools ─► answer check ─► grounding ─► cache store ─► annotate ─► shape

* ``restore`` puts real PII values back: everything after it sees what the
  client will see, and the cache before it only ever sees placeholders.
* ``tools`` runs the tool gate on the calls the model asked for; when none
  can run now, the turn ends with the gateway's explanation as text.
* ``answer check`` and ``grounding`` only run on a *final* answer: when the
  model asked for tools, its text is a preamble, not a factual claim.
* ``cache store`` skips answers that failed a check, and coalesced answers
  (another request already stored them).
* ``annotate`` adds the answer-check warning *after* grounding, so the
  warning itself is never fact-checked.

Buffered streams (cascades, and agent answers the answer check may change)
also use this pipeline and replay the result; see :mod:`.replay`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from groundwire.agents.answer_check import AnswerCheckReport
from groundwire.agents.policy import PolicyResult
from groundwire.cache.service import CacheLookup
from groundwire.core.config import AnswerCheckAction
from groundwire.domain.models import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    FinishReason,
    ToolCall,
)
from groundwire.guardrails.grounding import GroundingReport
from groundwire.safety.guard import PreparedRequest
from groundwire.services.chat.common import Answer, grounding_options, last_user_message
from groundwire.services.chat.dispatch import ProviderDispatcher
from groundwire.services.chat.stages import ChatStages

# Grounding metadata for an answer that is only tool calls (nothing to check).
GROUNDING_SKIPPED_TOOL_CALLS: dict[str, str] = {"verdict": "skipped", "reason": "tool_calls"}


@dataclass(slots=True)
class _Turn:
    """Everything one request accumulates on its way through the pipeline.

    Attributes:
        request: The client's request.
        route: Resolved route or cascade alias.
        prepared: Safety-processed request.
        lookup: Cache lookup (``None`` when caching is off).
        answer: The answer in redacted form.
        text: Answer text the client will see (PII restored).
        proposed: Tool calls the model asked for (PII restored).
        calls: Tool calls released to the client after the gate.
        policy: The gate's result, if it ran.
        finish_reason: Finish reason the client will see.
        meta: The response's ``groundwire`` metadata, built step by step.
        check: Answer-check report, if the check ran.
        report: Grounding report, if grounding ran.
    """

    request: ChatCompletionRequest
    route: str
    prepared: PreparedRequest
    lookup: CacheLookup | None
    answer: Answer
    text: str = ""
    proposed: list[ToolCall] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    policy: PolicyResult | None = None
    finish_reason: FinishReason | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    check: AnswerCheckReport | None = None
    report: GroundingReport | None = None


class CompletionPipeline:
    """Builds a full (non-streamed) response from the stages."""

    def __init__(self, stages: ChatStages, dispatcher: ProviderDispatcher) -> None:
        """Create the pipeline.

        Args:
            stages: Pipeline stages.
            dispatcher: Provider access.
        """
        self._stages = stages
        self._dispatcher = dispatcher

    async def run(self, request: ChatCompletionRequest, route: str) -> ChatCompletionResponse:
        """Serve one request end to end.

        Args:
            request: Client request.
            route: Resolved route or cascade alias.

        Returns:
            The OpenAI-compatible response with ``groundwire`` metadata.

        Raises:
            GatewayTimeoutError: Deadline exceeded.
            AllProvidersFailedError: Every provider failed.
            ProviderBadRequestError: Upstream rejected the request.
            GroundingViolationError: Grounding failed in ``block`` mode.
            GuardrailUnavailableError: Grounding could not run (fail-closed).
            PromptInjectionError: Injection detected and the policy is ``block``.
            ToolCallBlockedError: A call was blocked and ``on_violation`` is ``error``.
            AnswerCheckFailedError: The answer contradicts its tool results and
                the answer check is in ``block`` mode.
        """
        prepared = self._stages.safety.prepare(request)
        lookup = await self._stages.cache.lookup(prepared, route)
        answer = await self._obtain(route, prepared, lookup, request)
        turn = _Turn(request=request, route=route, prepared=prepared, lookup=lookup,
                     answer=answer)
        await self._apply_tools(turn)
        self._describe(turn)
        self._check_answer(turn)
        await self._ground(turn)
        await self._store(turn)
        self._annotate(turn)
        return self._shape(turn)

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    async def _obtain(self, route: str, prepared: PreparedRequest, lookup: CacheLookup | None,
                      request: ChatCompletionRequest) -> Answer:
        """The cached answer on a hit, otherwise a provider's."""
        if lookup is not None and lookup.hit and lookup.entry is not None:
            return Answer.from_cache(lookup)
        return await self._dispatcher.answer(route, prepared, lookup, request)

    async def _apply_tools(self, turn: _Turn) -> None:
        """Restore PII, then run the tool gate on the calls the model asked for."""
        safety = self._stages.safety
        completion = turn.answer.completion
        turn.text = safety.restore(completion.text, turn.prepared)
        turn.proposed = safety.restore_calls(completion.tool_calls, turn.prepared)
        turn.calls, turn.policy = await self._stages.tools.enforce(
            turn.proposed, turn.request, turn.prepared)
        turn.finish_reason = completion.finish_reason
        if turn.policy is not None and turn.proposed and not turn.calls:
            # No call can run now (blocked or waiting for approval): answer in text,
            # so the agent loop ends cleanly.
            prefix = turn.text + "\n\n" if turn.text.strip() else ""
            turn.text = prefix + turn.policy.explanation()
            turn.finish_reason = "stop"

    @staticmethod
    def _describe(turn: _Turn) -> None:
        """Start the ``groundwire`` metadata: who answered, and how."""
        answer = turn.answer
        meta: dict[str, Any] = {
            "route": turn.route,
            "provider": answer.provider,
            "upstream_model": answer.completion.upstream_model,
            "attempts": answer.attempts,
        }
        if answer.cascade is not None:
            meta["cascade"] = answer.cascade.as_dict()
        safety_meta = turn.prepared.metadata()
        if safety_meta:
            meta["safety"] = safety_meta
        if turn.lookup is not None:
            meta["cache"] = turn.lookup.as_dict()
        if turn.policy is not None:
            meta["tools"] = turn.policy.as_dict()
        turn.meta = meta

    def _check_answer(self, turn: _Turn) -> None:
        """Fact-check a final answer against the conversation's tool results."""
        if turn.proposed:
            return  # tool calls: the text is a preamble or the gateway's explanation
        turn.check = self._stages.answer_check.check(turn.text, turn.request)
        if turn.check is not None:
            turn.meta["answer_check"] = turn.check.as_dict()

    async def _ground(self, turn: _Turn) -> None:
        """Verify the restored answer against the original (unredacted) documents."""
        options = grounding_options(turn.request)
        if options is None:
            return
        if turn.proposed and not turn.answer.completion.text.strip():
            # A tool call is a request for action, not a factual claim: nothing
            # to check against documents (tool policies govern calls instead).
            turn.meta["grounding"] = dict(GROUNDING_SKIPPED_TOOL_CALLS)
            return
        # A coalesced answer was checked with another request's PII values,
        # so only reuse a precomputed report for our own answer.
        coalesced = turn.lookup is not None and turn.lookup.coalesced
        turn.report = await self._stages.grounding.check(
            turn.text, options, last_user_message(turn.request),
            precomputed=None if coalesced else turn.answer.report)
        turn.meta["grounding"] = turn.report.as_dict()

    async def _store(self, turn: _Turn) -> None:
        """Cache a fresh answer that passed its checks."""
        lookup = turn.lookup
        if lookup is None or lookup.hit or lookup.coalesced:
            return
        if turn.check is not None and turn.check.failed:
            return
        await self._stages.cache.store(lookup, turn.answer.completion, turn.answer.provider,
                                       turn.report)

    def _annotate(self, turn: _Turn) -> None:
        """Append the answer-check warning (``annotate`` mode), after grounding."""
        if (turn.check is not None and turn.check.failed and turn.text.strip()
                and self._stages.answer_check.action is AnswerCheckAction.ANNOTATE):
            turn.text = f"{turn.text}\n\n{turn.check.warning()}"

    @staticmethod
    def _shape(turn: _Turn) -> ChatCompletionResponse:
        """The public, OpenAI-compatible response."""
        message = AssistantMessage(content=turn.text if turn.text or not turn.calls else None,
                                   tool_calls=turn.calls or None)
        return ChatCompletionResponse(
            model=turn.request.model,
            choices=[Choice(message=message, finish_reason=turn.finish_reason)],
            usage=turn.answer.completion.usage,
            groundwire=turn.meta,
        )
