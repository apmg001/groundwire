"""The agent loop: think, call tools, answer, get checked, fix.

One user message is handled as::

    model ─► tool calls? ─yes─► run tools ─► model ─► ... (at most max_steps calls)
                │ no
                ▼
        final answer ─► Groundwire's answer check verdict
                          │ supported ─────────────► done (verified)
                          │ mismatch, retries left ─► tell the model what was wrong, loop
                          │ mismatch, no retries ───► apply the unverified policy

**Self-correction.** Groundwire compares every number and claimed action in
the final answer with the tool results in the conversation. When they do not
match, the model is told which kinds of problems were found (never the values)
and gets one more chance, *before* the user sees anything. If it still cannot
produce a matching answer, the policy decides: ``warn`` shows the answer with a
visible "could not verify" flag, ``withhold`` replaces it with a safe message.

**Bounded.** At most ``max_steps`` model calls per message, tools run one
at a time (memory writes stay ordered), and every model call has the gateway
client's timeout. A runaway loop costs a fixed, small amount.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from money_agent.agent.tools import ToolContext, ToolOutcome, ToolRegistry
from money_agent.core.config import UnverifiedAnswers
from money_agent.core.errors import AgentError
from money_agent.domain.ports import LLMClient, LLMReply

logger = logging.getLogger(__name__)

CORRECTION_PROMPT = (
    "[Automatic check] Your last answer did not match the tool results: {problems}. "
    "Answer again. Use only figures that appear in the tool results, exactly as they "
    "appear there (call a tool if you need a figure you do not have), and only say an "
    "action was done if its tool succeeded.")
FINAL_ANSWER_PROMPT = (
    "[Automatic check] You have used all your tool calls for this message. Answer now "
    "with the information you already have, and say what you could not find.")
WITHHELD_ANSWER = (
    "I couldn't verify my answer against your data, so I'm not showing it. Please try "
    "asking in a different way, or ask about one thing at a time.")


class VerificationStatus(StrEnum):
    """What the gateway's answer check said about the final answer."""

    VERIFIED = "verified"        # every number and action matched the tool results
    FAILED = "failed"            # mismatch remained after the retry
    NOT_NEEDED = "not_needed"    # no tool results to check against (e.g. "hello")
    UNAVAILABLE = "unavailable"  # the gateway did not run the check


@dataclass(frozen=True, slots=True)
class Verification:
    """The answer check's outcome, safe to show users.

    Attributes:
        status: Verdict.
        problems: Problem kinds found (``number_mismatch``, ``action_failed``...).
        corrected: True if a first answer failed and the retry passed.
    """

    status: VerificationStatus
    problems: tuple[str, ...] = ()
    corrected: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the API and stored message metadata."""
        return {"status": self.status.value, "problems": list(self.problems),
                "corrected": self.corrected}


@dataclass(frozen=True, slots=True)
class ToolTrace:
    """One tool call, for the response (no arguments or results: they are personal data)."""

    name: str
    ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Outcome of one user message.

    Attributes:
        answer: Text for the user.
        verification: The answer check's outcome.
        tools: Tool calls made, in order.
        model_calls: Model calls used.
        gateway: The final reply's gateway metadata (safety, provider...).
    """

    answer: str
    verification: Verification
    tools: tuple[ToolTrace, ...] = ()
    model_calls: int = 0
    gateway: dict[str, Any] = field(default_factory=dict)


def verification_of(reply: LLMReply) -> tuple[VerificationStatus, tuple[str, ...]]:
    """Read the gateway's answer-check metadata.

    "Verified" is a promise to the user, so it is only given when the check
    actually compared something: an answer with no figures and no claimed
    actions ("please upload a statement", or the backup model's "I can't reach
    the AI model") passes the check trivially and gets ``NOT_NEEDED``, not a
    verified badge. Missing counts are treated as zero for the same reason.

    Args:
        reply: A final model reply.

    Returns:
        The status and the problem kinds.
    """
    check = reply.gateway.get("answer_check")
    if not isinstance(check, dict):
        return VerificationStatus.UNAVAILABLE, ()
    verdict = check.get("verdict")
    if verdict == "supported":
        checked = _count(check.get("numbers_checked")) + _count(check.get("actions_checked"))
        return (VerificationStatus.VERIFIED if checked else VerificationStatus.NOT_NEEDED), ()
    if verdict == "mismatch":
        kinds = sorted({str(i.get("kind")) for i in check.get("issues", [])
                        if isinstance(i, dict)})
        return VerificationStatus.FAILED, tuple(kinds)
    return VerificationStatus.NOT_NEEDED, ()


def _count(value: Any) -> int:
    """A non-negative count from gateway metadata (anything else counts as zero)."""
    return value if isinstance(value, int) and value > 0 else 0


class AgentRunner:
    """Runs the loop for one message."""

    def __init__(self, llm: LLMClient, tools: ToolRegistry, *, max_steps: int = 6,
                 correction_retries: int = 1,
                 unverified: UnverifiedAnswers = UnverifiedAnswers.WARN) -> None:
        """Create the runner.

        Args:
            llm: Model access (through the gateway).
            tools: Tools offered to the model.
            max_steps: Most model calls per message (including retries).
            correction_retries: Extra attempts after a failed answer check.
            unverified: What to do when the check still fails.

        Raises:
            ValueError: If ``max_steps`` < 1 or ``correction_retries`` < 0.
        """
        if max_steps < 1 or correction_retries < 0:
            raise ValueError("max_steps must be >= 1 and correction_retries >= 0")
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps
        self._retries = correction_retries
        self._unverified = unverified

    async def run(self, context: ToolContext, messages: Sequence[dict[str, Any]]) -> AgentRun:
        """Answer the last user message in ``messages``.

        Args:
            context: Who the message is from.
            messages: System prompt, history and the new user message.

        Returns:
            The outcome.

        Raises:
            AgentError: The model produced no answer within the step budget.
            AssistantUnavailableError: The gateway failed.
            MessageRefusedError: The gateway refused the message.
        """
        conversation = list(messages)
        definitions = self._tools.definitions()
        traces: list[ToolTrace] = []
        retries_left = self._retries
        corrected = False
        for step in range(1, self._max_steps + 1):
            last_step = step == self._max_steps
            if last_step:
                conversation.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
            reply = await self._llm.complete(conversation, tools=[] if last_step
                                             else definitions, user=context.user_id)
            if reply.tool_calls and not last_step:
                conversation.append(_assistant_turn(reply))
                for call in reply.tool_calls:
                    outcome = await self._tools.invoke(context, call)
                    traces.append(_trace(outcome))
                    conversation.append(outcome.message())
                continue
            answer = (reply.content or "").strip()
            if not answer:
                break
            status, problems = verification_of(reply)
            if status is VerificationStatus.FAILED and retries_left and not last_step:
                retries_left -= 1
                corrected = True
                logger.warning("answer failed the check; asking the model to correct it",
                               extra={"problems": list(problems), "step": step})
                conversation.append({"role": "assistant", "content": answer})
                conversation.append({"role": "user", "content": CORRECTION_PROMPT.format(
                    problems=", ".join(problems) or "unspecified")})
                continue
            return self._finish(answer, status, problems, corrected, traces, step, reply)
        raise AgentError("no answer within the step budget",
                         details={"model_calls": self._max_steps})

    def _finish(self, answer: str, status: VerificationStatus, problems: tuple[str, ...],
                corrected: bool, traces: list[ToolTrace], steps: int,
                reply: LLMReply) -> AgentRun:
        verification = Verification(status, problems,
                                    corrected=corrected and status is VerificationStatus.VERIFIED)
        if status is VerificationStatus.FAILED:
            logger.warning("answer still unverified", extra={"problems": list(problems),
                                                             "policy": self._unverified.value})
            if self._unverified is UnverifiedAnswers.WITHHOLD:
                answer = WITHHELD_ANSWER
        logger.info("agent answered", extra={"model_calls": steps, "tools": len(traces),
                                             "verification": status.value})
        return AgentRun(answer=answer, verification=verification, tools=tuple(traces),
                        model_calls=steps, gateway=reply.gateway)


def _assistant_turn(reply: LLMReply) -> dict[str, Any]:
    return {"role": "assistant", "content": reply.content or None, "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in reply.tool_calls]}


def _trace(outcome: ToolOutcome) -> ToolTrace:
    error = None if outcome.ok else str(outcome.result.get("error", "error"))
    return ToolTrace(outcome.name, outcome.ok, error)
