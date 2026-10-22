"""Answer check: does an agent's final answer match what its tools actually returned?

The failure this catches was seen with a real model (Qwen 2.5 3B via Ollama):
every tool call was correct and allowed, the refund tool returned
``{"amount": 1, "status": "refunded"}``, and the model then told the user
*"I have refunded $49.99"*. Nothing upstream catches this: the tool policy
checks what an agent **does**, not what it **says**, and grounding is about
documents, not tool results.

What is checked (deterministic, no model call):

1. **Numbers next to a tool's words must come from that tool.** Each number
   in the answer is bound to the closest word that belongs to a tool (a part
   of its name, like *refund* in ``issue_refund``, or one of its result/argument
   field names, like *total*). The number must then appear in that tool's
   arguments or result (rounding to the answer's precision is allowed).
   ``refunded $49.99`` is bound to ``issue_refund``; its data holds ``1``, so
   this is a ``number_mismatch``.
2. **Other numbers must come from somewhere.** A significant number (money,
   percent, decimal, or at least ``min_unbound_value``) that appears in no
   tool data and no user/system message is an ``unsupported_number``.
3. **Claimed actions must have run, and succeeded.** A past-tense claim like
   *refunded* / *deleted* / *emailed*, not negated (*could not be refunded*)
   and not future (*will be refunded*), needs a call of a matching action tool
   in the conversation (else ``action_not_taken``) that did not return an
   error (else ``action_failed``). Read-only tools (``get_*``, ``lookup_*``,
   ``search_*``...) never count as actions.

Limits, stated plainly: this is a precise check for *numbers and completed
actions*, the claims an agent most often gets wrong and that cost the most
when wrong. It does not understand free-form claims ("the item is blue"),
irregular verbs (*sent* for ``send_email``) or numbers written as words. It
favours precision: anything it cannot bind or judge is left alone, and the
default action only flags. For free-form claims, use grounding (LettuceDetect)
with the tool results as context.

Privacy: issues carry the tool name, the kind of problem and the character
span in the answer, never the values themselves (tool data often holds
personal data, and spans are enough to highlight the problem).
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from groundwire.core.config import AnswerCheckAction, AnswerCheckConfig
from groundwire.domain.models import ChatCompletionRequest

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

# Tools whose name starts with one of these only read data; they make no actions.
_READ_VERBS = frozenset({
    "get", "lookup", "look", "fetch", "find", "search", "list", "read", "query", "check",
    "view", "show", "describe", "retrieve", "load", "count", "browse", "inspect",
})
# Name/field parts too generic to tell tools apart.
_GENERIC = _READ_VERBS | frozenset({
    "set", "put", "post", "issue", "make", "run", "call", "execute", "handle", "process",
    "perform", "tool", "api", "function", "the", "for", "and", "with", "from", "into", "all",
    "info", "data", "detail", "details", "result", "results", "value", "values", "type",
    "name", "error", "message", "code", "ids",
})
# Words that turn "refunded" into a non-claim: negation or not-yet-done.
_NOT_A_CLAIM = frozenset({
    # negation
    "not", "never", "no", "nothing", "none", "unable", "cannot", "can't", "couldn't",
    "didn't", "wasn't", "weren't", "isn't", "hasn't", "haven't", "won't", "failed",
    # not done yet / conditional
    "will", "would", "could", "should", "can", "may", "might", "shall", "to", "going",
    "if", "unless",
})
_CLAIM_LOOKBACK = 3  # words before the verb searched for _NOT_A_CLAIM

# Answers longer than this are only partly checked (bounded cost per request).
_MAX_ANSWER_CHARS = 50_000
_MAX_EVIDENCE_VALUES = 20_000

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

_NUMBER = re.compile(
    r"(?P<cur>[$€£₹¥])?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<pct>\s?%)?")
_APOSTROPHE = "\u2019"  # typographic apostrophe (U+2019), normalised to an ASCII one
_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019]*")
# Sentence ends: . ! ? not inside a number (49.99), plus ; and newlines.
_SENTENCE_END = re.compile(r"[.!?](?!\d)|[;\n]")
_NAME_PARTS = re.compile(r"[_\-.\s/]+|(?<=[a-z0-9])(?=[A-Z])")
_SUFFIXES = ("ing", "ed", "es", "s", "e")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class IssueKind(StrEnum):
    """Kinds of problem (stable codes for clients and metrics)."""

    NUMBER_MISMATCH = "number_mismatch"
    UNSUPPORTED_NUMBER = "unsupported_number"
    ACTION_NOT_TAKEN = "action_not_taken"
    ACTION_FAILED = "action_failed"


class CheckVerdict(StrEnum):
    """Outcome of one answer check."""

    SUPPORTED = "supported"
    MISMATCH = "mismatch"
    SKIPPED = "skipped"


_DETAILS: dict[IssueKind, str] = {
    IssueKind.NUMBER_MISMATCH: "a number in the answer differs from what {tool} returned",
    IssueKind.UNSUPPORTED_NUMBER: "a number in the answer appears in no tool result",
    IssueKind.ACTION_NOT_TAKEN: "the answer says an action was done, but {tool} never ran",
    IssueKind.ACTION_FAILED: "the answer says an action was done, but {tool} returned an error",
}


@dataclass(frozen=True, slots=True)
class AnswerIssue:
    """One problem in an answer.

    Attributes:
        kind: What is wrong.
        span: ``(start, end)`` character offsets of the claim in the answer.
        tool: The tool the claim was checked against, if any.
    """

    kind: IssueKind
    span: tuple[int, int]
    tool: str | None = None

    def detail(self) -> str:
        """Return a human-readable explanation (no values, only the tool name)."""
        return _DETAILS[self.kind].format(tool=self.tool or "the tool")

    def as_dict(self) -> dict[str, Any]:
        """Serialise for response metadata."""
        data: dict[str, Any] = {"kind": self.kind.value, "span": list(self.span),
                                "detail": self.detail()}
        if self.tool is not None:
            data["tool"] = self.tool
        return data


@dataclass(frozen=True, slots=True)
class AnswerCheckReport:
    """Outcome of checking one answer.

    Attributes:
        verdict: Supported, mismatch, or skipped (nothing to check against).
        issues: Problems found (empty unless ``verdict`` is mismatch).
        numbers_checked: Numbers in the answer that were checked.
        actions_checked: Completed-action claims that were checked.
        tool_results: Tool results the answer was checked against.
        reason: Why the check was skipped, if it was.
        latency_ms: Time spent checking.
    """

    verdict: CheckVerdict
    issues: tuple[AnswerIssue, ...] = ()
    numbers_checked: int = 0
    actions_checked: int = 0
    tool_results: int = 0
    reason: str | None = None
    latency_ms: float = 0.0

    @property
    def failed(self) -> bool:
        """True when the answer contradicts its tool results."""
        return self.verdict is CheckVerdict.MISMATCH

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the ``groundwire.answer_check`` metadata."""
        if self.verdict is CheckVerdict.SKIPPED:
            return {"verdict": self.verdict.value, "reason": self.reason}
        return {
            "verdict": self.verdict.value,
            "issues": [issue.as_dict() for issue in self.issues],
            "numbers_checked": self.numbers_checked,
            "actions_checked": self.actions_checked,
            "tool_results": self.tool_results,
            "latency_ms": round(self.latency_ms, 3),
        }

    def warning(self) -> str:
        """Text appended to the answer in ``annotate`` mode."""
        details = "; ".join(dict.fromkeys(issue.detail() for issue in self.issues))
        return (f"[Groundwire] Please verify this answer: it does not match the tool "
                f"results ({details}).")


# --------------------------------------------------------------------------- #
# Evidence: what the tools were given and returned
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolEvidence:
    """One executed tool call and its result, from the conversation history.

    Attributes:
        tool: Tool name.
        arguments: Decoded arguments (``None`` if not a JSON object).
        result: Decoded result (JSON value, or the raw text).
        failed: The result reports an error.
    """

    tool: str
    arguments: Any
    result: Any
    failed: bool


def collect_evidence(request: ChatCompletionRequest) -> list[ToolEvidence]:
    """Pair every tool result in the history with the call it answers.

    Args:
        request: Client request (unredacted, as the agent sent it).

    Returns:
        Evidence in conversation order. Results without a matching call keep
        the ``tool`` message's ``name`` (or ``"unknown"``) as the tool name.
    """
    calls: dict[str, tuple[str, Any]] = {}
    evidence: list[ToolEvidence] = []
    for message in request.messages:
        for call in message.tool_calls or ():
            calls[call.id] = (call.function.name, call.parsed_arguments())
        if message.role != "tool" or not message.tool_call_id:
            continue
        name, arguments = calls.get(message.tool_call_id, (message.name or "unknown", None))
        result = _decode(message.content)
        evidence.append(ToolEvidence(tool=name, arguments=arguments, result=result,
                                     failed=_reports_error(result)))
    return evidence


def _decode(content: str) -> Any:
    """Decode a tool result as JSON, falling back to the raw text."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        return content


def _reports_error(result: Any) -> bool:
    """Heuristic: does a tool result say the call failed?

    Args:
        result: Decoded tool result.

    Returns:
        True for ``{"error": ...}``, ``{"status": "error"|"failed"}``,
        ``{"success": false}``, or text starting with "error".
    """
    if isinstance(result, dict):
        if result.get("error"):
            return True
        if str(result.get("status", "")).lower() in {"error", "failed", "failure"}:
            return True
        return result.get("success") is False or result.get("ok") is False
    if isinstance(result, str):
        return result.strip().lower().startswith("error")
    return False


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Token:
    """A word or number in the answer, with its position.

    Attributes:
        start: Start offset in the answer.
        end: End offset.
        sentence: Sentence index (numbers bind within a sentence only).
        text: Lower-cased word (empty for numbers).
        stem: :func:`_stem` of ``text``, computed once.
        value: Numeric value (``None`` for words).
        places: Decimal places written in the answer.
        significant: Money, percent, decimal, or large enough to matter.
        percent: Written with a percent sign.
    """

    start: int
    end: int
    sentence: int
    text: str = ""
    stem: str = ""
    value: Decimal | None = None
    places: int = 0
    significant: bool = False
    percent: bool = False


def _stem(word: str) -> str:
    """Crude, predictable stemmer: refunded/refunds/refunding -> refund.

    Args:
        word: A word.

    Returns:
        The stem (lower case).
    """
    w = word.lower().replace(_APOSTROPHE, "'")
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]  # cancelled -> cancell -> cancel
    return w


def _name_parts(name: str) -> list[str]:
    """Split ``issue_refund`` / ``issueRefund`` / ``order-id`` into lower-case parts."""
    return [p.lower() for p in _NAME_PARTS.split(name) if p]


def _parse_number(match: re.Match[str]) -> tuple[Decimal, int] | None:
    """Turn a number match into ``(value, decimal_places)``."""
    raw = match.group("num").replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    places = len(raw.split(".", 1)[1]) if "." in raw else 0
    return value, places


def _tokenize(answer: str, min_unbound: Decimal) -> list[_Token]:
    """Split an answer into words and numbers, in order, with sentence indexes.

    Args:
        answer: The answer text.
        min_unbound: Threshold for a plain integer to count as significant.

    Returns:
        Tokens sorted by position.
    """
    ends = [m.start() for m in _SENTENCE_END.finditer(answer)]
    tokens: list[_Token] = []
    for match in _NUMBER.finditer(answer):
        parsed = _parse_number(match)
        if parsed is None:
            continue
        value, places = parsed
        start, end = match.start("num"), match.end()
        if match.group("cur"):
            start = match.start("cur")
        before = answer[match.start("num") - 1] if match.start("num") > 0 else ""
        if before.isalpha():
            continue  # part of an identifier like "qwen2" or "v2"
        percent = bool(match.group("pct"))
        significant = bool(match.group("cur")) or percent or places > 0 or value >= min_unbound
        tokens.append(_Token(start=start, end=end, sentence=bisect.bisect_left(ends, start),
                             value=value, places=places, significant=significant,
                             percent=percent))
    for match in _WORD.finditer(answer):
        text = match.group().lower().replace(_APOSTROPHE, "'")
        tokens.append(_Token(start=match.start(), end=match.end(),
                             sentence=bisect.bisect_left(ends, match.start()),
                             text=text, stem=_stem(text)))
    tokens.sort(key=lambda t: t.start)
    return tokens


def _numbers_in(value: Any) -> Iterator[Decimal]:
    """Yield every number inside a JSON value (including numbers inside strings).

    Args:
        value: Decoded JSON (or text).

    Yields:
        Numbers as ``Decimal`` (``bool`` is not a number).
    """
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, bool) or item is None:
            continue
        if isinstance(item, int | float):
            try:
                yield Decimal(str(item))
            except InvalidOperation:  # nan / inf
                continue
        elif isinstance(item, str):
            for match in _NUMBER.finditer(item):
                parsed = _parse_number(match)
                if parsed is not None:
                    yield parsed[0]
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, list | tuple):
            stack.extend(item)


def _field_names(value: Any) -> Iterator[str]:
    """Yield every object key inside a JSON value (iteratively, any depth)."""
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                yield str(key)
                stack.append(child)
        elif isinstance(item, list | tuple):
            stack.extend(item)


def _anchor_parts(parts: Iterable[str]) -> set[str]:
    """Stems of the name/field parts that can identify a tool."""
    return {_stem(p) for p in parts if len(p) >= 3 and p not in _GENERIC}


class _ValueSet:
    """Numbers from tool data, with fast "equal at the answer's precision" lookups.

    ``$1,234.57`` matches ``1234.5678``; ``15%`` matches ``15`` or ``0.15``.
    Rounded copies are built once per (precision, percent) and cached, so each
    number in the answer costs a set lookup, not a scan of every tool value.
    """

    __slots__ = ("_exact", "_rounded")

    def __init__(self, values: Iterable[Decimal]) -> None:
        """Create the set.

        Args:
            values: Numbers found in tool data (and messages).
        """
        self._exact: frozenset[Decimal] = frozenset(values)
        self._rounded: dict[tuple[int, bool], frozenset[Decimal]] = {}

    def contains(self, token: _Token) -> bool:
        """True if the answer's number equals a value, at the answer's precision.

        Args:
            token: Number token from the answer.

        Returns:
            Whether any value matches.
        """
        target = token.value
        if target is None:
            return False
        if target in self._exact:
            return True
        key = (token.places, token.percent)
        rounded = self._rounded.get(key)
        if rounded is None:
            rounded = self._rounded[key] = frozenset(self._round(*key))
        return target in rounded

    def _round(self, places: int, percent: bool) -> Iterator[Decimal]:
        """Yield every value rounded to ``places`` (and x100 for percentages)."""
        quantum = Decimal(1).scaleb(-places)
        for value in self._exact:
            for candidate in ((value, value * 100) if percent else (value,)):
                try:
                    yield candidate.quantize(quantum, rounding=ROUND_HALF_UP)
                except InvalidOperation:  # too large for the requested precision
                    continue


# --------------------------------------------------------------------------- #
# Checker
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _ToolIndex:
    """What the checker knows about the tools in one conversation.

    Attributes:
        anchors: Word stem -> tools it points to (name parts and field names).
        actions: Word stem -> action tools it names (``refund`` -> ``issue_refund``).
        values: Tool -> numbers in its arguments and results.
        evidence: Tool -> its executed calls.
        everything: All numbers in tool data and in user/system messages.
    """

    anchors: dict[str, set[str]] = field(default_factory=dict)
    actions: dict[str, set[str]] = field(default_factory=dict)
    values: dict[str, set[Decimal]] = field(default_factory=dict)
    evidence: dict[str, list[ToolEvidence]] = field(default_factory=dict)
    everything: set[Decimal] = field(default_factory=set)


class AnswerChecker:
    """Checks agents' final answers against their tool results.

    Thread-safe and stateless between calls: one instance serves the whole
    gateway.
    """

    def __init__(self, config: AnswerCheckConfig) -> None:
        """Create the checker.

        Args:
            config: Answer-check settings.
        """
        self._config = config
        self._min_unbound = Decimal(str(config.min_unbound_value))

    @property
    def enabled(self) -> bool:
        """Whether the check runs."""
        return self._config.enabled

    @property
    def action(self) -> AnswerCheckAction:
        """What happens to an answer with issues."""
        return self._config.action

    def applies_to(self, request: ChatCompletionRequest) -> bool:
        """True if this request's answer will be checked (it has tool results).

        Args:
            request: Client request.

        Returns:
            Whether :meth:`check` would have anything to check against.
        """
        return self.enabled and any(m.role == "tool" for m in request.messages)

    def check(self, answer: str, request: ChatCompletionRequest) -> AnswerCheckReport:
        """Check a final (text) answer against the conversation's tool results.

        Never raises for bad input: malformed tool data or an internal error
        yields a ``skipped`` report and a log line, so a heuristic can never
        take a request down (the answer is delivered unchecked).

        Args:
            answer: The final answer, PII-restored (what the user will read).
            request: The client's request, unredacted (holds the tool results).

        Returns:
            The report.
        """
        started = time.perf_counter()
        if not self.enabled:
            return AnswerCheckReport(verdict=CheckVerdict.SKIPPED, reason="disabled")
        if not answer.strip():
            return AnswerCheckReport(verdict=CheckVerdict.SKIPPED, reason="empty_answer")
        try:
            evidence = collect_evidence(request)
            if not evidence:
                return AnswerCheckReport(verdict=CheckVerdict.SKIPPED,
                                         reason="no_tool_results")
            index = self._index(evidence, request)
            tokens = _tokenize(answer[:_MAX_ANSWER_CHARS], self._min_unbound)
            issues, numbers = self._check_numbers(tokens, index)
            action_issues, actions = self._check_actions(tokens, index)
        except (ValueError, TypeError, ArithmeticError, RecursionError):
            logger.exception("answer check failed; answer delivered unchecked")
            return AnswerCheckReport(verdict=CheckVerdict.SKIPPED, reason="check_error")

        issues.extend(action_issues)
        issues.sort(key=lambda issue: issue.span)
        return AnswerCheckReport(
            verdict=CheckVerdict.MISMATCH if issues else CheckVerdict.SUPPORTED,
            issues=tuple(issues), numbers_checked=numbers, actions_checked=actions,
            tool_results=len(evidence), latency_ms=(time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _index(self, evidence: Sequence[ToolEvidence],
               request: ChatCompletionRequest) -> _ToolIndex:
        """Build the per-tool word and value index for one conversation.

        Args:
            evidence: Executed calls and results.
            request: Client request (offered tools, user/system messages).

        Returns:
            The index.
        """
        index = _ToolIndex()
        offered = [tool.function.name for tool in request.tools or ()]
        for name in dict.fromkeys([*offered, *(e.tool for e in evidence)]):
            parts = _name_parts(name)
            for stem in _anchor_parts(parts):
                index.anchors.setdefault(stem, set()).add(name)
                if parts and parts[0] not in _READ_VERBS:
                    index.actions.setdefault(stem, set()).add(name)

        budget = _MAX_EVIDENCE_VALUES
        for item in evidence:
            index.evidence.setdefault(item.tool, []).append(item)
            keys = [*_field_names(item.arguments), *_field_names(item.result)]
            for stem in _anchor_parts(part for key in keys for part in _name_parts(key)):
                index.anchors.setdefault(stem, set()).add(item.tool)
            numbers = [*_numbers_in(item.arguments), *_numbers_in(item.result)][:budget]
            budget -= len(numbers)
            index.values.setdefault(item.tool, set()).update(numbers)
            index.everything.update(numbers)
            if budget <= 0:
                logger.warning("answer check: tool data truncated to %d numbers",
                               _MAX_EVIDENCE_VALUES)
                break
        for message in request.messages:
            if message.role in ("user", "system"):
                index.everything.update(_numbers_in(message.content))
        return index

    def _check_numbers(self, tokens: Sequence[_Token],
                       index: _ToolIndex) -> tuple[list[AnswerIssue], int]:
        """Check every number in the answer (rules 1 and 2 in the module docs).

        Args:
            tokens: Answer tokens.
            index: Tool index.

        Returns:
            ``(issues, numbers_checked)``.
        """
        issues: list[AnswerIssue] = []
        checked = 0
        window = self._config.window_words
        everything = _ValueSet(index.everything)
        per_tools: dict[frozenset[str], _ValueSet] = {}  # built once per tool combination
        for position, token in enumerate(tokens):
            if token.value is None:
                continue
            bound = frozenset(self._bind(tokens, position, window, index))
            if bound:
                checked += 1
                values = per_tools.get(bound)
                if values is None:
                    values = per_tools[bound] = _ValueSet(
                        value for tool in bound for value in index.values.get(tool, ()))
                if not values.contains(token):
                    issues.append(AnswerIssue(IssueKind.NUMBER_MISMATCH,
                                              (token.start, token.end), tool=min(bound)))
            elif token.significant:
                checked += 1
                if not everything.contains(token):
                    issues.append(AnswerIssue(IssueKind.UNSUPPORTED_NUMBER,
                                              (token.start, token.end)))
        return issues, checked

    @staticmethod
    def _bind(tokens: Sequence[_Token], position: int, window: int,
              index: _ToolIndex) -> set[str]:
        """Find the tools whose words are closest to the number at ``position``.

        Only tools that actually ran are candidates (a number cannot be checked
        against a tool with no data), and only words in the same sentence and
        within ``window`` tokens count. Ties keep every tied tool, so a number
        passes if *any* of them supports it (precision over recall).

        Args:
            tokens: Answer tokens.
            position: Index of the number token.
            window: Maximum distance in tokens.
            index: Tool index.

        Returns:
            The bound tools (empty if none is close enough).
        """
        sentence = tokens[position].sentence
        for distance in range(1, window + 1):
            found: set[str] = set()
            for neighbour in (position - distance, position + distance):
                if not 0 <= neighbour < len(tokens):
                    continue
                token = tokens[neighbour]
                if token.sentence != sentence or not token.text:
                    continue
                tools = index.anchors.get(token.stem, set())
                found |= {tool for tool in tools if tool in index.evidence}
            if found:
                return found
        return set()

    @staticmethod
    def _check_actions(tokens: Sequence[_Token],
                       index: _ToolIndex) -> tuple[list[AnswerIssue], int]:
        """Check claims of completed actions (rule 3 in the module docs).

        Args:
            tokens: Answer tokens.
            index: Tool index.

        Returns:
            ``(issues, actions_checked)``.
        """
        issues: list[AnswerIssue] = []
        checked = 0
        for position, token in enumerate(tokens):
            if not token.text.endswith("ed") or len(token.text) < 5:
                continue
            tools = index.actions.get(token.stem)
            if not tools:
                continue
            before = [t.text for t in tokens[max(0, position - _CLAIM_LOOKBACK):position]
                      if t.text and t.sentence == token.sentence]
            if any(word in _NOT_A_CLAIM or word.endswith("n't") for word in before):
                continue  # "could not be refunded", "will be refunded"
            checked += 1
            runs = [e for tool in tools for e in index.evidence.get(tool, [])]
            if not runs:
                issues.append(AnswerIssue(IssueKind.ACTION_NOT_TAKEN,
                                          (token.start, token.end), tool=min(tools)))
            elif all(run.failed for run in runs):
                issues.append(AnswerIssue(IssueKind.ACTION_FAILED,
                                          (token.start, token.end), tool=min(tools)))
        return issues, checked
