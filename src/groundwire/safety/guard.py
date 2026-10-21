"""SafetyGuard: applies injection screening and PII redaction to a request.

The guard is the single entry point the chat service uses, so the service
never needs to know which detectors exist::

    prepared = guard.prepare(request, grounding_context)   # may raise
    routed   = await router.complete(route, prepared.request)
    text     = guard.restore(routed.completion.text, prepared)

Order matters:

1. **Injection screening runs on the original text**, before redaction, so
   placeholders can't hide an attack. Only user turns (direct injection) and
   grounding documents (indirect injection) are scanned; system prompts are
   written by the application, not attackers, and scanning them causes false
   positives.
2. **PII is redacted in every message** sent to the provider, including system
   prompts (RAG apps often paste customer records there).
3. Grounding context is *not* redacted: it never leaves the gateway, and the
   fact-checker needs the real values to compare against the restored answer.

Agents add two paths that need the same care:

* **Tool results** (``tool`` messages) are untrusted external data, like a web
  page or an email body, so they are scanned for *indirect* injection (when
  ``scan_context`` is on) and redacted like any other message.
* **Tool-call arguments** in the history are redacted, and arguments in the
  model's new tool calls are restored, so the real email reaches the real
  tool while the provider only ever saw ``[EMAIL_1]``. Both are done inside
  the JSON, so the arguments stay valid JSON.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from groundwire.core.config import InjectionDetectorType, SafetyAction, SafetyConfig
from groundwire.core.errors import GuardrailUnavailableError, PromptInjectionError
from groundwire.domain.models import ChatCompletionRequest, ChatMessage, FunctionCall, ToolCall
from groundwire.observability.metrics import GatewayMetrics
from groundwire.safety.injection import (
    ClassifierInjectionDetector,
    CombinedInjectionDetector,
    HeuristicInjectionDetector,
    InjectionDetector,
    InjectionReport,
)
from groundwire.safety.pii import (
    PIIRedactor,
    PIIType,
    PIIVault,
    RegexPIIDetector,
    StreamRestorer,
    transform_json_text,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedRequest:
    """A request after safety processing.

    Attributes:
        request: Request to send to providers (PII redacted if enabled).
        vault: Placeholder mapping for this request (empty if nothing redacted).
        injection: Injection report (``None`` if screening is off).
        injection_action: ``"none"``, ``"flagged"`` or ``"blocked"``.
    """

    request: ChatCompletionRequest
    vault: PIIVault = field(default_factory=PIIVault)
    injection: InjectionReport | None = None
    injection_action: str = "none"

    def metadata(self) -> dict[str, Any]:
        """Safety metadata for the response (no raw PII, no attack text)."""
        meta: dict[str, Any] = {}
        if self.injection is not None:
            meta["injection"] = {**self.injection.as_dict(), "action": self.injection_action}
        if self.vault:
            meta["pii_redacted"] = self.vault.counts()
        return meta


class SafetyGuard:
    """Applies the configured safety policy to requests and responses."""

    def __init__(self, config: SafetyConfig, *, metrics: GatewayMetrics | None = None,
                 injection_detector: InjectionDetector | None = None,
                 redactor: PIIRedactor | None = None) -> None:
        """Create the guard.

        Args:
            config: Safety settings.
            metrics: Metrics sink.
            injection_detector: Detector override (tests); built from config if omitted.
            redactor: Redactor override (tests); built from config if omitted.

        Raises:
            ConfigurationError: If a classifier is configured but cannot load.
        """
        self._config = config
        self._metrics = metrics
        self._detector = injection_detector
        if self._detector is None and config.injection.action is not SafetyAction.OFF:
            self._detector = build_injection_detector(config)
        self._redactor = redactor
        if self._redactor is None and config.pii.enabled:
            types = [PIIType(t) for t in config.pii.types]
            self._redactor = PIIRedactor(RegexPIIDetector(types))

    # ------------------------------------------------------------------ #
    # Request side
    # ------------------------------------------------------------------ #

    def prepare(self, request: ChatCompletionRequest,
                grounding_context: list[str] | None = None) -> PreparedRequest:
        """Screen for injection, then redact PII.

        Args:
            request: Client request.
            grounding_context: Documents supplied for grounding, if any.

        Returns:
            The prepared request.

        Raises:
            PromptInjectionError: If the injection score reaches the threshold
                and the action is ``block``.
            GuardrailUnavailableError: If the injection detector fails (the
                request is refused rather than passed through unchecked).
        """
        prepared = PreparedRequest(request=request)

        if self._detector is not None:
            prepared.injection = self._scan(request, grounding_context or [])
            prepared.injection_action = self._apply_injection_policy(prepared.injection)

        if self._redactor is not None:
            redacted_messages = [self._redact_message(message, prepared.vault)
                                 for message in request.messages]
            if prepared.vault:
                prepared.request = request.model_copy(update={"messages": redacted_messages})
                counts = prepared.vault.counts()
                logger.info("pii redacted", extra={"pii_counts": counts})
                if self._metrics:
                    for pii_type, count in counts.items():
                        self._metrics.pii_redactions.labels(type=pii_type).inc(count)
        return prepared

    def _redact_message(self, message: ChatMessage, vault: PIIVault) -> ChatMessage:
        """Redact a message's text and any tool-call arguments it carries.

        Args:
            message: Message.
            vault: Request vault (updated in place).

        Returns:
            The redacted copy.
        """
        assert self._redactor is not None  # guarded by caller
        redactor = self._redactor
        update: dict[str, object] = {"content": redactor.redact(message.content, vault)}
        if message.tool_calls:
            update["tool_calls"] = [
                call.model_copy(update={"function": FunctionCall(
                    name=call.function.name,
                    arguments=transform_json_text(call.function.arguments,
                                                  lambda text: redactor.redact(text, vault)))})
                for call in message.tool_calls]
        return message.model_copy(update=update)

    def _scan(self, request: ChatCompletionRequest, context: list[str]) -> InjectionReport:
        assert self._detector is not None  # guarded by caller
        settings = self._config.injection
        report = InjectionReport()
        try:
            if settings.scan_user_messages:
                for message in request.messages:
                    if message.role == "user":
                        report.merge(self._detector.scan(message.content, source="user"))
            if settings.scan_context:
                for document in context:
                    report.merge(self._detector.scan(document, source="context"))
                # Tool results are outside data too: the main way agents get attacked.
                for message in request.messages:
                    if message.role == "tool" and message.content:
                        report.merge(self._detector.scan(message.content, source="tool"))
        except GuardrailUnavailableError:
            logger.exception("injection detector failed; refusing request")
            raise
        return report

    def _apply_injection_policy(self, report: InjectionReport) -> str:
        settings = self._config.injection
        triggered = report.score >= settings.threshold
        action = ("blocked" if triggered and settings.action is SafetyAction.BLOCK
                  else "flagged" if report.findings else "none")
        if self._metrics:
            self._metrics.injection_checks.labels(action=action).inc()
        if report.findings:
            logger.warning("possible prompt injection", extra={
                "score": round(report.score, 3), "action": action,
                "rules": sorted({f.rule for f in report.findings}),
                "sources": sorted({f.source for f in report.findings})})
        if action == "blocked":
            raise PromptInjectionError(
                "request blocked: it looks like a prompt-injection attempt",
                details={"injection": report.as_dict()},
            )
        return action

    # ------------------------------------------------------------------ #
    # Response side
    # ------------------------------------------------------------------ #

    def restore(self, text: str, prepared: PreparedRequest) -> str:
        """Put original PII values back into model output.

        Args:
            text: Model output.
            prepared: The prepared request.

        Returns:
            Restored text (unchanged if restoration is disabled or nothing was redacted).
        """
        if not self._config.pii.restore_output:
            return text
        return PIIRedactor.restore(text, prepared.vault)

    def restore_tool_calls(self, calls: list[ToolCall],
                           prepared: PreparedRequest) -> list[ToolCall]:
        """Put original PII values back into tool-call arguments (JSON-safe).

        Args:
            calls: Tool calls from the model.
            prepared: The prepared request.

        Returns:
            Restored calls (unchanged if nothing was redacted).
        """
        if not calls or not prepared.vault or not self._config.pii.restore_output:
            return calls
        return [call.model_copy(update={"function": FunctionCall(
                    name=call.function.name,
                    arguments=transform_json_text(
                        call.function.arguments,
                        lambda text: PIIRedactor.restore(text, prepared.vault)))})
                for call in calls]

    def stream_restorer(self, prepared: PreparedRequest) -> StreamRestorer | None:
        """Create a restorer for streaming output.

        Args:
            prepared: The prepared request.

        Returns:
            A restorer, or ``None`` if there is nothing to restore.
        """
        if not prepared.vault or not self._config.pii.restore_output:
            return None
        return StreamRestorer(prepared.vault)


def build_injection_detector(config: SafetyConfig) -> InjectionDetector:
    """Build the configured injection detector.

    Args:
        config: Safety settings.

    Returns:
        The detector.

    Raises:
        ConfigurationError: If the classifier is required but unavailable.
    """
    kind = config.injection.detector
    if kind is InjectionDetectorType.HEURISTIC:
        return HeuristicInjectionDetector()
    classifier = ClassifierInjectionDetector.load(config.injection.classifier_model)
    if kind is InjectionDetectorType.CLASSIFIER:
        return classifier
    return CombinedInjectionDetector([HeuristicInjectionDetector(), classifier])
