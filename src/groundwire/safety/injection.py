"""Prompt-injection detection.

Two detectors behind one interface:

* :class:`HeuristicInjectionDetector`: weighted rules for well-known attack
  patterns (instruction override, role hijack, system-prompt extraction, fake
  chat-template markers, data exfiltration via links). Dependency-free, about
  a millisecond, and every finding is explainable ("rule X matched here").
* :class:`ClassifierInjectionDetector`: a fine-tuned text classifier
  (default ``protectai/deberta-v3-base-prompt-injection-v2``, Apache-2.0)
  that generalises beyond fixed phrases. Needs the ``nli`` extra.
  Its model card notes it is English-only, does not target jailbreaks, and
  is not recommended for scanning system prompts.

:class:`CombinedInjectionDetector` takes the maximum of several detectors.

Text is normalised before matching (Unicode NFKC, zero-width characters
removed, whitespace collapsed, lower-cased) so trivial obfuscation such as
``"ig\\u200bnore previous instructions"`` still matches.

**No detector stops prompt injection on its own.** This is one layer; the
others (least privilege, deterministic decisions outside the model, output
checks such as grounding) are architectural. See the README.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from groundwire.core.errors import ConfigurationError, GuardrailUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))


def normalise(text: str) -> str:
    """Normalise text so cosmetic obfuscation does not hide an attack.

    Args:
        text: Raw text.

    Returns:
        NFKC-normalised, zero-width-free, whitespace-collapsed, lower-case text.
    """
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    """One reason a text looks like an injection attempt.

    Attributes:
        rule: Rule or model identifier.
        category: Attack category.
        weight: Contribution to the score, in [0, 1].
        source: Where the text came from (``"user"``, ``"context"``...).
        excerpt: Short excerpt of the matched text (for audit logs).
    """

    rule: str
    category: str
    weight: float
    source: str
    excerpt: str


@dataclass(slots=True)
class InjectionReport:
    """Aggregate result of scanning one or more texts.

    Attributes:
        score: Overall likelihood of injection in [0, 1].
        findings: Individual findings.
    """

    score: float = 0.0
    findings: list[InjectionFinding] = field(default_factory=list)

    def merge(self, other: InjectionReport) -> None:
        """Fold another report into this one (score = max).

        Args:
            other: Report to merge.
        """
        self.score = max(self.score, other.score)
        self.findings.extend(other.findings)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (rule ids only, no raw excerpts)."""
        return {
            "score": round(self.score, 4),
            "findings": [{"rule": f.rule, "category": f.category, "source": f.source,
                          "weight": round(f.weight, 3)} for f in self.findings],
        }


class InjectionDetector(ABC):
    """Scores text for prompt-injection attempts."""

    name: str = "base"

    @abstractmethod
    def scan(self, text: str, *, source: str) -> InjectionReport:
        """Scan one text.

        Args:
            text: Text to scan.
            source: Label for where the text came from.

        Returns:
            The report (score 0.0 means nothing suspicious found).

        Raises:
            GuardrailUnavailableError: If the detector cannot run.
        """


# --------------------------------------------------------------------------- #
# Heuristic rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class InjectionRule:
    """A weighted regex rule applied to normalised text.

    Attributes:
        id: Stable rule identifier.
        category: Attack category.
        weight: Confidence that a match is malicious, in (0, 1].
        pattern: Compiled pattern (matched against normalised text).
    """

    id: str
    category: str
    weight: float
    pattern: re.Pattern[str]


_TARGET = r"(?:instructions?|prompts?|rules|directions|guidelines|guardrails|context)"
_FILLER = r"(?:(?:all|any|the|of|your|my|these|those|every)\s+)"
_PRIOR = r"(?:previous|prior|above|earlier|preceding|original|initial|system|safety)"

DEFAULT_RULES: tuple[InjectionRule, ...] = (
    InjectionRule("override.ignore_instructions", "instruction_override", 0.85, re.compile(
        rf"\b(?:ignore|disregard|forget|override|bypass)\s+{_FILLER}{{0,3}}"
        rf"(?:{_PRIOR}\s+)?(?:\w+\s+)?{_TARGET}\b")),
    InjectionRule("override.ignore_above", "instruction_override", 0.6, re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|everything\s+)?(?:the\s+)?"
        r"(?:above|everything (?:before|above))\b")),
    InjectionRule("override.new_instructions", "instruction_override", 0.6, re.compile(
        r"\b(?:new|updated|real|actual)\s+(?:system\s+)?instructions?\s*(?::|are\b)")),
    InjectionRule("override.do_not_follow", "instruction_override", 0.7, re.compile(
        rf"\b(?:do not|don't|stop)\s+(?:follow|obey|listen to)\s+{_FILLER}{{0,2}}"
        rf"(?:{_PRIOR}\s+)?{_TARGET}\b")),
    InjectionRule("role.you_are_now", "role_hijack", 0.55, re.compile(
        r"\byou are (?:now|no longer)\b|\bfrom now on,? you (?:are|will|must)\b")),
    InjectionRule("role.jailbreak_persona", "role_hijack", 0.8, re.compile(
        r"\b(?:dan mode|do anything now|developer mode|jailbreak(?:ed)?|god mode|"
        r"unrestricted mode|unfiltered mode)\b")),
    InjectionRule("exfil.system_prompt", "prompt_extraction", 0.8, re.compile(
        r"\b(?:reveal|show|print|repeat|output|display|leak|tell me|what (?:is|are|was))\b"
        r"(?:\s+\w+){0,3}?\s+(?:(?:the|your)\s+)?(?:system prompt|hidden (?:prompt|instructions?)|"
        r"initial (?:prompt|instructions?)|original (?:prompt|instructions?)|"
        r"your (?:system |original |initial )?prompt)\b")),
    InjectionRule("markers.chat_template", "fake_role_marker", 0.75, re.compile(
        r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[/?inst\]|<<sys>>|"
        r"(?:^|\s)#{2,}\s*(?:system|assistant)\s*:|(?:^|\s)system\s*:\s*you\b")),
    InjectionRule("exfil.markdown_link", "data_exfiltration", 0.7, re.compile(
        r"!\[[^\]]*\]\(https?://[^)\s]*[?&][^)\s]*=")),
    InjectionRule("exfil.send_data", "data_exfiltration", 0.4, re.compile(
        r"\b(?:send|post|upload|forward|leak)\b(?:\s+\w+){0,4}?\s+"
        r"(?:to|at)\s+(?:https?://\S+)")),
    InjectionRule("context.addressed_to_ai", "indirect_injection", 0.5, re.compile(
        r"\b(?:attention|note|message|instructions?) (?:to|for) (?:the )?(?:ai|assistant|"
        r"chatbot|llm|language model)\b|\bif you are an? (?:ai|assistant|llm|language model)\b")),
)


class HeuristicInjectionDetector(InjectionDetector):
    """Explainable rule-based detector.

    Scores combine like independent evidence ("noisy-OR"):
    ``score = 1 - prod(1 - weight)`` over the distinct rules that matched.
    """

    name = "heuristic"

    def __init__(self, rules: Sequence[InjectionRule] = DEFAULT_RULES) -> None:
        """Create the detector.

        Args:
            rules: Rules to apply.
        """
        self._rules = tuple(rules)

    def scan(self, text: str, *, source: str) -> InjectionReport:
        """Scan one text. See :meth:`InjectionDetector.scan`."""
        normalised = normalise(text)
        findings: list[InjectionFinding] = []
        remaining = 1.0
        for rule in self._rules:
            match = rule.pattern.search(normalised)
            if match is None:
                continue
            findings.append(InjectionFinding(rule.id, rule.category, rule.weight, source,
                                             match.group(0)[:80]))
            remaining *= 1.0 - rule.weight
        return InjectionReport(score=1.0 - remaining, findings=findings)


# --------------------------------------------------------------------------- #
# ML classifier
# --------------------------------------------------------------------------- #


class ClassifierInjectionDetector(InjectionDetector):
    """Fine-tuned sequence classifier returning P(injection).

    Long texts are scored in overlapping character windows and the maximum is
    kept, so an attack buried deep in a document is not truncated away.
    """

    name = "classifier"

    def __init__(self, *, tokenizer: Any, model: Any, torch_module: Any,
                 max_length: int = 512, window_chars: int = 1500,
                 model_name: str = "custom") -> None:
        """Wrap a loaded tokenizer and classification model.

        Args:
            tokenizer: Hugging Face tokenizer.
            model: Sequence-classification model with ``config.id2label``.
            torch_module: The ``torch`` module.
            max_length: Token limit per window.
            window_chars: Characters per window.
            model_name: Label for findings and logs.

        Raises:
            ConfigurationError: If settings are invalid or the model has no
                recognisable injection label.
        """
        if max_length < 16 or window_chars < 100:
            raise ConfigurationError("classifier max_length >= 16 and window_chars >= 100")
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._max_length = max_length
        self._window = window_chars
        self.model_name = model_name
        self._label_idx = self._find_injection_index(model.config.id2label)

    @staticmethod
    def _find_injection_index(id2label: dict[int, str]) -> int:
        """Locate the 'injection' class.

        Args:
            id2label: Model's index -> label mapping.

        Returns:
            Index of the injection label (falls back to 1 for binary models
            whose labels are ``LABEL_0``/``LABEL_1``).

        Raises:
            ConfigurationError: If no injection label can be identified.
        """
        for idx, label in id2label.items():
            if "inject" in str(label).lower() or str(label).lower() in ("unsafe", "malicious"):
                return int(idx)
        if len(id2label) == 2 and {str(v).upper() for v in id2label.values()} == {
                "LABEL_0", "LABEL_1"}:
            return 1
        raise ConfigurationError(f"cannot find an injection label in {id2label}")

    @classmethod
    def load(cls, model_name: str = DEFAULT_INJECTION_MODEL, *,
             num_threads: int | None = None) -> ClassifierInjectionDetector:
        """Download (or read from cache) and load the classifier.

        Args:
            model_name: Hugging Face model id or local directory.
            num_threads: CPU threads for torch.

        Returns:
            A ready detector.

        Raises:
            ConfigurationError: If the ``nli`` extra is missing or the model
                cannot be loaded.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "the injection classifier needs the 'nli' extra: make install-nli") from exc
        if num_threads is not None:
            torch.set_num_threads(max(1, num_threads))
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot load injection model '{model_name}': {exc}") from exc
        model.eval()
        logger.info("injection classifier loaded", extra={"model": model_name})
        return cls(tokenizer=tokenizer, model=model, torch_module=torch, model_name=model_name)

    def _windows(self, text: str) -> list[str]:
        if len(text) <= self._window:
            return [text]
        step = self._window * 3 // 4  # 25% overlap so a phrase on a boundary is still seen
        return [text[i:i + self._window] for i in range(0, len(text) - self._window // 4, step)]

    def scan(self, text: str, *, source: str) -> InjectionReport:
        """Scan one text. See :meth:`InjectionDetector.scan`."""
        if not text.strip():
            return InjectionReport()
        windows = self._windows(text)
        try:
            encoded = self._tokenizer(windows, truncation=True, padding=True,
                                      max_length=self._max_length, return_tensors="pt")
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
            probs = self._torch.softmax(logits, dim=-1)[:, self._label_idx].tolist()
        except (RuntimeError, ValueError) as exc:
            raise GuardrailUnavailableError(f"injection classifier failed: {exc}") from exc
        best = max(range(len(probs)), key=probs.__getitem__)
        score = float(probs[best])
        findings = [InjectionFinding(f"classifier:{self.model_name}", "classifier", score,
                                     source, windows[best][:80])] if score >= 0.5 else []
        return InjectionReport(score=score, findings=findings)


class CombinedInjectionDetector(InjectionDetector):
    """Runs several detectors and keeps the highest score."""

    name = "combined"

    def __init__(self, detectors: Sequence[InjectionDetector]) -> None:
        """Create the detector.

        Args:
            detectors: Detectors to run (at least one).

        Raises:
            ConfigurationError: If ``detectors`` is empty.
        """
        if not detectors:
            raise ConfigurationError("CombinedInjectionDetector needs at least one detector")
        self._detectors = tuple(detectors)

    def scan(self, text: str, *, source: str) -> InjectionReport:
        """Scan with every detector. See :meth:`InjectionDetector.scan`."""
        report = InjectionReport()
        for detector in self._detectors:
            report.merge(detector.scan(text, source=source))
        return report
