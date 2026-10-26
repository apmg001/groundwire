"""Prompt-injection screening and PII redaction settings.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    Field,
    field_validator,
)

from groundwire.core.config.base import StrictModel


class SafetyAction(StrEnum):
    """What to do when a safety check fires."""

    OFF = "off"  # do not run the check
    FLAG = "flag"  # run it and report findings in the response metadata
    BLOCK = "block"  # refuse the request (HTTP 400)


class InjectionDetectorType(StrEnum):
    """Available prompt-injection detectors."""

    HEURISTIC = "heuristic"  # explainable rules, no dependencies
    CLASSIFIER = "classifier"  # fine-tuned classifier; needs the `nli` extra
    BOTH = "both"  # max of heuristic and classifier


class InjectionConfig(StrictModel):
    """Prompt-injection screening.

    Attributes:
        action: off | flag | block.
        detector: Which detector to run.
        threshold: Score at or above which the action applies.
        scan_user_messages: Scan user turns (direct injection).
        scan_context: Scan grounding context documents (indirect injection).
        classifier_model: Model id for the classifier detector.
    """

    action: SafetyAction = SafetyAction.FLAG
    detector: InjectionDetectorType = InjectionDetectorType.HEURISTIC
    threshold: Annotated[float, Field(gt=0, le=1)] = 0.7
    scan_user_messages: bool = True
    scan_context: bool = True
    classifier_model: str = "protectai/deberta-v3-base-prompt-injection-v2"


class PIIConfig(StrictModel):
    """PII redaction before requests reach model providers.

    Attributes:
        enabled: Redact PII in every message sent to providers.
        types: Categories to redact (default: all supported).
        restore_output: Put original values back into the model's answer.
    """

    enabled: bool = False
    types: list[str] = Field(default_factory=lambda: [
        "email", "phone", "card", "aadhaar", "pan", "ip_address"])
    restore_output: bool = True

    @field_validator("types")
    @classmethod
    def _check_types(cls, value: list[str]) -> list[str]:
        allowed = {"email", "phone", "card", "aadhaar", "pan", "ip_address"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown PII types {unknown}; choose from {sorted(allowed)}")
        return value


class SafetyConfig(StrictModel):
    """Input-side safety settings.

    Attributes:
        injection: Prompt-injection screening.
        pii: PII redaction.
    """

    injection: InjectionConfig = InjectionConfig()
    pii: PIIConfig = PIIConfig()
