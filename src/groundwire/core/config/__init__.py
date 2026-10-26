"""Configuration loading and validation.

Configuration is split in two on purpose:

* :class:`Settings` comes from environment variables (``GROUNDWIRE_*``).
  It holds process-level knobs: bind address, log level, and the path to the
  gateway config file. This is what you change per deployment/container.

* :class:`GatewayConfig` comes from a YAML file. It describes *behaviour*:
  providers, routes (fallback chains), resilience policy and guardrails. It is
  versioned with the code and reviewed like code.

Secrets never live in YAML. A provider declares ``api_key_env: OPENAI_API_KEY``
and the key is read from the environment at startup. Missing keys for enabled
providers fail the boot instead of failing the first request.

All validation happens at load time so a bad config never reaches traffic.

Layout (one module per feature, so a change to cache settings never touches
the safety settings file)::

    base        StrictModel: unknown keys are errors
    settings    Settings (environment)
    providers   ProviderConfig, mock behaviour, resolve_api_key
    resilience  retry, circuit breaker, admission (load shedding)
    grounding   grounding verification and evidence retrieval
    safety      prompt injection and PII
    cache       response cache
    agents      tool policy, approvals, answer check
    routing     cascades
    dashboard   live dashboard
    gateway     GatewayConfig: composes the above, checks cross-references
    loader      load_gateway_config, parse_gateway_config

Import from this package (``from groundwire.core.config import CacheConfig``);
the submodules are an internal layout and may move.
"""

from __future__ import annotations

from groundwire.core.config.agents import (
    AnswerCheckAction,
    AnswerCheckConfig,
    ApprovalConfig,
    ApprovalStoreType,
    ArgumentConstraint,
    ToolAction,
    ToolPolicyConfig,
    ToolRule,
    ToolViolationHandling,
)
from groundwire.core.config.base import StrictModel
from groundwire.core.config.cache import CacheConfig
from groundwire.core.config.dashboard import DashboardConfig
from groundwire.core.config.gateway import GatewayConfig
from groundwire.core.config.grounding import (
    AnswerAggregation,
    EvidenceConfig,
    EvidenceRetrieverType,
    GroundingConfig,
    GroundingFailurePolicy,
    GroundingScorerType,
)
from groundwire.core.config.loader import load_gateway_config, parse_gateway_config
from groundwire.core.config.providers import (
    MockBehaviour,
    MockToolCall,
    ProviderConfig,
    ProviderType,
    resolve_api_key,
)
from groundwire.core.config.resilience import AdmissionConfig, CircuitBreakerConfig, RetryConfig
from groundwire.core.config.routing import CascadeCheckFailure, CascadeConfig, CascadeTier
from groundwire.core.config.safety import (
    InjectionConfig,
    InjectionDetectorType,
    PIIConfig,
    SafetyAction,
    SafetyConfig,
)
from groundwire.core.config.settings import Settings

__all__ = [
    "AdmissionConfig",
    "AnswerAggregation",
    "AnswerCheckAction",
    "AnswerCheckConfig",
    "ApprovalConfig",
    "ApprovalStoreType",
    "ArgumentConstraint",
    "CacheConfig",
    "CascadeCheckFailure",
    "CascadeConfig",
    "CascadeTier",
    "CircuitBreakerConfig",
    "DashboardConfig",
    "EvidenceConfig",
    "EvidenceRetrieverType",
    "GatewayConfig",
    "GroundingConfig",
    "GroundingFailurePolicy",
    "GroundingScorerType",
    "InjectionConfig",
    "InjectionDetectorType",
    "MockBehaviour",
    "MockToolCall",
    "PIIConfig",
    "ProviderConfig",
    "ProviderType",
    "RetryConfig",
    "SafetyAction",
    "SafetyConfig",
    "Settings",
    "StrictModel",
    "ToolAction",
    "ToolPolicyConfig",
    "ToolRule",
    "ToolViolationHandling",
    "load_gateway_config",
    "parse_gateway_config",
    "resolve_api_key",
]
