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
"""

from __future__ import annotations

import logging
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from groundwire.core.errors import ConfigurationError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Process settings (environment)
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Process-level settings read from ``GROUNDWIRE_*`` environment variables.

    Attributes:
        config_path: Path to the gateway YAML config.
        host: Interface to bind.
        port: Port to bind.
        log_level: Root log level name.
        log_json: Emit JSON logs (production) instead of human-readable text.
        workers: Uvicorn worker processes.
    """

    model_config = SettingsConfigDict(env_prefix="GROUNDWIRE_", extra="ignore")

    config_path: Path = Path("config/groundwire.yaml")
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    log_level: str = "INFO"
    log_json: bool = True
    workers: Annotated[int, Field(ge=1, le=64)] = 1

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level '{value}'")
        return level


# --------------------------------------------------------------------------- #
# Gateway config (YAML)
# --------------------------------------------------------------------------- #


class _StrictModel(BaseModel):
    """Base for config models: unknown keys are errors (catches typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderType(StrEnum):
    """Supported provider adapter types."""

    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI, llama.cpp server, Ollama, vLLM, Groq...
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # deterministic fake for tests, load and chaos testing


class MockToolCall(_StrictModel):
    """A tool call a ``mock`` provider makes when the request offers that tool.

    Attributes:
        name: Tool (function) name.
        arguments: Arguments object.
    """

    name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    arguments: dict[str, Any] = Field(default_factory=dict)


class MockBehaviour(_StrictModel):
    """Behaviour of a ``mock`` provider.

    Attributes:
        response_text: Text the mock returns.
        latency_s: Artificial latency per call.
        failure_rate: Probability in [0, 1] that a call fails with a server error.
        seed: Seed for the failure RNG, for reproducible chaos tests.
        tool_call: Call this tool when it is offered and the last message is not
            already a tool result (so an agent loop ends with ``response_text``).
    """

    response_text: str = "This is a mock response from Groundwire."
    latency_s: Annotated[float, Field(ge=0, le=60)] = 0.0
    failure_rate: Annotated[float, Field(ge=0, le=1)] = 0.0
    seed: int | None = None
    tool_call: MockToolCall | None = None


class ProviderConfig(_StrictModel):
    """One upstream provider.

    Attributes:
        name: Unique provider name used in routes, logs and metrics.
        type: Adapter type.
        model: Upstream model identifier sent to the provider.
        base_url: Provider base URL (required for HTTP providers).
        api_key_env: Name of the env var holding the API key, if any.
        timeout_s: Per-call timeout (connect + read), in seconds.
        max_connections: Connection-pool size for this provider.
        enabled: Disabled providers are skipped and need no API key.
        default_max_tokens: Used when the client sends no ``max_tokens``
            (Anthropic requires the field).
        mock: Behaviour for ``mock`` providers.
    """

    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.\-]+$")]
    type: ProviderType
    model: Annotated[str, Field(min_length=1)]
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: Annotated[float, Field(gt=0, le=600)] = 60.0
    max_connections: Annotated[int, Field(ge=1, le=1000)] = 50
    enabled: bool = True
    default_max_tokens: Annotated[int, Field(ge=1, le=200_000)] = 1024
    mock: MockBehaviour = MockBehaviour()

    @model_validator(mode="after")
    def _check_base_url(self) -> Self:
        if self.type is not ProviderType.MOCK and not self.base_url:
            raise ValueError(f"provider '{self.name}': base_url is required for type '{self.type}'")
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"provider '{self.name}': base_url must start with http:// or https://")
        return self


class RetryConfig(_StrictModel):
    """Retry policy applied per provider before failing over.

    Attributes:
        max_attempts: Total attempts per provider (1 disables retries).
        base_delay_s: Initial back-off delay.
        max_delay_s: Cap on any single back-off delay.
    """

    max_attempts: Annotated[int, Field(ge=1, le=10)] = 2
    base_delay_s: Annotated[float, Field(ge=0, le=30)] = 0.2
    max_delay_s: Annotated[float, Field(ge=0, le=120)] = 2.0


class CircuitBreakerConfig(_StrictModel):
    """Circuit-breaker policy applied per provider.

    Attributes:
        failure_threshold: Consecutive failures that open the circuit.
        recovery_timeout_s: Time the circuit stays open before a probe.
        half_open_max_calls: Concurrent probe calls allowed while half-open.
    """

    failure_threshold: Annotated[int, Field(ge=1, le=1000)] = 5
    recovery_timeout_s: Annotated[float, Field(gt=0, le=3600)] = 30.0
    half_open_max_calls: Annotated[int, Field(ge=1, le=100)] = 1


class GroundingScorerType(StrEnum):
    """Available grounding scorers."""

    LEXICAL = "lexical"  # fast, dependency-free; checks content words and numbers
    NLI = "nli"  # transformer NLI model; needs the `nli` extra
    HYBRID = "hybrid"  # numeric gate + NLI; needs the `nli` extra
    TOKEN = "token"  # noqa: S105 - fine-tuned token classifier (LettuceDetect), `nli` extra


class GroundingFailurePolicy(StrEnum):
    """What to do when the grounding check itself cannot run."""

    FAIL_OPEN = "fail_open"  # return the answer, marked as unverified
    FAIL_CLOSED = "fail_closed"  # refuse the request with 503


class AnswerAggregation(StrEnum):
    """How per-claim support scores combine into one verdict for the answer."""

    RATIO = "ratio"  # pass if >= min_supported_ratio of claims meet support_threshold
    MIN = "min"  # score = weakest claim
    LOWEST_TWO = "lowest_two"  # score = mean of the two weakest claims
    MEAN = "mean"  # score = mean of all claims


class GroundingConfig(_StrictModel):
    """Grounding-verification settings.

    Attributes:
        scorer: Which scorer to use.
        nli_model: Hugging Face model id for the NLI scorer.
        token_model: Hugging Face model id for the token scorer.
        token_max_length: Maximum tokens (prompt + answer) for the token scorer.
        support_threshold: Per-claim score needed to count as supported.
        min_supported_ratio: Fraction of claims that must be supported for
            the answer to pass.
        min_claim_tokens: Sentences with fewer content tokens are not
            treated as factual claims (e.g. "Sure!").
        aggregation: How claim scores become an answer verdict. ``ratio`` uses
            ``min_supported_ratio``; the score-based modes compare the
            aggregated score with ``support_threshold``.
        evidence_passages: 0 scores each claim against every passage
            separately (best one wins). ``k > 0`` first picks the ``k`` passages
            sharing the most words with the claim and scores the claim once
            against them joined together: far fewer model calls, and claims
            that combine facts from several passages can be supported.
        on_error: Policy when the scorer fails.
    """

    scorer: GroundingScorerType = GroundingScorerType.LEXICAL
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    token_model: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"  # noqa: S105 - model id
    token_max_length: Annotated[int, Field(ge=32, le=32_768)] = 4096
    support_threshold: Annotated[float, Field(ge=0, le=1)] = 0.6
    min_supported_ratio: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_claim_tokens: Annotated[int, Field(ge=1, le=50)] = 3
    aggregation: AnswerAggregation = AnswerAggregation.RATIO
    evidence_passages: Annotated[int, Field(ge=0, le=10)] = 0
    on_error: GroundingFailurePolicy = GroundingFailurePolicy.FAIL_OPEN


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


class InjectionConfig(_StrictModel):
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


class PIIConfig(_StrictModel):
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


class SafetyConfig(_StrictModel):
    """Input-side safety settings.

    Attributes:
        injection: Prompt-injection screening.
        pii: PII redaction.
    """

    injection: InjectionConfig = InjectionConfig()
    pii: PIIConfig = PIIConfig()


class CacheConfig(_StrictModel):
    """Response cache (exact + optional semantic matching).

    Attributes:
        enabled: Turn the cache on.
        semantic: Also match *reworded* questions using sentence embeddings
            (needs the ``nli`` extra). Off = exact matches only, no extra deps.
        embedding_model: Hugging Face sentence-embedding model id.
        embedding_max_length: Token cap for the embedded question.
        similarity_threshold: Minimum cosine similarity for a semantic hit.
            Higher = fewer wrong answers served, fewer hits. The default 0.96
            is the threshold tuned on QQP train for a 1% false-hit budget with
            MiniLM; on QQP validation it served wrong answers to 0.9% of
            different questions and reused 15% of rewordings (see README).
            Re-run ``make bench-cache`` if you change the embedding model.
        ttl_s: Seconds an entry stays valid.
        max_entries: Entries kept in memory; least recently used are evicted.
        max_temperature: Requests sampled hotter than this are not cached
            (the caller asked for variety). ``None`` caches any temperature.
        lookup_timeout_s: Budget for a lookup; slower lookups are skipped and
            the request goes to the provider (the cache never adds big latency).
        coalesce: Merge identical requests that arrive while one is in flight
            into a single provider call (protects providers from stampedes).
        partition_by_user: Keep a separate cache per OpenAI ``user`` field
            (multi-tenant isolation) instead of sharing answers across users.
        namespace: Free-form version tag mixed into every key. Change it to
            invalidate the whole cache, e.g. after changing a system prompt.
    """

    enabled: bool = False
    semantic: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_max_length: Annotated[int, Field(ge=16, le=8192)] = 256
    similarity_threshold: Annotated[float, Field(gt=0, le=1)] = 0.96
    ttl_s: Annotated[float, Field(gt=0, le=30 * 24 * 3600)] = 3600.0
    max_entries: Annotated[int, Field(ge=1, le=1_000_000)] = 10_000
    max_temperature: Annotated[float, Field(ge=0, le=2)] | None = 1.0
    lookup_timeout_s: Annotated[float, Field(gt=0, le=10)] = 0.5
    coalesce: bool = True
    partition_by_user: bool = False
    namespace: Annotated[str, Field(min_length=1, max_length=64)] = "v1"


class ToolAction(StrEnum):
    """What a tool rule does with a matching call."""

    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"  # hold the call until a human approves it


class ToolViolationHandling(StrEnum):
    """What the gateway does with a blocked tool call."""

    DROP = "drop"  # remove the call; other calls continue; text explains if none remain
    ERROR = "error"  # refuse the whole response (HTTP 422 tool_call_blocked)


class ArgumentConstraint(_StrictModel):
    """Limits on one argument (all set limits must hold).

    Attributes:
        required: The argument must be present.
        min: Smallest allowed number.
        max: Largest allowed number.
        enum: Allowed values.
        pattern: Regular expression the whole string value must match.
        max_length: Longest allowed string.
    """

    required: bool = False
    min: float | None = None
    max: float | None = None
    enum: list[Any] | None = None
    pattern: str | None = None
    max_length: Annotated[int, Field(ge=0)] | None = None

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid regular expression {value!r}: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) is greater than max ({self.max})")
        return self


class ToolRule(_StrictModel):
    """One ordered rule; the first rule that matches a call decides.

    A rule matches when the tool name matches one of ``tools`` *and* every
    ``when`` condition holds; otherwise evaluation falls through to the next
    rule (firewall-style). ``when`` makes amount-based tiers possible::

        - {tools: [issue_refund], action: require_approval, when: {amount: {min: 500}}}
        - {tools: [issue_refund], arguments: {amount: {max: 5000}}}

    Attributes:
        tools: Tool-name patterns (shell-style: ``*``, ``?``, ``[abc]``).
        action: Allow, block, or hold for human approval.
        when: Argument conditions that select this rule. A condition holds only
            when the argument is present and satisfies the limits.
        arguments: Argument name (dotted path for nested values) -> limits that
            must hold. Checked for allowed calls and for calls that need
            approval (a call over a hard limit is blocked, never sent to a
            reviewer).
        name: Label shown in audit logs, approvals and response metadata.
    """

    tools: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]
    action: ToolAction = ToolAction.ALLOW
    when: dict[str, ArgumentConstraint] = Field(default_factory=dict)
    arguments: dict[str, ArgumentConstraint] = Field(default_factory=dict)
    name: Annotated[str, Field(max_length=64)] | None = None


class ToolPolicyConfig(_StrictModel):
    """Guardrails for tool calls requested by models (OWASP LLM06, LLM10).

    Checked in code on every tool call, before the agent receives it, so the
    model can neither forget nor be talked out of them.

    Attributes:
        enabled: Turn enforcement on.
        rules: Ordered rules; the first matching rule decides.
        default_action: Action when no rule matches.
        on_violation: Drop blocked calls (agent continues) or refuse the response.
        injection_lockdown: When the request contains a suspected prompt
            injection, block every tool call (text answers still go through).
        max_tool_rounds: Assistant tool-call turns allowed in one conversation
            before further calls are blocked (stops runaway agent loops).
        validate_arguments: Check arguments are a JSON object matching the
            tool's declared JSON Schema (types, required, enum, ranges).
        reject_unknown_tools: Block calls to tools the request did not offer.
    """

    enabled: bool = True
    rules: list[ToolRule] = Field(default_factory=list)
    default_action: ToolAction = ToolAction.ALLOW
    on_violation: ToolViolationHandling = ToolViolationHandling.DROP
    injection_lockdown: bool = True
    max_tool_rounds: Annotated[int, Field(ge=1, le=1000)] = 25
    validate_arguments: bool = True
    reject_unknown_tools: bool = True


class ApprovalStoreType(StrEnum):
    """Where pending approvals are kept."""

    MEMORY = "memory"  # single replica; lost on restart (agents simply ask again)


class ApprovalConfig(_StrictModel):
    """Human-in-the-loop approval for tool calls (rules with ``action: require_approval``).

    An approval is permission for one *exact* call (same tool, same arguments,
    same ``user``), used at most once.

    Attributes:
        enabled: Allow rules to hold calls for approval. Rules that require
            approval are a configuration error when this is off.
        store: Storage backend.
        ttl_s: How long a request waits for a decision before it expires.
        grant_ttl_s: How long an approved call stays usable before it expires.
        retention_s: How long decided records stay visible for audit before
            they are purged from the store.
        max_records: Capacity of the store (all statuses).
        max_pending_per_scope: Pending requests one ``user`` may have open at
            once, so a looping agent cannot flood reviewers.
        reviewer_token_env: Environment variable holding the reviewer API token.
            When unset, the reviewer endpoints are disabled (fail closed).
    """

    enabled: bool = True
    store: ApprovalStoreType = ApprovalStoreType.MEMORY
    ttl_s: Annotated[float, Field(ge=10, le=7 * 24 * 3600)] = 900.0
    grant_ttl_s: Annotated[float, Field(ge=10, le=7 * 24 * 3600)] = 600.0
    retention_s: Annotated[float, Field(ge=60, le=90 * 24 * 3600)] = 86_400.0
    max_records: Annotated[int, Field(ge=10, le=1_000_000)] = 10_000
    max_pending_per_scope: Annotated[int, Field(ge=1, le=10_000)] = 20
    reviewer_token_env: Annotated[str, Field(pattern=r"^[A-Z_][A-Z0-9_]{0,127}$")] = (
        "GROUNDWIRE_APPROVER_TOKEN")  # noqa: S105 - name of an env var, not a secret


class AnswerCheckAction(StrEnum):
    """What the gateway does when an agent's answer contradicts its tool results."""

    FLAG = "flag"  # deliver unchanged; report issues in ``groundwire.answer_check``
    ANNOTATE = "annotate"  # append a visible warning to the answer
    BLOCK = "block"  # refuse the answer (HTTP 422 answer_check_failed)


class AnswerCheckConfig(_StrictModel):
    """Fact-check an agent's final answer against the tool results it was given.

    Catches the agent that runs the right tools and then reports the wrong
    outcome ("refunded $49.99" when the refund tool returned ``amount: 1``),
    and claims of actions that never ran or failed. Deterministic: no model
    call, so it adds microseconds, not seconds.

    Attributes:
        enabled: Run the check on final answers in conversations with tool results.
        action: Flag, annotate or block an answer with issues.
        window_words: How far (in words) a number may be from a tool's words
            (its name or result field names) to be checked against that tool.
        min_unbound_value: Numbers not near any tool's words are only reported
            when they are at least this large, or have a currency sign,
            percent sign or decimal point (small counts like "2 items" are noise).
    """

    enabled: bool = True
    action: AnswerCheckAction = AnswerCheckAction.FLAG
    window_words: Annotated[int, Field(ge=1, le=50)] = 6
    min_unbound_value: Annotated[float, Field(ge=0)] = 100.0


class CascadeTier(StrEnum):
    """The two tiers of a cascade."""

    CHEAP = "cheap"
    STRONG = "strong"


class CascadeCheckFailure(StrEnum):
    """What a cascade does when the fact-check cannot score the cheap answer."""

    ESCALATE = "escalate"  # correctness first: ask the strong tier
    SERVE_CHEAP = "serve_cheap"  # cost first: deliver the unchecked cheap answer


class CascadeConfig(_StrictModel):
    """Cheap-first routing that escalates when the fact-check fails.

    Measured on RAGTruth (``make bench-cascade-token``): GPT-3.5 -> GPT-4 matched
    GPT-4's hallucination rate at ~15% of its cost, but Llama-2-7B -> GPT-4 only
    halved the cost while doubling errors. Savings are capped by how often the
    cheap tier is right, so enable a cascade only where the cheap model is decent,
    and watch ``groundwire_cascade_requests_total``.

    Attributes:
        cheap: Provider chain tried first (normal retries and failover apply).
        strong: Provider chain used when the cheap answer fails the check.
        escalate_below: Escalate when the verifier's answer score is below this.
            Its meaning follows ``grounding.aggregation`` (with LettuceDetect and
            ``min``: 0.5 = escalate if any token is more likely made up than not).
        on_check_error: Behaviour when the check cannot run.
        without_context: Tier for requests that carry no grounding documents
            (nothing to check against).
    """

    cheap: Annotated[list[str], Field(min_length=1)]
    strong: Annotated[list[str], Field(min_length=1)]
    escalate_below: Annotated[float, Field(ge=0, le=1)] = 0.5
    on_check_error: CascadeCheckFailure = CascadeCheckFailure.ESCALATE
    without_context: CascadeTier = CascadeTier.STRONG


class GatewayConfig(_StrictModel):
    """Root of the YAML gateway configuration.

    Attributes:
        providers: All upstream providers.
        routes: Route alias -> ordered fallback chain of provider names.
            Clients select a route via the OpenAI ``model`` field.
        default_route: Route used when a client sends ``model: "default"``
            or the alias is omitted. Must exist in ``routes`` or ``cascades``.
        request_timeout_s: End-to-end deadline across retries and failover.
        retry: Retry policy.
        circuit_breaker: Circuit-breaker policy.
        grounding: Grounding-verification settings.
        safety: Prompt-injection and PII settings.
        cache: Response cache settings.
        cascades: Cascade alias -> cheap-first routing policy. Clients select a
            cascade via the ``model`` field, like a route.
        tool_policy: Guardrails for tool calls made by agents.
        answer_check: Fact-check of agents' final answers against tool results.
        approvals: Human-in-the-loop approval for held tool calls.
    """

    providers: Annotated[list[ProviderConfig], Field(min_length=1)]
    routes: Annotated[dict[str, list[str]], Field(min_length=1)]
    default_route: str = "default"
    request_timeout_s: Annotated[float, Field(gt=0, le=1800)] = 120.0
    retry: RetryConfig = RetryConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    grounding: GroundingConfig = GroundingConfig()
    safety: SafetyConfig = SafetyConfig()
    cache: CacheConfig = CacheConfig()
    cascades: dict[str, CascadeConfig] = Field(default_factory=dict)
    tool_policy: ToolPolicyConfig = ToolPolicyConfig()
    answer_check: AnswerCheckConfig = AnswerCheckConfig()
    approvals: ApprovalConfig = ApprovalConfig()

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        names = [p.name for p in self.providers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate provider names: {duplicates}")

        known = set(names)
        enabled = {p.name for p in self.providers if p.enabled}
        for route, chain in self.routes.items():
            if not chain:
                raise ValueError(f"route '{route}' has an empty provider chain")
            unknown = [p for p in chain if p not in known]
            if unknown:
                raise ValueError(f"route '{route}' references unknown providers: {unknown}")
            if len(set(chain)) != len(chain):
                raise ValueError(f"route '{route}' lists a provider more than once")
            if not any(p in enabled for p in chain):
                raise ValueError(f"route '{route}' has no enabled providers")

        for name, cascade in self.cascades.items():
            if name in self.routes or name == "default":
                raise ValueError(f"cascade '{name}' clashes with a route alias")
            overlap = sorted(set(cascade.cheap) & set(cascade.strong))
            if overlap:
                raise ValueError(f"cascade '{name}' lists {overlap} in both tiers")
            for tier, chain in (("cheap", cascade.cheap), ("strong", cascade.strong)):
                unknown = [p for p in chain if p not in known]
                if unknown:
                    raise ValueError(
                        f"cascade '{name}' {tier} tier references unknown providers: {unknown}")
                if len(set(chain)) != len(chain):
                    raise ValueError(f"cascade '{name}' {tier} tier lists a provider twice")
                if not any(p in enabled for p in chain):
                    raise ValueError(f"cascade '{name}' {tier} tier has no enabled providers")

        if self.default_route not in self.routes and self.default_route not in self.cascades:
            raise ValueError(f"default_route '{self.default_route}' is not defined in routes "
                             "or cascades")

        policy = self.tool_policy
        needs_approval = [rule.name or ",".join(rule.tools) for rule in policy.rules
                          if rule.action is ToolAction.REQUIRE_APPROVAL]
        if policy.default_action is ToolAction.REQUIRE_APPROVAL:
            needs_approval.append("default_action")
        if needs_approval and not self.approvals.enabled:
            raise ValueError(f"tool rules {needs_approval} require approval, but "
                             "approvals.enabled is false")
        return self

    def enabled_providers(self) -> list[ProviderConfig]:
        """Return only the providers marked ``enabled``."""
        return [p for p in self.providers if p.enabled]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_gateway_config(path: Path) -> GatewayConfig:
    """Load and validate the gateway YAML configuration.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If the file is missing, unreadable, not valid YAML,
            or fails schema/cross-reference validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read config file {path}: {exc}") from exc

    try:
        raw: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"config root in {path} must be a mapping")

    return parse_gateway_config(raw, source=str(path))


def parse_gateway_config(raw: dict[str, Any], *, source: str = "<dict>") -> GatewayConfig:
    """Validate an already-parsed config mapping.

    Args:
        raw: Parsed configuration mapping.
        source: Label used in error messages.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If validation fails.
    """
    try:
        config = GatewayConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(f"invalid gateway config ({source}): {problems}") from exc

    logger.info(
        "gateway config loaded",
        extra={
            "source": source,
            "providers": [p.name for p in config.enabled_providers()],
            "routes": list(config.routes),
        },
    )
    return config


def resolve_api_key(provider: ProviderConfig) -> SecretStr | None:
    """Read a provider's API key from the environment.

    Args:
        provider: Provider whose ``api_key_env`` should be resolved.

    Returns:
        The key wrapped in :class:`SecretStr` (so it never appears in logs or
        reprs), or ``None`` when the provider needs no key.

    Raises:
        ConfigurationError: If ``api_key_env`` is set but the variable is
            missing or empty.
    """
    if not provider.api_key_env:
        return None
    value = os.environ.get(provider.api_key_env, "").strip()
    if not value:
        raise ConfigurationError(
            f"provider '{provider.name}' needs env var '{provider.api_key_env}', "
            "which is not set (or set enabled: false to skip this provider)"
        )
    return SecretStr(value)
