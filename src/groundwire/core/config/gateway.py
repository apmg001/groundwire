"""Root of the YAML gateway configuration and its cross-reference checks.

Each feature's settings live in their own module; this one composes them and
validates references *between* them (routes naming providers, approval rules
needing approvals enabled).
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from groundwire.core.config.agents import (
    AnswerCheckConfig,
    ApprovalConfig,
    ToolAction,
    ToolPolicyConfig,
)
from groundwire.core.config.base import StrictModel
from groundwire.core.config.cache import CacheConfig
from groundwire.core.config.dashboard import DashboardConfig
from groundwire.core.config.grounding import GroundingConfig
from groundwire.core.config.providers import ProviderConfig
from groundwire.core.config.resilience import AdmissionConfig, CircuitBreakerConfig, RetryConfig
from groundwire.core.config.routing import CascadeConfig
from groundwire.core.config.safety import SafetyConfig


class GatewayConfig(StrictModel):
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
        admission: Load shedding when the gateway is overloaded.
        dashboard: Live dashboard at ``/dashboard``.
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
    admission: AdmissionConfig = AdmissionConfig()
    dashboard: DashboardConfig = DashboardConfig()

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        """Validate references between sections (run by pydantic after parsing).

        Returns:
            The validated config.

        Raises:
            ValueError: On the first broken reference (pydantic reports it as a
                validation error, which the loader turns into ``ConfigurationError``).
        """
        known, enabled = self._provider_names()
        for route, chain in self.routes.items():
            _check_chain(f"route '{route}'", chain, known, enabled,
                         repeated="lists a provider more than once")
        for name, cascade in self.cascades.items():
            self._check_cascade(name, cascade, known, enabled)
        if self.default_route not in self.routes and self.default_route not in self.cascades:
            raise ValueError(f"default_route '{self.default_route}' is not defined in routes "
                             "or cascades")
        self._check_approvals()
        return self

    def _provider_names(self) -> tuple[set[str], set[str]]:
        """All provider names and the enabled ones.

        Returns:
            ``(known, enabled)``.

        Raises:
            ValueError: If two providers share a name.
        """
        names = [p.name for p in self.providers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate provider names: {duplicates}")
        return set(names), {p.name for p in self.providers if p.enabled}

    def _check_cascade(self, name: str, cascade: CascadeConfig, known: set[str],
                       enabled: set[str]) -> None:
        """Validate one cascade's alias and both of its tiers.

        Args:
            name: Cascade alias.
            cascade: Its config.
            known: All provider names.
            enabled: Enabled provider names.

        Raises:
            ValueError: If the alias clashes with a route, a provider sits in
                both tiers, or a tier's chain is invalid.
        """
        if name in self.routes or name == "default":
            raise ValueError(f"cascade '{name}' clashes with a route alias")
        overlap = sorted(set(cascade.cheap) & set(cascade.strong))
        if overlap:
            raise ValueError(f"cascade '{name}' lists {overlap} in both tiers")
        for tier, chain in (("cheap", cascade.cheap), ("strong", cascade.strong)):
            _check_chain(f"cascade '{name}' {tier} tier", chain, known, enabled,
                         repeated="lists a provider twice")

    def _check_approvals(self) -> None:
        """Rules that hold calls for a human need the approval workflow switched on.

        Raises:
            ValueError: If any rule (or the default action) requires approval
                while ``approvals.enabled`` is false.
        """
        policy = self.tool_policy
        needs_approval = [rule.name or ",".join(rule.tools) for rule in policy.rules
                          if rule.action is ToolAction.REQUIRE_APPROVAL]
        if policy.default_action is ToolAction.REQUIRE_APPROVAL:
            needs_approval.append("default_action")
        if needs_approval and not self.approvals.enabled:
            raise ValueError(f"tool rules {needs_approval} require approval, but "
                             "approvals.enabled is false")

    def enabled_providers(self) -> list[ProviderConfig]:
        """Return only the providers marked ``enabled``."""
        return [p for p in self.providers if p.enabled]


def _check_chain(label: str, chain: list[str], known: set[str], enabled: set[str], *,
                 repeated: str) -> None:
    """Validate one ordered provider chain (a route or a cascade tier).

    Args:
        label: What the chain belongs to, for the error message.
        chain: Provider names in fallback order.
        known: All provider names.
        enabled: Enabled provider names.
        repeated: Message suffix when a provider appears twice (kept as it
            was per section, since operators and tests match on it).

    Raises:
        ValueError: If the chain is empty, names an unknown provider, repeats
            one, or has no enabled provider.
    """
    if not chain:
        raise ValueError(f"{label} has an empty provider chain")
    unknown = [p for p in chain if p not in known]
    if unknown:
        raise ValueError(f"{label} references unknown providers: {unknown}")
    if len(set(chain)) != len(chain):
        raise ValueError(f"{label} {repeated}")
    if not any(p in enabled for p in chain):
        raise ValueError(f"{label} has no enabled providers")
