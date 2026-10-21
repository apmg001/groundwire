"""Unit tests for cascade routing: config validation, router tiers, CascadeRunner."""

from __future__ import annotations

from typing import Any

import pytest

from conftest import ScriptedProvider, chat_request, make_config
from groundwire.core.config import (
    CascadeCheckFailure,
    CascadeTier,
    GatewayConfig,
    GroundingConfig,
)
from groundwire.core.errors import (
    AllProvidersFailedError,
    ConfigurationError,
    GuardrailUnavailableError,
    ModelNotFoundError,
    ProviderBadRequestError,
    ProviderServerError,
)
from groundwire.guardrails.grounding import GroundingReport, GroundingVerifier, Verdict
from groundwire.guardrails.scorers import LexicalScorer
from groundwire.observability.metrics import GatewayMetrics
from groundwire.routing.cascade import CascadeDecision, CascadeReason, CascadeRunner
from groundwire.routing.router import Router, tier_route

DOCS = ["The refund for order 7781 is 49.99 USD and was issued on 2 May."]
GOOD = "The refund for order 7781 is 49.99 USD."
BAD = "The refund for order 7781 is 89.99 USD."
DOWN = ProviderServerError("down", provider="x")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _config(**cascade: Any) -> GatewayConfig:
    providers = [{"name": n, "type": "mock", "model": "m"}
                 for n in ("primary", "secondary", "cheap", "cheap2", "strong")]
    return make_config(providers=providers,
                       cascades={"smart": {"cheap": ["cheap", "cheap2"], "strong": ["strong"],
                                           **cascade}})


def _providers(cheap: Any = GOOD, cheap2: Any = DOWN, strong: Any = "strong answer"
               ) -> dict[str, ScriptedProvider]:
    return {name: ScriptedProvider(name, *(value if isinstance(value, tuple) else (value,)))
            for name, value in (("primary", "p"), ("secondary", "s"), ("cheap", cheap),
                                ("cheap2", cheap2), ("strong", strong))}


class ExplodingVerifier(GroundingVerifier):
    """Verifier whose scorer is down and whose policy is fail-closed."""

    def __init__(self) -> None:
        super().__init__(GroundingConfig(), LexicalScorer())

    async def verify(self, answer: str, context: list[str], *, question: str | None = None,
                     prompt: str | None = None) -> GroundingReport:
        raise GuardrailUnavailableError("scorer crashed")


class UnverifiedVerifier(GroundingVerifier):
    """Verifier whose scorer is down and whose policy is fail-open."""

    def __init__(self) -> None:
        super().__init__(GroundingConfig(), LexicalScorer())

    async def verify(self, answer: str, context: list[str], *, question: str | None = None,
                     prompt: str | None = None) -> GroundingReport:
        return GroundingReport(verdict=Verdict.UNVERIFIED, scorer="lexical",
                               supported_ratio=0.0, answer_score=0.0, error="down")


def _runner(config: GatewayConfig, providers: dict[str, ScriptedProvider], *,
            verifier: GroundingVerifier | None = None,
            metrics: GatewayMetrics | None = None) -> CascadeRunner:
    router = Router(config, providers)  # type: ignore[arg-type]
    return CascadeRunner(router, verifier or GroundingVerifier(GroundingConfig(),
                                                               LexicalScorer()),
                         metrics=metrics)


async def _run(runner: CascadeRunner, context: list[str] | None = DOCS, **kwargs: Any) -> Any:
    return await runner.complete("smart", chat_request("How much is the refund?"),
                                 context=context, question="How much is the refund?", **kwargs)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("cascades", "message"), [
    ({"smart": {"cheap": ["cheap"], "strong": ["cheap"]}}, "both tiers"),
    ({"smart": {"cheap": ["nope"], "strong": ["strong"]}}, "unknown providers"),
    ({"smart": {"cheap": ["cheap", "cheap"], "strong": ["strong"]}}, "twice"),
    ({"default": {"cheap": ["cheap"], "strong": ["strong"]}}, "clashes"),
    ({"smart": {"cheap": [], "strong": ["strong"]}}, "cheap"),
])
def test_cascade_config_validation(cascades: dict[str, Any], message: str) -> None:
    providers = [{"name": n, "type": "mock", "model": "m"}
                 for n in ("primary", "secondary", "cheap", "strong")]
    with pytest.raises(ConfigurationError, match=message):
        make_config(providers=providers, cascades=cascades)


def test_cascade_tier_needs_an_enabled_provider() -> None:
    providers = [{"name": "primary", "type": "mock", "model": "m"},
                 {"name": "secondary", "type": "mock", "model": "m"},
                 {"name": "cheap", "type": "mock", "model": "m", "enabled": False},
                 {"name": "strong", "type": "mock", "model": "m"}]
    with pytest.raises(ConfigurationError, match="no enabled providers"):
        make_config(providers=providers, cascades={"smart": {"cheap": ["cheap"],
                                                             "strong": ["strong"]}})


def test_default_route_may_be_a_cascade() -> None:
    config = _config()
    assert make_config(providers=[p.model_dump() for p in config.providers],
                       cascades={"smart": {"cheap": ["cheap"], "strong": ["strong"]}},
                       default_route="smart").default_route == "smart"


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def test_router_exposes_cascades_and_resolves_them() -> None:
    router = Router(_config(), _providers())  # type: ignore[arg-type]
    assert router.resolve("smart") == "smart"
    assert router.cascades == {"smart": {"cheap": ["cheap", "cheap2"], "strong": ["strong"]}}
    assert "smart" not in router.routes  # cascades are not plain routes
    assert router.cascade_config("default") is None
    with pytest.raises(ModelNotFoundError) as info:
        router.resolve("nope")
    assert "smart" in info.value.details["available"]
    assert tier_route("smart", CascadeTier.CHEAP) == "smart:cheap"


async def test_router_drops_disabled_providers_from_tiers() -> None:
    providers = _providers()
    del providers["cheap2"]  # not built = disabled
    router = Router(_config(), providers)  # type: ignore[arg-type]
    assert router.cascades["smart"]["cheap"] == ["cheap"]


# --------------------------------------------------------------------------- #
# Runner: main paths
# --------------------------------------------------------------------------- #


async def test_supported_cheap_answer_is_served() -> None:
    metrics = GatewayMetrics()
    providers = _providers(cheap=GOOD)
    outcome = await _run(_runner(_config(), providers, metrics=metrics))
    decision = outcome.decision
    assert outcome.routed.provider == "cheap" and outcome.routed.completion.text == GOOD
    assert (decision.tier, decision.reason, decision.escalated) == \
        (CascadeTier.CHEAP, CascadeReason.PASSED, False)
    assert decision.score == 1.0 and outcome.report is not None  # reusable report
    assert providers["strong"].calls == 0
    assert metrics.registry.get_sample_value(
        "groundwire_cascade_requests_total",
        {"cascade": "smart", "tier": "cheap", "reason": "passed"}) == 1


async def test_unsupported_cheap_answer_escalates() -> None:
    providers = _providers(cheap=BAD)
    outcome = await _run(_runner(_config(), providers))
    assert outcome.routed.provider == "strong"
    assert outcome.decision.reason is CascadeReason.LOW_SUPPORT and outcome.decision.escalated
    assert outcome.decision.score == 0.0 and outcome.report is None
    assert [(a["tier"], a["provider"]) for a in outcome.attempts] == \
        [("cheap", "cheap"), ("strong", "strong")]


async def test_threshold_is_respected() -> None:
    half = "The refund for order 7781 is 49.99 USD. Shipping was free of charge for everyone."
    low = await _run(_runner(_config(escalate_below=0.9), _providers(cheap=half)))
    high = await _run(_runner(_config(escalate_below=0.1), _providers(cheap=half)))
    assert low.decision.tier is CascadeTier.STRONG and high.decision.tier is CascadeTier.CHEAP


async def test_check_runs_on_restored_text() -> None:
    providers = _providers(cheap="Refund sent to [EMAIL_1]: 49.99 USD for order 7781.")
    docs = ["Refund for order 7781 of 49.99 USD goes to rahul@example.com."]
    outcome = await _run(_runner(_config(), providers), context=docs,
                         restore=lambda t: t.replace("[EMAIL_1]", "rahul@example.com"))
    assert outcome.decision.reason is CascadeReason.PASSED


# --------------------------------------------------------------------------- #
# Runner: failures
# --------------------------------------------------------------------------- #


async def test_cheap_tier_down_serves_strong() -> None:
    providers = _providers(cheap=DOWN, cheap2=DOWN)
    outcome = await _run(_runner(_config(), providers))
    assert outcome.decision.reason is CascadeReason.CHEAP_UNAVAILABLE
    assert outcome.routed.provider == "strong"
    assert {a["tier"] for a in outcome.attempts} == {"cheap", "strong"}


async def test_cheap_tier_fails_over_within_the_tier_first() -> None:
    outcome = await _run(_runner(_config(), _providers(cheap=DOWN, cheap2=GOOD)))
    assert outcome.routed.provider == "cheap2" and outcome.decision.reason is CascadeReason.PASSED


async def test_strong_down_on_escalation_serves_cheap() -> None:
    outcome = await _run(_runner(_config(), _providers(cheap=BAD, strong=DOWN)))
    assert outcome.routed.provider == "cheap"
    assert outcome.decision.reason is CascadeReason.STRONG_UNAVAILABLE
    assert outcome.decision.tier is CascadeTier.CHEAP and not outcome.decision.escalated


async def test_both_tiers_down_reports_every_attempt() -> None:
    runner = _runner(_config(), _providers(cheap=DOWN, cheap2=DOWN, strong=DOWN))
    with pytest.raises(AllProvidersFailedError) as info:
        await _run(runner)
    assert info.value.details["route"] == "smart"
    assert {a["provider"] for a in info.value.details["attempts"]} == \
        {"cheap", "cheap2", "strong"}


async def test_client_fault_is_not_retried_on_the_strong_tier() -> None:
    providers = _providers(cheap=ProviderBadRequestError("bad", provider="cheap"))
    with pytest.raises(ProviderBadRequestError):
        await _run(_runner(_config(), providers))
    assert providers["strong"].calls == 0


@pytest.mark.parametrize("verifier", [ExplodingVerifier(), UnverifiedVerifier()])
async def test_check_failure_escalates_by_default(verifier: GroundingVerifier) -> None:
    outcome = await _run(_runner(_config(), _providers(), verifier=verifier))
    assert outcome.decision.reason is CascadeReason.CHECK_FAILED
    assert outcome.decision.tier is CascadeTier.STRONG and outcome.decision.escalated


async def test_check_failure_can_serve_cheap() -> None:
    config = _config(on_check_error=CascadeCheckFailure.SERVE_CHEAP.value)
    outcome = await _run(_runner(config, _providers(), verifier=ExplodingVerifier()))
    assert outcome.decision.tier is CascadeTier.CHEAP
    assert outcome.decision.reason is CascadeReason.CHECK_FAILED


# --------------------------------------------------------------------------- #
# Runner: no documents
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("tier", "provider"), [("strong", "strong"), ("cheap", "cheap")])
async def test_without_context_uses_configured_tier(tier: str, provider: str) -> None:
    providers = _providers()
    outcome = await _run(_runner(_config(without_context=tier), providers), context=None)
    assert outcome.routed.provider == provider
    assert outcome.decision.reason is CascadeReason.NO_CONTEXT


@pytest.mark.parametrize(("tier", "reason"), [("strong", CascadeReason.STRONG_UNAVAILABLE),
                                              ("cheap", CascadeReason.CHEAP_UNAVAILABLE)])
async def test_without_context_falls_back_to_the_other_tier(tier: str,
                                                            reason: CascadeReason) -> None:
    providers = _providers(cheap=DOWN if tier == "cheap" else GOOD,
                           cheap2=DOWN, strong=DOWN if tier == "strong" else "s")
    outcome = await _run(_runner(_config(without_context=tier), providers), context=[])
    assert outcome.decision.reason is reason


async def test_without_context_both_down() -> None:
    runner = _runner(_config(), _providers(cheap=DOWN, cheap2=DOWN, strong=DOWN))
    with pytest.raises(AllProvidersFailedError):
        await _run(runner, context=None)


async def test_unknown_cascade() -> None:
    with pytest.raises(ModelNotFoundError):
        await _runner(_config(), _providers()).complete("default", chat_request(), context=None,
                                                         question=None)


def test_decision_metadata() -> None:
    decision = CascadeDecision("smart", CascadeTier.STRONG, CascadeReason.LOW_SUPPORT, 0.5,
                               score=0.123456, check_ms=12.345)
    assert decision.as_dict() == {"cascade": "smart", "tier": "strong", "reason": "low_support",
                                  "escalated": True, "threshold": 0.5, "cheap_score": 0.1235,
                                  "check_ms": 12.3}
