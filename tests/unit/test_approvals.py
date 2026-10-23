"""Unit tests for human-in-the-loop approvals: model, store, service, policy, gate, config."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import Any

import pytest

from conftest import make_config
from groundwire.agents.gate import ToolGate
from groundwire.agents.policy import BlockReason, ToolPolicy, ToolVerdict
from groundwire.approvals.models import (
    Approval,
    ApprovalStatus,
    InvalidTransitionError,
    call_fingerprint,
    new_approval_id,
)
from groundwire.approvals.service import ApprovalService, Resolution, scope_of
from groundwire.approvals.store import ApprovalStore, InMemoryApprovalStore
from groundwire.container import _reviewer_token
from groundwire.core.config import ApprovalConfig, ToolPolicyConfig
from groundwire.core.errors import (
    ApprovalConflictError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalStoreError,
    ConfigurationError,
    ToolCallBlockedError,
)
from groundwire.domain.models import ChatCompletionRequest, FunctionCall, ToolCall
from groundwire.observability.metrics import GatewayMetrics

REFUND = {"order_id": 4417, "amount": 1200}


class FakeClock:
    """Controllable wall clock."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _ids() -> Any:
    counter = itertools.count(1)
    return lambda: f"apr_{next(counter):032x}"


def _service(clock: FakeClock | None = None, *, store: ApprovalStore | None = None,
             metrics: GatewayMetrics | None = None, **config: Any) -> ApprovalService:
    return ApprovalService(store or InMemoryApprovalStore(), ApprovalConfig(**config),
                           metrics=metrics, clock=clock or FakeClock(), id_factory=_ids())


async def _resolve(service: ApprovalService, *, scope: str = "alice",
                   arguments: dict[str, Any] | None = None) -> Any:
    return await service.resolve(tool="issue_refund", arguments=arguments or REFUND,
                                 scope=scope, rule="refund-approval")


def _record(status: ApprovalStatus = ApprovalStatus.PENDING, **changes: Any) -> Approval:
    base: dict[str, Any] = {
        "id": new_approval_id(), "scope": "alice", "fingerprint": "f" * 64,
        "tool": "issue_refund", "arguments": REFUND, "rule": "r", "status": status,
        "created_at": 100.0, "expires_at": 200.0, "updated_at": 100.0}
    base.update(changes)
    return Approval(**base)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def test_fingerprint_ignores_key_order_and_int_float_spelling() -> None:
    a = call_fingerprint("issue_refund", {"order_id": 4417, "amount": 1200})
    b = call_fingerprint("issue_refund", {"amount": 1200.0, "order_id": 4417})
    assert a == b


@pytest.mark.parametrize("other", [
    ("issue_refund", {"order_id": 4417, "amount": 1201}),
    ("issue_refund", {"order_id": 4418, "amount": 1200}),
    ("issue_credit", {"order_id": 4417, "amount": 1200}),
    ("issue_refund", {"order_id": 4417, "amount": 1200, "note": "x"}),
])
def test_fingerprint_changes_with_any_difference(other: tuple[str, dict[str, Any]]) -> None:
    assert call_fingerprint("issue_refund", REFUND) != call_fingerprint(*other)


def test_fingerprint_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        call_fingerprint("t", {"x": float("nan")})


def test_ids_are_unguessable_and_well_formed() -> None:
    ids = {new_approval_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(len(i) == 36 and i.startswith("apr_") for i in ids)


def test_lifecycle_transitions() -> None:
    pending = _record()
    approved = pending.transition(ApprovalStatus.APPROVED, 150.0, reviewer="anita")
    assert (approved.status, approved.updated_at, approved.reviewer) == (
        ApprovalStatus.APPROVED, 150.0, "anita")
    assert approved.transition(ApprovalStatus.CONSUMED, 160.0).status is ApprovalStatus.CONSUMED
    for closed in (ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED, ApprovalStatus.CONSUMED):
        with pytest.raises(InvalidTransitionError):
            _record(closed).transition(ApprovalStatus.APPROVED, 1.0)
    with pytest.raises(InvalidTransitionError):
        pending.transition(ApprovalStatus.CONSUMED, 1.0)  # never released without approval


def test_agent_view_hides_arguments_and_scope() -> None:
    summary = _record().summary()
    assert set(summary) == {"id", "status", "tool", "expires_at"}
    assert "arguments" in _record().as_dict()


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


async def test_store_indexes_latest_record_per_call() -> None:
    store = InMemoryApprovalStore()
    first, second = _record(created_at=1.0), _record(created_at=2.0)
    await store.add(first)
    await store.add(second)
    assert await store.find_latest("alice", "f" * 64) == second
    assert await store.find_latest("bob", "f" * 64) is None
    with pytest.raises(ApprovalStoreError, match="duplicate"):
        await store.add(first)


async def test_compare_and_set_rejects_stale_writers() -> None:
    store = InMemoryApprovalStore()
    pending = _record()
    await store.add(pending)
    approved = pending.transition(ApprovalStatus.APPROVED, 150.0)
    rejected = pending.transition(ApprovalStatus.REJECTED, 150.0)
    assert await store.compare_and_set(pending, approved)
    assert not await store.compare_and_set(pending, rejected)  # lost the race
    assert (await store.get(pending.id)) == approved
    assert await store.count(ApprovalStatus.PENDING) == 0
    assert await store.count(ApprovalStatus.APPROVED, scope="alice") == 1


async def test_store_lists_newest_first_and_purges_closed_only() -> None:
    store = InMemoryApprovalStore()
    old = _record(ApprovalStatus.REJECTED, created_at=1.0, updated_at=10.0)
    open_ = _record(created_at=2.0, updated_at=10.0)
    await store.add(old)
    await store.add(open_)
    assert [r.id for r in await store.list()] == [open_.id, old.id]
    assert [r.id for r in await store.list(status=ApprovalStatus.REJECTED)] == [old.id]
    assert await store.purge_closed(before=11.0) == 1
    assert await store.get(old.id) is None
    assert await store.get(open_.id) == open_  # open records are never purged


async def test_full_store_evicts_closed_records_but_never_open_ones() -> None:
    store = InMemoryApprovalStore(max_records=2)
    closed = _record(ApprovalStatus.CONSUMED, updated_at=1.0)
    await store.add(closed)
    await store.add(_record())
    await store.add(_record())  # evicts the closed record
    assert await store.get(closed.id) is None
    with pytest.raises(ApprovalStoreError, match="full"):
        await store.add(_record())


def test_store_needs_positive_capacity() -> None:
    with pytest.raises(ValueError):
        InMemoryApprovalStore(max_records=0)


# --------------------------------------------------------------------------- #
# Service: the lifecycle
# --------------------------------------------------------------------------- #


async def test_first_request_is_held_and_repeats_are_deduplicated() -> None:
    service = _service()
    first = await _resolve(service)
    again = await _resolve(service)
    assert first.resolution is Resolution.PENDING
    assert again.resolution is Resolution.PENDING
    assert again.approval.id == first.approval.id  # one request, not two


async def test_approved_call_is_released_exactly_once() -> None:
    service = _service()
    pending = (await _resolve(service)).approval
    approved = await service.approve(pending.id, reviewer="anita", note="ok")
    assert (approved.status, approved.reviewer, approved.note) == (
        ApprovalStatus.APPROVED, "anita", "ok")

    released = await _resolve(service)
    assert released.resolution is Resolution.RELEASED
    assert released.approval.status is ApprovalStatus.CONSUMED

    replay = await _resolve(service)
    assert replay.resolution is Resolution.PENDING  # a new approval is needed
    assert replay.approval.id != pending.id


async def test_approval_is_bound_to_the_exact_arguments() -> None:
    service = _service()
    pending = (await _resolve(service)).approval
    await service.approve(pending.id, reviewer="anita")
    bigger = await _resolve(service, arguments={"order_id": 4417, "amount": 60_000})
    assert bigger.resolution is Resolution.PENDING
    assert bigger.approval.id != pending.id


async def test_approvals_never_cross_scopes() -> None:
    service = _service()
    pending = (await _resolve(service, scope="alice")).approval
    await service.approve(pending.id, reviewer="anita")
    assert (await _resolve(service, scope="carol")).resolution is Resolution.PENDING
    assert (await _resolve(service, scope="alice")).resolution is Resolution.RELEASED


async def test_rejected_call_stays_blocked() -> None:
    service = _service()
    pending = (await _resolve(service)).approval
    await service.reject(pending.id, reviewer="anita", note="duplicate claim")
    outcome = await _resolve(service)
    assert outcome.resolution is Resolution.REJECTED
    assert outcome.approval.note == "duplicate claim"


async def test_pending_request_expires() -> None:
    clock = FakeClock()
    service = _service(clock, ttl_s=60)
    pending = (await _resolve(service)).approval
    clock.advance(61)
    with pytest.raises(ApprovalExpiredError):
        await service.approve(pending.id, reviewer="anita")
    assert (await service.get(pending.id)).status is ApprovalStatus.EXPIRED
    fresh = await _resolve(service)
    assert fresh.resolution is Resolution.PENDING and fresh.approval.id != pending.id


async def test_unused_approval_expires() -> None:
    clock = FakeClock()
    service = _service(clock, grant_ttl_s=30)
    pending = (await _resolve(service)).approval
    await service.approve(pending.id, reviewer="anita")
    clock.advance(31)
    outcome = await _resolve(service)
    assert outcome.resolution is Resolution.PENDING  # stale grants are not honoured
    assert (await service.get(pending.id)).status is ApprovalStatus.EXPIRED


async def test_decisions_are_final() -> None:
    service = _service()
    pending = (await _resolve(service)).approval
    await service.approve(pending.id, reviewer="anita")
    with pytest.raises(ApprovalConflictError, match="already approved"):
        await service.reject(pending.id, reviewer="bob")
    with pytest.raises(ApprovalNotFoundError):
        await service.approve("apr_" + "0" * 32, reviewer="anita")


async def test_one_scope_cannot_flood_reviewers() -> None:
    service = _service(max_pending_per_scope=2)
    for amount in (600, 700):
        assert (await _resolve(service, arguments={"amount": amount})).resolution is (
            Resolution.PENDING)
    flood = await _resolve(service, arguments={"amount": 800})
    assert flood.resolution is Resolution.UNAVAILABLE
    other = await _resolve(service, scope="bob", arguments={"amount": 800})
    assert other.resolution is Resolution.PENDING


class _BrokenStore(InMemoryApprovalStore):
    async def find_latest(self, scope: str, fingerprint: str) -> Approval | None:
        raise ApprovalStoreError("connection refused")


async def test_store_failure_fails_closed() -> None:
    outcome = await _resolve(_service(store=_BrokenStore()))
    assert outcome.resolution is Resolution.UNAVAILABLE
    assert outcome.approval is None


class _NetworkLikeStore(InMemoryApprovalStore):
    """Yields to the event loop *after* every read, like a network round trip.

    The pause comes after the read so the caller holds a possibly stale record
    while others run: exactly the window compare-and-set must protect.
    Without it, coroutines never interleave and "concurrency" tests prove nothing.
    """

    async def get(self, approval_id: str) -> Approval | None:
        record = await super().get(approval_id)
        await asyncio.sleep(0)
        return record

    async def find_latest(self, scope: str, fingerprint: str) -> Approval | None:
        record = await super().find_latest(scope, fingerprint)
        await asyncio.sleep(0)
        return record


async def test_concurrent_requests_release_one_approval_once() -> None:
    service = _service(store=_NetworkLikeStore())
    pending = (await _resolve(service)).approval
    await service.approve(pending.id, reviewer="anita")
    outcomes = await asyncio.gather(*(_resolve(service) for _ in range(5)))
    released = [o for o in outcomes if o.resolution is Resolution.RELEASED]
    assert len(released) == 1
    assert all(o.resolution is Resolution.PENDING for o in outcomes if o not in released)


async def test_concurrent_reviewers_cannot_both_decide() -> None:
    service = _service(store=_NetworkLikeStore())
    pending = (await _resolve(service)).approval
    results = await asyncio.gather(service.approve(pending.id, reviewer="a"),
                                   service.reject(pending.id, reviewer="b"),
                                   return_exceptions=True)
    assert sum(isinstance(r, Approval) for r in results) == 1
    assert sum(isinstance(r, ApprovalConflictError) for r in results) == 1


async def test_list_applies_expiry() -> None:
    clock = FakeClock()
    service = _service(clock, ttl_s=60)
    await _resolve(service)
    assert len(await service.list(status=ApprovalStatus.PENDING)) == 1
    clock.advance(61)
    assert await service.list(status=ApprovalStatus.PENDING) == []
    assert len(await service.list(status=ApprovalStatus.EXPIRED)) == 1


async def test_metrics_and_audit_log_without_argument_values(
        caplog: pytest.LogCaptureFixture) -> None:
    metrics = GatewayMetrics()
    service = _service(metrics=metrics)
    secret = {"order_id": 4417, "amount": 1200, "iban": "DE89370400440532013000"}
    with caplog.at_level(logging.INFO, logger="groundwire.audit"):
        pending = (await _resolve(service, arguments=secret)).approval
        await service.approve(pending.id, reviewer="anita")
        await _resolve(service, arguments=secret)
    events = [r.__dict__.get("event") for r in caplog.records if r.name == "groundwire.audit"]
    assert events == ["approval.requested", "approval.approved", "approval.released"]
    assert "DE89370400440532013000" not in caplog.text
    text = metrics.render().decode()
    assert 'groundwire_approvals_total{event="released"} 1.0' in text
    assert "groundwire_approvals_pending 0.0" in text


def test_scope_of() -> None:
    assert scope_of(None) == scope_of("  ") == "anonymous"
    assert scope_of(" alice ") == "alice"


# --------------------------------------------------------------------------- #
# Policy: require_approval and `when`
# --------------------------------------------------------------------------- #


def _call(amount: float | str, cid: str = "c1", name: str = "issue_refund") -> ToolCall:
    args = amount if isinstance(amount, str) else json.dumps({"order_id": 1, "amount": amount})
    return ToolCall(id=cid, function=FunctionCall(name=name, arguments=args))


def _request(user: str | None = "alice") -> ChatCompletionRequest:
    tools = [{"type": "function", "function": {"name": n, "parameters": {
        "type": "object", "properties": {"order_id": {"type": "integer"},
                                         "amount": {"type": "number"}}}}}
             for n in ("issue_refund", "delete_user")]
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}],
                                 tools=tools, user=user)


TIERED: dict[str, Any] = {"rules": [
    {"name": "refund-approval", "tools": ["issue_refund"], "action": "require_approval",
     "when": {"amount": {"min": 500}}, "arguments": {"amount": {"max": 5000}}},
    {"name": "refund-cap", "tools": ["issue_refund"], "arguments": {"amount": {"max": 5000}}},
]}


@pytest.mark.parametrize(("amount", "verdict", "rule"), [
    (49.99, ToolVerdict.ALLOWED, "refund-cap"),        # `when` does not hold: falls through
    (500, ToolVerdict.NEEDS_APPROVAL, "refund-approval"),
    (4999, ToolVerdict.NEEDS_APPROVAL, "refund-approval"),
    (100_000, ToolVerdict.BLOCKED, "refund-approval"),  # hard limit beats approval
])
def test_tiers(amount: float, verdict: ToolVerdict, rule: str) -> None:
    policy = ToolPolicy(ToolPolicyConfig(**TIERED))
    (decision,) = policy.evaluate([_call(amount)], _request(),
                                  injection_suspected=False).decisions
    assert (decision.verdict, decision.rule) == (verdict, rule)


def test_when_never_matches_a_missing_argument() -> None:
    policy = ToolPolicy(ToolPolicyConfig(**TIERED))
    (decision,) = policy.evaluate([_call('{"order_id": 1}')], _request(),
                                  injection_suspected=False).decisions
    assert decision.rule == "refund-cap" and decision.allowed


def test_approval_does_not_bypass_other_checks() -> None:
    policy = ToolPolicy(ToolPolicyConfig(**TIERED))
    (decision,) = policy.evaluate([_call(1200)], _request(), injection_suspected=True).decisions
    assert decision.reason is BlockReason.INJECTION_LOCKDOWN


def test_default_action_can_require_approval() -> None:
    policy = ToolPolicy(ToolPolicyConfig(default_action="require_approval"))
    (decision,) = policy.evaluate([_call(1, name="delete_user")], _request(),
                                  injection_suspected=False).decisions
    assert decision.verdict is ToolVerdict.NEEDS_APPROVAL


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


async def test_gate_holds_releases_and_explains() -> None:
    service = _service()
    gate = ToolGate(ToolPolicy(ToolPolicyConfig(**TIERED)), approvals=service)
    calls, result = await gate.enforce([_call(49.99, "small"), _call(1200, "big")],
                                       _request(), injection_suspected=False)
    assert [c.id for c in calls] == ["small"]
    assert result is not None
    (held,) = result.held
    assert held.approval_id and held.call.id == "big"
    assert result.as_dict()["pending_approval"] == 1
    assert f"approval {held.approval_id}" in result.explanation()
    assert "1200" not in result.explanation()

    await service.approve(held.approval_id, reviewer="anita")
    calls, result = await gate.enforce([_call(1200, "big2")], _request(),
                                       injection_suspected=False)
    assert [c.id for c in calls] == ["big2"]
    assert result is not None and result.decisions[0].approval_id == held.approval_id


async def test_gate_without_approval_service_fails_closed() -> None:
    gate = ToolGate(ToolPolicy(ToolPolicyConfig(**TIERED)), approvals=None)
    calls, result = await gate.enforce([_call(1200)], _request(), injection_suspected=False)
    assert calls == []
    assert result is not None and result.blocked[0].reason is BlockReason.APPROVAL_UNAVAILABLE


async def test_held_calls_are_not_violations_but_rejections_are() -> None:
    service = _service()
    gate = ToolGate(ToolPolicy(ToolPolicyConfig(**TIERED, on_violation="error")),
                    approvals=service)
    _, result = await gate.enforce([_call(1200)], _request(), injection_suspected=False)
    assert result is not None
    await service.reject(result.held[0].approval_id or "", reviewer="anita")
    with pytest.raises(ToolCallBlockedError) as info:
        await gate.enforce([_call(1200)], _request(), injection_suspected=False)
    assert info.value.details["tools"]["decisions"][0]["reason"] == "approval_rejected"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_rules_requiring_approval_need_approvals_enabled() -> None:
    with pytest.raises(ConfigurationError, match="require approval"):
        make_config(tool_policy=TIERED, approvals={"enabled": False})
    make_config(tool_policy=TIERED)  # enabled by default


def test_reviewer_token_env_must_look_like_an_env_var() -> None:
    with pytest.raises(ValueError):
        ApprovalConfig(reviewer_token_env="not valid")


def test_reviewer_token_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ApprovalConfig(reviewer_token_env="TEST_APPROVER_TOKEN")
    monkeypatch.delenv("TEST_APPROVER_TOKEN", raising=False)
    assert _reviewer_token(config) is None  # unset: reviewer API off
    monkeypatch.setenv("TEST_APPROVER_TOKEN", "short")
    with pytest.raises(ConfigurationError, match="at least 16"):
        _reviewer_token(config)
    monkeypatch.setenv("TEST_APPROVER_TOKEN", "a-long-enough-token")
    token = _reviewer_token(config)
    assert token is not None and token.get_secret_value() == "a-long-enough-token"
    assert _reviewer_token(ApprovalConfig(enabled=False)) is None
