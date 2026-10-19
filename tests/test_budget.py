"""成本预算：两道阈值 + 预扣-校正。

硬上限是止损线，软阈值是「体验换成本」；两者缺一不可 ——
只有软阈值时用量仍会无限涨，只有硬上限时业务会被直接掐断。
"""

from __future__ import annotations

from src.infra.quota.budget import AgentBudgetGuard, BudgetLimits, InMemoryBudgetStore


def _guard(*, daily: int = 1000, ratio: float = 0.8) -> AgentBudgetGuard:
    return AgentBudgetGuard(
        InMemoryBudgetStore(), limits=BudgetLimits(daily_tokens_per_agent=daily, downgrade_ratio=ratio)
    )


def test_scope_is_service_plus_agent():
    assert AgentBudgetGuard.scope_of(service="cs-service", agent="customer_agent") == "cs-service/customer_agent"
    assert AgentBudgetGuard.scope_of(service=None, agent=None) == "unknown/default"


async def test_below_soft_threshold_stays_on_full_model():
    guard = _guard(daily=1000, ratio=0.8)
    decision = await guard.decide(service="cs-service", agent="a1", estimated_tokens=100)

    assert decision.allowed is True
    assert decision.downgrade is False
    assert decision.ratio == 0.1


async def test_soft_threshold_triggers_downgrade_without_blocking():
    guard = _guard(daily=1000, ratio=0.8)
    decision = await guard.decide(service="cs-service", agent="a1", estimated_tokens=850)

    assert decision.allowed is True
    assert decision.downgrade is True


async def test_hard_limit_denies_and_rolls_back_precharge():
    guard = _guard(daily=200, ratio=0.8)
    await guard.decide(service="cs-service", agent="a1", estimated_tokens=150)
    denied = await guard.decide(service="cs-service", agent="a1", estimated_tokens=100)

    assert denied.allowed is False

    # 被拒的预扣必须回滚，否则额度会被拒绝的请求慢慢吞掉
    usage = await guard.usage(service="cs-service", agent="a1")
    assert usage["used_tokens"] == 150


async def test_settle_corrects_reservation_towards_real_usage():
    guard = _guard(daily=1000)
    decision = await guard.decide(service="cs-service", agent="a1", estimated_tokens=400)
    await guard.settle(service="cs-service", agent="a1", reserved_tokens=decision.used_tokens, actual_tokens=120)

    usage = await guard.usage(service="cs-service", agent="a1")
    assert usage["used_tokens"] == 120


async def test_disabled_budget_always_allows():
    guard = AgentBudgetGuard(
        InMemoryBudgetStore(),
        limits=BudgetLimits(daily_tokens_per_agent=0, downgrade_ratio=0.8, enabled=False),
    )
    decision = await guard.decide(service="cs-service", agent="a1", estimated_tokens=10**9)

    assert decision.allowed is True
    assert decision.downgrade is False


async def test_store_failure_does_not_block_business():
    class BrokenStore:
        async def add(self, key: str, tokens: int, *, ttl_s: int) -> int:
            raise ConnectionError("redis is down")

    guard = AgentBudgetGuard(BrokenStore(), limits=BudgetLimits(daily_tokens_per_agent=1000, downgrade_ratio=0.8))
    decision = await guard.decide(service="cs-service", agent="a1", estimated_tokens=500)

    assert decision.allowed is True
    assert decision.degraded is True
