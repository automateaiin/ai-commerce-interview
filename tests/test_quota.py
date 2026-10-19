"""配额：预留-实扣的不变量。

核心不变量只有一条：**预留必须有且只有一次结算或释放**。
任何「既不结算也不释放」的路径都会把配额永久吞掉，
所以对外只暴露 quota_scope() 这个上下文管理器，异常路径自动释放。
"""

from __future__ import annotations

import pytest

from src.infra.core.errors import QuotaExceededError
from src.infra.quota.store import InMemoryQuotaStore, QuotaLimits, QuotaManager

WINDOW = 3600


def _manager(requests: int = 10, tokens: int = 1000) -> QuotaManager:
    limits = QuotaLimits(requests_per_window=requests, tokens_per_window=tokens, window_s=WINDOW, reservation_ttl_s=60)
    return QuotaManager(InMemoryQuotaStore(), default_limits=limits)


async def test_settlement_uses_real_usage_not_reservation():
    """多退少补：预留是上界估计，结算以真实 usage 为准。"""
    manager = _manager()
    limits = manager.limits_for()

    async with manager.quota_scope("p1", estimated_tokens=500, limits=limits) as reservation:
        reservation.meta["actual_tokens"] = 120

    usage = await manager.store.usage("p1", WINDOW)
    assert usage["requests"] == 1
    assert usage["tokens"] == 120  # 500 预留被退回，只扣 120


async def test_failed_request_releases_reservation():
    manager = _manager()
    limits = manager.limits_for()

    with pytest.raises(RuntimeError):
        async with manager.quota_scope("p2", estimated_tokens=500, limits=limits):
            raise RuntimeError("模型调用炸了")

    usage = await manager.store.usage("p2", WINDOW)
    # 失败请求不能留下任何痕迹，否则重试会莫名其妙撞配额
    assert usage["requests"] == 0
    assert usage["tokens"] == 0


async def test_exceeding_quota_raises_429_with_retry_after():
    manager = _manager(requests=1)
    limits = manager.limits_for()

    async with manager.quota_scope("p3", estimated_tokens=10, limits=limits):
        pass

    with pytest.raises(QuotaExceededError) as captured:
        async with manager.quota_scope("p3", estimated_tokens=10, limits=limits):
            pass

    error = captured.value
    assert error.http_status == 429
    assert error.retry_after == 60
    assert error.detail["quota_type"] == "requests"

    # 被拒的请求不能把余额扣走（否则余额会被永久吞掉）
    usage = await manager.store.usage("p3", WINDOW)
    assert usage["requests"] == 1


async def test_token_quota_is_enforced_separately_from_request_quota():
    manager = _manager(requests=100, tokens=50)
    limits = manager.limits_for()

    with pytest.raises(QuotaExceededError) as captured:
        async with manager.quota_scope("p4", estimated_tokens=200, limits=limits):
            pass

    assert captured.value.detail["quota_type"] == "tokens"


async def test_settle_and_release_are_idempotent():
    """重复结算/释放必须是无操作：否则一次请求会把配额退两次。"""
    manager = _manager()
    limits = manager.limits_for()

    async with manager.quota_scope("p5", estimated_tokens=300, limits=limits) as reservation:
        reservation.meta["actual_tokens"] = 100

    await manager.store.settle(reservation, actual_tokens=100)
    await manager.store.release(reservation)

    usage = await manager.store.usage("p5", WINDOW)
    assert usage["requests"] == 1
    assert usage["tokens"] == 100


async def test_disabled_quota_does_not_touch_the_store():
    manager = QuotaManager(InMemoryQuotaStore(), default_limits=_manager().limits_for(), enabled=False)
    async with manager.quota_scope("p6", estimated_tokens=999) as reservation:
        assert reservation.backend == "disabled"

    usage = await manager.store.usage("p6", WINDOW)
    assert usage == {"requests": 0.0, "tokens": 0.0}
