"""限流：滑动窗口语义 + Redis 故障 fail-open。

两条必须守住的不变量：
1) 同一窗口内放行次数**恰好**等于 limit，边界不双倍放行（固定窗口的经典漏洞）；
2) 限流后端故障时**放行**并显式标记 degraded —— 保护性能力不故障升级成全站不可用。
"""

from __future__ import annotations

from src.infra.ratelimit.limiter import (
    InMemorySlidingWindowStore,
    RateLimiter,
    build_rate_limiter,
    hash_key,
)

WINDOW = 60


async def test_sliding_window_allows_exactly_limit_then_denies():
    store = InMemorySlidingWindowStore()
    for _ in range(3):
        allowed, count, retry = await store.hit("service:cs", limit=3, window_s=WINDOW)
        assert allowed is True
        assert count <= 3

    allowed, count, retry = await store.hit("service:cs", limit=3, window_s=WINDOW)
    assert allowed is False
    assert count == 3
    assert retry >= 1  # 拒绝时必须给出退避时长，否则调用方只能盲重试


async def test_wrong_dimension_key_does_not_cross_contaminate():
    """IP 维度与 Service 维度必须各自计数：混在一起会让「谁在刷接口」无法定位。"""
    store = InMemorySlidingWindowStore()
    assert (await store.hit("ip:1.2.3.4", limit=1, window_s=WINDOW))[0] is True
    assert (await store.hit("service:cs", limit=1, window_s=WINDOW))[0] is True
    assert (await store.hit("ip:1.2.3.4", limit=1, window_s=WINDOW))[0] is False


async def test_backend_failure_is_fail_open():
    class BrokenStore:
        async def hit(self, key: str, *, limit: int, window_s: int):
            raise ConnectionError("redis is down")

    limiter = RateLimiter(BrokenStore())
    decision = await limiter.check(dimension="service", key="cs", limit=1, window_s=WINDOW)

    assert decision.allowed is True
    assert decision.degraded is True
    assert decision.backend == "degraded"


async def test_disabled_limiter_short_circuits():
    limiter = RateLimiter(InMemorySlidingWindowStore(), enabled=False)
    first = await limiter.check(dimension="ip", key="1.2.3.4", limit=1, window_s=WINDOW)
    second = await limiter.check(dimension="ip", key="1.2.3.4", limit=1, window_s=WINDOW)

    assert first.allowed and second.allowed
    assert second.backend == "disabled"


async def test_rejected_decision_carries_hashed_key_not_raw_ip():
    limiter = RateLimiter(InMemorySlidingWindowStore())
    await limiter.check(dimension="ip", key="203.0.113.7", limit=1, window_s=WINDOW)
    decision = await limiter.check(dimension="ip", key="203.0.113.7", limit=1, window_s=WINDOW)

    assert decision.allowed is False
    assert decision.key_hash == hash_key("ip:203.0.113.7")
    assert "203.0.113.7" not in decision.key_hash  # 日志/指标里不落 IP 原文


def test_build_rate_limiter_falls_back_to_memory_without_redis():
    class Settings:
        ratelimit_enabled = True

    limiter = build_rate_limiter(Settings(), redis_client=None)
    assert limiter.backend == "InMemorySlidingWindowStore"
