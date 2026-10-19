"""重试机制：分类、退避、幂等约束。

面试要点：重试的正确性不在于「重试几次」，
而在于「哪些错误才允许重试」——重试不可重试的错误只会放大故障。
"""

from __future__ import annotations

import pytest

from src.infra.core.errors import (
    AuthenticationError,
    UpstreamRateLimitError,
    UpstreamTimeoutError,
    ValidationError,
)
from src.infra.core.retry import RetryPolicy, retry_async


FAST = RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter_ratio=0.0, max_delay_s=0.0)


async def test_retryable_error_is_retried_until_success():
    seen: list[int] = []

    async def operation(attempt: int) -> str:
        seen.append(attempt)
        if attempt < 3:
            raise UpstreamTimeoutError("下游超时")
        return "ok"

    assert await retry_async(operation, name="t", policy=FAST) == "ok"
    assert seen == [1, 2, 3]


async def test_non_retryable_error_fails_fast():
    seen: list[int] = []

    async def operation(attempt: int) -> str:
        seen.append(attempt)
        raise ValidationError("参数不合法")

    with pytest.raises(ValidationError):
        await retry_async(operation, name="t", policy=FAST)
    # 确定性错误重试 3 次只是白白放大故障
    assert seen == [1]


async def test_non_idempotent_operation_is_not_retried():
    seen: list[int] = []

    async def operation(attempt: int) -> str:
        seen.append(attempt)
        raise UpstreamTimeoutError("超时")

    with pytest.raises(UpstreamTimeoutError):
        await retry_async(operation, name="t", policy=FAST, idempotent=False)
    assert seen == [1]


async def test_auth_error_from_upstream_is_not_retried():
    """上游 401 对本服务来说是配置问题：换多少次结果都一样。"""

    async def operation(attempt: int) -> str:
        raise AuthenticationError("环境变量未配置")

    with pytest.raises(AuthenticationError):
        await retry_async(operation, name="t", policy=FAST)


async def test_total_budget_stops_retrying():
    seen: list[int] = []
    exhausted = RetryPolicy(max_attempts=5, base_delay_s=0.0, jitter_ratio=0.0, total_budget_s=0.0)

    async def operation(attempt: int) -> str:
        seen.append(attempt)
        raise UpstreamTimeoutError("超时")

    with pytest.raises(UpstreamTimeoutError):
        await retry_async(operation, name="t", policy=exhausted)
    # 预算为 0：不允许把请求长时间挂住
    assert seen == [1]


async def test_retry_if_hook_can_narrow_the_condition():
    seen: list[int] = []

    async def operation(attempt: int) -> str:
        seen.append(attempt)
        raise UpstreamTimeoutError("超时")

    with pytest.raises(UpstreamTimeoutError):
        await retry_async(operation, name="t", policy=FAST, retry_if=lambda error: error.code == "upstream_error")
    assert seen == [1]


def test_backoff_is_exponential_and_respects_retry_after():
    policy = RetryPolicy(base_delay_s=0.2, max_delay_s=4.0, jitter_ratio=0.0)

    # 指数退避：0.2 → 0.4 → 0.8，且被 max_delay 封顶
    assert policy.delay_for(1) == pytest.approx(0.2)
    assert policy.delay_for(2) == pytest.approx(0.4)
    assert policy.delay_for(5) == pytest.approx(3.2)

    # 下游明确给了 Retry-After 就尊重它，但仍受 max_delay 约束
    assert policy.delay_for(1, retry_after_s=1.0) == pytest.approx(1.0)
    assert policy.delay_for(1, retry_after_s=99.0) == pytest.approx(4.0)


def test_jitter_keeps_delay_in_expected_band():
    """抖动是为了让并发请求错峰重试，避免重试风暴；但必须在可预期区间内。"""
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=1.0, jitter_ratio=0.3)
    samples = [policy.delay_for(3) for _ in range(50)]
    assert all(0.7 <= value <= 1.3 for value in samples)
    assert len({round(value, 6) for value in samples}) > 1


def test_rate_limit_error_carries_retry_after():
    error = UpstreamRateLimitError("限流", retry_after_s=2.5)
    assert error.retryable is True and error.retry_after_s == 2.5
    assert error.http_status == 429
