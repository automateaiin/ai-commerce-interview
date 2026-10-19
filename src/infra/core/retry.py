"""重试机制：只重试「值得重试」的错误，并带上退避、抖动、预算三道约束。

三条原则（面试可直接讲）：
1) 先分类再重试。参数错误、鉴权失败、业务拒绝属于确定性错误，重试只会放大故障；
   只有网络超时、限流、5xx 这类暂时性错误才进重试路径 —— 依赖 InfraError.retryable 标记。
2) 退避必须带抖动，否则大量请求会在同一时刻同时重试，把下游二次打垮（重试风暴）。
3) 非幂等操作默认不重试。想重试必须由调用方显式声明 idempotent=True，
   并在业务层用幂等键保证「重复执行等价于执行一次」。
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from .errors import InfraError, UpstreamRateLimitError, normalize_exception
from .logging import get_logger

logger = get_logger("infra.retry")

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略。所有参数可配置，禁止在业务代码里写死。"""

    max_attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 4.0
    jitter_ratio: float = 0.3
    total_budget_s: float = 15.0
    """整段重试（含等待）的总预算，超过即放弃，避免请求长时间挂住。"""

    def delay_for(self, attempt: int, *, retry_after_s: float | None = None) -> float:
        """计算第 attempt 次失败后的等待时间（attempt 从 1 开始）。"""
        if retry_after_s is not None:
            # 下游明确给了 Retry-After，优先尊重它，但要受 max_delay 约束
            return min(max(retry_after_s, 0.0), self.max_delay_s)
        exponential = self.base_delay_s * (2 ** (attempt - 1))
        capped = min(exponential, self.max_delay_s)
        jitter = capped * random.uniform(-self.jitter_ratio, self.jitter_ratio)
        return max(0.0, capped + jitter)


DEFAULT_POLICY = RetryPolicy()


async def retry_async(
    operation: Callable[[int], Awaitable[T]],
    *,
    name: str,
    policy: RetryPolicy = DEFAULT_POLICY,
    idempotent: bool = True,
    retry_if: Callable[[InfraError], bool] | None = None,
) -> T:
    """执行带重试的异步操作。

    Args:
        operation: 接收 attempt（1-based）并返回 awaitable 的函数。
        name: 操作名，只用于日志与指标。
        policy: 重试策略（次数/退避/抖动/总预算）。
        idempotent: 该操作是否幂等。False 时任何错误都不重试，直接抛出。
        retry_if: 额外的重试判定钩子（例如「仅当错误码为 5xx 才重试」）。

    Raises:
        InfraError: 所有尝试耗尽后抛出最后一次归一化后的异常。
    """
    started = time.monotonic()
    last_error: InfraError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation(attempt)
        except BaseException as exc:  # noqa: BLE001 - 必须捕获后归一，再决定是否重试
            error = normalize_exception(exc)
            last_error = error

            exhausted = attempt >= policy.max_attempts
            budget_left = policy.total_budget_s - (time.monotonic() - started)

            if not error.retryable:
                raise error from exc
            if not idempotent:
                logger.info(
                    f"{name}.skip_retry_non_idempotent",
                    extra={"event": "retry_skipped", "attempt": attempt, "error_code": error.code, "reason": "non_idempotent"},
                )
                raise error from exc
            if retry_if is not None and not retry_if(error):
                raise error from exc
            if exhausted or budget_left <= 0:
                raise error from exc

            retry_after = getattr(error, "retry_after_s", None) if isinstance(error, UpstreamRateLimitError) else None
            delay = min(policy.delay_for(attempt, retry_after_s=retry_after), max(budget_left, 0.0))
            logger.warning(
                f"{name}.retry",
                extra={
                    "event": "retry",
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "sleep_ms": round(delay * 1000, 2),
                    "error_code": error.code,
                    "reason": "retryable_error",
                },
            )
            await asyncio.sleep(delay)

    # 理论上循环内已经 return 或 raise，这里兜底
    raise last_error or InfraError(f"{name} failed without error")


def is_transient(error: InfraError) -> bool:
    """常用的重试判定：只放行明确标记为暂时性的错误。"""
    return error.retryable


__all__ = ["RetryPolicy", "DEFAULT_POLICY", "retry_async", "is_transient", "Any"]
