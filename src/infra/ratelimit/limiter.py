"""滑动窗口限流：Redis ZSET + Lua 原子实现，内存降级，Redis 故障 fail-open。

三个必须讲清的点（面试高频追问）：

1) **为什么是滑动窗口，不是固定窗口**
   固定窗口在窗口边界的「跨窗双倍放行」是经典漏洞：59s 打满 100 次、61s 再打满 100 次，
   1 秒过 200 次但两个窗口各自「没超」；滑动窗口按时间戳淘汰，任何时刻都只看最近 N 秒。

2) **为什么用 ZSET + Lua 而不是应用层读改写**
   读-判-写如果不原子，多实例下会同时放行；把「淘汰过期 + 计数 + 写入」放进一次
   `EVAL`，Redis 单线程执行脚本，天然原子，也省一趟 RTT。

3) **Redis 挂了怎么办：fail-open**
   限流是**保护性**能力，不是业务能力。它故障时若选择拒绝所有请求（fail-close），
   等于把缓存故障升级成全站不可用；所以这里放行并**显式标记 degraded**，
   由监控告警 + 配额层兜底，而不是让核心客服链路跟着挂。

与 quota/ 的分工：quota 管「窗口内总量与计费」，本模块管「瞬时速率」。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.logging import get_logger
from ..core.metrics import metrics

logger = get_logger("infra.ratelimit")


@dataclass(slots=True)
class RateLimitDecision:
    """一次限流判定的结论。remaining / retry_after 都要能直接回给调用方。"""

    allowed: bool
    dimension: str
    key_hash: str
    limit: int
    remaining: int
    retry_after_s: int = 0
    backend: str = "memory"
    degraded: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def hash_key(key: str) -> str:
    """把限流主键（可能是 IP）哈希后用于日志与指标。

    IP 属于个人可识别信息，日志里不应该出现原文；哈希后仍可聚合统计，
    需要精确定位时再用 trace 关联。
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


# 原子滑动窗口：淘汰过期成员 → 计数 → 未超限则写入并续期。
# 返回 {allowed, count, retry_after_ms}，全部在一次 EVAL 内完成。
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window_ms)
local count = redis.call('ZCARD', key)

if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry = window_ms
  if oldest[2] then
    retry = (tonumber(oldest[2]) + window_ms) - now
  end
  if retry < 0 then retry = 0 end
  return {0, count, math.floor(retry)}
end

redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, window_ms)
return {1, count + 1, 0}
"""


class SlidingWindowStore(Protocol):
    async def hit(self, key: str, *, limit: int, window_s: int) -> tuple[bool, int, int]: ...


class InMemorySlidingWindowStore:
    """进程内实现：单实例或本地演示。asyncio.Lock 保证「判-写」原子。"""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def hit(self, key: str, *, limit: int, window_s: int) -> tuple[bool, int, int]:
        now = time.monotonic()
        window = float(window_s)
        async with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(hits[0] + window - now) + 1 if hits else 1
                return False, len(hits), max(retry, 1)
            hits.append(now)
            return True, len(hits), 0


class RedisSlidingWindowStore:
    """Redis 实现：ZSET 按请求时间戳打分，成员用「时间戳-序号」保证唯一。"""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._seq = 0

    async def hit(self, key: str, *, limit: int, window_s: int) -> tuple[bool, int, int]:
        self._seq += 1
        now_ms = int(time.time() * 1000)
        window_ms = int(window_s * 1000)
        member = f"{now_ms}-{self._seq}"
        allowed, count, retry_ms = await self._redis.eval(
            _SLIDING_WINDOW_LUA, 1, f"ratelimit:{key}", now_ms, window_ms, limit, member
        )
        return int(allowed) == 1, int(count), max(1, round(int(retry_ms) / 1000)) if int(allowed) == 0 else 0


class RateLimiter:
    """限流门面：统一 fail-open 语义与观测出口。"""

    def __init__(self, store: SlidingWindowStore, *, enabled: bool = True, telemetry: Any | None = None) -> None:
        self._store = store
        self._enabled = enabled
        self._telemetry = telemetry

    @property
    def backend(self) -> str:
        return type(self._store).__name__

    async def check(self, *, dimension: str, key: str, limit: int, window_s: int) -> RateLimitDecision:
        key_hash = hash_key(f"{dimension}:{key}")
        if not self._enabled:
            return RateLimitDecision(
                allowed=True, dimension=dimension, key_hash=key_hash, limit=limit, remaining=limit, backend="disabled"
            )
        try:
            allowed, count, retry_after_s = await self._store.hit(f"{dimension}:{key}", limit=limit, window_s=window_s)
        except Exception as exc:  # noqa: BLE001 - 降级路径必须兜住全部异常
            metrics.incr("ratelimit_degraded", dimension)
            logger.warning(
                "ratelimit.degraded",
                extra={
                    "event": "ratelimit",
                    "status": "degraded",
                    "degraded": True,
                    "dimension": dimension,
                    "key_hash": key_hash,
                    "reason": type(exc).__name__,
                },
            )
            return RateLimitDecision(
                allowed=True,
                dimension=dimension,
                key_hash=key_hash,
                limit=limit,
                remaining=limit,
                backend="degraded",
                degraded=True,
                meta={"reason": type(exc).__name__},
            )

        remaining = max(0, limit - count)
        decision = RateLimitDecision(
            allowed=allowed,
            dimension=dimension,
            key_hash=key_hash,
            limit=limit,
            remaining=remaining,
            retry_after_s=retry_after_s,
            backend=self.backend,
        )
        if not allowed:
            metrics.incr("ratelimit_rejected", dimension)
            logger.warning(
                "ratelimit.rejected",
                extra={
                    "event": "ratelimit",
                    "status": "error",
                    "error_code": "rate_limited",
                    "dimension": dimension,
                    "key_hash": key_hash,
                    "limit": limit,
                    "remaining": remaining,
                    "retry_after_s": retry_after_s,
                },
            )
            if self._telemetry is not None:
                self._telemetry.event("ratelimit.rejected", dimension=dimension, limit=limit)
        return decision


def build_rate_limiter(settings: Any, redis_client: Any | None = None, telemetry: Any | None = None) -> RateLimiter:
    """有 Redis 用 ZSET（多实例一致），否则退化为进程内滑动窗口。"""
    store: SlidingWindowStore = (
        RedisSlidingWindowStore(redis_client) if redis_client is not None else InMemorySlidingWindowStore()
    )
    return RateLimiter(store, enabled=getattr(settings, "ratelimit_enabled", True), telemetry=telemetry)


__all__ = [
    "RateLimitDecision",
    "RateLimiter",
    "InMemorySlidingWindowStore",
    "RedisSlidingWindowStore",
    "build_rate_limiter",
    "hash_key",
]
