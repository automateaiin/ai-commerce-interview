"""配额：预留-实扣两步结算。

为什么必须两步（面试追问「并发超发怎么防」的答案）：
如果等模型返回后才扣真实用量，那么在请求返回前的这段时间里，
并发请求看到的都是「还没扣」的余额，会一起放行 → 超发。
所以请求前先按「输入长度 + max_tokens」估一个上界做**预留**，
结束后用真实 usage **结算**（多退少补），失败/超时则**释放**预留。

关键不变量：reserved 只有在 settle 或 release 时才会归还，任何路径都不能既不结算也不释放 ——
因此对外只暴露 `quota_scope()` 上下文管理器，异常路径自动释放。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from ..core.errors import QuotaExceededError
from ..core.logging import get_logger
from ..core.metrics import metrics

logger = get_logger("infra.quota")


@dataclass(slots=True)
class QuotaLimits:
    requests_per_window: int
    tokens_per_window: int
    window_s: int
    reservation_ttl_s: int


@dataclass(slots=True)
class Reservation:
    """一次预留。settle/release 必须且只能发生一次。"""

    reservation_id: str
    project_id: str
    requests: int = 1
    tokens: int = 0
    window_s: int = 3600
    settled: bool = False
    backend: str = "memory"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def window_start(self) -> int:
        return int(time.time()) // self.window_s * self.window_s


class QuotaStore(Protocol):
    async def reserve(self, project_id: str, *, requests: int, tokens: int, limits: QuotaLimits) -> Reservation: ...
    async def settle(self, reservation: Reservation, *, actual_tokens: int) -> None: ...
    async def release(self, reservation: Reservation) -> None: ...
    async def usage(self, project_id: str, window_s: int) -> dict[str, float]: ...


def _check_limits(current_requests: float, current_tokens: float, limits: QuotaLimits) -> None:
    if current_requests > limits.requests_per_window:
        raise QuotaExceededError(
            f"请求数配额已用尽（窗口 {limits.window_s}s）",
            detail={
                "limit": limits.requests_per_window,
                "used": current_requests,
                "quota_type": "requests",
            },
        )
    if current_tokens > limits.tokens_per_window:
        raise QuotaExceededError(
            f"Token 配额已用尽（窗口 {limits.window_s}s）",
            detail={
                "limit": limits.tokens_per_window,
                "used": current_tokens,
                "quota_type": "tokens",
            },
        )


class InMemoryQuotaStore:
    """进程内实现：单实例部署或本地演示使用。

    asyncio.Lock 保证「读-判-加」是原子的 —— 这正是多请求同时放行的根因所在。
    多实例部署必须换 Redis，因为进程内内存无法跨实例共享。
    """

    def __init__(self) -> None:
        self._requests: dict[tuple[str, int], float] = {}
        self._tokens: dict[tuple[str, int], float] = {}
        self._lock = asyncio.Lock()

    def _keys(self, project_id: str, window_s: int) -> tuple[tuple[str, int], tuple[str, int]]:
        window_start = int(time.time()) // window_s * window_s
        return (project_id, window_start), (project_id, window_start)

    async def reserve(self, project_id: str, *, requests: int, tokens: int, limits: QuotaLimits) -> Reservation:
        reservation = Reservation(
            reservation_id=f"rsv_{project_id}_{int(time.time()*1000)}",
            project_id=project_id,
            requests=requests,
            tokens=tokens,
            window_s=limits.window_s,
            backend="memory",
        )
        request_key, token_key = self._keys(project_id, limits.window_s)
        async with self._lock:
            next_requests = self._requests.get(request_key, 0) + requests
            next_tokens = self._tokens.get(token_key, 0) + tokens
            try:
                _check_limits(next_requests, next_tokens, limits)
            except QuotaExceededError:
                # 检查不通过时不能留下任何痕迹，否则余额会被永久吞掉
                metrics.incr("quota_rejected", project_id)
                raise
            self._requests[request_key] = next_requests
            self._tokens[token_key] = next_tokens
        return reservation

    async def settle(self, reservation: Reservation, *, actual_tokens: int) -> None:
        """多退少补：把预留与真实的差额还回去（真实可以小于也可以大于预留）。"""
        if reservation.settled:
            return
        _, token_key = self._keys(reservation.project_id, reservation.window_s)
        delta = reservation.tokens - actual_tokens  # >0 表示退回，<0 表示补扣
        async with self._lock:
            self._tokens[token_key] = self._tokens.get(token_key, 0) - delta
        reservation.settled = True

    async def release(self, reservation: Reservation) -> None:
        if reservation.settled:
            return
        async with self._lock:
            request_key, token_key = self._keys(reservation.project_id, reservation.window_s)
            self._requests[request_key] = max(0.0, self._requests.get(request_key, 0) - reservation.requests)
            self._tokens[token_key] = max(0.0, self._tokens.get(token_key, 0) - reservation.tokens)
        reservation.settled = True

    async def usage(self, project_id: str, window_s: int) -> dict[str, float]:
        request_key, token_key = self._keys(project_id, window_s)
        return {
            "requests": self._requests.get(request_key, 0.0),
            "tokens": self._tokens.get(token_key, 0.0),
        }


# 原子「检查并扣减」脚本：把 INCR、比较、回滚放进一次服务端执行，
# 避免「先加后判」在多实例并发下产生超发窗口。
_RESERVE_LUA = """
local r = redis.call('INCRBY', KEYS[1], ARGV[1])
local t = redis.call('INCRBY', KEYS[2], ARGV[2])
if r > tonumber(ARGV[3]) or t > tonumber(ARGV[4]) then
  redis.call('DECRBY', KEYS[1], ARGV[1])
  redis.call('DECRBY', KEYS[2], ARGV[2])
  return 0
end
redis.call('EXPIRE', KEYS[1], ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[5])
return 1
"""


class RedisQuotaStore:
    """Redis 实现：多实例部署时必须使用。

    用 Lua 脚本做原子「检查+扣减」；TTL 让窗口自然过期，不会无限增长。
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    @staticmethod
    def _keys(project_id: str, window_s: int) -> tuple[str, str]:
        window_start = int(time.time()) // window_s * window_s
        base = f"quota:{project_id}:{window_start}"
        return f"{base}:requests", f"{base}:tokens"

    async def reserve(self, project_id: str, *, requests: int, tokens: int, limits: QuotaLimits) -> Reservation:
        request_key, token_key = self._keys(project_id, limits.window_s)
        result = await self._redis.eval(
            _RESERVE_LUA,
            2,
            request_key,
            token_key,
            requests,
            tokens,
            limits.requests_per_window,
            limits.tokens_per_window,
            limits.window_s + limits.reservation_ttl_s,
        )
        if int(result) != 1:
            metrics.incr("quota_rejected", project_id)
            usage = await self.usage(project_id, limits.window_s)
            quota_type = "requests" if usage["requests"] > limits.requests_per_window else "tokens"
            raise QuotaExceededError(
                f"{quota_type} 配额已用尽", detail={"quota_type": quota_type, "usage": usage}
            )
        return Reservation(
            reservation_id=f"rsv_{project_id}_{int(time.time()*1000)}",
            project_id=project_id,
            requests=requests,
            tokens=tokens,
            window_s=limits.window_s,
            backend="redis",
        )

    async def settle(self, reservation: Reservation, *, actual_tokens: int) -> None:
        if reservation.settled:
            return
        _, token_key = self._keys(reservation.project_id, reservation.window_s)
        delta = reservation.tokens - actual_tokens
        if delta:
            await self._redis.incrbyfloat(token_key, -delta)
        reservation.settled = True

    async def release(self, reservation: Reservation) -> None:
        if reservation.settled:
            return
        request_key, token_key = self._keys(reservation.project_id, reservation.window_s)
        await self._redis.decrby(request_key, reservation.requests)
        if reservation.tokens:
            await self._redis.incrbyfloat(token_key, -reservation.tokens)
        reservation.settled = True

    async def usage(self, project_id: str, window_s: int) -> dict[str, float]:
        request_key, token_key = self._keys(project_id, window_s)
        values = await self._redis.mget(request_key, token_key)
        return {
            "requests": float(values[0] or 0),
            "tokens": float(values[1] or 0),
        }


class QuotaManager:
    """配额门面：对外只暴露 quota_scope，保证「预留必有结算或释放」。"""

    def __init__(self, store: QuotaStore, *, default_limits: QuotaLimits, enabled: bool = True) -> None:
        self._store = store
        self._default_limits = default_limits
        self._enabled = enabled

    @property
    def backend(self) -> str:
        return type(self._store).__name__

    @property
    def store(self) -> QuotaStore:
        return self._store

    def limits_for(self, *, requests: int | None = None, tokens: int | None = None) -> QuotaLimits:
        return QuotaLimits(
            requests_per_window=requests or self._default_limits.requests_per_window,
            tokens_per_window=tokens or self._default_limits.tokens_per_window,
            window_s=self._default_limits.window_s,
            reservation_ttl_s=self._default_limits.reservation_ttl_s,
        )

    @asynccontextmanager
    async def quota_scope(
        self,
        project_id: str,
        *,
        estimated_tokens: int = 0,
        requests: int = 1,
        limits: QuotaLimits | None = None,
    ) -> AsyncIterator[Reservation]:
        """预留 → 执行 → 结算。

        用法：
            async with quota.quota_scope(project_id, estimated_tokens=n) as rsv:
                result = await call_model()
                rsv.meta["actual_tokens"] = result.usage.total_tokens

        结算优先级：meta["actual_tokens"] > 预留值（无真实用量时按预留结算）。
        异常路径自动 release，绝不吞配额。
        """
        if not self._enabled:
            yield Reservation(reservation_id="disabled", project_id=project_id, backend="disabled")
            return

        reservation = await self._store.reserve(
            project_id, requests=requests, tokens=estimated_tokens, limits=limits or self._default_limits
        )
        logger.info(
            "quota.reserved",
            extra={
                "event": "quota_reserve",
                "quota_reserved": estimated_tokens,
                "project_id": project_id,
                "quota_backend": reservation.backend,
            },
        )
        try:
            yield reservation
        except BaseException:
            await self._store.release(reservation)
            logger.warning(
                "quota.released",
                extra={"event": "quota_release", "project_id": project_id, "reason": "request_failed"},
            )
            raise
        else:
            actual = int(reservation.meta.get("actual_tokens", reservation.tokens))
            await self._store.settle(reservation, actual_tokens=actual)
            metrics.incr("quota_settled", project_id)
            logger.info(
                "quota.settled",
                extra={
                    "event": "quota_settle",
                    "quota_reserved": reservation.tokens,
                    "quota_settled": actual,
                    "project_id": project_id,
                },
            )


def build_quota_manager(settings: Any, redis_client: Any | None = None) -> QuotaManager:
    """按可用性选择后端：有 Redis 用 Redis（多实例安全），否则退化为进程内实现。

    探测失败时降级并**记录日志**，不做静默降级 —— 静默降级会让多实例下的超发变得不可见。
    """
    limits = QuotaLimits(
        requests_per_window=settings.quota_requests_per_window,
        tokens_per_window=settings.quota_tokens_per_window,
        window_s=settings.quota_window_s,
        reservation_ttl_s=settings.quota_reservation_ttl_s,
    )
    if redis_client is not None:
        return QuotaManager(RedisQuotaStore(redis_client), default_limits=limits, enabled=settings.quota_enabled)
    return QuotaManager(InMemoryQuotaStore(), default_limits=limits, enabled=settings.quota_enabled)
