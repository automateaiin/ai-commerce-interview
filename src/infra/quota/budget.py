"""成本预算：单 Agent 日预算 + 软阈值降级 Lite 模型。

为什么要按 **Agent** 聚合而不是按项目：项目是一整条业务线，里面可能同时跑着
客服 Agent、质检 Agent、运营报告 Agent。只按项目设一个总预算，
某个 Agent 逻辑写错（例如循环里每次调用都带全量上下文）就能把整条线的额度吃光，
而且事后根本分不清是谁花的。按 Agent 切分后，跑飞的那个 Agent 会先撞到自己的天花板。

两道阈值（面试要能讲清为什么不是一道）：
- **软阈值（downgrade_ratio）**：如 80% 用量后自动降级到 Lite 模型。
  它不中断业务 —— 客服答得稍差但仍可用，是「体验换成本」；
- **硬上限（daily_tokens）**：预估用量超过即直接拒绝（429）。这是止损线，
  没有它，软阈值形同虚设：降级只是变便宜，不代表不会继续涨。

预扣-校正是预算侧的必要动作：用量只能在模型返回后才知道，
所以先按「输入估算 + max_tokens」预扣，返回后用真实 usage 修正（与配额同一思路）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from ..core.logging import get_logger
from ..core.metrics import metrics

logger = get_logger("infra.budget")


@dataclass(slots=True)
class BudgetLimits:
    daily_tokens_per_agent: int
    downgrade_ratio: float
    enabled: bool = True


@dataclass(slots=True)
class BudgetDecision:
    """预算判定：allowed=false 表示已触及硬上限，downgrade=true 表示应切 Lite。"""

    allowed: bool
    downgrade: bool
    scope: str
    used_tokens: int
    limit_tokens: int
    ratio: float
    backend: str
    degraded: bool = False


def _utc_day(now: float | None = None) -> str:
    """按 UTC 自然日切分。

    用 UTC 而不是本地时区：多地域部署时本地日界各不相同，
    「谁的日预算」会变成一笔糊涂账，UTC 是唯一一致的基准。
    """
    current = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    return current.strftime("%Y%m%d")


def _seconds_to_next_utc_day(now: float | None = None) -> int:
    reference = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    tomorrow = (reference + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - reference).total_seconds()))


class BudgetStore(Protocol):
    async def add(self, key: str, tokens: int, *, ttl_s: int) -> int: ...


class InMemoryBudgetStore:
    """进程内实现：单实例或本地演示。累加在同一事件循环内是原子的。"""

    def __init__(self) -> None:
        self._usage: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def add(self, key: str, tokens: int, *, ttl_s: int) -> int:
        async with self._lock:
            next_value = max(0, self._usage.get(key, 0) + tokens)
            self._usage[key] = next_value
            return next_value


class RedisBudgetStore:
    """Redis 实现：INCRBY 原子累加，TTL 到下一个 UTC 日界自动清零（无清理任务）。"""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def add(self, key: str, tokens: int, *, ttl_s: int) -> int:
        value = int(await self._redis.incrby(key, tokens))
        if value < 0:  # 校正回退不应把用量打成负数
            await self._redis.set(key, 0, ex=ttl_s)
            return 0
        await self._redis.expire(key, ttl_s)
        return value


class AgentBudgetGuard:
    """预算门面：decide（预扣 + 判定）→ settle（按真实用量校正）。"""

    def __init__(self, store: BudgetStore, *, limits: BudgetLimits, telemetry: Any | None = None) -> None:
        self._store = store
        self._limits = limits
        self._telemetry = telemetry

    @property
    def backend(self) -> str:
        return type(self._store).__name__

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    @staticmethod
    def scope_of(*, service: str | None, agent: str | None) -> str:
        return f"{service or 'unknown'}/{agent or 'default'}"

    def _key(self, scope: str) -> str:
        return f"budget:{_utc_day()}:{scope}"

    async def decide(self, *, service: str | None, agent: str | None, estimated_tokens: int) -> BudgetDecision:
        """预扣并判定。预算未启用时一律放行（含 downgrade=False）。"""
        scope = self.scope_of(service=service, agent=agent)
        limit = self._limits.daily_tokens_per_agent
        if not self._limits.enabled or limit <= 0:
            return BudgetDecision(
                allowed=True, downgrade=False, scope=scope, used_tokens=0, limit_tokens=limit,
                ratio=0.0, backend="disabled",
            )

        ttl_s = _seconds_to_next_utc_day()
        try:
            used = await self._store.add(self._key(scope), estimated_tokens, ttl_s=ttl_s)
        except Exception as exc:  # noqa: BLE001 - 预算故障不应阻断业务，降级放行
            metrics.incr("budget_degraded", scope)
            logger.warning(
                "budget.degraded",
                extra={
                    "event": "budget",
                    "status": "degraded",
                    "degraded": True,
                    "budget_scope": scope,
                    "reason": type(exc).__name__,
                },
            )
            return BudgetDecision(
                allowed=True, downgrade=False, scope=scope, used_tokens=0, limit_tokens=limit,
                ratio=0.0, backend="degraded", degraded=True,
            )

        ratio = round(used / limit, 4)
        if used > limit:
            # 硬上限：把预扣回滚，避免拒绝的请求也留下痕迹（否则额度会被慢慢吞掉）
            await self._store.add(self._key(scope), -estimated_tokens, ttl_s=ttl_s)
            metrics.incr("budget_rejected", scope)
            logger.warning(
                "budget.exceeded",
                extra={
                    "event": "budget",
                    "status": "error",
                    "error_code": "quota_exceeded",
                    "budget_scope": scope,
                    "budget_used": used,
                    "budget_limit": limit,
                    "budget_ratio": ratio,
                },
            )
            return BudgetDecision(
                allowed=False, downgrade=False, scope=scope, used_tokens=used, limit_tokens=limit,
                ratio=ratio, backend=self.backend,
            )

        downgrade = ratio >= self._limits.downgrade_ratio
        if downgrade:
            metrics.incr("budget_downgrade", scope)
            logger.info(
                "budget.downgrade",
                extra={
                    "event": "budget",
                    "status": "ok",
                    "budget_scope": scope,
                    "budget_used": used,
                    "budget_limit": limit,
                    "budget_ratio": ratio,
                    "downgrade": True,
                },
            )
            if self._telemetry is not None:
                self._telemetry.event("budget.downgrade", budget_scope=scope, budget_ratio=ratio)
        return BudgetDecision(
            allowed=True, downgrade=downgrade, scope=scope, used_tokens=used, limit_tokens=limit,
            ratio=ratio, backend=self.backend,
        )

    async def settle(self, *, service: str | None, agent: str | None, reserved_tokens: int, actual_tokens: int) -> int:
        """用真实用量校正预扣（多退少补），返回校正后的当日累计。"""
        if not self._limits.enabled or self._limits.daily_tokens_per_agent <= 0:
            return 0
        scope = self.scope_of(service=service, agent=agent)
        delta = reserved_tokens - actual_tokens
        if not delta:
            return 0
        try:
            return await self._store.add(self._key(scope), -delta, ttl_s=_seconds_to_next_utc_day())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "budget.settle_failed",
                extra={"event": "budget", "status": "degraded", "budget_scope": scope, "reason": type(exc).__name__},
            )
            return 0

    async def usage(self, *, service: str | None, agent: str | None) -> dict[str, Any]:
        """只读查询当日用量（不回滚、不写入）。"""
        scope = self.scope_of(service=service, agent=agent)
        limit = self._limits.daily_tokens_per_agent
        try:
            used = await self._store.add(self._key(scope), 0, ttl_s=_seconds_to_next_utc_day())
        except Exception:  # noqa: BLE001
            used = 0
        return {
            "scope": scope,
            "used_tokens": used,
            "limit_tokens": limit,
            "ratio": round(used / limit, 4) if limit else 0.0,
            "backend": self.backend,
        }


def build_budget_guard(settings: Any, redis_client: Any | None = None, telemetry: Any | None = None) -> AgentBudgetGuard:
    limits = BudgetLimits(
        daily_tokens_per_agent=getattr(settings, "budget_daily_tokens_per_agent", 0),
        downgrade_ratio=getattr(settings, "budget_downgrade_ratio", 1.0),
        enabled=getattr(settings, "budget_enabled", True),
    )
    store: BudgetStore = RedisBudgetStore(redis_client) if redis_client is not None else InMemoryBudgetStore()
    return AgentBudgetGuard(store, limits=limits, telemetry=telemetry)


__all__ = [
    "BudgetLimits",
    "BudgetDecision",
    "BudgetStore",
    "AgentBudgetGuard",
    "InMemoryBudgetStore",
    "RedisBudgetStore",
    "build_budget_guard",
]
