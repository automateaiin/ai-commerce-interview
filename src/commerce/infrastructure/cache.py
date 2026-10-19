"""缓存：Redis 优先，未配置则**显式降级**为进程内 TTL/LRU。

为什么必须显式降级而不是静默：
缓存是优化而非正确性依赖，生产环境 Redis 挂了业务必须继续跑 ——
所以这里降级不抛异常，但会打 `degraded=True` 的结构化日志。
运维看日志就知道「现在跑的是本地缓存」，多实例部署时缓存不共享这件事才不会变成隐性故障。

为什么不用「一个 try/except 包住所有读写」的写法：
降级的正确位置在**装配时**（容器里 `_try_connect_redis` 已经把不可用判定了），
读写路径再去 try 一遍只会把 Redis 抖动变成每个请求的额外延迟。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from typing import Any, Protocol

from ...infra.core.logging import get_logger

logger = get_logger("commerce.cache")


class CacheBackend(Protocol):
    backend: str

    async def get_json(self, key: str) -> dict[str, Any] | None: ...

    async def set_json(self, key: str, value: dict[str, Any], ttl_s: int = 60) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr_window(self, key: str, *, window_s: int) -> int:
        """固定窗口计数。用于「会员 × SKU 下单频率」这类业务限流。"""
        ...


class InMemoryTTLCache:
    """进程内缓存。带容量上限 + LRU 淘汰：无上限的 dict 缓存等于内存泄漏。"""

    backend = "memory"

    def __init__(self, *, max_entries: int = 1024) -> None:
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max = max_entries
        self._lock = asyncio.Lock()

    def _purge(self, key: str) -> None:
        entry = self._store.get(key)
        if entry and entry[0] <= time.monotonic():
            self._store.pop(key, None)

    async def get_json(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            self._purge(key)
            entry = self._store.get(key)
            if entry is None:
                return None
            self._store.move_to_end(key)
            return entry[1]

    async def set_json(self, key: str, value: dict[str, Any], ttl_s: int = 60) -> None:
        async with self._lock:
            self._store[key] = (time.monotonic() + ttl_s, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def incr_window(self, key: str, *, window_s: int) -> int:
        async with self._lock:
            entry = self._store.get(key)
            now = time.monotonic()
            if entry is None or entry[0] <= now:
                self._store[key] = (now + window_s, {"count": 1})
                self._store.move_to_end(key)
                return 1
            current = int(entry[1].get("count", 0)) + 1
            self._store[key] = (entry[0], {"count": current})
            return current


class RedisCache:
    """Redis 实现。窗口计数用 INCR + EXPIRE，天然是「首次计数时才设过期」。"""

    backend = "redis"

    def __init__(self, client: Any, *, prefix: str = "commerce:") -> None:
        self._client = client
        self._prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get_json(self, key: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._k(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 值坏了就删掉：留着只会让后续每次读都解析失败
            await self.delete(key)
            return None

    async def set_json(self, key: str, value: dict[str, Any], ttl_s: int = 60) -> None:
        await self._client.set(self._k(key), json.dumps(value, ensure_ascii=False), ex=ttl_s)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._k(key))

    async def incr_window(self, key: str, *, window_s: int) -> int:
        full = self._k(key)
        count = await self._client.incr(full)
        if count == 1:
            await self._client.expire(full, window_s)
        return int(count)


def build_cache(redis_client: Any | None) -> CacheBackend:
    if redis_client is not None:
        logger.info("cache.ready", extra={"event": "startup", "backend": "redis", "degraded": False})
        return RedisCache(redis_client)
    logger.warning(
        "cache.degraded",
        extra={"event": "startup", "backend": "memory", "degraded": True, "reason": "redis_unavailable"},
    )
    return InMemoryTTLCache()


__all__ = ["CacheBackend", "InMemoryTTLCache", "RedisCache", "build_cache"]
