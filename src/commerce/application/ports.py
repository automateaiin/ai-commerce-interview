"""应用层端口：依赖倒置的接入点。

三个端口对应三种「应用层需要、但不该自己实现」的能力：
- `UnitOfWork`：事务边界的抽象。用例写的是「五件事同生共死」，
  它不知道背后是 SQLite 还是 PostgreSQL，也不知道仓储是 ORM 还是别的；
- `CachePort`：缓存抽象。Redis 可用就用 Redis，不可用就进程内 —— 用例无感；
- `AIHookPort`：AI 侧缝。M1 只有 `NoopHook`，后续换成真实实现时
  用例代码一行都不用改（这正是把钩子做成端口的意义）。

注意这里**只放 Protocol，不放实现**：实现分别在 infrastructure 与
（后续的）harness/dataplatform 侧。用例只 import 本文件。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from ..domain.events import DomainEvent
from ..domain.repository import (
    CartRepository,
    CouponRepository,
    IdempotencyRepository,
    MemberRepository,
    OrderRepository,
    OutboxRepository,
    ProductRepository,
    StockRepository,
)


class UnitOfWork(Protocol):
    """一个事务内的仓储集合。用例必须 `async with` 使用，并显式 `commit()`。"""

    products: ProductRepository
    stocks: StockRepository
    cart: CartRepository
    orders: OrderRepository
    members: MemberRepository
    coupons: CouponRepository
    idempotency: IdempotencyRepository
    outbox: OutboxRepository

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class CachePort(Protocol):
    backend: str

    async def get_json(self, key: str) -> dict[str, Any] | None: ...

    async def set_json(self, key: str, value: dict[str, Any], ttl_s: int = 60) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr_window(self, key: str, *, window_s: int) -> int:
        """固定窗口计数，用于「会员 × SKU 下单频率」业务限流。"""
        ...


class AIHookPort(Protocol):
    """AI 钩子。**调用方必须在事务提交之后**才触发它 —— 见 usecases/order_uc.py。"""

    async def on_product_published(self, event: DomainEvent) -> None: ...

    async def on_order_placed(self, event: DomainEvent) -> None: ...

    async def on_order_cancelled(self, event: DomainEvent) -> None: ...


UowFactory = Callable[[], UnitOfWork]
EventHandler = Callable[[DomainEvent], Awaitable[None]]


__all__ = ["AIHookPort", "CachePort", "EventHandler", "UnitOfWork", "UowFactory"]
