"""工作单元（Unit of Work）：把「一次业务操作 = 一个事务」这件事变成可复用的载体。

为什么需要它：
用例要写「扣库存 + 建订单 + 核销券 + 记幂等 + 落事件」五件事，
它们**必须同生共死**。如果没有 UoW，用例就得自己 new session、自己 new 五个仓储、
自己管 commit/rollback —— 每个用例重复一遍，且很容易漏掉某个仓储用了别的 session。
UoW 把「同一事务内的仓储集合」一次性提供出来，用例只关心业务顺序。

为什么 `commit()` 由用例显式调用，而不是 `__aexit__` 里自动提交：
显式提交是一个**断点**：读到这一行就知道「事务在这里结束、副作用从这里开始」。
自动提交会让「忘写 commit 导致数据没落库」这种 bug 变成静默的。
`__aexit__` 只负责**出错时回滚**与关闭会话。
"""

from __future__ import annotations

from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from .db.session import CommerceDatabase
from .repositories import (
    SqlCartRepository,
    SqlCouponRepository,
    SqlIdempotencyRepository,
    SqlMemberRepository,
    SqlOrderRepository,
    SqlOutboxRepository,
    SqlProductRepository,
    SqlStockRepository,
)


class SqlUnitOfWork:
    """SQLAlchemy 版工作单元。属性在 `__aenter__` 时绑定到同一个 session。"""

    products: SqlProductRepository
    stocks: SqlStockRepository
    cart: SqlCartRepository
    orders: SqlOrderRepository
    members: SqlMemberRepository
    coupons: SqlCouponRepository
    idempotency: SqlIdempotencyRepository
    outbox: SqlOutboxRepository

    def __init__(self, db: CommerceDatabase) -> None:
        self._db = db
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        assert self._session is not None, "UoW 必须在 async with 上下文中使用"
        return self._session

    async def __aenter__(self) -> "SqlUnitOfWork":
        self._session = self._db.session()
        session = self._session
        self.products = SqlProductRepository(session)
        self.stocks = SqlStockRepository(session)
        self.cart = SqlCartRepository(session)
        self.orders = SqlOrderRepository(session)
        self.members = SqlMemberRepository(session)
        self.coupons = SqlCouponRepository(session)
        self.idempotency = SqlIdempotencyRepository(session)
        self.outbox = SqlOutboxRepository(session)
        return self

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if exc_type is not None:
                await self.rollback()
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None


def build_uow_factory(db: CommerceDatabase) -> Callable[[], SqlUnitOfWork]:
    """返回一个「每次调用都给一个新 UoW」的工厂。

    工厂而不是共享实例：UoW 持有 session，共享它等于把事务串成一条线，
    并发请求会互相干扰（前一个请求没提交，后一个的写入就落在别人的事务里）。
    """

    return lambda: SqlUnitOfWork(db)


__all__ = ["SqlUnitOfWork", "build_uow_factory"]
