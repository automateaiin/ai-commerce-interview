"""业务库连接与建表：**独立于记忆库**的引擎。

为什么要独立库（而不是复用 `MemoryStore` 的库）：
业务表（商品/库存/订单）与记忆表（消息/画像）的**生命周期完全不同** ——
业务数据要长期保全、要做备份与只读副本；记忆数据可以随时清理。
放在同一个库里，任何一次「清记忆」都可能误伤订单，测试也无法真隔离。

并发要点（面试必问「SQLite 下并发写会不会炸」）：
- `busy_timeout`：SQLite 默认遇到写锁**立刻**返回 "database is locked"，
  设成等待 30s 之后，并发写会排队而不是失败 —— 这是并发抢购测试能稳定复现的前提；
- `journal_mode=WAL`：读写不再互相阻塞，读请求不会被正在写的请求挡住；
- 这两个设置只对 SQLite 生效，PostgreSQL 下天然是行级锁，无需此配置。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from .models import Base


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


class CommerceDatabase:
    """业务库句柄。写法与 `infra/memory/store.py` 一致，但**不共享其引擎**。"""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, url: str) -> "CommerceDatabase":
        connect_args: dict[str, Any] = {"timeout": 30} if _is_sqlite(url) else {}
        engine = create_async_engine(url, pool_pre_ping=True, connect_args=connect_args)
        if _is_sqlite(url):
            cls._install_sqlite_pragmas(engine)
        return cls(engine)

    @staticmethod
    def _install_sqlite_pragmas(engine: AsyncEngine) -> None:
        @event.listens_for(engine.sync_engine, "connect")
        def _pragmas(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover - 由驱动回调
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
            # 关掉 pysqlite 的隐式事务管理，改由下面的 begin 钩子显式接管。
            # 不做这一步，SQLite 会在「先读后写」的事务里出现 SHARED→RESERVED 的
            # 锁升级死锁：两个并发事务都持有读锁时，谁都无法升级成写锁，
            # 直接返回 "database is locked"，busy_timeout 也救不了。
            dbapi_connection.isolation_level = None

        @event.listens_for(engine.sync_engine, "begin")
        def _begin_immediate(conn: Any) -> None:  # pragma: no cover - 由引擎回调
            # BEGIN IMMEDIATE 在**事务开始时就取写锁**，并发写因此严格串行排队，
            # 而不是「读到一半才发现要写」。这样并发抢购在 SQLite 下也能稳定复现，
            # 与 PostgreSQL 的行锁语义保持一致。
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    def session(self) -> AsyncSession:
        return self._sessionmaker()

    async def init_schema(self) -> None:
        """建表。沿用 create_all（M1 不做 alembic 迁移）—— 见方案第 10 节减法清单。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()


__all__ = ["CommerceDatabase"]
