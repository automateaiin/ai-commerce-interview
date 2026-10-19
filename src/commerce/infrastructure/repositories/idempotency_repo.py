"""幂等仓储。

**并发语义在这里定死**：
`save` 直接 INSERT，撞唯一键（`uq_commerce_idem_project_key`）会抛 `IntegrityError`。
调用方捕获后去 `get` 已存响应 —— 于是「同一幂等键的两个并发请求只会有一个真正下单」，
这个保证来自**数据库唯一约束**，而不是应用层的查询-判断（后者在并发下必然漏）。

`request_hash` 的作用是防「同键不同体」：拿一个旧键发一笔新订单，
如果不校验请求体哈希，会返回上一笔的订单给用户 —— 那是典型的资金级事故。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.idempotency import hash_request
from ..db.models import IdempotencyRecord
from .product_repo import new_id


class SqlIdempotencyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, project_id: str, key: str) -> dict[str, object] | None:
        row = (
            await self._session.execute(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.project_id == project_id, IdempotencyRecord.idempotency_key == key
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return {"request_hash": row.request_hash, "response": dict(row.response or {})}

    async def save(self, *, project_id: str, key: str, request_hash: str, response: dict[str, object]) -> None:
        self._session.add(
            IdempotencyRecord(
                record_id=new_id("idem"),
                project_id=project_id,
                idempotency_key=key,
                request_hash=request_hash,
                response=dict(response),
            )
        )
        await self._session.flush()


__all__ = ["SqlIdempotencyRepository", "hash_request"]
