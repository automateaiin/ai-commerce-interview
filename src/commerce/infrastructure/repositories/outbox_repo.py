"""发件箱（Outbox）仓储。

存在的意义：**「业务写库」与「事件投递」之间不能有空档**。
若采用「先提交订单事务，再 publish 事件」的写法，进程在这两步之间崩溃，
事件就永久丢了（订单有了，推荐/知识抽取永远不触发）。
把事件行与订单放进**同一个事务**插入，再由消费者异步补发，
就得到了「至少一次投递」的保证；消费端用 `event_id` 去重即可收敛为「恰好一次」。
"""

from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import OutboxRecord, utcnow


class SqlOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(self, *, topic: str, payload: dict[str, Any], event_id: str) -> None:
        self._session.add(OutboxRecord(event_id=event_id, topic=topic, payload=dict(payload)))
        await self._session.flush()

    async def pending(self, limit: int = 50) -> Sequence[dict[str, Any]]:
        rows = (
            (
                await self._session.execute(
                    select(OutboxRecord)
                    .where(OutboxRecord.published_at.is_(None))
                    .order_by(OutboxRecord.created_at)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return [
            {"event_id": row.event_id, "topic": row.topic, "payload": dict(row.payload or {})} for row in rows
        ]

    async def mark_published(self, event_id: str) -> None:
        row = (
            await self._session.execute(select(OutboxRecord).where(OutboxRecord.event_id == event_id))
        ).scalar_one_or_none()
        if row is not None:
            row.published_at = utcnow()
            await self._session.flush()


__all__ = ["SqlOutboxRepository"]
