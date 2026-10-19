"""订单仓储。所有状态变更都走 **CAS（条件 UPDATE）**，绝不做「先读状态、再写状态」。

为什么：一个超时 sweeper 与一个用户付款请求可能同时到达。
如果两者都「读到 PENDING_PAYMENT → 各自写一个新状态」，后写的会覆盖先写的，
出现「已付款订单被取消」这类事故。条件里带上期望的旧状态，
数据库就替我们做了原子判定 —— `rowcount == 0` 意味着「有人先改了」，直接放弃即可。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.models import Order, OrderItem, OrderStatus
from ...domain.values import Money
from ..db.models import OrderItemRecord, OrderRecord


class SqlOrderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _load(self, record: OrderRecord) -> Order:
        rows = (
            (
                await self._session.execute(
                    select(OrderItemRecord)
                    .where(OrderItemRecord.order_id == record.order_id)
                    .order_by(OrderItemRecord.order_item_id)
                )
            )
            .scalars()
            .all()
        )
        return Order(
            order_id=record.order_id,
            member_id=record.member_id,
            items=[
                OrderItem(
                    sku_id=row.sku_id,
                    product_id=row.product_id,
                    name=row.name,
                    quantity=row.quantity,
                    unit_price=Money(row.unit_price_cents, row.currency),
                )
                for row in rows
            ],
            status=OrderStatus(record.status),
            coupon_id=record.coupon_id,
            discount=Money(record.discount_cents, record.currency),
            carrier=record.carrier,
            created_at=record.created_at,
            expires_at=record.expires_at,
            paid_at=record.paid_at,
            cancelled_at=record.cancelled_at,
            cancel_reason=record.cancel_reason,
            idempotency_key=record.idempotency_key,
        )

    async def save(self, order: Order, *, project_id: str = "shop") -> None:
        self._session.add(
            OrderRecord(
                order_id=order.order_id,
                project_id=project_id,
                member_id=order.member_id,
                status=order.status.value,
                coupon_id=order.coupon_id,
                goods_cents=order.goods_total().cents,
                discount_cents=order.discount.cents,
                payable_cents=order.payable().cents,
                currency=order.goods_total().currency,
                carrier=order.carrier,
                idempotency_key=order.idempotency_key,
                created_at=order.created_at,
                expires_at=order.expires_at,
                paid_at=order.paid_at,
                cancelled_at=order.cancelled_at,
                cancel_reason=order.cancel_reason,
            )
        )
        for item in order.items:
            self._session.add(
                OrderItemRecord(
                    order_item_id=f"oi_{order.order_id}_{item.sku_id}",
                    order_id=order.order_id,
                    sku_id=item.sku_id,
                    product_id=item.product_id,
                    name=item.name,
                    quantity=item.quantity,
                    unit_price_cents=item.unit_price.cents,
                    currency=item.unit_price.currency,
                )
            )
        await self._session.flush()

    async def get(self, order_id: str) -> Order | None:
        record = (
            await self._session.execute(select(OrderRecord).where(OrderRecord.order_id == order_id))
        ).scalar_one_or_none()
        return await self._load(record) if record else None

    async def update_status(self, order_id: str, status: str, expected: str | None = None, **fields: Any) -> int:
        conditions = [OrderRecord.order_id == order_id]
        if expected is not None:
            conditions.append(OrderRecord.status == expected)
        result = await self._session.execute(
            update(OrderRecord).where(*conditions).values(status=status, **fields).execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)

    async def list_by_member(self, member_id: str, *, offset: int = 0, limit: int = 20) -> list[Order]:
        records = (
            (
                await self._session.execute(
                    select(OrderRecord)
                    .where(OrderRecord.member_id == member_id)
                    .order_by(OrderRecord.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return [await self._load(record) for record in records]

    async def list_all(self, *, status: str | None = None, offset: int = 0, limit: int = 20) -> list[Order]:
        """运营侧跨会员订单列表。`status` 走**字面值**过滤（与 DB 存储一致），不做标签翻译。"""
        stmt = select(OrderRecord)
        if status:
            stmt = stmt.where(OrderRecord.status == status)
        records = (
            (await self._session.execute(stmt.order_by(OrderRecord.created_at.desc()).offset(offset).limit(limit)))
            .scalars()
            .all()
        )
        return [await self._load(record) for record in records]

    async def cancel_expired(self, now: datetime, *, limit: int = 200) -> list[Order]:
        """超时未付款订单：先选候选，再**逐单 CAS** 抢占取消权。

        只返回真正被本次抢占到的订单（rowcount==1），
        库存回滚由调用方按这批结果执行 —— 这样并发跑两个 sweeper 也不会重复回滚。
        """
        candidates = (
            (
                await self._session.execute(
                    select(OrderRecord.order_id)
                    .where(
                        OrderRecord.status == OrderStatus.PENDING_PAYMENT.value,
                        OrderRecord.expires_at.is_not(None),
                        OrderRecord.expires_at < now,
                    )
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        cancelled: list[Order] = []
        for order_id in candidates:
            matched = await self.update_status(
                order_id,
                OrderStatus.CANCELLED.value,
                expected=OrderStatus.PENDING_PAYMENT.value,
                cancelled_at=now,
                cancel_reason="timeout",
            )
            if matched == 1:
                order = await self.get(order_id)
                if order is not None:
                    cancelled.append(order)
        return cancelled

    async def stats(self, *, since: datetime | None = None) -> dict[str, Any]:
        """报表口径：GMV 只统计**未取消**订单的应付金额，取消单不计入营收。"""
        base = select(
            func.count(OrderRecord.order_id),
            func.coalesce(func.sum(OrderRecord.payable_cents), 0),
        ).where(OrderRecord.status != OrderStatus.CANCELLED.value)
        if since is not None:
            base = base.where(OrderRecord.created_at >= since)
        count, gmv = (await self._session.execute(base)).one()

        status_rows = (
            await self._session.execute(
                select(OrderRecord.status, func.count(OrderRecord.order_id)).group_by(OrderRecord.status)
            )
        ).all()
        return {
            "orders": int(count or 0),
            "gmv_cents": int(gmv or 0),
            "by_status": {row[0]: int(row[1]) for row in status_rows},
            "since": since.isoformat() if since else None,
        }


__all__ = ["SqlOrderRepository"]
