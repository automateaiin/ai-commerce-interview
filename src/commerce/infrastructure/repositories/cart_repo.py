"""购物车仓储。

加购的「合并」语义落在**唯一键 + upsert**上，而不是「先查再决定插还是改」：
后者在两个并发加购之间会插出两行同 SKU（数据脏了，后面所有数量统计都错）。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.models import Cart, CartItem
from ...domain.values import Money
from ..db.models import CartItemRecord
from .product_repo import new_id


class SqlCartRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_member(self, member_id: str) -> Cart:
        rows = (
            (
                await self._session.execute(
                    select(CartItemRecord)
                    .where(CartItemRecord.member_id == member_id)
                    .order_by(CartItemRecord.added_at)
                )
            )
            .scalars()
            .all()
        )
        return Cart(
            cart_id=f"cart_{member_id}",
            member_id=member_id,
            items=[
                CartItem(
                    sku_id=row.sku_id,
                    product_id=row.product_id,
                    quantity=row.quantity,
                    price_snapshot=Money(row.price_cents, row.currency),
                    added_at=row.added_at,
                )
                for row in rows
            ],
        )

    async def save_item(
        self, member_id: str, sku_id: str, product_id: str, quantity: int, price_cents: int, currency: str = "USD"
    ) -> None:
        row = (
            await self._session.execute(
                select(CartItemRecord).where(
                    CartItemRecord.member_id == member_id, CartItemRecord.sku_id == sku_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            self._session.add(
                CartItemRecord(
                    cart_item_id=new_id("ci"),
                    member_id=member_id,
                    sku_id=sku_id,
                    product_id=product_id,
                    quantity=quantity,
                    price_cents=price_cents,
                    currency=currency,
                )
            )
        else:
            row.quantity += quantity
            # 价格快照按最新价刷新，避免用户按历史价结算；库存/价格随下单前校验兜底
            row.price_cents = price_cents
            row.currency = currency
        await self._session.flush()

    async def remove_item(self, member_id: str, sku_id: str) -> None:
        row = (
            await self._session.execute(
                select(CartItemRecord).where(
                    CartItemRecord.member_id == member_id, CartItemRecord.sku_id == sku_id
                )
            )
        ).scalar_one_or_none()
        if row is not None:
            await self._session.delete(row)
            await self._session.flush()

    async def clear(self, member_id: str) -> None:
        rows = (
            (await self._session.execute(select(CartItemRecord).where(CartItemRecord.member_id == member_id)))
            .scalars()
            .all()
        )
        for row in rows:
            await self._session.delete(row)
        await self._session.flush()


__all__ = ["SqlCartRepository"]
