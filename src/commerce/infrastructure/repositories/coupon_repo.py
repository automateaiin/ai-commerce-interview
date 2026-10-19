"""优惠券仓储（模板 + 已领券）。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.models import Coupon, CouponKind, CouponStatus, CouponTemplate, utcnow
from ...domain.values import Money
from ..db.models import CouponRecord, CouponTemplateRecord


class SqlCouponRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- 模板 --------------------------------------------------------------- #
    @staticmethod
    def _template(row: CouponTemplateRecord) -> CouponTemplate:
        return CouponTemplate(
            template_id=row.template_id,
            name=row.name,
            kind=CouponKind(row.kind),
            threshold=Money(row.threshold_cents),
            value_cents=row.value_cents,
            total=row.total,
            per_member_limit=row.per_member_limit,
            stackable=row.stackable,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            issued=row.issued,
        )

    async def get_template(self, template_id: str) -> CouponTemplate | None:
        row = (
            await self._session.execute(
                select(CouponTemplateRecord).where(CouponTemplateRecord.template_id == template_id)
            )
        ).scalar_one_or_none()
        return self._template(row) if row else None

    async def list_templates(self) -> list[CouponTemplate]:
        rows = (await self._session.execute(select(CouponTemplateRecord))).scalars().all()
        return [self._template(row) for row in rows]

    async def save_template(self, template: CouponTemplate) -> None:
        row = (
            await self._session.execute(
                select(CouponTemplateRecord).where(CouponTemplateRecord.template_id == template.template_id)
            )
        ).scalar_one_or_none()
        if row is None:
            row = CouponTemplateRecord(template_id=template.template_id)
            self._session.add(row)
        row.name = template.name
        row.kind = template.kind.value
        row.threshold_cents = template.threshold.cents
        row.value_cents = template.value_cents
        row.total = template.total
        row.per_member_limit = template.per_member_limit
        row.stackable = template.stackable
        row.valid_from = template.valid_from
        row.valid_to = template.valid_to
        await self._session.flush()

    async def bump_issued(self, template_id: str) -> int:
        """领券计数用条件 UPDATE：`issued < total` 是**发券上限的并发闸门**。

        先查后改会让两个并发领取各自以为还有余量，超出总发放量。
        """
        result = await self._session.execute(
            update(CouponTemplateRecord)
            .where(
                CouponTemplateRecord.template_id == template_id,
                (CouponTemplateRecord.total == 0) | (CouponTemplateRecord.issued < CouponTemplateRecord.total),
            )
            .values(issued=CouponTemplateRecord.issued + 1)
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)

    # -- 券 ------------------------------------------------------------------ #
    @staticmethod
    def _coupon(row: CouponRecord) -> Coupon:
        return Coupon(
            coupon_id=row.coupon_id,
            template_id=row.template_id,
            member_id=row.member_id,
            status=CouponStatus(row.status),
            received_at=row.received_at,
            used_at=row.used_at,
            order_id=row.order_id,
        )

    async def issue(self, coupon: Coupon) -> None:
        self._session.add(
            CouponRecord(
                coupon_id=coupon.coupon_id,
                template_id=coupon.template_id,
                member_id=coupon.member_id,
                status=coupon.status.value,
                received_at=coupon.received_at,
            )
        )
        await self._session.flush()

    async def get(self, coupon_id: str) -> Coupon | None:
        row = (
            await self._session.execute(select(CouponRecord).where(CouponRecord.coupon_id == coupon_id))
        ).scalar_one_or_none()
        return self._coupon(row) if row else None

    async def list_by_member(self, member_id: str) -> list[Coupon]:
        rows = (
            (
                await self._session.execute(
                    select(CouponRecord)
                    .where(CouponRecord.member_id == member_id)
                    .order_by(CouponRecord.received_at.desc())
                )
            )
            .scalars()
            .all()
        )
        return [self._coupon(row) for row in rows]

    async def count_by_member(self, member_id: str, template_id: str) -> int:
        return int(
            (
                await self._session.execute(
                    select(func.count()).select_from(CouponRecord).where(
                        CouponRecord.member_id == member_id, CouponRecord.template_id == template_id
                    )
                )
            ).scalar_one()
        )

    async def mark_used(self, coupon_id: str, order_id: str, *, now: datetime | None = None) -> int:
        result = await self._session.execute(
            update(CouponRecord)
            .where(CouponRecord.coupon_id == coupon_id, CouponRecord.status == CouponStatus.RECEIVED.value)
            .values(status=CouponStatus.USED.value, order_id=order_id, used_at=now or utcnow())
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)

    async def mark_released(self, coupon_id: str) -> int:
        """订单取消 → 券退回。条件里限定 USED，避免把已过期的券又变成可用。"""
        result = await self._session.execute(
            update(CouponRecord)
            .where(CouponRecord.coupon_id == coupon_id, CouponRecord.status == CouponStatus.USED.value)
            .values(status=CouponStatus.RECEIVED.value, order_id=None, used_at=None)
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)


__all__ = ["SqlCouponRepository"]
