"""优惠券用例：领券 / 我的券 / 券模板管理。

一个关键并发点（面试可讲）：**「每人限领 N 张」与「总量有限」是两种不同的约束**，
必须分别判定：
- 每人限领：领之前 `count_by_member` 判断，属于**软约束**，并发下可能短暂超领，
  因此要在同一事务里配合总量兜底；
- 总量：`bump_issued` 是一次条件 UPDATE（`issued < total`），rowcount=0 就说明被抢光，
  这是**硬约束**，也是唯一能挡住并发的闸门。

为什么不在领域层做总量判断：`assert_receivable(owned)` 只能看到「已领数量」这个快照，
而真实的总量控制必须落在数据库的单语句上，否则两个请求都会读到 `issued=99` 然后各自 +1。
"""

from __future__ import annotations

from src.infra.core.errors import NotFoundError
from src.infra.core.logging import get_logger

from ...domain.errors import CouponNotApplicableError
from ...domain.ids import new_id
from ...domain.models import Coupon, CouponKind, CouponTemplate
from ...domain.values import Money
from ..ports import UowFactory

logger = get_logger("commerce.usecase.coupon")


class CouponUseCase:
    def __init__(self, uow_factory: UowFactory) -> None:
        self._uow = uow_factory

    async def list_templates(self) -> list[CouponTemplate]:
        async with self._uow() as uow:
            templates = await uow.coupons.list_templates()
            await uow.commit()
        return templates

    async def receive(self, member_id: str, template_id: str) -> Coupon:
        async with self._uow() as uow:
            template = await uow.coupons.get_template(template_id)
            if template is None:
                raise NotFoundError(f"券模板不存在: {template_id}", detail={"template_id": template_id})
            owned = await uow.coupons.count_by_member(member_id, template_id)
            template.assert_receivable(owned)  # 软约束：每人限领 / 有效期
            if await uow.coupons.bump_issued(template_id) != 1:
                # 硬约束：并发下总量被抢光，只有条件 UPDATE 的 rowcount 能判定
                raise CouponNotApplicableError("券已领完", detail={"template_id": template_id})
            coupon = Coupon(coupon_id=new_id("cpn"), template_id=template_id, member_id=member_id)
            await uow.coupons.issue(coupon)
            await uow.commit()
        logger.info(
            "coupon.received",
            extra={"event": "coupon", "status": "ok", "member_id": member_id, "template_id": template_id},
        )
        return coupon

    async def mine(self, member_id: str) -> list[Coupon]:
        async with self._uow() as uow:
            coupons = await uow.coupons.list_by_member(member_id)
            await uow.commit()
        return coupons

    async def create_template(
        self,
        *,
        name: str,
        kind: str = CouponKind.FULL_REDUCTION.value,
        threshold_usd: float,
        value_cents: int,
        total: int = 0,
        per_member_limit: int = 1,
        stackable: bool = False,
    ) -> CouponTemplate:
        template = CouponTemplate(
            template_id=new_id("ct"),
            name=name,
            kind=CouponKind(kind),
            threshold=Money.of(threshold_usd),
            value_cents=int(value_cents),
            total=int(total),
            per_member_limit=int(per_member_limit),
            stackable=stackable,
        )
        async with self._uow() as uow:
            await uow.coupons.save_template(template)
            await uow.commit()
        return template


__all__ = ["CouponUseCase"]
