"""无状态领域服务：当规则「不属于某一个实体」时，把它放在服务里。

判别标准（面试可讲）：如果一个规则需要同时读多个聚合、或者本质上是一段纯粹的计算，
它就不该塞进任何单个聚合 —— 塞进去会让聚合承担它不需要知道的协作知识。
反过来，只要规则能被单个聚合完整表达（订单状态机），就绝不上提到服务层。

三个服务：
- `PricingPolicy`：会员折扣 + 券折扣 + 应付金额，纯函数式，可脱库单测；
- `CouponRule`：券的门槛/叠加/时效校验（分工明确：`PricingPolicy` 算钱，`CouponRule` 判能不能用）；
- `StockAllocator`：把「色号 + 规格」的选择解析成真实 SKU。M1 只做解析与展示，
  不做库存池分配算法（那是 M2 秒杀系统的事）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .errors import CouponNotApplicableError, DomainValidationError, ProductNotSellableError
from .models import Coupon, CouponKind, CouponTemplate, Member, MemberLevel, Product, utcnow
from .values import Money


@dataclass(frozen=True, slots=True)
class PriceBreakdown:
    """价格拆解。把每一步都显式列出来，而不是只给一个最终价 ——
    用户投诉「为什么是这个价」时，客服需要能逐项解释。"""

    goods_total: Money
    member_discount: Money
    coupon_discount: Money
    payable: Money
    member_level: MemberLevel

    def to_dict(self) -> dict[str, object]:
        return {
            "goods_total": self.goods_total.to_dict(),
            "member_discount": self.member_discount.to_dict(),
            "coupon_discount": self.coupon_discount.to_dict(),
            "payable": self.payable.to_dict(),
            "member_level": self.member_level.value,
        }


class PricingPolicy:
    """定价策略。

    顺序很重要：**先算会员折扣，再在折后金额上判券门槛**。
    反过来（先判券门槛再打折）会出现「券门槛按原价算、实付却更低」的漏洞，
    这是电商里非常经典的价格套利口子。
    """

    def quote(
        self,
        *,
        goods_total: Money,
        member: Member,
        coupon: Coupon | None = None,
        template: CouponTemplate | None = None,
    ) -> PriceBreakdown:
        rate = member.level.discount_rate
        member_discount = Money(int(round(goods_total.cents * rate)), goods_total.currency)
        after_member = (goods_total - member_discount).clamp_at_zero()

        coupon_discount = Money(0, goods_total.currency)
        if coupon is not None and template is not None:
            CouponRule().assert_applicable(coupon=coupon, template=template, amount=after_member)
            coupon_discount = CouponRule().discount_for(template=template, amount=after_member)

        payable = (after_member - coupon_discount).clamp_at_zero()
        return PriceBreakdown(
            goods_total=goods_total,
            member_discount=member_discount,
            coupon_discount=coupon_discount,
            payable=payable,
            member_level=member.level,
        )


class CouponRule:
    """券规则。只回答「能不能用」与「减多少」，不负责收款。"""

    def assert_applicable(
        self,
        *,
        coupon: Coupon,
        template: CouponTemplate,
        amount: Money,
        now: datetime | None = None,
    ) -> None:
        coupon.assert_usable()
        moment = now or utcnow()
        if template.valid_to and moment > template.valid_to:
            raise CouponNotApplicableError("券已过期", detail={"coupon_id": coupon.coupon_id})
        if amount < template.threshold:
            raise CouponNotApplicableError(
                f"未满 {template.threshold.amount} 不能使用该券",
                detail={"coupon_id": coupon.coupon_id, "threshold": str(template.threshold.amount)},
            )

    def discount_for(self, *, template: CouponTemplate, amount: Money) -> Money:
        if template.kind is CouponKind.FULL_REDUCTION:
            return Money(template.value_cents, amount.currency)
        # 百分比券：value_cents 表示「折扣基点」（1000 = 9 折），避免用浮点表达折扣率
        cut = Money(int(round(amount.cents * template.value_cents / 10000)), amount.currency)
        return cut.clamp_at_zero()

    def combine(self, *, first: Money, second: Money, stackable: bool) -> Money:
        """券叠加。不可叠加时只取优惠更大的一张 ——
        静默丢弃一张会让用户觉得「我的券被吞了」，所以要显式返回并解释。"""
        if not stackable:
            return first if first.cents >= second.cents else second
        return first + second


class StockAllocator:
    """把维度选择解析成真实 SKU。

    M1 刻意只做**解析与展示**：不做「多仓库存池分配」「就近仓」这类算法，
    它们属于履约系统，塞进来会让商品域承担它不该有的复杂度。
    """

    def resolve(self, product: Product, choice: dict[str, str] | None) -> str:
        product.assert_sellable()
        if product.matrix is None:
            if not product.skus:
                raise ProductNotSellableError("商品没有可售规格", detail={"product_id": product.product_id})
            return product.skus[0].sku_id
        if not choice:
            raise DomainValidationError("多规格商品必须指定色号与规格", detail={"axes": product.matrix.axis_names()})
        if not product.matrix.is_sellable(choice):
            raise DomainValidationError(
                f"该组合暂不售卖: {choice}", detail={"choice": choice, "product_id": product.product_id}
            )
        target = tuple(choice[axis] for axis in product.matrix.axes)
        for sku in product.skus:
            if (sku.shade, sku.spec) == target:
                return sku.sku_id
        raise ProductNotSellableError("可售组合没有对应的 SKU", detail={"choice": choice})


__all__ = ["CouponRule", "PriceBreakdown", "PricingPolicy", "StockAllocator"]
