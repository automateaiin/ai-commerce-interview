"""领域聚合：Product / Sku / Member / Cart / Order / Coupon 等。

聚合的设计边界（面试必问「为什么这么切」）：**一致性边界**。
- 订单是聚合根：`OrderItem` 只存在于订单内部，绝不会被单独查询或修改，
  因此它们一起被一个事务保护；
- SKU 的**库存**不是订单聚合的一部分：库存会被多个订单并发争抢，
  若把它塞进订单聚合，就得在应用层加锁，反而放大争用。所以库存单独成
  `Stock`（见 infrastructure 用条件 UPDATE 原子扣减）。
- 优惠券是独立聚合：券的领取/核销与下单是不同生命周期，强行合并会让券模板改动
  牵动订单模型。

状态机写在聚合内部（`transition_to` / `publish`），而不是散在用例里：
非法迁移必须**在领域层就被拒绝**，否则并发与多入口迟早绕过校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .errors import (
    CouponNotApplicableError,
    DomainValidationError,
    InvalidStateTransitionError,
    ProductNotSellableError,
)
from .security import verify_password
from .values import BatchNo, InciIngredient, Money, SkuMatrix, TrialSample


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# 枚举
# --------------------------------------------------------------------------- #
class ProductStatus(str, Enum):
    DRAFT = "DRAFT"
    ON_SALE = "ON_SALE"
    OFF_SHELF = "OFF_SHELF"

    @property
    def label(self) -> str:
        return {"DRAFT": "草稿", "ON_SALE": "在售", "OFF_SHELF": "已下架"}[self.value]


class OrderStatus(str, Enum):
    PENDING_PAYMENT = "PENDING_PAYMENT"
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

    @property
    def label(self) -> str:
        return {
            "PENDING_PAYMENT": "待付款",
            "PAID": "已付款",
            "SHIPPED": "已发货",
            "COMPLETED": "已完成",
            "CANCELLED": "已取消",
        }[self.value]

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.COMPLETED, OrderStatus.CANCELLED)


# 合法的状态迁移表。用表驱动而不是 if 链：
# 一眼能看出「哪些迁移合法」，新增状态时也不会漏改某个分支。
_ORDER_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING_PAYMENT: frozenset({OrderStatus.PAID, OrderStatus.CANCELLED}),
    OrderStatus.PAID: frozenset({OrderStatus.SHIPPED, OrderStatus.CANCELLED}),
    OrderStatus.SHIPPED: frozenset({OrderStatus.COMPLETED}),
    OrderStatus.COMPLETED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
}

_PRODUCT_TRANSITIONS: dict[ProductStatus, frozenset[ProductStatus]] = {
    ProductStatus.DRAFT: frozenset({ProductStatus.ON_SALE}),
    ProductStatus.ON_SALE: frozenset({ProductStatus.OFF_SHELF}),
    ProductStatus.OFF_SHELF: frozenset({ProductStatus.ON_SALE}),
}


class MemberLevel(str, Enum):
    NORMAL = "NORMAL"
    SILVER = "SILVER"
    GOLD = "GOLD"

    @property
    def discount_rate(self) -> float:
        """会员价折扣率（0 表示无折扣）。规则属于领域知识，不放在用例里。"""
        return {"NORMAL": 0.0, "SILVER": 0.03, "GOLD": 0.06}[self.value]


class CouponKind(str, Enum):
    FULL_REDUCTION = "FULL_REDUCTION"
    PERCENT = "PERCENT"


class CouponStatus(str, Enum):
    RECEIVED = "RECEIVED"
    USED = "USED"
    EXPIRED = "EXPIRED"


# --------------------------------------------------------------------------- #
# 商品域
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Sku:
    """最小可售单元。库存单独成 Stock 聚合，这里只保留静态属性。"""

    sku_id: str
    product_id: str
    shade: str
    spec: str
    price: Money
    barcode: str = ""
    trial: TrialSample | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "product_id": self.product_id,
            "shade": self.shade,
            "spec": self.spec,
            "price": self.price.to_dict(),
        }


@dataclass(slots=True)
class Product:
    product_id: str
    name: str
    brand: str
    category: str
    market: str
    price: Money
    skin_types: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    usage: str = ""
    status: ProductStatus = ProductStatus.DRAFT
    matrix: SkuMatrix | None = None
    skus: list[Sku] = field(default_factory=list)
    inci: list[InciIngredient] = field(default_factory=list)
    batches: list[BatchNo] = field(default_factory=list)
    version: int = 1

    # -- 状态机 ------------------------------------------------------------- #
    def transition_to(self, target: ProductStatus) -> None:
        if target not in _PRODUCT_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"商品状态不允许 {self.status.value} → {target.value}",
                detail={"product_id": self.product_id, "from": self.status.value, "to": target.value},
            )
        if target is ProductStatus.ON_SALE:
            # 上架前的前置条件在这里统一校验：没有价格或没有可售组合的商品不该出现在货架
            if self.price.cents <= 0:
                raise DomainValidationError("未定价的商品不能上架", detail={"product_id": self.product_id})
            if self.matrix is not None and not self.matrix.combinations():
                raise DomainValidationError("没有可售组合的商品不能上架", detail={"product_id": self.product_id})
        self.status = target

    def assert_sellable(self) -> None:
        if self.status is not ProductStatus.ON_SALE:
            raise ProductNotSellableError(
                f"商品当前不可售（{self.status.label}）",
                detail={"product_id": self.product_id, "status": self.status.value},
            )

    def inci_normalized(self) -> list[InciIngredient]:
        from .values import normalize_inci

        return normalize_inci(self.inci)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "market": self.market,
            "price": self.price.to_dict(),
            "skin_types": list(self.skin_types),
            "highlights": list(self.highlights),
            "usage": self.usage,
            "status": self.status.value,
            "status_label": self.status.label,
            "skus": [sku.to_dict() for sku in self.skus],
            "inci": [item.to_dict() for item in self.inci_normalized()],
            "batches": [batch.to_dict() for batch in self.batches],
            "matrix": (
                {"axes": {k: list(v) for k, v in self.matrix.axes.items()}, "sellable": len(self.matrix.sellable)}
                if self.matrix
                else None
            ),
            "version": self.version,
        }


@dataclass(slots=True)
class Stock:
    """库存聚合。`available` 是可售量，`reserved` 是已下单未付款的占用。"""

    sku_id: str
    available: int
    reserved: int = 0
    version: int = 0
    threshold: int = 5

    @property
    def sellable(self) -> int:
        return self.available

    def is_low(self) -> bool:
        return self.available <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "available": self.available,
            "reserved": self.reserved,
            "version": self.version,
            "low": self.is_low(),
        }


# --------------------------------------------------------------------------- #
# 会员域
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Address:
    address_id: str
    member_id: str
    receiver: str
    phone: str
    country: str
    city: str
    line1: str
    is_default: bool = False

    def masked(self) -> dict[str, Any]:
        """对外展示时手机号必须打码：日志与响应里都不该出现完整手机号。"""
        tail = self.phone[-4:] if len(self.phone) >= 4 else "****"
        return {
            "address_id": self.address_id,
            "receiver": self.receiver,
            "phone_masked": f"****{tail}",
            "country": self.country,
            "city": self.city,
            "line1": self.line1,
            "is_default": self.is_default,
        }


@dataclass(slots=True)
class Member:
    member_id: str
    email: str
    nickname: str = ""
    level: MemberLevel = MemberLevel.NORMAL
    salt_hex: str = ""
    hash_hex: str = ""
    created_at: datetime = field(default_factory=utcnow)
    addresses: list[Address] = field(default_factory=list)

    def verify_password(self, password: str) -> bool:
        return verify_password(password, salt_hex=self.salt_hex, hash_hex=self.hash_hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "email": self.email,
            "nickname": self.nickname,
            "level": self.level.value,
            "discount_rate": self.level.discount_rate,
            "addresses": [address.masked() for address in self.addresses],
        }


# --------------------------------------------------------------------------- #
# 购物车域
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CartItem:
    sku_id: str
    product_id: str
    quantity: int
    price_snapshot: Money
    added_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "price_snapshot": self.price_snapshot.to_dict(),
            "subtotal": (self.price_snapshot * self.quantity).to_dict(),
            "added_at": self.added_at.isoformat(),
        }


@dataclass(slots=True)
class Cart:
    """购物车聚合。同一个 SKU 只能有一行——重复加购必须**合并数量**而不是堆行。"""

    cart_id: str
    member_id: str
    items: list[CartItem] = field(default_factory=list)
    updated_at: datetime = field(default_factory=utcnow)

    def find(self, sku_id: str) -> CartItem | None:
        return next((item for item in self.items if item.sku_id == sku_id), None)

    def add(self, item: CartItem) -> None:
        existing = self.find(item.sku_id)
        if existing is None:
            self.items.append(item)
        else:
            existing.quantity += item.quantity
            # 价格快照必须刷新为最新价：否则用户看到的是历史价，结算时金额对不上
            existing.price_snapshot = item.price_snapshot
        self.updated_at = utcnow()

    def remove(self, sku_id: str) -> None:
        self.items = [item for item in self.items if item.sku_id != sku_id]
        self.updated_at = utcnow()

    def clear(self) -> None:
        self.items = []
        self.updated_at = utcnow()

    def line_total(self) -> Money:
        total = Money(0)
        for item in self.items:
            total = total + item.price_snapshot * item.quantity
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "cart_id": self.cart_id,
            "member_id": self.member_id,
            "items": [item.to_dict() for item in self.items],
            "item_count": len(self.items),
            "total": self.line_total().to_dict(),
            "updated_at": self.updated_at.isoformat(),
        }


# --------------------------------------------------------------------------- #
# 订单域
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class OrderItem:
    sku_id: str
    product_id: str
    name: str
    quantity: int
    unit_price: Money

    def subtotal(self) -> Money:
        return self.unit_price * self.quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "product_id": self.product_id,
            "name": self.name,
            "quantity": self.quantity,
            "unit_price": self.unit_price.to_dict(),
            "subtotal": self.subtotal().to_dict(),
        }


@dataclass(slots=True)
class Order:
    order_id: str
    member_id: str
    items: list[OrderItem]
    status: OrderStatus = OrderStatus.PENDING_PAYMENT
    coupon_id: str | None = None
    discount: Money = field(default_factory=lambda: Money(0))
    carrier: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    expires_at: datetime | None = None
    paid_at: datetime | None = None
    cancelled_at: datetime | None = None
    cancel_reason: str | None = None
    idempotency_key: str | None = None

    def goods_total(self) -> Money:
        total = Money(0)
        for item in self.items:
            total = total + item.subtotal()
        return total

    def payable(self) -> Money:
        return (self.goods_total() - self.discount).clamp_at_zero()

    def transition_to(self, target: OrderStatus) -> None:
        if target not in _ORDER_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"订单状态不允许 {self.status.value} → {target.value}",
                detail={"order_id": self.order_id, "from": self.status.value, "to": target.value},
            )
        self.status = target

    def assert_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or utcnow()) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "member_id": self.member_id,
            "status": self.status.value,
            "status_label": self.status.label,
            "items": [item.to_dict() for item in self.items],
            "goods_total": self.goods_total().to_dict(),
            "discount": self.discount.to_dict(),
            "payable": self.payable().to_dict(),
            "coupon_id": self.coupon_id,
            "carrier": self.carrier,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "cancel_reason": self.cancel_reason,
        }


# --------------------------------------------------------------------------- #
# 营销域（优惠券）
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CouponTemplate:
    template_id: str
    name: str
    kind: CouponKind
    threshold: Money
    value_cents: int
    total: int = 0
    per_member_limit: int = 1
    stackable: bool = False
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    issued: int = 0

    def assert_receivable(self, owned: int, now: datetime | None = None) -> None:
        moment = now or utcnow()
        if self.total and self.issued >= self.total:
            raise CouponNotApplicableError("券已领完", detail={"template_id": self.template_id})
        if owned >= self.per_member_limit:
            raise CouponNotApplicableError(
                f"每人限领 {self.per_member_limit} 张", detail={"template_id": self.template_id}
            )
        if self.valid_from and moment < self.valid_from:
            raise CouponNotApplicableError("活动尚未开始", detail={"template_id": self.template_id})
        if self.valid_to and moment > self.valid_to:
            raise CouponNotApplicableError("活动已结束", detail={"template_id": self.template_id})

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "kind": self.kind.value,
            "threshold": self.threshold.to_dict(),
            "value_cents": self.value_cents,
            "total": self.total,
            "issued": self.issued,
            "per_member_limit": self.per_member_limit,
            "stackable": self.stackable,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }


@dataclass(slots=True)
class Coupon:
    coupon_id: str
    template_id: str
    member_id: str
    status: CouponStatus = CouponStatus.RECEIVED
    received_at: datetime = field(default_factory=utcnow)
    used_at: datetime | None = None
    order_id: str | None = None

    def assert_usable(self) -> None:
        if self.status is not CouponStatus.RECEIVED:
            raise CouponNotApplicableError(
                f"券不可用（{self.status.value}）", detail={"coupon_id": self.coupon_id, "status": self.status.value}
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coupon_id": self.coupon_id,
            "template_id": self.template_id,
            "member_id": self.member_id,
            "status": self.status.value,
            "received_at": self.received_at.isoformat(),
            "used_at": self.used_at.isoformat() if self.used_at else None,
            "order_id": self.order_id,
        }


__all__ = [
    "Address",
    "Cart",
    "CartItem",
    "Coupon",
    "CouponKind",
    "CouponStatus",
    "CouponTemplate",
    "Member",
    "MemberLevel",
    "Order",
    "OrderItem",
    "OrderStatus",
    "Product",
    "ProductStatus",
    "Sku",
    "Stock",
    "utcnow",
]
