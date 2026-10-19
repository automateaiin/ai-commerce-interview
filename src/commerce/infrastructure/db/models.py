"""业务库 ORM 表映射（PostgreSQL / SQLite 通用）。

命名统一带 `commerce_` 前缀：业务库与记忆库共用同一套表名空间时，
加前缀能让「这条 SQL 打的是哪个库」在日志里一眼可辨。

三处刻意的设计（面试可讲）：
1. **库存独立成表 `commerce_stock`**：见 domain/models.py 的说明 ——
   库存是并发争用的热点，必须能被单条条件 UPDATE 原子扣减；
2. **`commerce_idempotency` 有唯一键 `(project_id, idempotency_key)`**：
   幂等不是「查一下有没有」，而是**靠数据库唯一约束兜住并发**，
   否则两个并发重放会双双查不到、双双下单；
3. **`commerce_outbox` 与业务写在同一事务提交**：事件投递不能依赖「先提交再发消息」，
   否则进程在两步之间崩溃，事件就永久丢了。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------- #
# PMS：商品域
# --------------------------------------------------------------------------- #
class ProductRecord(Base):
    __tablename__ = "commerce_product"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    brand: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(16), default="US")
    price_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    skin_types: Mapped[list[str]] = mapped_column(JSON, default=list)
    highlights: Mapped[list[str]] = mapped_column(JSON, default=list)
    usage_text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="DRAFT")
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_commerce_product_category_status", "category", "status"),
        Index("ix_commerce_product_market_status", "market", "status"),
    )


class ShadeRecord(Base):
    """色号字典。色号是**共享的命名资源**（一季多个商品共用同一批色号名），
    因此独立成表而不是塞进 SKU 的字符串里，便于统一改名与排序。"""

    __tablename__ = "commerce_shade"

    shade_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(64))
    code: Mapped[str] = mapped_column(String(32), default="")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)


class SkuRecord(Base):
    __tablename__ = "commerce_sku"

    sku_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), index=True)
    shade: Mapped[str] = mapped_column(String(64), default="")
    spec: Mapped[str] = mapped_column(String(64), default="")
    price_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    barcode: Mapped[str] = mapped_column(String(64), default="")
    is_trial: Mapped[bool] = mapped_column(Boolean, default=False)
    trial_volume_ml: Mapped[float | None] = mapped_column(Float, nullable=True)
    trial_attached_to: Mapped[str | None] = mapped_column(String(64), nullable=True)


class StockRecord(Base):
    """库存表。`version` 每次扣减自增，用于事后核对「扣了几次」。"""

    __tablename__ = "commerce_stock"

    sku_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    available: Mapped[int] = mapped_column(Integer, default=0)
    reserved: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=0)
    threshold: Mapped[int] = mapped_column(Integer, default=5)


class BatchRecord(Base):
    __tablename__ = "commerce_batch"

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sku_id: Mapped[str] = mapped_column(String(64), index=True)
    code: Mapped[str] = mapped_column(String(32))
    produced_on: Mapped[date] = mapped_column(Date)
    shelf_life_months: Mapped[int] = mapped_column(Integer)
    expires_on: Mapped[date] = mapped_column(Date, index=True)


class InciRecord(Base):
    __tablename__ = "commerce_inci"

    inci_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    normalized: Mapped[str] = mapped_column(String(128), index=True)
    function: Mapped[str] = mapped_column(String(128), default="")
    cas_no: Mapped[str | None] = mapped_column(String(32), nullable=True)
    concentration: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (UniqueConstraint("product_id", "normalized", name="uq_commerce_inci_product_norm"),)


# --------------------------------------------------------------------------- #
# UMS：会员域
# --------------------------------------------------------------------------- #
class MemberRecord(Base):
    __tablename__ = "commerce_member"

    member_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(191), unique=True)
    nickname: Mapped[str] = mapped_column(String(128), default="")
    level: Mapped[str] = mapped_column(String(16), default="NORMAL")
    salt_hex: Mapped[str] = mapped_column(String(64), default="")
    hash_hex: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AddressRecord(Base):
    __tablename__ = "commerce_address"

    address_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    member_id: Mapped[str] = mapped_column(String(64), index=True)
    receiver: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(32))
    country: Mapped[str] = mapped_column(String(64), default="")
    city: Mapped[str] = mapped_column(String(64), default="")
    line1: Mapped[str] = mapped_column(String(256), default="")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


# --------------------------------------------------------------------------- #
# 购物车域
# --------------------------------------------------------------------------- #
class CartItemRecord(Base):
    """同一会员的同一 SKU 只允许一行：合并加购靠唯一键兜底，不靠应用层判断。"""

    __tablename__ = "commerce_cart_item"

    cart_item_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    member_id: Mapped[str] = mapped_column(String(64), index=True)
    sku_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    price_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("member_id", "sku_id", name="uq_commerce_cart_member_sku"),)


# --------------------------------------------------------------------------- #
# OMS：订单域
# --------------------------------------------------------------------------- #
class OrderRecord(Base):
    __tablename__ = "commerce_order"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), default="shop")
    member_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default="PENDING_PAYMENT")
    coupon_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    goods_cents: Mapped[int] = mapped_column(Integer, default=0)
    discount_cents: Mapped[int] = mapped_column(Integer, default=0)
    payable_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    carrier: Mapped[str | None] = mapped_column(String(64), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    __table_args__ = (
        # 超时取消的扫描条件就是这两列，覆盖索引让 sweeper 不必全表扫
        Index("ix_commerce_order_status_expires", "status", "expires_at"),
        Index("ix_commerce_order_member_created", "member_id", "created_at"),
    )


class OrderItemRecord(Base):
    __tablename__ = "commerce_order_item"

    order_item_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    sku_id: Mapped[str] = mapped_column(String(64))
    product_id: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(256), default="")
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")


# --------------------------------------------------------------------------- #
# SMS：营销域
# --------------------------------------------------------------------------- #
class CouponTemplateRecord(Base):
    __tablename__ = "commerce_coupon_template"

    template_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(24), default="FULL_REDUCTION")
    threshold_cents: Mapped[int] = mapped_column(Integer, default=0)
    value_cents: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    issued: Mapped[int] = mapped_column(Integer, default=0)
    per_member_limit: Mapped[int] = mapped_column(Integer, default=1)
    stackable: Mapped[bool] = mapped_column(Boolean, default=False)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CouponRecord(Base):
    __tablename__ = "commerce_coupon"

    coupon_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    template_id: Mapped[str] = mapped_column(String(64), index=True)
    member_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="RECEIVED")
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


# --------------------------------------------------------------------------- #
# 工程化支撑表：幂等 + 发件箱
# --------------------------------------------------------------------------- #
class IdempotencyRecord(Base):
    """写幂等表。唯一键是**并发正确性的关键**：
    两个并发的同幂等键请求里，只有一个能 INSERT 成功，另一个撞唯一键后
    去读已存的响应 —— 这样「重放不重复扣减」不依赖任何应用层锁。
    """

    __tablename__ = "commerce_idempotency"

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    response: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("project_id", "idempotency_key", name="uq_commerce_idem_project_key"),
    )


class OutboxRecord(Base):
    """发件箱。与业务写**同一事务**落库，保证「事务提交了，事件就一定存在」。"""

    __tablename__ = "commerce_outbox"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    topic: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_commerce_outbox_topic_published", "topic", "published_at"),)


__all__ = [
    "AddressRecord",
    "Base",
    "BatchRecord",
    "CartItemRecord",
    "CouponRecord",
    "CouponTemplateRecord",
    "IdempotencyRecord",
    "InciRecord",
    "MemberRecord",
    "OrderItemRecord",
    "OrderRecord",
    "OutboxRecord",
    "ProductRecord",
    "ShadeRecord",
    "SkuRecord",
    "StockRecord",
    "utcnow",
]
