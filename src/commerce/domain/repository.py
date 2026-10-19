"""仓储 Protocol：接口定义在 domain，实现在 infrastructure。

为什么接口不放基础设施层：
如果 domain 依赖「某个 ORM 的 Repository」，领域就被存储技术绑死了，
想换数据库就得改业务规则。把 Protocol 放在 domain，依赖方向就被翻转成
「基础设施实现领域接口」（依赖倒置），领域保持纯净、可单测。

只定义 M1 真正用到的方法：接口一旦写宽，实现方就会为了「以后可能用到」而写一堆
没人调的空实现，反而掩盖了真实的查询需求。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence

from .models import (
    Cart,
    Coupon,
    CouponTemplate,
    Member,
    Order,
    Product,
    Stock,
)


class ProductRepository(Protocol):
    async def get(self, product_id: str) -> Product | None: ...

    async def upsert(self, product: Product) -> None: ...

    async def search(
        self,
        *,
        keyword: str | None = None,
        category: str | None = None,
        skin_type: str | None = None,
        market: str | None = None,
        status: str | None = None,
        max_price_cents: int | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Product], int]: ...


class StockRepository(Protocol):
    async def get(self, sku_id: str) -> Stock | None: ...

    async def upsert(self, stock: Stock) -> None: ...

    async def try_reserve(self, sku_id: str, quantity: int) -> bool:
        """条件 UPDATE 扣减。返回 False 表示库存不足（不是异常）。"""
        ...

    async def release(self, sku_id: str, quantity: int) -> int:
        """回滚库存，返回受影响行数。天然幂等。"""
        ...


class MemberRepository(Protocol):
    async def get(self, member_id: str) -> Member | None: ...

    async def by_email(self, email: str) -> Member | None: ...

    async def save(self, member: Member) -> None: ...

    async def save_address(self, member_id: str, address: Any) -> None: ...


class CartRepository(Protocol):
    async def get_by_member(self, member_id: str) -> Cart: ...

    async def save_item(self, member_id: str, sku_id: str, product_id: str, quantity: int, price_cents: int) -> None: ...

    async def remove_item(self, member_id: str, sku_id: str) -> None: ...

    async def clear(self, member_id: str) -> None: ...


class OrderRepository(Protocol):
    async def save(self, order: Order, *, project_id: str = "shop") -> None: ...

    async def get(self, order_id: str) -> Order | None: ...

    async def update_status(self, order_id: str, status: str, **fields: Any) -> int: ...

    async def list_by_member(self, member_id: str, *, offset: int = 0, limit: int = 20) -> list[Order]: ...

    async def list_all(
        self, *, status: str | None = None, offset: int = 0, limit: int = 20
    ) -> list[Order]:
        """运营侧跨会员订单列表（可按状态过滤）。买家侧只能看自己的，这条只服务后台。"""
        ...

    async def cancel_expired(self, now: datetime) -> list[Order]:
        """扫描超时未付款订单并置为取消，返回被取消的订单（供库存回滚）。"""
        ...

    async def stats(self, *, since: datetime | None = None) -> dict[str, Any]: ...


class CouponRepository(Protocol):
    async def get_template(self, template_id: str) -> CouponTemplate | None: ...

    async def list_templates(self) -> list[CouponTemplate]: ...

    async def save_template(self, template: CouponTemplate) -> None: ...

    async def bump_issued(self, template_id: str) -> None: ...

    async def issue(self, coupon: Coupon) -> None: ...

    async def get(self, coupon_id: str) -> Coupon | None: ...

    async def list_by_member(self, member_id: str) -> list[Coupon]: ...

    async def count_by_member(self, member_id: str, template_id: str) -> int: ...

    async def mark_used(self, coupon_id: str, order_id: str) -> int: ...

    async def mark_released(self, coupon_id: str) -> int: ...


class IdempotencyRecord(Protocol):
    project_id: str
    idempotency_key: str
    request_hash: str
    response: dict[str, Any]


class IdempotencyRepository(Protocol):
    async def get(self, project_id: str, key: str) -> dict[str, Any] | None: ...

    async def save(self, *, project_id: str, key: str, request_hash: str, response: dict[str, Any]) -> None: ...


class OutboxRepository(Protocol):
    async def enqueue(self, *, topic: str, payload: dict[str, Any], event_id: str) -> None: ...

    async def pending(self, limit: int = 50) -> Sequence[dict[str, Any]]: ...

    async def mark_published(self, event_id: str) -> None: ...


__all__ = [
    "CartRepository",
    "CouponRepository",
    "IdempotencyRecord",
    "IdempotencyRepository",
    "MemberRepository",
    "OrderRepository",
    "OutboxRepository",
    "ProductRepository",
    "StockRepository",
]
