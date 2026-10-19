"""仓储实现集合。domain 定义 Protocol，这里给出 SQLAlchemy 版本。

每个仓储的构造参数都是**一个 session**，由用例层控制事务边界 ——
仓储不自己开连接，是「扣库存 + 建订单」能落进同一事务的前提。
"""

from __future__ import annotations

from .cart_repo import SqlCartRepository
from .coupon_repo import SqlCouponRepository
from .idempotency_repo import SqlIdempotencyRepository, hash_request
from .member_repo import SqlMemberRepository
from .order_repo import SqlOrderRepository
from .outbox_repo import SqlOutboxRepository
from .product_repo import SqlProductRepository, SqlStockRepository, new_id

__all__ = [
    "SqlCartRepository",
    "SqlCouponRepository",
    "SqlIdempotencyRepository",
    "SqlMemberRepository",
    "SqlOrderRepository",
    "SqlOutboxRepository",
    "SqlProductRepository",
    "SqlStockRepository",
    "hash_request",
    "new_id",
]
