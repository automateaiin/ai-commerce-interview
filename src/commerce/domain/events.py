"""领域事件：已发生的事实（过去式命名）。

为什么不用「回调」而用事件：
下单事务体里**绝不该**去调推荐、知识抽取这些副作用 —— 它们慢、可能失败，
一旦失败会把下单本身拖挂。正确做法是把「已发生的事实」记成事件，
事务提交后由订阅方异步消费（见 application/events.py 的 outbox）。
事件是过去式、不可变、可序列化 —— 这三点是它能被投递与重放的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """事件基类。`event_id` 用于消费端幂等去重，`occurred_at` 用于排序与重放。"""

    event_id: str
    occurred_at: str = field(default_factory=_now_iso)

    @property
    def name(self) -> str:
        return type(self).__name__

    def payload(self) -> dict[str, Any]:
        # 事件是 `slots=True` 的不可变 dataclass，没有 `__dict__` ——
        # 必须按字段取值，否则投递到 Outbox 时会在序列化这一步炸掉
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class ProductPublished(DomainEvent):
    """商品上架。AI 接缝：触发知识抽取流水线（本仓为 dataplatform）。"""

    product_id: str = ""
    name: str = ""
    market: str = ""


@dataclass(frozen=True, slots=True)
class OrderPlaced(DomainEvent):
    """下单成功。AI 接缝：触发推荐/售后子 Agent。"""

    order_id: str = ""
    member_id: str = ""
    amount_cents: int = 0
    currency: str = "USD"
    item_summary: str = ""


@dataclass(frozen=True, slots=True)
class OrderCancelled(DomainEvent):
    """订单取消（主动取消或超时取消）。库存已回滚。"""

    order_id: str = ""
    reason: str = ""
    released_items: int = 0


@dataclass(frozen=True, slots=True)
class StockLow(DomainEvent):
    """库存触达预警线。供补货策略消费。"""

    sku_id: str = ""
    available: int = 0
    threshold: int = 0


__all__ = [
    "DomainEvent",
    "OrderCancelled",
    "OrderPlaced",
    "ProductPublished",
    "StockLow",
]
