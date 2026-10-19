"""事件的「发布」与「消费」：两边分离，且都不在业务事务里做重活。

两条纪律（面试可直接讲）：

1) **发布只写 Outbox**：`OutboxPublisher` 把领域事件写进 `commerce_outbox` 表，
   与「扣库存 + 建订单」**同一个事务**提交。它绝不在事务里去调推荐、知识抽取、
   发短信 —— 那些副作用慢且可能失败，一旦失败会把下单本身拖挂。
2) **消费在提交之后**：`EventRouter` 拿到的是「已经提交成功」的事件列表，
   逐个路由到 `AIHookPort`。钩子抛异常只记日志、不回溯事务 —— 事件已经落库，
   补发由 Outbox 的消费者负责，业务事务不该为钩子的失败买单。

`RepositoryOutbox` 这个**写入口**放在 application 而不是 infrastructure：
它是「用例决定要发布什么」的表达，而具体落到哪张表由仓储实现。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..domain.events import DomainEvent, OrderCancelled, OrderPlaced, ProductPublished, StockLow
from ..domain.repository import OutboxRepository
from .ports import AIHookPort


def event_topic(event: DomainEvent) -> str:
    """主题名 = 事件类型名。消费端按主题订阅即可，不需要额外的注册表。"""
    return event.name


class OutboxPublisher:
    """把一个事务内要发布的事件**收集起来**，在 commit 前一次性写进 Outbox。

    为什么先收集再写：事件可能在事务中途才产生（例如扣完库存才知道触达预警线），
    逐条写也可以；但收集后统一落库能让「本次事务发布了哪些事件」在一个地方可见，
    排查「订单有了但事件没发」时直接看这一处。
    """

    def __init__(self, outbox: OutboxRepository) -> None:
        self._outbox = outbox
        self._pending: list[DomainEvent] = []

    def add(self, event: DomainEvent) -> None:
        self._pending.append(event)

    @property
    def pending(self) -> Sequence[DomainEvent]:
        return tuple(self._pending)

    async def flush(self) -> int:
        """写入 Outbox（**必须在 commit 之前调用**）。返回写入条数。"""
        for event in self._pending:
            await self._outbox.enqueue(topic=event_topic(event), payload=event.payload(), event_id=event.event_id)
        written = len(self._pending)
        self._pending.clear()
        return written


class EventRouter:
    """提交后的事件路由：把事件投给 AI 钩子。

    只认识「钩子关心的三类事件」；其余事件（如 `StockLow`）目前没有订阅方，
    沉默丢弃是**有意为之**：它们在 Outbox 里仍在，接补货策略时再订阅即可。
    """

    def __init__(self, hook: AIHookPort) -> None:
        self._hook = hook

    async def dispatch(self, events: Sequence[DomainEvent]) -> dict[str, int]:
        delivered: dict[str, int] = {}
        for event in events:
            if isinstance(event, ProductPublished):
                await self._hook.on_product_published(event)
            elif isinstance(event, OrderPlaced):
                await self._hook.on_order_placed(event)
            elif isinstance(event, OrderCancelled):
                await self._hook.on_order_cancelled(event)
            else:
                continue
            delivered[event_topic(event)] = delivered.get(event_topic(event), 0) + 1
        return delivered


def summarize(events: Sequence[DomainEvent]) -> list[dict[str, Any]]:
    """对外可读的事件摘要（REST 响应里回给调用方，便于对账）。"""
    return [{"topic": event_topic(event), "event_id": event.event_id, "occurred_at": event.occurred_at} for event in events]


__all__ = ["EventRouter", "OutboxPublisher", "event_topic", "summarize"]
