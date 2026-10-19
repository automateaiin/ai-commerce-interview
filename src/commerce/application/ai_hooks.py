"""AI 钩子：为「把项目一/二/三接进来」预留的缝，M1 只提供 Protocol 与空实现。

为什么 M1 就要把这个缝留出来（哪怕不做实现）：
如果业务代码里直接写 `await knowledge.extract(product)`，那么接入 AI 的那一刻
就必须回头改下单/上架用例 —— 而这时候业务逻辑通常已经在生产上跑了。
留一个 `AIHookPort`，后续只需在装配处把 NoopHook 换成真实实现，
**业务用例一行不改**。这是「AI 能力不对业务造成侵入」的具体落地方式。

`NoopHook` 刻意只写日志不发通知：
一个「什么都不做的实现」如果连日志都不打，线上就会看不出「钩子到底有没有被触发」，
排查「为什么上架后没有生成知识」时会变成无头案。
"""

from __future__ import annotations

from typing import Any

from ...infra.core.logging import get_logger
from ..domain.events import DomainEvent

logger = get_logger("commerce.ai_hooks")


class NoopHook:
    """空实现。只记录「事件已到达钩子」，供后续替换与排障。"""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    async def _note(self, channel: str, event: DomainEvent) -> None:
        if not self._enabled:
            return
        logger.info(
            "commerce.ai_hook.noop",
            extra={"event": "business", "channel": channel, "hook": event.name, "status": "ok"},
        )

    async def on_product_published(self, event: DomainEvent) -> None:
        await self._note("product_published", event)

    async def on_order_placed(self, event: DomainEvent) -> None:
        await self._note("order_placed", event)

    async def on_order_cancelled(self, event: DomainEvent) -> None:
        await self._note("order_cancelled", event)


class RecordingHook:
    """测试用：把触发过的钩子记下来，供断言「钩子确实在事务提交后被调用了」。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _record(self, channel: str, event: DomainEvent) -> None:
        self.calls.append({"channel": channel, "event": event.name, "payload": event.payload()})

    async def on_product_published(self, event: DomainEvent) -> None:
        self._record("product_published", event)

    async def on_order_placed(self, event: DomainEvent) -> None:
        self._record("order_placed", event)

    async def on_order_cancelled(self, event: DomainEvent) -> None:
        self._record("order_cancelled", event)


__all__ = ["NoopHook", "RecordingHook"]
