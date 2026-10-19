"""MockProvider：可脚本化的离线模型。

作用（面试高频追问「MockProvider 干什么用」）：
让业务流程、多步工具循环、异常分支都能在没有网络、不烧真钱的前提下被稳定复现。
比如：第 1 次调用返回一个工具调用，第 2 次调用返回终答 —— 这正是 Agent Loop 的关键路径。

诚实边界：Mock 通过不等于真实模型效果正确。工具调用协议差异、生成质量、
延迟和真实 token 用量仍然必须用集成测试覆盖。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from ..core.errors import InfraError
from .types import Chunk, LLMResponse, Message, ToolCall, Usage


@dataclass
class MockTurn:
    """一次预设的响应。三选一：终答 / 工具调用 / 抛错。"""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: InfraError | None = None
    finish_reason: str = "stop"
    delay_s: float = 0.0
    prompt_tokens: int = 120
    completion_tokens: int = 40


def final(text: str, **kwargs: Any) -> MockTurn:
    return MockTurn(text=text, finish_reason="stop", **kwargs)


def calls(*tool_calls: tuple[str, dict[str, Any]] | ToolCall, **kwargs: Any) -> MockTurn:
    """构造一次工具调用轮次。用法：calls(("search_products", {"query": "精华"}))"""
    normalized: list[ToolCall] = []
    for index, item in enumerate(tool_calls):
        if isinstance(item, ToolCall):
            normalized.append(item)
        else:
            name, args = item
            normalized.append(ToolCall(id=f"call_mock_{index}_{name}", name=name, args=args))
    return MockTurn(tool_calls=normalized, finish_reason="tool_calls", **kwargs)


def failing(error: InfraError, **kwargs: Any) -> MockTurn:
    return MockTurn(error=error, **kwargs)


class MockProvider:
    """按脚本依次返回预设响应；脚本用尽后生成确定性的兜底终答。

    `history` 记录每次收到的 messages，便于测试断言「上下文是否正确回填」。
    """

    def __init__(self, name: str = "mock", turns: Sequence[MockTurn] | None = None, *, latency_s: float = 0.0) -> None:
        self.name = name
        self._turns: deque[MockTurn] = deque(turns or [])
        self._latency_s = latency_s
        self.history: list[list[Message]] = []
        self.call_count = 0

    # -- 脚本控制 ----------------------------------------------------------- #
    def enqueue(self, *turns: MockTurn) -> "MockProvider":
        self._turns.extend(turns)
        return self

    def reset(self, turns: Sequence[MockTurn] | None = None) -> "MockProvider":
        self._turns = deque(turns or [])
        self.history.clear()
        self.call_count = 0
        return self

    # -- 内部 --------------------------------------------------------------- #
    async def _next_turn(self, messages: Sequence[Message], model: str) -> MockTurn:
        self.call_count += 1
        self.history.append(list(messages))
        await asyncio.sleep(self._latency_s)
        if self._turns:
            return self._turns.popleft()
        return final(self._default_answer(messages, model))

    @staticmethod
    def _default_answer(messages: Sequence[Message], model: str) -> str:
        last_user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tool_results:
            return f"[mock:{model}] 已根据 {len(tool_results)} 条工具结果生成回答：{last_user}"
        return f"[mock:{model}] 收到问题：{last_user}"

    @staticmethod
    def _usage(turn: MockTurn) -> Usage:
        total = turn.prompt_tokens + turn.completion_tokens
        return Usage(
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
            total_tokens=total,
            cost_usd=0.0,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        turn = await self._next_turn(messages, model)
        if turn.error is not None:
            raise turn.error
        return LLMResponse(
            text=turn.text,
            tool_calls=list(turn.tool_calls),
            usage=self._usage(turn),
            finish_reason=turn.finish_reason,
            model=model,
            provider=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            raw={"mock": True, "tools_provided": len(tools or [])},
        )

    async def stream(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[Chunk]:
        response = await self.complete(messages, model, tools=tools)
        text = response.text or ""
        for index in range(0, len(text), 16):
            yield Chunk(delta_text=text[index : index + 16])
        if response.tool_calls:
            yield Chunk(
                tool_call_delta={
                    "index": 0,
                    "id": response.tool_calls[0].id,
                    "name": response.tool_calls[0].name,
                    "arguments": json.dumps(response.tool_calls[0].args, ensure_ascii=False),
                }
            )
        yield Chunk(finish_reason=response.finish_reason, usage=response.usage)
