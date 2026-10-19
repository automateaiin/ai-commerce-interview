"""模型调用的统一数据结构（协议无关）。

这是 Provider 抽象的第二层：「统一响应结构」。
各家返回的文本、工具调用、用量、结束原因形状都不一样，
适配器负责把它们归一到这里，主循环（Agent Loop）只认这几个类型，
所以换供应商时上层代码零改动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

# 统一消息结构：只有这几个键是跨供应商通用的
Message = dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    """归一化后的工具调用。

    差异在这里被抹平：
    - OpenAI 兼容：tool_calls[i].function.arguments 是 JSON **字符串**，需要 json.loads
    - Anthropic：tool_use block 的 input 直接就是对象
    统一成 args: dict，调用方不需要知道供应商细节。
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    """原始 arguments 字符串，解析失败时保留用于排查，且不会泄漏给模型。"""


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def merged_with(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(slots=True)
class LLMResponse:
    """统一模型响应。"""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    logical_model: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None
    """供应商原始响应，仅用于服务端排查，默认不写日志、不进模型上下文。"""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        """把响应转回 assistant 消息，便于回填进对话历史。"""
        message: Message = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in self.tool_calls
            ]
        return message


@dataclass(slots=True)
class Chunk:
    """流式增量。finish_reason 与 usage 可能只在最后一帧出现。"""

    delta_text: str = ""
    tool_call_delta: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
