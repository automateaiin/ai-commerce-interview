"""模型输出 → 归一化决策。

Harness 的第一条铁律：**模型返回的不是命令，而是「决策提议」**。
所以这里只做一件事：把供应商差异（文本/工具调用混排、finish_reason 语义、arguments 字符串）
收敛成一个内部类型，后续所有处理都基于这个类型 ——
校验、护栏、观测、交接都不再关心是哪家模型。

三类输出都要能正确处理（面试常被追问「模型返回纯文本怎么办」）：
1) 只有工具调用            → TOOL_CALLS
2) 只有文本                → FINAL
3) 文本 + 工具调用同时出现 → TOOL_CALLS，但把文本记为 preamble（有些供应商会边解释边调工具）
4) 两者都没有              → 解析失败，属于不可恢复的脏输出，交给上层终止并交接
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.infra.core.errors import InfraError
from src.infra.llm.types import LLMResponse


class DecisionKind(str, Enum):
    FINAL = "final"
    TOOL_CALLS = "tool_calls"


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """归一化后的一个工具调用提议（**尚未**通过校验）。"""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    """arguments 不是合法 JSON 时保留原文：用于报可恢复错误，且不回填给模型避免二次污染。"""

    def fingerprint(self) -> str:
        """调用指纹：用于「打转检测」。

        刻意对参数做**排序后**哈希 —— 同一组参数换个键顺序就是同一次调用，
        否则模型只要换个字段顺序就能绕过重复检测。
        """
        canonical = json.dumps(self.args, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha1(f"{self.name}:{canonical}".encode()).hexdigest()
        return f"{self.name}#{digest[:12]}"


@dataclass(slots=True)
class LLMDecision:
    kind: DecisionKind
    text: str = ""
    calls: tuple[PlannedCall, ...] = ()
    mixed: bool = False
    """文本与工具调用同时存在。记录它是为了可观测：混排输出往往意味着 Prompt 约束不够紧。"""

    finish_reason: str = "stop"
    logical_model: str = ""
    usage: Any = None
    latency_ms: float = 0.0

    def describe(self) -> dict[str, Any]:
        """给日志/轨迹用的安全摘要：只记结构，不记参数原文。"""
        return {
            "kind": self.kind.value,
            "mixed": self.mixed,
            "calls": [{"tool": call.name, "args_digest": call.fingerprint()} for call in self.calls],
            "text_chars": len(self.text),
            "finish_reason": self.finish_reason,
        }


class DecisionParseError(InfraError):
    """模型输出无法解析为任何决策（脏输出）。不可恢复：重试同一个输出没有意义。"""

    code = "decision_invalid"
    http_status = 502
    safe_message = "模型未返回可用的决策"


def parse_decision(response: LLMResponse) -> LLMDecision:
    """把统一模型响应解析成决策。绝不猜测：解析不出来就报错。"""
    calls = tuple(
        PlannedCall(
            id=call.id,
            name=call.name,
            args=dict(call.args or {}),
            raw_arguments=call.raw_arguments,
        )
        for call in response.tool_calls
    )
    text = (response.text or "").strip()

    if calls:
        # 有工具调用就以工具调用为准：文本只作为「解释性前言」记录，不参与终止判断
        return LLMDecision(
            kind=DecisionKind.TOOL_CALLS,
            text=text,
            calls=calls,
            mixed=bool(text),
            finish_reason=response.finish_reason,
            logical_model=response.logical_model,
            usage=response.usage,
            latency_ms=response.latency_ms,
        )

    if text:
        return LLMDecision(
            kind=DecisionKind.FINAL,
            text=text,
            finish_reason=response.finish_reason,
            logical_model=response.logical_model,
            usage=response.usage,
            latency_ms=response.latency_ms,
        )

    raise DecisionParseError(
        "模型既没有返回文本也没有返回工具调用",
        detail={"finish_reason": response.finish_reason, "logical_model": response.logical_model},
    )
