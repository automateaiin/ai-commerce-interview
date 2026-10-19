"""工具结果 → 模型可消费的观察（observation）。

这一步是「Agent 能不能自己纠错」的分水岭，三个细节必须做对：

1) **对齐 call_id**：observation 必须以 `role=tool` + `tool_call_id` 回填，
   否则模型看到的是一堆「不知道对应哪次调用」的文本，也就无从判断该改哪个参数；
2) **控制长度**：工具可能返回几百条商品，整段塞回去会挤爆上下文并且花钱。
   这里的做法是**分页 + 显式声明被截断**，并给出缩小范围的提示 ——
   既保住上下文，又不让模型误以为「只有这么多结果」；
3) **只回安全摘要**：错误只回错误码与安全文案，绝不回堆栈或内部结构；
   但参数类错误要把 issues（哪个字段、错在哪）带上，让模型能自己改对。

另外历史观察会随轮数增长，所以要按「轮」为单位做尾部保留 + 老观察压缩，
而不是简单按条数裁剪 —— 按条数裁会把同一次调用的结果和它的调用切开，
产生「孤儿 tool 消息」，直接导致下一次请求被供应商判为非法。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from src.infra.tools.base import ToolResult

from .decision import PlannedCall
from .validate import Violation

_STRING_LIMIT = 400
_DICT_LIMIT = 20
_HISTORY_LIMIT = 160


class ObservationBuilder:
    def __init__(self, *, max_chars: int = 1500, page_size: int = 20, max_depth: int = 4) -> None:
        self._max_chars = max_chars
        self._page_size = page_size
        self._max_depth = max_depth

    # -- 对外接口 ---------------------------------------------------------- #

    def from_result(self, call: PlannedCall, result: ToolResult) -> dict[str, Any]:
        payload = result.to_observation()
        if not result.ok and result.meta:
            # ToolResult 的 observation 不含 meta；参数问题必须补上 issues 才有自纠错能力
            issues = result.meta.get("issues")
            if issues:
                payload["issues"] = list(issues)[:5]
        return self._message(call.id, call.name, payload)

    def from_violation(self, violation: Violation) -> dict[str, Any]:
        payload = violation.to_observation()
        return self._message(violation.call_id, violation.tool, payload, call=violation)

    def from_guardrail(self, detail: dict[str, Any]) -> dict[str, Any]:
        """护栏拦截也需要让模型「看到」自己为什么被打断，否则它会重复同样的动作。"""
        payload = {"status": "error", "error_code": "guardrail_tripped", **detail}
        return {"role": "user", "content": self._render(payload)}

    def trim(self, messages: Sequence[dict[str, Any]], *, keep_rounds: int = 6) -> list[dict[str, Any]]:
        system = [message for message in messages if message.get("role") == "system"][:1]
        rounds: list[list[dict[str, Any]]] = []
        for message in messages:
            if message.get("role") == "system":
                continue
            if message.get("role") == "assistant" or not rounds:
                rounds.append([message])
            else:
                rounds[-1].append(message)

        kept = rounds[-keep_rounds:]
        for group in kept[:-1]:
            # 保留的旧轮次里，把大的工具结果压成摘要：位置留着（协议合法），内容不占预算
            for message in group:
                if message.get("role") == "tool" and len(str(message.get("content", ""))) > _HISTORY_LIMIT:
                    message["content"] = f"{str(message['content'])[:_HISTORY_LIMIT]}…[历史观察已压缩]"
        return [*system, *(message for group in kept for message in group)]

    # -- 内部 -------------------------------------------------------------- #

    def _message(
        self,
        call_id: str | None,
        tool: str | None,
        payload: dict[str, Any],
        *,
        call: Violation | None = None,
    ) -> dict[str, Any]:
        # 校验失败可能发生在「模型提了调用但没执行」时，但仍然要挂回对应的 tool_call_id，
        # 否则协议上会出现「assistant 声明了 tool_calls 却没有对应 tool 消息」的缺口
        return {
            "role": "tool",
            "tool_call_id": call_id if call_id is not None else _fallback_id(tool, call),
            "name": tool or "unknown_tool",
            "content": self._render(payload),
        }

    def _render(self, payload: dict[str, Any]) -> str:
        text = json.dumps(self._shrink(payload), ensure_ascii=False, default=str)
        if len(text) <= self._max_chars:
            return text
        return f"{text[: self._max_chars]}…[observation_truncated]"

    def _shrink(self, value: Any, depth: int = 0) -> Any:
        if depth >= self._max_depth:
            return "…"
        if isinstance(value, str):
            return value if len(value) <= _STRING_LIMIT else f"{value[:_STRING_LIMIT]}…"
        if isinstance(value, dict):
            return {key: self._shrink(item, depth + 1) for key, item in list(value.items())[:_DICT_LIMIT]}
        if isinstance(value, (list, tuple)):
            items = list(value)
            if len(items) <= self._page_size:
                return [self._shrink(item, depth + 1) for item in items]
            return {
                "items": [self._shrink(item, depth + 1) for item in items[: self._page_size]],
                "total": len(items),
                "truncated": True,
                "hint": "结果已截断，请用更精确的参数缩小范围后重试",
            }
        return value


def _fallback_id(tool: str | None, call: Violation | None) -> str:
    """没有原始 call_id 时（不该发生）给一个稳定占位，保证协议字段不缺失。"""
    return f"call_{tool or 'unknown'}_{call.code if call else 'na'}"
