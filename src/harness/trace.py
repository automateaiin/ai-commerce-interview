"""结构化执行轨迹。

轨迹有三个消费者，这也是它必须结构化的原因：
1) **排障**：出错时要知道第几轮调了什么工具、参数指纹是什么、失败码是什么；
2) **前端可视化**：把「模型想了什么、调了什么、看到什么」流式展示给业务人员；
3) **离线评估**：轨迹可以直接当评估样本（哪一步开始跑偏），所以字段要稳定、可比。

写轨迹时的两条纪律：
- **不写参数原文**，只写指纹。参数可能含用户隐私（地址、手机号），轨迹会被长期保留；
- 出错只写错误码与安全摘要，堆栈只进服务端日志（infra 的 JsonFormatter 已有白名单与脱敏）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.infra.core.logging import scrub

_CLIP = 240


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= _CLIP else f"{value[:_CLIP]}…"
    if isinstance(value, dict):
        return {key: _clip(item) for key, item in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_clip(item) for item in list(value)[:20]]
    return value


@dataclass(slots=True)
class ExecutionTrace:
    """一次运行的轨迹。字段刻意保持扁平，方便直接落库做离线分析。"""

    trace_id: str
    rounds: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    stop_reason: str | None = None
    _events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, step: int, kind: str, **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {"step": step, "type": kind}
        for key, value in fields.items():
            if value is None:
                continue
            event[key] = _clip(scrub(value) if isinstance(value, (str, dict)) else value)
        self._events.append(event)
        return event

    def termination(
        self, *, step: int, reason: str, handoff: str | None = None, detail: dict[str, Any] | None = None
    ) -> None:
        self.stop_reason = reason
        self.record(
            step=step,
            kind="termination",
            status="ok" if reason == "completed" else "error",
            stop_reason=reason,
            handoff=handoff,
            detail=detail,
        )

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def digest(self) -> dict[str, Any]:
        """回放/评估用的紧凑摘要：一轮一行不需要看全部事件就能判断这次运行健康与否。"""
        tools = [
            {"tool": event.get("tool"), "status": event.get("status"), "error_code": event.get("error_code")}
            for event in self._events
            if event["type"] == "tool"
        ]
        errors: dict[str, int] = {}
        for item in tools:
            if item["status"] == "error" and item["error_code"]:
                errors[item["error_code"]] = errors.get(item["error_code"], 0) + 1
        return {
            "trace_id": self.trace_id,
            "rounds": self.rounds,
            "stop_reason": self.stop_reason,
            "tools": tools,
            "error_distribution": errors,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_ms": round(self.llm_ms, 2),
            "tool_ms": round(self.tool_ms, 2),
        }
