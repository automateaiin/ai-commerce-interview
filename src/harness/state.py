"""运行状态与终止原因。

把状态显式成一个对象（而不是散在局部变量里）的理由：
**可中断性**。人工接管、超时熔断、进程重启续跑都需要「当时跑到哪了」这个快照，
散落的局部变量做不到这一点。

状态里刻意**不含业务数据原文**，只有摘要与指纹：状态会被序列化进轨迹、进交接包、进日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.infra.llm.types import Usage
from src.infra.tools.base import ToolContext

from .handoff import HandoffPacket
from .trace import ExecutionTrace


class StopReason(str, Enum):
    """终止原因。取值稳定，因为它是对外契约（前端换文案、监控分桶、离线评估都靠它）。"""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    REPEATED_CALL_DETECTED = "repeated_call_detected"
    ERROR_THRESHOLD_EXCEEDED = "error_threshold_exceeded"
    TERMINAL_TOOL_ERROR = "terminal_tool_error"
    DECISION_INVALID = "decision_invalid"
    HANDOFF_REQUIRED = "handoff_required"
    NO_TOOLS = "no_tools"


@dataclass(slots=True)
class RunState:
    task: str
    ctx: ToolContext
    trace: ExecutionTrace
    messages: list[dict[str, Any]]
    logical_model: str | None
    max_steps: int
    deadline: float
    """以 time.perf_counter() 为基准的绝对截止时刻 —— 用单调时钟，避免系统时间被调整后护栏失效。"""

    token_budget: int
    cost_budget_usd: float

    usage: Usage = field(default_factory=Usage)
    step: int = 0
    """已完成决策的轮数（从 0 开始计，进入第 N+1 轮时等于 N）。"""

    fingerprints: dict[str, int] = field(default_factory=dict)
    error_streak: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    answer: str | None = None
    handoff: HandoffPacket | None = None

    def remaining_ms(self, now: float) -> float:
        return max(0.0, (self.deadline - now) * 1000)

    def note(self, **fields: Any) -> dict[str, Any]:
        """往轨迹与对外 transcript 同步记一笔（两者面向的读者不同，但事实必须一致）。"""
        entry: dict[str, Any] = {"step": self.step}
        entry.update({key: value for key, value in fields.items() if value is not None})
        self.transcript.append(entry)
        return entry
