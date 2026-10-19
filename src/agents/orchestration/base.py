"""编排模式的公共契约。

三种模式（Sequential / Fan-out-Fan-in / Handoff）的差别只有「谁先跑、谁跟谁有依赖」，
但它们的**输出形状必须一致**，否则编排层与前端就要为每种模式写一套解析。
所以这里把共同点沉淀成三样东西：

1) `AgentRunner`：把「agent 名 + 任务」真正执行掉的可调用对象（由 Orchestrator 注入）；
2) `ModeOutcome`：模式跑完后吐出的统一结果（子任务结果 + 事件 + 终止原因 + 交接包）；
3) 两个小工具：`to_subtask`（HarnessResult → SubTaskResult）与 `merge_usage`（子任务用量求和）。

编排模式本身**不认识具体业务**：它只认「任务有依赖就串行、无依赖可并行、遇到交接就转派」，
因此新增一个业务 Agent 不需要改任何模式代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from src.harness.engine import HarnessResult
from src.infra.llm.types import Usage
from src.infra.tools.base import ToolContext

from ..schemas import AgentInvocation, OrchestrationEvent, SubTaskResult, usage_payload

# 执行一次子任务的回调：由 Orchestrator 提供（它才知道 agent 名到实现的映射）
AgentRunner = Callable[[AgentInvocation, ToolContext], Awaitable[HarnessResult]]


@dataclass(slots=True)
class ModeOutcome:
    """模式跑完的统一结果。"""

    subtasks: list[SubTaskResult] = field(default_factory=list)
    events: list[OrchestrationEvent] = field(default_factory=list)
    stop_reason: str = "completed"
    handoff: dict[str, Any] | None = None

    @property
    def has_output(self) -> bool:
        return any(task.answer for task in self.subtasks)


class OrchestrationMode(Protocol):
    """编排模式协议。新增模式只需实现 `name` 与 `run`，再注册进 Registry。"""

    name: str

    async def run(
        self,
        *,
        tasks: list[AgentInvocation],
        ctx: ToolContext,
        runner: AgentRunner,
    ) -> ModeOutcome: ...


def to_subtask(*, task_id: str, invocation: AgentInvocation, result: HarnessResult) -> SubTaskResult:
    """HarnessResult → 编排子任务结果。用量与轨迹只保留摘要，不搬运参数原文。"""
    return SubTaskResult(
        id=task_id,
        agent=invocation.agent,
        ok=result.answer is not None,
        answer=result.answer,
        stop_reason=result.stop_reason,
        steps=result.steps,
        usage=usage_payload(result.usage),
        tool_calls=result.tool_calls,
        handoff=result.handoff.to_payload() if result.handoff else None,
    )


def merge_usage(results: list[HarnessResult]) -> Usage:
    """编排花的钱 = 各子 Agent 之和；只报最后一次会让成本统计系统性偏低。"""
    total = Usage()
    for result in results:
        total = total.merged_with(result.usage)
    return total


def event(step: int, kind: str, *, agent: str | None = None, status: str = "ok", detail: str | None = None) -> OrchestrationEvent:
    return OrchestrationEvent(step=step, kind=kind, agent=agent, status=status, detail=detail)


__all__ = [
    "AgentRunner",
    "ModeOutcome",
    "OrchestrationMode",
    "to_subtask",
    "merge_usage",
    "event",
]
