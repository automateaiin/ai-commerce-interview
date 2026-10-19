"""Sequential 编排：按依赖顺序串行执行。

适用场景：后一步**必须**用到前一步的结论（例如先查库存、再据库存决定是否下单）。
关键点不是「串行」本身，而是**依赖注入**：`depends_on` 显式声明谁依赖谁，
模式负责把前序输出拼进后序任务的 `notes`，否则后序 Agent 根本看不到前序结果
（这是多 Agent 系统里最常见的「上下文断了」故障）。
"""

from __future__ import annotations

from src.infra.tools.base import ToolContext

from ..schemas import AgentInvocation
from .base import AgentRunner, ModeOutcome, event, to_subtask


class SequentialMode:
    name = "sequential"

    async def run(
        self,
        *,
        tasks: list[AgentInvocation],
        ctx: ToolContext,
        runner: AgentRunner,
    ) -> ModeOutcome:
        outcome = ModeOutcome()
        outputs: dict[str, str] = {}

        for index, invocation in enumerate(tasks, start=1):
            depends = invocation.depends_on or (
                [tasks[index - 2].agent] if index > 1 and tasks[index - 2].agent in outputs else []
            )
            notes = _compose_notes(invocation, outputs, depends)
            planned = AgentInvocation(
                agent=invocation.agent,
                task=invocation.task,
                scenario=invocation.scenario,
                notes=notes,
                depends_on=depends,
                confirmed=invocation.confirmed,
                max_steps=invocation.max_steps,
            )
            outcome.events.append(
                event(index, "dispatch", agent=invocation.agent, detail=f"depends_on={','.join(depends) or '-'}")
            )
            result = await runner(planned, ctx)
            task_id = f"{index}.{invocation.agent}"
            outcome.subtasks.append(to_subtask(task_id=task_id, invocation=invocation, result=result))
            if result.answer:
                outputs[invocation.agent] = result.answer
            outcome.events.append(
                event(index, "completed", agent=invocation.agent, status="ok" if result.answer else "error",
                      detail=result.stop_reason)
            )
            if result.answer is None:
                # 串行链上任何一环没跑完，后面的任务就没有可信输入 —— 停下并交接，不硬跑下去
                outcome.stop_reason = result.stop_reason
                outcome.handoff = result.handoff.to_payload() if result.handoff else None
                break

        outcome.stop_reason = "completed" if outcome.stop_reason == "completed" else outcome.stop_reason
        return outcome


def _compose_notes(invocation: AgentInvocation, outputs: dict[str, str], depends: list[str]) -> str:
    """把依赖方的结论拼成 notes。没有依赖时沿用调用方自带的 notes。"""
    if not depends:
        return invocation.notes
    parts = [f"[前序结论 · {agent}]\n{outputs[agent]}" for agent in depends if agent in outputs]
    if invocation.notes:
        parts.insert(0, invocation.notes)
    return "\n\n".join(parts)


def mode() -> SequentialMode:
    return SequentialMode()


__all__ = ["SequentialMode", "mode"]
