"""Handoff 编排：主 Agent 跑不完（高风险待确认 / 越权 / 缺信息）时转派专家。

与 Sequential 的区别：Sequential 是「一步接一步往前推」，Handoff 是「换个人接着干」。
转派必须携带**交接包**（reason + 已完成步骤 + 失败原因 + 建议动作），
否则接手方从零开始，等于把问题重问一遍 —— harness 的 handoff 包就是为这个设计的。

必须有**转派次数上限**：没有上限的转派链在权限/风险场景会来回踢皮球，
最终把一次请求变成一串无界模型调用。
"""

from __future__ import annotations

from src.infra.tools.base import ToolContext

from ..schemas import AgentInvocation
from .base import AgentRunner, ModeOutcome, event, to_subtask


class HandoffMode:
    name = "handoff"

    def __init__(self, *, max_handoffs: int = 2) -> None:
        self._max_handoffs = max_handoffs

    async def run(
        self,
        *,
        tasks: list[AgentInvocation],
        ctx: ToolContext,
        runner: AgentRunner,
    ) -> ModeOutcome:
        outcome = ModeOutcome()
        if not tasks:
            outcome.stop_reason = "invalid_plan"
            outcome.events.append(event(1, "plan_invalid", status="error", detail="handoff 模式需要至少一个主 Agent"))
            return outcome

        index = 0
        hops = 0
        step = 0
        transfer_notes = ""

        while True:
            invocation = tasks[index]
            planned = AgentInvocation(
                agent=invocation.agent,
                task=invocation.task,
                scenario=invocation.scenario,
                notes=transfer_notes or invocation.notes,
                depends_on=invocation.depends_on,
                confirmed=invocation.confirmed,
                max_steps=invocation.max_steps,
            )
            step += 1
            outcome.events.append(
                event(step, "dispatch", agent=invocation.agent, detail="handoff" if transfer_notes else "primary")
            )
            result = await runner(planned, ctx)
            outcome.subtasks.append(
                to_subtask(task_id=f"{step}.{invocation.agent}", invocation=invocation, result=result)
            )

            if result.answer is not None:
                outcome.events.append(event(step, "completed", agent=invocation.agent, detail=result.stop_reason))
                outcome.stop_reason = "completed"
                return outcome

            packet = result.handoff
            outcome.stop_reason = result.stop_reason
            outcome.handoff = packet.to_payload() if packet else None
            outcome.events.append(
                event(step, "handoff_required", agent=invocation.agent, status="error",
                      detail=packet.reason if packet else result.stop_reason)
            )
            hops += 1
            if hops > self._max_handoffs or index + 1 >= len(tasks):
                # 没有可接手的专家，或转派已到上限：停在交接态，交由人工或上游处理
                outcome.events.append(event(step, "handoff_exhausted", status="error", detail=f"hops={hops}"))
                return outcome

            index += 1
            target = tasks[index]
            transfer_notes = _transfer_note(packet, invocation)
            outcome.events.append(event(step, "handoff", agent=target.agent, detail=f"from={invocation.agent}"))


def _transfer_note(packet, previous: AgentInvocation) -> str:
    """把交接包翻译成接手方能用的上下文（只搬运摘要与建议动作，不搬运敏感原文）。"""
    if packet is None:
        return f"[转派上下文]\n原始任务：{previous.task}\n前序 Agent 未给出终答，请接手完成。"
    actions = "；".join(packet.next_actions) if packet.next_actions else "无"
    return (
        "[转派上下文]\n"
        f"原始任务：{previous.task}\n"
        f"交接原因：{packet.reason}\n"
        f"已完成步骤：{packet.completed_steps}\n"
        f"失败说明：{packet.failure or '无'}\n"
        f"建议动作：{actions}"
    )


def mode(*, max_handoffs: int = 2) -> HandoffMode:
    return HandoffMode(max_handoffs=max_handoffs)


__all__ = ["HandoffMode", "mode"]
