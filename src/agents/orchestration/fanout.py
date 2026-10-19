"""Fan-out-Fan-in 编排：无依赖的子任务并行跑，最后收敛成统一结论。

为什么必须「先按依赖分层，再并行」而不是无脑 `asyncio.gather` 全部：
gather 会同时发起所有子任务，但如果某个子任务依赖另一个的输出，它拿不到任何上下文 ——
并行只对**彼此独立**的工作安全。所以这里按依赖拓扑分层，层内并行、层间串行。

Fan-in（收敛）也由模式负责：把各分支结论拼给一个合成的 Agent（默认取首个任务的 Agent），
产出统一答复。多分支各答各的、没人收敛，等于把麻烦丢给用户。
"""

from __future__ import annotations

import asyncio

from src.infra.tools.base import ToolContext

from ..schemas import AgentInvocation
from .base import AgentRunner, ModeOutcome, event, to_subtask


class FanOutFanInMode:
    name = "fanout"

    async def run(
        self,
        *,
        tasks: list[AgentInvocation],
        ctx: ToolContext,
        runner: AgentRunner,
    ) -> ModeOutcome:
        outcome = ModeOutcome()
        outputs: dict[str, str] = {}
        done: set[int] = set()
        step = 0
        index_list = list(range(len(tasks)))

        while len(done) < len(tasks):
            ready = [i for i in index_list if i not in done and all(dep in outputs for dep in tasks[i].depends_on)]
            if not ready:
                # 依赖成环或引用了不存在的 Agent：显式失败，绝不静默空转
                outcome.events.append(event(step + 1, "plan_invalid", status="error", detail="依赖未满足或成环"))
                outcome.stop_reason = "invalid_plan"
                return outcome

            step += 1
            outcome.events.append(event(step, "fan_out", detail=f"parallel={len(ready)}"))
            planned = [
                AgentInvocation(
                    agent=tasks[i].agent,
                    task=tasks[i].task,
                    scenario=tasks[i].scenario,
                    notes=_notes_from(tasks[i], outputs),
                    depends_on=tasks[i].depends_on,
                    confirmed=tasks[i].confirmed,
                    max_steps=tasks[i].max_steps,
                )
                for i in ready
            ]
            results = await asyncio.gather(*(runner(item, ctx) for item in planned))
            for i, invocation, result in zip(ready, planned, results):
                done.add(i)
                task_id = f"{i}.{invocation.agent}"
                outcome.subtasks.append(to_subtask(task_id=task_id, invocation=invocation, result=result))
                outcome.events.append(
                    event(step, "branch_done", agent=invocation.agent,
                          status="ok" if result.answer else "error", detail=result.stop_reason)
                )
                if result.answer:
                    outputs[invocation.agent] = result.answer
                elif result.handoff is not None:
                    outcome.handoff = result.handoff.to_payload()
                    outcome.stop_reason = result.stop_reason

        if not outputs:
            outcome.stop_reason = outcome.stop_reason if outcome.stop_reason != "completed" else "no_branches"
            return outcome

        step += 1
        outcome.events.append(event(step, "fan_in", detail=f"merge={len(outputs)}"))
        synthesizer = AgentInvocation(
            agent=tasks[0].agent,
            task="综合以上各分支的结论，给出统一答复；分支之间若有冲突，明确说明以哪一条为准及理由。",
            scenario=tasks[0].scenario,
            notes="\n\n".join(f"[分支结论 · {agent}]\n{answer}" for agent, answer in outputs.items()),
        )
        merged = await runner(synthesizer, ctx)
        outcome.subtasks.append(
            to_subtask(task_id=f"{step}.{synthesizer.agent}.merge", invocation=synthesizer, result=merged)
        )
        if merged.answer is None:
            outcome.stop_reason = merged.stop_reason
            outcome.handoff = merged.handoff.to_payload() if merged.handoff else outcome.handoff
        else:
            outcome.stop_reason = "completed"
        return outcome


def _notes_from(invocation: AgentInvocation, outputs: dict[str, str]) -> str:
    if not invocation.depends_on:
        return invocation.notes
    parts = [f"[前序结论 · {agent}]\n{outputs[agent]}" for agent in invocation.depends_on if agent in outputs]
    if invocation.notes:
        parts.insert(0, invocation.notes)
    return "\n\n".join(parts)


def mode() -> FanOutFanInMode:
    return FanOutFanInMode()


__all__ = ["FanOutFanInMode", "mode"]
