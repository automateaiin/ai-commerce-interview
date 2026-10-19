"""项目一（agents）的编排契约。

这一层只放**跨模块共享的数据结构**：编排子任务、事件、结果契约与路由计划。
把它单独成文件的原因与 harness 里的 `HarnessResult` 一样：
一次编排调用要同时服务三类读者 —— 业务前端（answer）、计费（usage）、排障与评估（events/stop_reason）。
让所有模式（Sequential / Fan-out / Handoff）吐同一种形状，上层与前端就不必为每种模式写一套解析。

两条纪律沿用 harness：
- 事件里**不写用户正文**，只写 agent 名、状态、摘要；
- 结果契约必须**汇总 token 用量**：编排花的钱是各子 Agent 之和，不能只报最后一次。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.infra.llm.types import Usage


def usage_payload(usage: Usage) -> dict[str, Any]:
    """把 Usage 归一成对外信封形状（与 bridge 的 response 保持一致）。"""
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost_usd": round(usage.cost_usd, 6),
    }


@dataclass(slots=True)
class AgentInvocation:
    """一次子 Agent 调用的入参。

    `depends_on` 是**显式依赖声明**：没有它，后序 Agent 看不到前序任何输出。
    编排模式负责按依赖把前序结果注入 `notes`，Agent 自己不猜测上下文。
    """

    agent: str
    task: str
    scenario: str | None = None
    notes: str = ""
    depends_on: list[str] = field(default_factory=list)
    confirmed: bool = False
    max_steps: int | None = None


@dataclass(slots=True)
class SubTaskResult:
    """一个子任务的结构化结果（编排结果里每一项都长这样）。"""

    id: str
    agent: str
    ok: bool
    answer: str | None
    stop_reason: str
    steps: int
    usage: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    handoff: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "ok": self.ok,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "handoff": self.handoff,
        }


@dataclass(slots=True)
class OrchestrationEvent:
    """编排事件流的最小单元。三种模式共用，前端可据此画时序图。"""

    step: int
    kind: str
    agent: str | None = None
    status: str = "ok"
    detail: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"step": self.step, "kind": self.kind, "status": self.status}
        if self.agent is not None:
            payload["agent"] = self.agent
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


@dataclass(slots=True)
class RoutePlan:
    """复杂度路由的产物。

    `kind=direct` 表示单域查询直连白名单 Tool（**不产生 Agent 推理**，`llm_calls=0`）；
    `kind=agent` 表示跨域/多步任务，需要进 Agent 循环，带出编排模式与子任务列表。
    """

    kind: str
    reason: str
    domain: str | None = None
    tool: str | None = None
    tool_args: dict[str, Any] | None = None
    mode: str | None = None
    tasks: list[AgentInvocation] = field(default_factory=list)
    llm_calls: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "domain": self.domain,
            "tool": self.tool,
            "mode": self.mode,
            "llm_calls": self.llm_calls,
            "tasks": [
                {"agent": task.agent, "scenario": task.scenario, "depends_on": task.depends_on} for task in self.tasks
            ],
        }


@dataclass(slots=True)
class GroundingVerdict:
    """Grounding Gate 的判定。

    `enforced=True` 表示原回答被**改写**（拒答或降级）—— 交易类无来源结论不允许原样返回。
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)
    sanitized_answer: str | None = None
    enforced: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "enforced": self.enforced, "reasons": self.reasons}


@dataclass(slots=True)
class OrchestrationResult:
    """一次编排调用的对外结果契约（三种模式统一）。"""

    success: bool
    answer: str | None
    stop_reason: str
    kind: str
    mode: str | None
    steps: int
    usage: Usage
    subtasks: list[SubTaskResult] = field(default_factory=list)
    events: list[OrchestrationEvent] = field(default_factory=list)
    grounding: dict[str, Any] | None = None
    route: dict[str, Any] | None = None
    handoff: dict[str, Any] | None = None
    trace_id: str = ""
    duration_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "kind": self.kind,
            "mode": self.mode,
            "steps": self.steps,
            "usage": usage_payload(self.usage),
            "subtasks": [task.to_payload() for task in self.subtasks],
            "events": [event.to_payload() for event in self.events],
            "grounding": self.grounding,
            "route": self.route,
            "handoff": self.handoff,
            "trace_id": self.trace_id,
            "duration_ms": round(self.duration_ms, 2),
        }


__all__ = [
    "AgentInvocation",
    "SubTaskResult",
    "OrchestrationEvent",
    "RoutePlan",
    "GroundingVerdict",
    "OrchestrationResult",
    "usage_payload",
]
