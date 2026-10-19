"""运营工作流的节点契约与运行记录。

为什么节点输出必须是 Pydantic 模型（而不是让节点返回一段自由文本）：
- 下游节点要用上游的**字段**，而不是把上游整段文字塞进 prompt —— 后者会让依赖关系
  退化成字符串拼接：上游改一句措辞，下游就悄悄坏掉，且没有任何地方报错；
- 结构化输出可以被程序校验：不合法就重试 / 回退 / 记录原因，自由文本只能靠人眼看。

运行记录（NodeRecord）单独建模，是因为它同时服务三种读者：
Checkpointer 用它做续跑判定、接口用它做回放、排障用它看第几次尝试失败的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# 节点输出契约（Structured Output 的 schema 来源）
# --------------------------------------------------------------------------- #
class TopicPlan(BaseModel):
    """选题节点输出：候选选题 + 选择理由。"""

    model_config = ConfigDict(extra="forbid")

    topics: list[str] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=1, max_length=500)


class SellingPoint(BaseModel):
    """单条成分卖点：claim 必须能挂到证据切片上，否则会被合规校验拦下。"""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=300)
    evidence: list[str] = Field(default_factory=list)
    compliance: str = Field(default="ok", max_length=32)


class SellingPoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[SellingPoint] = Field(min_length=1, max_length=10)


class CopyDraft(BaseModel):
    """短文案：轻量模型负责的那条路径。"""

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=1200)
    hashtags: list[str] = Field(default_factory=list, max_length=12)
    cta: str = Field(default="", max_length=120)


class ImageBrief(BaseModel):
    """产品图生成提示词：本仓库只产出提示词与尺寸，不真的渲染。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=1000)
    aspect: str = Field(default="portrait_4_3", max_length=32)
    negative: str = Field(default="", max_length=300)


# --------------------------------------------------------------------------- #
# 运行记录
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class NodeAttempt:
    """一次节点尝试（重试与回退的最小留痕单位）。"""

    attempt: int
    status: str
    logical_model: str | None = None
    error: str | None = None
    fallback_reason: str | None = None
    duration_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "status": self.status,
            "logical_model": self.logical_model,
            "error": self.error,
            "fallback_reason": self.fallback_reason,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass(slots=True)
class NodeRecord:
    """节点级状态：Checkpointer 的持久化单位。"""

    node: str
    status: str = NodeStatus.PENDING.value
    attempt: int = 0
    output: dict[str, Any] | None = None
    error: str | None = None
    logical_model: str | None = None
    fallback_reason: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    history: list[NodeAttempt] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.started_at is None or self.finished_at is None:
            return 0.0
        return round((self.finished_at - self.started_at) * 1000, 2)

    def to_payload(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "status": self.status,
            "attempt": self.attempt,
            "logical_model": self.logical_model,
            "output": self.output,
            "error": self.error,
            "fallback_reason": self.fallback_reason,
            "attempts": [item.to_payload() for item in self.history],
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True)
class WorkflowResult:
    """一次工作流运行的对外结果。"""

    run_id: str
    workflow: str
    status: str
    records: list[NodeRecord]
    artifacts: dict[str, Any]
    resumed_from: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow": self.workflow,
            "status": self.status,
            "resolved": [record.node for record in self.records if record.status == NodeStatus.SUCCEEDED.value],
            "resumed_from": list(self.resumed_from),
            "records": [record.to_payload() for record in self.records],
            "artifacts": self.artifacts,
            "duration_ms": round(self.duration_ms, 2),
        }


__all__ = [
    "CopyDraft",
    "ImageBrief",
    "NodeAttempt",
    "NodeRecord",
    "NodeStatus",
    "SellingPoint",
    "SellingPoints",
    "TopicPlan",
    "WorkflowResult",
]
