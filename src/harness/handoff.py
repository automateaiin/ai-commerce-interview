"""终止即交接：把运行状态打包给「人」或「上层 Agent」。

为什么终止一定要带交接包（这是 Agent 落地最容易被忽视的地方）：
Agent 的失败与普通接口的失败不一样 —— 它通常已经**做了一半**（查了商品、锁了库存）。
如果只是返回一个 500，接手的业务人员必须从零重问一遍，前面几轮的 token 全白花。

所以交接包必须能回答四个问题：
1) 用户到底要什么（summary）；
2) 已经做到哪一步（completed_steps + tool_results）；
3) 为什么停（reason + failure）；
4) 接手后建议怎么做（next_actions）。

它不是日志，而是**给下一位执行者的输入**，所以字段是稳定的结构化契约，不是自由文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.infra.core.logging import scrub


class HandoffReason(str, Enum):
    """需要人工/上层接管的典型原因。"""

    BUDGET = "budget"
    """Token/成本预算耗尽：不是错误，是「钱花完了但任务没完」。"""

    ERROR_THRESHOLD = "error_threshold"
    """连续失败熔断：工具或环境坏了，继续调用只会烧预算。"""

    PERMISSION = "permission"
    """权限不足：需要业务人员授权或换凭证，模型自己解决不了。"""

    HIGH_RISK = "high_risk_confirmation"
    """高风险动作待确认：下单、退款这类有业务副作用的动作不能由模型拍板。"""

    LOOP_DETECTED = "loop_detected"
    """检测到打转：模型在重复同一调用，继续下去也不会收敛。"""

    INVALID_DECISION = "invalid_decision"
    """模型返回脏输出：没有可执行的决策。"""

    NO_TOOLS = "no_tools"
    """当前身份下没有可用工具：配置/权限问题，重试无意义。"""


@dataclass(slots=True)
class HandoffPacket:
    reason: str
    summary: str
    completed_steps: int
    trace_id: str
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    failure: str | None = None
    next_actions: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "summary": scrub(self.summary),
            "completed_steps": self.completed_steps,
            "tool_results": scrub(self.tool_results),
            "failure": self.failure,
            "next_actions": self.next_actions,
            "trace_id": self.trace_id,
        }
