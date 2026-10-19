"""Customer Agent：消费者侧的四类任务（肤质测试 / 成分咨询 / 方案推荐 / 订单售后）。

它做的事只有三件，且都必须在**进模型之前**完成：
1) **场景判定**：四类任务的 Prompt 片段、可用工具、风险策略都不同，先判场景再组装；
2) **上下文回填**：会话历史 + 用户画像（复用 infra/memory 的 token 预算与摘要策略），
   并把「上游 Agent 结论」/「历史内容」统一过一遍围栏，避免间接注入与跨用户污染；
3) **交给共享 Harness 跑**：本 Agent **自己不含 Agent Loop** —— 循环、护栏、校验、
   轨迹、交接全在 harness 里。项目一与项目二共用同一条执行链路，行为才不会分叉。

工具白名单也在这里收口：消费者侧只允许「只读查询 + 画像写入 + 下单（高风险，需确认）」，
运营侧的报表工具不会暴露给它。
"""

from __future__ import annotations

import re
from typing import Any

from src.harness.engine import Harness, HarnessResult
from src.infra.core.logging import get_logger
from src.infra.prompts.registry import PromptRegistry
from src.infra.tools.base import ToolContext

from .prompt_guard import PromptGuard
from .schemas import AgentInvocation
from .session import SessionManager

logger = get_logger("agents.customer")

_ORDER_STATUS = re.compile(r"(订单|物流|发货|到哪|签收|运单)", re.IGNORECASE)
_AFTER_SALES = re.compile(r"(退货|退款|换货|售后|投诉|补发|赔付)")
_SKIN_TEST = re.compile(r"(肤质|敏感肌|干皮|油皮|混合皮|痘痘肌|测试)")


class CustomerAgent:
    name = "customer"
    SCENARIOS = ("product_consulting", "skin_test", "after_sales", "order_status")
    # 消费者侧白名单：读商品/知识/订单 + 写画像 + 下单（高风险，须显式确认）
    DEFAULT_TOOLS = (
        "search_products",
        "get_product_detail",
        "get_order_status",
        "search_knowledge",
        "get_conversation_history",
        "update_user_profile",
        "create_order",
    )

    def __init__(
        self,
        *,
        harness: Harness,
        prompts: PromptRegistry,
        sessions: SessionManager,
        guard: PromptGuard,
        max_steps: int,
        tools: tuple[str, ...] | None = None,
    ) -> None:
        self._harness = harness
        self._prompts = prompts
        self._sessions = sessions
        self._guard = guard
        self._max_steps = max_steps
        self._tools = list(tools or self.DEFAULT_TOOLS)

    async def handle(self, invocation: AgentInvocation, ctx: ToolContext) -> HarnessResult:
        scenario = invocation.scenario if invocation.scenario in self.SCENARIOS else self.classify(invocation.task)
        max_steps = invocation.max_steps or self._max_steps
        base_system = self._prompts.compose("customer", scenario=scenario, max_steps=max_steps)

        session_id = ctx.session_id or "default"
        snapshot = await self._sessions.rehydrate(
            project_id=ctx.project_id,
            session_id=session_id,
            user_id=ctx.user_id,
            current_message=invocation.task,
            system_prompt=base_system,
        )
        # 首条 system 是 rehydrate 组装好的成品（已 harden）；画像单独补在末尾
        system_prompt = snapshot.messages[0]["content"]
        if snapshot.user_profile:
            system_prompt = f"{system_prompt}{self._profile_block(snapshot.user_profile)}"

        history = list(snapshot.messages[1:-1])  # 去掉首条 system 与末条当前问题
        if invocation.notes:
            history.append(
                {
                    "role": "system",
                    "content": self._guard.fence(invocation.notes, source="upstream_agent", scope=snapshot.scope),
                }
            )

        logger.info(
            "agents.customer.dispatch",
            extra={"event": "agent_run", "reason": scenario, "tool": ",".join(self._tools)},
        )
        return await self._harness.run(
            task=invocation.task,
            ctx=ctx,
            max_steps=max_steps,
            tool_names=self._tools,
            confirmed=invocation.confirmed,
            system_prompt=system_prompt,
            history=history,
        )

    @staticmethod
    def classify(task: str) -> str:
        """规则优先的场景判定：确定性、零成本、可解释。

        规则命中不了才落到默认「成分/方案咨询」—— 这一步不做模型调用，
        因为场景判定只影响 Prompt 组装，判错的代价远小于每轮多一次 LLM 调用。
        """
        if _AFTER_SALES.search(task):
            return "after_sales"
        if _ORDER_STATUS.search(task):
            return "order_status"
        if _SKIN_TEST.search(task):
            return "skin_test"
        return "product_consulting"

    @staticmethod
    def _profile_block(profile: dict[str, Any]) -> str:
        """把用户画像作为**服务端可信事实**拼进系统提示（不入围栏：它不是外部内容）。"""
        items = "；".join(f"{key}={value}" for key, value in sorted(profile.items()))
        return f"\n\n[已知用户画像]\n{items}"


__all__ = ["CustomerAgent"]
