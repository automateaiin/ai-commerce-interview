"""Operations Agent：品牌运营侧（经营分析 / 客服质检 / 商品卖点 / 运营报告）。

与 Customer Agent 的关键差别在于**数据来源**：
消费者侧的一切事实都走白名单 Tool（工具即权限边界）；
运营侧的经营指标属于**内部只读分析**，通过 `OpsDataPort` 直接取业务库聚合结果 ——
不用 Text-to-SQL，也不给模型一条能跑到用户私有数据的通道。

指标先取回来、再作为 `notes` 注入，让模型只做「解释与成文」，
而不是让它自己去想「该查什么表」：后者既不可控（SQL 注入/越权），又不可复现。
"""

from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

from src.harness.engine import Harness, HarnessResult
from src.infra.core.logging import get_logger
from src.infra.prompts.registry import PromptRegistry
from src.infra.tools.base import ToolContext

from .prompt_guard import PromptGuard
from .schemas import AgentInvocation

logger = get_logger("agents.operations")


class OpsDataPort(Protocol):
    """运营侧只读数据端口（由组合根注入 commerce 实现，依赖倒置）。"""

    async def order_stats(self, *, days: int) -> dict[str, Any]: ...
    async def low_stock(self) -> list[dict[str, Any]]: ...
    async def top_catalog(self, *, limit: int) -> list[dict[str, Any]]: ...


class OperationsAgent:
    name = "operations"
    SCENARIOS = ("analysis", "report", "review")
    DEFAULT_TOOLS = ("search_knowledge", "search_products", "get_product_detail")

    def __init__(
        self,
        *,
        harness: Harness,
        prompts: PromptRegistry,
        data: OpsDataPort,
        guard: PromptGuard,
        max_steps: int,
        tools: Sequence[str] | None = None,
    ) -> None:
        self._harness = harness
        self._prompts = prompts
        self._data = data
        self._guard = guard
        self._max_steps = max_steps
        self._tools = list(tools or self.DEFAULT_TOOLS)

    async def handle(self, invocation: AgentInvocation, ctx: ToolContext) -> HarnessResult:
        scenario = invocation.scenario if invocation.scenario in self.SCENARIOS else "analysis"
        max_steps = invocation.max_steps or self._max_steps
        role = "inspector" if scenario == "review" else "operations"
        system_prompt = self._guard.harden(
            self._prompts.compose(role, scenario=scenario, max_steps=max_steps)
        )

        evidence = await self._collect_evidence(scenario, invocation.task)
        notes = self._compose_notes(evidence, invocation.notes, ctx)
        history = [{"role": "system", "content": notes}] if notes else None

        logger.info(
            "agents.operations.dispatch",
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

    async def _collect_evidence(self, scenario: str, task: str) -> dict[str, Any]:
        """按场景取指标。质检场景不需要经营指标（它分析的是会话本身）。"""
        if scenario == "review":
            return {}
        evidence: dict[str, Any] = {"order_stats": await self._data.order_stats(days=30)}
        evidence["low_stock"] = await self._data.low_stock()
        if scenario == "report":
            evidence["top_products"] = await self._data.top_catalog(limit=5)
        return evidence

    def _compose_notes(self, evidence: dict[str, Any], upstream: str, ctx: ToolContext) -> str:
        blocks: list[str] = []
        if upstream:
            blocks.append(self._guard.fence(upstream, source="upstream_agent", scope=ctx.project_id))
        if evidence:
            blocks.append(
                self._guard.fence(
                    json.dumps(evidence, ensure_ascii=False), source="commerce_metrics", scope=ctx.project_id
                )
            )
        return "\n\n".join(blocks)


__all__ = ["OperationsAgent", "OpsDataPort"]
