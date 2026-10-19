"""Orchestrator：复杂度路由 + 分派。

为什么要有它：把一个「能聊天的 Agent」做成「能扛住成本与延迟的系统」，关键在**分流**。
不是每个问题都值得走一遍多步 Agent 循环：

- **单域查询**（给一个订单号问状态、给一个商品号问详情、问一条政策）→ 直连白名单 Tool 取数，
  **零模型推理**。这类请求占多数，全部塞进 Agent 循环会让成本与 P95 延迟无意义地上涨；
- **跨域/多步任务**（「这个精华适合我吗，顺便看下有没有优惠，再帮我下单」）→ 才进 Agent，
  并按依赖关系在 Sequential / Fan-out / Handoff 之间选一种模式。

分流本身也是分层的：**先规则、后模型**。
规则能判的（有订单号/商品号、明确的知识问题）绝不多花一次 LLM 调用；
只有规则判不了才让模型做规划（`llm_calls=1`），且模型给出的计划要**逐项校验**
（Agent 是否存在、场景是否合法、模式是否注册）—— 模型可以提议，但不能发明执行路径。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Awaitable, Callable, Mapping

from src.harness.engine import HarnessResult
from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage
from src.infra.llm.types import Usage
from src.infra.tools.base import ToolContext, ToolExecutor

from .grounding import GroundingGate
from .orchestration import AgentRunner, ModeRegistry
from .schemas import AgentInvocation, OrchestrationResult, RoutePlan, SubTaskResult, usage_payload

logger = get_logger("agents.orchestrator")

_ORDER_ID = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
_PRODUCT_ID = re.compile(r"\bSKU-\d+\b", re.IGNORECASE)
_CROSS_DOMAIN = re.compile(r"(并且|同时|然后|接着|再帮我|还要|顺便|以及|另外)")
_KNOWLEDGE_INTENT = re.compile(r"(成分|功效|作用|使用方法|怎么用|搭配|政策|退换|有效期|注意事项)")
_SHOP_INTENT = re.compile(r"(推荐|买|购买|预算|下单|有货|现货|价格|多少钱|优惠|折扣)")
_OPS_INTENT = re.compile(r"(经营|报表|复盘|质检|卖点|转化|销量|GMV|客诉|库存预警|运营报告)")

_AGENT_NAMES = ("customer", "operations")
_MODES = ("sequential", "fanout", "handoff")

_PLANNER_SYSTEM = """你是编排规划器。根据用户任务，输出**唯一一个 JSON 对象**（不要解释、不要代码围栏）：
{"domain": "customer|operations", "mode": "sequential|fanout|handoff", "tasks": [{"agent": "customer|operations", "scenario": "...", "task": "...", "depends_on": ["agent名"]}]}
规则：
1) 无依赖、可并行的子任务用 fanout；有前后依赖用 sequential；需要转派专家（高风险/越权/缺信息）用 handoff；
2) 每个子任务的 task 要自洽可执行；depends_on 只填真正的上游 agent 名，没有就留空数组；
3) 单域问题也要给出 1 个任务。
可选 scenario：customer: product_consulting|skin_test|after_sales|order_status；operations: analysis|report|review。"""


class ComplexityRouter:
    """规则优先、模型兜底的路由器。"""

    def __init__(
        self,
        *,
        gateway: Any,
        modes: ModeRegistry,
        agents: Mapping[str, Any] | None = None,
        planner_model: str | None = None,
        max_tasks: int = 4,
    ) -> None:
        self._gateway = gateway
        self._modes = modes
        self._agents = dict(agents or {})
        self._planner_model = planner_model
        self._max_tasks = max_tasks

    async def plan(
        self,
        task: str,
        *,
        scenario_hint: str | None = None,
        mode_hint: str | None = None,
        domain_hint: str | None = None,
    ) -> RoutePlan:
        if mode_hint:
            # 调用方显式指定模式：跳过规则，但仍要校验合法性与补齐子任务
            return self._explicit(task, scenario_hint, mode_hint, domain_hint)

        rule = self._rule_route(task)
        if rule is not None:
            return rule
        return await self._model_route(task, scenario_hint=scenario_hint, domain_hint=domain_hint)

    # -- 规则层 ------------------------------------------------------------- #
    def _rule_route(self, task: str) -> RoutePlan | None:
        cross = bool(_CROSS_DOMAIN.search(task))
        order = _ORDER_ID.search(task)
        product = _PRODUCT_ID.search(task)

        if order and not cross:
            return RoutePlan(
                kind="direct",
                reason="rule:order_id",
                domain="commerce",
                tool="get_order_status",
                tool_args={"order_id": order.group(0)},
            )
        if product and not cross:
            return RoutePlan(
                kind="direct",
                reason="rule:product_id",
                domain="commerce",
                tool="get_product_detail",
                tool_args={"product_id": product.group(0)},
            )
        if not cross and _KNOWLEDGE_INTENT.search(task) and not _SHOP_INTENT.search(task):
            return RoutePlan(
                kind="direct",
                reason="rule:knowledge_question",
                domain="knowledge",
                tool="search_knowledge",
                tool_args={"query": task, "top_k": 3},
            )
        return None

    # -- 显式模式 ----------------------------------------------------------- #
    def _explicit(self, task: str, scenario: str | None, mode: str, domain: str | None) -> RoutePlan:
        if not self._modes.has(mode):
            raise ValueError(f"未知编排模式: {mode}")
        agent = self._normalize_agent(domain) or ("operations" if _OPS_INTENT.search(task) else "customer")
        return RoutePlan(
            kind="agent",
            reason=f"explicit:{mode}",
            domain=agent,
            mode=mode,
            tasks=[AgentInvocation(agent=agent, task=task, scenario=scenario)],
        )

    # -- 模型规划层 --------------------------------------------------------- #
    async def _model_route(self, task: str, *, scenario_hint: str | None, domain_hint: str | None) -> RoutePlan:
        planner_messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": task},
        ]
        plan: dict[str, Any] | None = None
        try:
            async with stage("agents.route_plan"):
                response = await self._gateway.complete(planner_messages, logical_model=self._planner_model)
            plan = _parse_plan(response.text)
        except Exception as exc:  # noqa: BLE001 - 规划失败必须降级而不是让整个请求失败
            logger.warning(
                "agents.router.plan_failed",
                extra={"event": "route", "status": "degraded", "reason": type(exc).__name__},
            )
            plan = None

        if plan is None:
            return self._heuristic(task, scenario_hint, domain_hint, reason="fallback:planner_unparsed")

        tasks = self._sanitize_tasks(plan.get("tasks"), task, scenario_hint, domain_hint)
        mode = plan.get("mode") if plan.get("mode") in _MODES else "sequential"
        domain = self._normalize_agent(plan.get("domain"))
        return RoutePlan(kind="agent", reason="model:planner", domain=domain, mode=mode, tasks=tasks, llm_calls=1)

    def _heuristic(self, task: str, scenario_hint: str | None, domain_hint: str | None, *, reason: str) -> RoutePlan:
        domain = self._normalize_agent(domain_hint) or ("operations" if _OPS_INTENT.search(task) else "customer")
        if domain == "operations" and not scenario_hint:
            scenario_hint = "report" if re.search(r"(报告|复盘)", task) else "analysis"
        return RoutePlan(
            kind="agent",
            reason=reason,
            domain=domain,
            mode="sequential",
            tasks=[AgentInvocation(agent=domain, task=task, scenario=scenario_hint)],
        )

    def _sanitize_tasks(
        self, raw: Any, task: str, scenario_hint: str | None, domain_hint: str | None
    ) -> list[AgentInvocation]:
        """逐项校验模型给出的计划：Agent 必须存在、依赖必须指向真实上游、数量要有上限。"""
        if not isinstance(raw, list) or not raw:
            return self._heuristic(task, scenario_hint, domain_hint, reason="fallback:no_tasks").tasks
        tasks: list[AgentInvocation] = []
        seen: set[str] = set()
        for item in raw[: self._max_tasks]:
            if not isinstance(item, dict):
                continue
            agent = self._normalize_agent(item.get("agent"))
            if agent is None:
                continue
            depends = [dep for dep in _as_list(item.get("depends_on")) if dep in seen]
            tasks.append(
                AgentInvocation(
                    agent=agent,
                    task=str(item.get("task") or task),
                    scenario=item.get("scenario") if isinstance(item.get("scenario"), str) else scenario_hint,
                    depends_on=depends,
                )
            )
            seen.add(agent)
        if not tasks:
            return self._heuristic(task, scenario_hint, domain_hint, reason="fallback:no_valid_tasks").tasks
        return tasks

    def _normalize_agent(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return normalized if normalized in _AGENT_NAMES else None


class Orchestrator:
    """把路由结果落成执行：direct 直连 Tool，agent 走编排模式。"""

    def __init__(
        self,
        *,
        router: ComplexityRouter,
        modes: ModeRegistry,
        runners: Mapping[str, Callable[[AgentInvocation, ToolContext], Awaitable[HarnessResult]]],
        executor: ToolExecutor,
        grounding: GroundingGate,
    ) -> None:
        self._router = router
        self._modes = modes
        self._runners = dict(runners)
        self._executor = executor
        self._grounding = grounding

    async def route(self, task: str, **hints: Any) -> RoutePlan:
        """只路由不执行（调试/可观测用）。"""
        return await self._router.plan(task, **hints)

    async def handle(
        self,
        task: str,
        ctx: ToolContext,
        *,
        scenario: str | None = None,
        mode: str | None = None,
        domain: str | None = None,
        confirmed: bool = False,
    ) -> OrchestrationResult:
        started = time.perf_counter()
        plan = await self._router.plan(task, scenario_hint=scenario, mode_hint=mode, domain_hint=domain)
        if confirmed:
            plan.tasks = [
                AgentInvocation(
                    agent=item.agent,
                    task=item.task,
                    scenario=item.scenario,
                    notes=item.notes,
                    depends_on=item.depends_on,
                    confirmed=True,
                    max_steps=item.max_steps,
                )
                for item in plan.tasks
            ]

        if plan.kind == "direct":
            result = await self._direct(plan, ctx)
        else:
            result = await self._agentic(plan, ctx)

        result.route = plan.to_payload()
        result.kind = plan.kind
        result.trace_id = ctx.trace_id
        result.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "agents.orchestrator.completed",
            extra={
                "event": "orchestration",
                "reason": plan.reason,
                "status": "ok" if result.success else "error",
                "duration_ms": round(result.duration_ms, 2),
            },
        )
        return result

    # -- 直连路径 ----------------------------------------------------------- #
    async def _direct(self, plan: RoutePlan, ctx: ToolContext) -> OrchestrationResult:
        tool = plan.tool or ""
        tool_result = await self._executor.execute(tool, dict(plan.tool_args or {}), ctx)
        answer = _render_direct(tool, tool_result)
        verdict = self._grounding.check(
            answer=answer,
            tool_results=[{"tool": tool, "ok": tool_result.ok, "data": tool_result.data}],
        )
        subtask = SubTaskResult(
            id=f"{tool}",
            agent="direct_tool",
            ok=tool_result.ok,
            answer=answer,
            stop_reason="direct_tool" if tool_result.ok else (tool_result.error_code or "tool_error"),
            steps=0,
            usage=usage_payload(Usage()),
            tool_calls=[{"id": tool, "name": tool, "ok": tool_result.ok, "error_code": tool_result.error_code}],
        )
        return OrchestrationResult(
            success=tool_result.ok,
            answer=verdict.sanitized_answer,
            stop_reason="direct_tool" if tool_result.ok else (tool_result.error_code or "tool_error"),
            kind="direct",
            mode=None,
            steps=0,
            usage=Usage(),
            subtasks=[subtask],
            grounding=verdict.to_payload(),
        )

    # -- Agent 路径 --------------------------------------------------------- #
    async def _agentic(self, plan: RoutePlan, ctx: ToolContext) -> OrchestrationResult:
        mode_name = plan.mode or "sequential"
        mode = self._modes.get(mode_name)
        outcome = await mode.run(tasks=plan.tasks, ctx=ctx, runner=self._runner)

        final_answer = _final_answer(outcome.subtasks)
        evidence = [
            {"tool": call.get("name"), "ok": bool(call.get("ok"))}
            for task in outcome.subtasks
            for call in task.tool_calls
        ]
        verdict = self._grounding.check(
            answer=final_answer,
            tool_results=evidence,
            intent=plan.tasks[0].scenario if plan.tasks else None,
        )

        answer = verdict.sanitized_answer
        stop_reason = "grounding_blocked" if verdict.enforced else outcome.stop_reason
        if answer is None and outcome.handoff is None:
            stop_reason = stop_reason or "handoff_no_target"
        return OrchestrationResult(
            success=answer is not None and not verdict.enforced,
            answer=answer,
            stop_reason=stop_reason,
            kind="agent",
            mode=mode_name,
            steps=sum(task.steps for task in outcome.subtasks),
            usage=_aggregate(outcome.subtasks),
            subtasks=outcome.subtasks,
            events=outcome.events,
            grounding=verdict.to_payload(),
            handoff=outcome.handoff,
        )

    async def _runner(self, invocation: AgentInvocation, ctx: ToolContext) -> HarnessResult:
        runner = self._runners.get(invocation.agent)
        if runner is None:
            raise ValueError(f"未注册的 Agent: {invocation.agent}")
        return await runner(invocation, ctx)


def _aggregate(subtasks: list[SubTaskResult]) -> Usage:
    """编排成本 = 各子任务之和（含 fan-out 的并行分支与 fan-in 的收敛）。"""
    total = Usage()
    for task in subtasks:
        usage = task.usage
        total = total.merged_with(
            Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                cost_usd=float(usage.get("cost_usd", 0.0)),
            )
        )
    return total


def _final_answer(subtasks: list[SubTaskResult]) -> str | None:
    for task in reversed(subtasks):
        if task.answer:
            return task.answer
    return None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_plan(text: str | None) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON。

    模型常带代码围栏或前后解释，所以先剥围栏，再取第一个完整 `{...}` 片段，
    避免因为多一句寒暄就判成「规划失败」。
    """
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _render_direct(tool: str, result: Any) -> str:
    """直连路径的答案渲染：确定性、无模型参与。

    直连本身就是「不花推理」的设计，答案就不能再依赖模型生成 ——
    这里用模板把结构化结果转成人话，数字完全来自本次 Tool Result。
    """
    if not result.ok:
        return f"查询未成功：{result.error_message or result.error_code or '下游不可用'}。"
    data = result.data
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)

    if tool == "get_order_status":
        return (
            f"订单 {data.get('order_id', '-')} 当前状态：{data.get('status', '未知')}"
            f"；承运商：{data.get('carrier') or '待分配'}；金额：{data.get('total_usd', data.get('total', '-'))}。"
        )
    if tool == "get_product_detail":
        stock = data.get("stock", data.get("available"))
        return (
            f"{data.get('name', data.get('product_id', '-'))}（{data.get('product_id', '-')}）："
            f"售价 {data.get('price_usd', data.get('price', '-'))}，当前库存 {stock if stock is not None else '未知'}。"
        )
    if tool == "search_knowledge":
        chunks = data.get("chunks") or []
        if not chunks:
            return "知识库没有检索到可用证据，不做推断。"
        top = chunks[0]
        return f"检索到 {len(chunks)} 条证据，最相关：{top.get('title') or top.get('chunk_id')}。"
    if tool == "search_products":
        items = data.get("items") or []
        names = "、".join(str(item.get("name")) for item in items[:3]) or "无"
        return f"检索到 {len(items)} 个商品：{names}。"
    return json.dumps(data, ensure_ascii=False)


__all__ = ["Orchestrator", "ComplexityRouter"]
