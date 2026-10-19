"""项目二：通用 Harness / Agent Loop 的三重护栏 + 校验 + 交接 + 轨迹。

这里的循环不是「一次 LLM 调用」，而是 决策→校验→派发→观察 的循环。
真正决定能不能上线的是三件事，逐条单独打出来：
1) 三重护栏（静态上限 / 打转检测 / 连续失败熔断）；
2) 派发前校验（未知工具、参数非法、数量上限 → 全部回填成可自纠错的 observation）；
3) 终止即交接（高风险确认、越权、预算耗尽都要带 HandoffPacket，而不是静默停止）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.harness.engine import Harness
from src.harness.guardrails import Guardrails
from src.harness.observation import ObservationBuilder
from src.harness.state import RunState
from src.harness.trace import ExecutionTrace
from src.infra.core.errors import ForbiddenError, NotFoundError
from src.infra.core.settings import Settings
from src.infra.llm.mock import MockProvider, calls, final
from src.infra.llm.types import Usage
from src.infra.prompts.registry import build_default_registry
from src.infra.tools.base import FunctionTool, ToolContext, ToolExecutor, ToolRegistry, ToolRisk


class _SearchArgs(BaseModel):
    query: str


class _OrderArgs(BaseModel):
    product_id: str
    quantity: int = 1
    idempotency_key: str


class _StubGateway:
    """只实现 Harness 需要的那一个方法，并记录每次暴露给模型的工具集。"""

    def __init__(self, provider: MockProvider) -> None:
        self.provider = provider
        self.seen_tools: list[Any] = []

    async def complete(self, messages, *, logical_model=None, task=None, tools=None, **_):  # noqa: ANN001
        self.seen_tools.append(tools)
        return await self.provider.complete(messages, "stub-model", tools=tools)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    async def search_products(args: _SearchArgs, ctx: ToolContext) -> dict[str, Any]:
        return {"items": [{"product_id": "SKU-1001", "name": "雪绒花修护精华"}], "query": args.query}

    async def search_knowledge(args: _SearchArgs, ctx: ToolContext) -> dict[str, Any]:
        return {"chunks": [{"chunk_id": "kb_001", "text": args.query}]}

    async def create_order(args: _OrderArgs, ctx: ToolContext) -> dict[str, Any]:
        return {"order_id": "ORD-1", "idempotency_key": args.idempotency_key}

    async def flaky(args: _SearchArgs, ctx: ToolContext) -> dict[str, Any]:
        raise NotFoundError("下游依赖不可用", detail={"resource": args.query})

    async def guarded(args: _SearchArgs, ctx: ToolContext) -> dict[str, Any]:
        raise ForbiddenError("该资源不允许当前身份访问", detail={"resource": args.query})

    registry.register(
        FunctionTool(
            name="search_products",
            description="按关键词检索在售商品",
            args_model=_SearchArgs,
            handler=search_products,
            requires_scope="tools:read",
            risk=ToolRisk.READ_ONLY,
        )
    )
    registry.register(
        FunctionTool(
            name="guarded",
            description="业务级越权：返回 ForbiddenError",
            args_model=_SearchArgs,
            handler=guarded,
            requires_scope="tools:read",
            risk=ToolRisk.READ_ONLY,
        )
    )
    registry.register(
        FunctionTool(
            name="search_knowledge",
            description="检索商品知识库",
            args_model=_SearchArgs,
            handler=search_knowledge,
            requires_scope="tools:knowledge",
            risk=ToolRisk.READ_ONLY,
        )
    )
    registry.register(
        FunctionTool(
            name="create_order",
            description="创建待付款订单（写操作，必须带幂等键）",
            args_model=_OrderArgs,
            handler=create_order,
            requires_scope="tools:write",
            risk=ToolRisk.HIGH_RISK,
            parallel_safe=False,
        )
    )
    registry.register(
        FunctionTool(
            name="flaky",
            description="会持续失败的下游调用",
            args_model=_SearchArgs,
            handler=flaky,
            requires_scope="tools:read",
            risk=ToolRisk.READ_ONLY,
        )
    )
    return registry


def _harness(**overrides: Any) -> tuple[Harness, _StubGateway, MockProvider]:
    provider = MockProvider()
    gateway = _StubGateway(provider)
    registry = _registry()
    guardrails = Guardrails(
        max_steps=overrides.pop("max_steps", 6),
        deadline_s=overrides.pop("deadline_s", 10.0),
        token_budget=overrides.pop("token_budget", 20000),
        cost_budget_usd=overrides.pop("cost_budget_usd", 0.0),
        max_consecutive_errors=overrides.pop("max_consecutive_errors", 3),
        repeat_window_steps=overrides.pop("repeat_window_steps", 3),
        repeat_exempt_tools=frozenset(overrides.pop("repeat_exempt_tools", ())),
        max_calls_per_turn=overrides.pop("max_calls_per_turn", 4),
        max_observation_chars=overrides.pop("max_observation_chars", 1500),
    )
    harness = Harness(
        gateway=gateway,  # type: ignore[arg-type]
        registry=registry,
        executor=ToolExecutor(registry, max_concurrency=4),
        prompts=build_default_registry(),
        guardrails=guardrails,
    )
    return harness, gateway, provider


def _ctx(*scopes: str) -> ToolContext:
    return ToolContext(trace_id="tr_harness_test", project_id="cs-bot", session_id="s1", user_id="u1", scopes=frozenset(scopes))


# --------------------------------------------------------------------------- #
# 正常路径：循环与结果契约
# --------------------------------------------------------------------------- #
async def test_multi_step_loop_returns_structured_contract():
    harness, _, provider = _harness()
    provider.reset(
        [
            calls(("search_products", {"query": "精华"})),
            calls(("search_knowledge", {"query": "成分"})),
            final("雪绒花修护精华适合敏感肌。"),
        ]
    )

    result = await harness.run(task="给敏感肌用户推荐一款精华", ctx=_ctx("tools:read", "tools:knowledge"))

    assert result.stop_reason == "completed"
    assert result.answer == "雪绒花修护精华适合敏感肌。"
    assert result.steps == 3  # 两次决策调用工具 + 一次终答
    assert [call["name"] for call in result.tool_calls] == ["search_products", "search_knowledge"]
    assert result.usage.total_tokens > 0
    assert result.handoff is None
    # 结果契约必须能支撑「计费 / 前端可视化 / 排障」三类读者
    payload = result.to_payload(trace_id="tr_x", duration_ms=1.0)
    assert payload["success"] is True
    assert payload["trace"]["stop_reason"] == "completed"


async def test_read_only_tools_in_one_turn_are_executed_together():
    harness, _, provider = _harness()
    provider.reset(
        [
            calls(("search_products", {"query": "精华"}), ("search_knowledge", {"query": "成分"})),
            final("综合商品与知识给出结论。"),
        ]
    )

    result = await harness.run(task="同时查商品与知识", ctx=_ctx("tools:read", "tools:knowledge"))

    assert [call["name"] for call in result.tool_calls] == ["search_products", "search_knowledge"]
    assert all(call["ok"] for call in result.tool_calls)
    assert result.stop_reason == "completed"


async def test_only_authorized_tools_are_exposed_to_the_model():
    harness, gateway, provider = _harness()
    provider.reset([final("没有可用工具，直接回答。")])

    await harness.run(task="下单", ctx=_ctx("tools:read"), tool_names=["search_products", "create_order"])

    exposed = gateway.seen_tools[0]
    assert exposed is not None
    names = [item["function"]["name"] for item in exposed]
    # create_order 需要 tools:write：越权工具根本不暴露给模型（而不是暴露后再拒绝）
    assert names == ["search_products"]


async def test_no_visible_tools_stops_with_explicit_reason():
    harness, _, _ = _harness()
    result = await harness.run(task="没有可用工具", ctx=_ctx("tools:read"), tool_names=["create_order"])

    assert result.stop_reason == "no_tools"
    assert result.steps == 0


# --------------------------------------------------------------------------- #
# 三重护栏
# --------------------------------------------------------------------------- #
async def test_repeated_call_is_detected_and_stops_the_loop():
    harness, _, provider = _harness()
    provider.reset(
        [
            calls(("search_products", {"query": "面霜"})),
            calls(("search_products", {"query": "面霜"})),
            final("不应该走到这里"),
        ]
    )

    result = await harness.run(task="反复查同一件事", ctx=_ctx("tools:read"))

    assert result.stop_reason == "repeated_call_detected"
    assert result.answer is None
    assert result.handoff is not None and result.handoff.reason == "loop_detected"


async def test_repeat_exempt_tool_is_allowed_to_poll():
    """合法轮询豁免：轮询类工具反复调用不该被判成打转。"""
    harness, _, provider = _harness(repeat_exempt_tools=["search_products"])
    provider.reset(
        [
            calls(("search_products", {"query": "面霜"})),
            calls(("search_products", {"query": "面霜"})),
            final("查完了。"),
        ]
    )

    result = await harness.run(task="轮询同一状态", ctx=_ctx("tools:read"))

    assert result.stop_reason == "completed"


async def test_max_steps_guard_limits_iterations():
    harness, _, provider = _harness(max_steps=2)
    provider.reset([calls(("search_products", {"query": f"第{i}轮"})) for i in range(6)])

    result = await harness.run(task="一直查下去", ctx=_ctx("tools:read"))

    assert result.stop_reason == "max_steps"
    assert result.steps == 2


async def test_token_budget_guard_stops_cost_growth():
    harness, _, provider = _harness(token_budget=1)  # 第一次调用就会超出预算
    provider.reset([calls(("search_products", {"query": "第1轮"})), calls(("search_products", {"query": "第2轮"}))])

    result = await harness.run(task="烧钱任务", ctx=_ctx("tools:read"))

    assert result.stop_reason == "token_budget_exceeded"
    assert result.handoff is not None and result.handoff.reason == "budget"


async def test_cost_budget_guard_trips_on_money_not_tokens():
    """按钱熔断比按 token 更贴业务：预算是钱。这里直接对护栏做单测（Mock 成本为 0，打不出真实路径）。"""
    guardrails = Guardrails(token_budget=10_000, cost_budget_usd=0.01)
    state = RunState(
        task="贵任务",
        ctx=_ctx("tools:read"),
        trace=ExecutionTrace(trace_id="tr_x"),
        messages=[],
        logical_model=None,
        max_steps=6,
        deadline=1e18,
        token_budget=10_000,
        cost_budget_usd=0.01,
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.02),
    )

    trip = guardrails.check_limits(state, now=0.0)

    assert trip is not None and trip.stop_reason.value == "cost_budget_exceeded"
    assert trip.handoff_reason == "budget"


async def test_deadline_guard_prevents_hanging_requests():
    harness, _, provider = _harness(deadline_s=-1.0)
    provider.reset([calls(("search_products", {"query": "第1轮"}))])

    result = await harness.run(task="卡住的编排", ctx=_ctx("tools:read"))

    assert result.stop_reason == "deadline_exceeded"
    assert result.steps == 0


async def test_consecutive_failures_trip_the_circuit_breaker():
    harness, _, provider = _harness(max_consecutive_errors=2)
    provider.reset([calls(("flaky", {"query": f"r{i}"})) for i in range(4)])

    result = await harness.run(task="下游一直挂", ctx=_ctx("tools:read"))

    assert result.stop_reason == "error_threshold_exceeded"
    assert result.handoff is not None and result.handoff.reason == "error_threshold"


# --------------------------------------------------------------------------- #
# 派发前校验：可恢复错误回填成 observation，让模型自纠错
# --------------------------------------------------------------------------- #
async def test_unknown_tool_is_fed_back_and_loop_recovers():
    harness, _, provider = _harness()
    provider.reset([calls(("not_a_real_tool", {})), final("换个工具再答。")])

    result = await harness.run(task="用不存在的工具", ctx=_ctx("tools:read"))

    assert result.stop_reason == "completed"
    assert result.answer == "换个工具再答。"
    violations = [item for item in result.transcript if item["type"] == "violation"]
    assert violations and violations[0]["error_code"] == "not_found"


async def test_invalid_arguments_are_fed_back_with_field_issues():
    """参数问题必须回带上「哪个字段错」，否则模型只能瞎猜。"""
    harness, gateway, provider = _harness()
    provider.reset([calls(("search_products", {})), final("补齐参数后再答。")])

    result = await harness.run(task="缺少必填参数", ctx=_ctx("tools:read"))

    assert result.stop_reason == "completed"
    # 第二轮的 messages 里应包含带 issues 的 tool 观察
    second_round = gateway.provider.history[-1]
    tool_messages = [m for m in second_round if m.get("role") == "tool"]
    assert tool_messages and "issues" in tool_messages[0]["content"]


async def test_per_turn_call_limit_is_enforced():
    harness, _, provider = _harness(max_calls_per_turn=1)
    provider.reset(
        [
            calls(("search_products", {"query": "精华"}), ("search_knowledge", {"query": "成分"})),
            final("只执行了允许的数量。"),
        ]
    )

    result = await harness.run(task="一次派发两个调用", ctx=_ctx("tools:read", "tools:knowledge"))

    assert len(result.tool_calls) == 1
    assert any(item["type"] == "violation" and item["error_code"] == "call_limit_exceeded" for item in result.transcript)


# --------------------------------------------------------------------------- #
# 终止即交接：高风险确认与越权
# --------------------------------------------------------------------------- #
async def test_high_risk_action_requires_confirmation_and_hands_off():
    harness, _, provider = _harness()
    provider.reset([calls(("create_order", {"product_id": "SKU-1001", "quantity": 1, "idempotency_key": "k12345678"}))])

    result = await harness.run(task="帮用户下单", ctx=_ctx("tools:read", "tools:write"))

    assert result.stop_reason == "handoff_required"
    assert result.handoff is not None
    assert result.handoff.reason == "high_risk_confirmation"
    assert result.handoff.next_actions  # 必须告诉接手人下一步怎么做
    assert result.tool_calls == []  # 一个动作都没执行


async def test_confirmed_high_risk_action_is_executed():
    harness, _, provider = _harness()
    provider.reset(
        [
            calls(("create_order", {"product_id": "SKU-1001", "quantity": 1, "idempotency_key": "k12345678"})),
            final("已人工确认并下单。"),
        ]
    )

    result = await harness.run(task="人工确认后下单", ctx=_ctx("tools:read", "tools:write"), confirmed=True)

    assert result.stop_reason == "completed"
    assert [call["name"] for call in result.tool_calls] == ["create_order"]


async def test_business_level_permission_error_is_terminal_and_hands_off():
    """业务级越权（如订单不属于当前用户）在执行器里被标记 terminal，必须终止而不是继续重试。"""
    harness, _, provider = _harness()
    provider.reset([calls(("guarded", {"query": "他人订单"})), final("不该到达")])

    result = await harness.run(task="越权查询", ctx=_ctx("tools:read"))

    assert result.stop_reason == "terminal_tool_error"
    assert result.answer is None
    assert result.handoff is not None and result.handoff.reason == "permission"


async def test_handoff_packet_carries_completed_steps_and_failure():
    harness, _, provider = _harness(max_consecutive_errors=1)
    provider.reset([calls(("flaky", {"query": "r0"})), calls(("flaky", {"query": "r1"}))])

    result = await harness.run(task="下游挂了", ctx=_ctx("tools:read"))

    payload = result.handoff.to_payload() if result.handoff else {}
    assert payload["completed_steps"] >= 1
    assert payload["trace_id"] == "tr_harness_test"
    assert any(item["status"] == "error" for item in payload["tool_results"])


# --------------------------------------------------------------------------- #
# 轨迹与观察压缩
# --------------------------------------------------------------------------- #
async def test_trace_digest_summarizes_a_run():
    harness, _, provider = _harness()
    provider.reset([calls(("search_products", {"query": "精华"})), final("结论。")])

    result = await harness.run(task="查商品", ctx=_ctx("tools:read"))
    digest = result.trace.digest()

    assert digest["rounds"] == 2
    assert digest["stop_reason"] == "completed"
    assert digest["tools"][0]["tool"] == "search_products"
    assert digest["tokens"] > 0


async def test_trace_records_fingerprint_not_raw_arguments():
    """轨迹会被长期保留：只记参数指纹，不记参数原文（可能含用户隐私）。"""
    harness, _, provider = _harness()
    provider.reset([calls(("flaky", {"query": "13800138000"})), final("结束。")])

    result = await harness.run(task="查一个手机号", ctx=_ctx("tools:read"))
    tools = [event for event in result.trace.events() if event["type"] == "tool"]

    assert tools and "13800138000" not in str(result.trace.events())
    assert tools[0]["fingerprint"].startswith("flaky#")


def test_observation_trim_removes_old_rounds_and_compresses_history():
    builder = ObservationBuilder(max_chars=100, page_size=2)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for index in range(8):
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{index}", "content": "x" * 500})

    trimmed = builder.trim(messages, keep_rounds=2)

    assert trimmed[0] == {"role": "system", "content": "sys"}
    # 只保留末 2 轮，且老一轮的 tool 内容被压缩
    assert sum(1 for m in trimmed if m.get("role") == "tool") == 2
    compressed = next(m for m in trimmed if m.get("role") == "tool")
    assert compressed["content"].endswith("[历史观察已压缩]")


def test_observation_truncation_declares_it_explicitly():
    """被截断的结果必须显式声明，否则模型会误以为「只有这么多结果」。"""
    builder = ObservationBuilder(max_chars=200, page_size=1)
    payload = builder._render({"items": [{"a": i} for i in range(10)], "count": 10})

    assert "truncated" in payload or "[observation_truncated]" in payload


# --------------------------------------------------------------------------- #
# factory：配置 → 护栏参数
# --------------------------------------------------------------------------- #
def test_build_harness_reads_guardrails_from_settings():
    from src.harness.factory import build_guardrails, build_harness

    settings = Settings(
        agent_max_steps=9,
        agent_total_timeout_s=12.0,
        agent_token_budget=5000,
        harness_cost_budget_usd=0.5,
        harness_max_consecutive_errors=2,
        harness_repeat_window_steps=4,
        harness_repeat_exempt_tools="get_order_status, search_products",
        harness_max_calls_per_turn=2,
        harness_max_observation_chars=800,
    )
    guardrails = build_guardrails(settings)

    assert guardrails.max_steps == 9
    assert guardrails.cost_budget_usd == 0.5
    assert guardrails.repeat_window_steps == 4
    assert guardrails.repeat_exempt_tools == frozenset({"get_order_status", "search_products"})
    assert guardrails.max_calls_per_turn == 2
    assert guardrails.max_observation_chars == 800

    registry = _registry()
    built = build_harness(
        settings=settings,
        gateway=_StubGateway(MockProvider()),
        registry=registry,
        executor=ToolExecutor(registry, max_concurrency=4),
        prompts=build_default_registry(),
    )
    assert built.guardrails.max_steps == 9
