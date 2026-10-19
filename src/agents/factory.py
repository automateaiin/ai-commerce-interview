"""项目一装配：由 Settings + 既有基建构造出 AgentService。

装配顺序就是依赖顺序，读这个文件能一眼看清「谁依赖谁」：
    护栏 → 会话 → 两个角色 Agent → 编排模式注册表 → 路由器 → 编排器
         → 直连/Agent 两条路径共用同一个 ToolExecutor（工具权限链只有一条）
         → HITL / 实时语音 / 检索重写 / 反馈 / A2A
"""

from __future__ import annotations

from typing import Any

from src.harness.engine import Harness
from src.infra.core.settings import Settings
from src.infra.memory.store import MemoryStore
from src.infra.prompts.registry import PromptRegistry
from src.infra.rag.service import KnowledgeService
from src.infra.tools.base import ToolExecutor

from .a2a import A2AService
from .customer_agent import CustomerAgent
from .feedback import FeedbackCollector
from .grounding import GroundingGate
from .hitl import HITLManager, HITLStore
from .operations_agent import OperationsAgent
from .orchestration import build_mode_registry
from .orchestrator import ComplexityRouter, Orchestrator
from .prompt_guard import PromptGuard, build_prompt_guard
from .realtime import RealtimePipeline, build_realtime_ports
from .retrieval_strategy import QueryRewriter
from .service import AgentService
from .session import SessionManager


class _CommerceOpsData:
    """`OpsDataPort` 的 commerce 适配器：只读聚合，给运营侧做分析用。

    运营侧读的是**内部经营指标**，不走消费者侧那套白名单 Tool ——
    工具是「对外能力边界」，不等于「内部不许直接读库」。
    """

    def __init__(self, commerce: Any) -> None:
        self._commerce = commerce

    async def order_stats(self, *, days: int) -> dict[str, Any]:
        stats = await self._commerce.orders.stats()
        return {"window_days": days, **(stats if isinstance(stats, dict) else {"stats": stats})}

    async def low_stock(self) -> list[dict[str, Any]]:
        stocks = await self._commerce.catalog.low_stock()
        return [
            {"sku_id": stock.sku_id, "available": stock.available, "threshold": stock.threshold}
            for stock in stocks
        ]

    async def top_catalog(self, *, limit: int) -> list[dict[str, Any]]:
        products, total = await self._commerce.catalog.search(offset=0, limit=limit)
        return [
            {
                "product_id": product.product_id,
                "name": product.name,
                "status": product.status.value,
                "price_usd": str(product.price.amount()),
            }
            for product in products
        ] + [{"total": total}]


def build_agents_service(
    *,
    settings: Settings,
    harness: Harness,
    gateway: Any,
    prompts: PromptRegistry,
    memory: MemoryStore,
    knowledge: KnowledgeService,
    executor: ToolExecutor,
    commerce: Any,
) -> AgentService:
    guard: PromptGuard = build_prompt_guard()
    sessions = SessionManager(
        memory=memory,
        guard=guard,
        token_budget=settings.memory_context_token_budget,
        recent_turns=settings.memory_recent_turns,
    )

    customer = CustomerAgent(
        harness=harness, prompts=prompts, sessions=sessions, guard=guard, max_steps=settings.agents_max_steps
    )
    operations = OperationsAgent(
        harness=harness,
        prompts=prompts,
        data=_CommerceOpsData(commerce),
        guard=guard,
        max_steps=settings.agents_max_steps,
    )

    modes = build_mode_registry(max_handoffs=settings.agents_max_handoffs)
    router = ComplexityRouter(
        gateway=gateway,
        modes=modes,
        agents={"customer": customer, "operations": operations},
        planner_model=settings.agents_planner_model or None,
        max_tasks=settings.agents_max_tasks,
    )
    orchestrator = Orchestrator(
        router=router,
        modes=modes,
        runners={"customer": customer.handle, "operations": operations.handle},
        executor=executor,
        grounding=GroundingGate(enabled=settings.agents_grounding_enabled),
    )

    rtc, asr, tts = build_realtime_ports()
    hitl = HITLManager(
        executor=executor,
        store=HITLStore(max_pending=settings.agents_hitl_max_pending),
        secret=settings.resolved_agents_hitl_secret(),
        # 高风险白名单：只有真实业务副作用、且模型不该自行拍板的动作才需要人工确认
        high_risk_tools={"create_order"},
    )
    return AgentService(
        orchestrator=orchestrator,
        customer=customer,
        operations=operations,
        sessions=sessions,
        hitl=hitl,
        realtime=RealtimePipeline(rtc=rtc, asr=asr, tts=tts, orchestrator=orchestrator),
        rtc=rtc,
        feedback=FeedbackCollector(limit=settings.agents_feedback_limit),
        retrieval=QueryRewriter(
            knowledge=knowledge, gateway=gateway, model=settings.agents_planner_model or None
        ),
        a2a=A2AService(orchestrator=orchestrator, internal_token=settings.resolved_internal_service_token()),
        guard=guard,
    )


__all__ = ["build_agents_service"]
