"""依赖装配容器（组合根）。

所有组件的构造顺序、单例边界、启动/关闭动作集中在这里，
业务模块只声明「我需要什么」，不自己 new 依赖 —— 这是可测试性的前提：
测试可以直接用 Container 的字段替换掉任意一个组件（例如把 Gateway 换成 Mock 脚本化的那个）。
"""

from __future__ import annotations

from typing import Any

from src.agents.factory import build_agents_service
from src.agents.service import AgentService
from src.commerce.factory import build_commerce_service
from src.commerce.service import CommerceService
from src.commerce.tools_port import CommerceToolsAdapter
from src.dataplatform.service import DataPlatformService
from src.harness.engine import Harness
from src.harness.factory import build_harness
from src.ops.factory import build_ops_service
from src.ops.service import OpsService

from .bridge.auth import ApiKeyAuthenticator
from .bridge.invoker import InProcessInvoker, ToolInvoker
from .core.audit import AuditLog
from .core.logging import get_logger, setup_logging
from .core.settings import Settings, get_settings
from .eval.pipeline import EvalLimits, EvaluationPipeline, HarnessEvalRunner
from .llm.catalog import ModelCatalog
from .llm.factory import build_catalog_snapshot, build_providers
from .llm.gateway import LLMGateway
from .mcp.server import build_mcp_servers
from .memory.store import MemoryStore
from .prompts.registry import PromptRegistry, build_default_registry
from .quota.budget import AgentBudgetGuard, build_budget_guard
from .quota.store import QuotaManager, build_quota_manager
from .core.config import ConfigSnapshot
from .core.retry import RetryPolicy
from .rag.service import HashingEmbedder, KnowledgeService
from .ratelimit.limiter import RateLimiter, build_rate_limiter
from .telemetry.port import TelemetryPort, build_telemetry
from .tools.base import ToolExecutor, ToolRegistry
from .tools.factory import build_registry

logger = get_logger("infra.container")


class Container:
    """运行期依赖集合。created once at app startup。"""

    def __init__(
        self,
        *,
        settings: Settings,
        catalog: ConfigSnapshot[ModelCatalog],
        gateway: LLMGateway,
        registry: ToolRegistry,
        executor: ToolExecutor,
        invoker: ToolInvoker,
        knowledge: KnowledgeService,
        memory: MemoryStore,
        prompts: PromptRegistry,
        quota: QuotaManager,
        authenticator: ApiKeyAuthenticator,
        harness: Harness,
        dataplatform: DataPlatformService,
        commerce: CommerceService,
        mcp_servers: dict[str, Any],
        mcp_paths: list[str],
        telemetry: TelemetryPort | None = None,
        ratelimit: RateLimiter | None = None,
        audit: AuditLog | None = None,
        budget: AgentBudgetGuard | None = None,
        eval_pipeline: EvaluationPipeline | None = None,
        redis_client: Any | None = None,
        agents: AgentService | None = None,
        ops: OpsService | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.gateway = gateway
        self.registry = registry
        self.executor = executor
        self.invoker = invoker
        self.knowledge = knowledge
        self.memory = memory
        self.prompts = prompts
        self.quota = quota
        self.authenticator = authenticator
        self.harness = harness
        self.dataplatform = dataplatform
        self.commerce = commerce
        self.mcp_servers = mcp_servers
        self.mcp_paths = mcp_paths
        # 阶段 1 新增的治理组件：默认给空实现，便于单测按需替换成精简容器
        self.telemetry = telemetry or build_telemetry(settings)
        self.ratelimit = ratelimit or build_rate_limiter(settings)
        self.audit = audit or AuditLog()
        self.budget = budget or build_budget_guard(settings)
        self.eval_pipeline = eval_pipeline
        self._redis_client = redis_client
        self.agents = agents
        self.ops = ops

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "Container":
        active = settings or get_settings()
        setup_logging(active.log_level, json_output=active.log_json)

        # 0) 可观测出口：最早建立，后续组件可以直接打点（OTel 依赖缺失时自动降级到 console）
        telemetry = build_telemetry(active)

        # 1) 配置快照 + Redis 连接（提前到网关之前：限流与成本预算都要用它）
        catalog = build_catalog_snapshot(active)
        redis_client = await cls._try_connect_redis(active)

        # 2) 限流（滑动窗口）+ 成本预算：有 Redis 则多实例一致，否则进程内降级
        ratelimit = build_rate_limiter(active, redis_client, telemetry)
        budget = build_budget_guard(active, redis_client, telemetry)

        # 3) 模型网关（Provider 注册表按逻辑模型绑定端点）+ 预算守卫（超阈值自动切 Lite）
        providers = build_providers(catalog.current(), active)
        gateway = LLMGateway(
            catalog,
            providers,
            retry_policy=RetryPolicy(max_attempts=active.llm_max_retries),
            budget=budget,
            lite_model=active.budget_lite_model,
            lite_max_output_tokens=active.budget_lite_max_output_tokens,
        )

        # 4) 存储：记忆库（本地默认 SQLite，生产 PostgreSQL）
        memory = MemoryStore.from_url(active.resolved_database_url())
        await memory.init_schema()

        # 5) 检索：知识索引（Embedder 可替换为真实 embedding 服务）
        knowledge = KnowledgeService(HashingEmbedder())
        await knowledge.build_index()

        # 6) Prompt 模板中心：优先 YAML，缺失时显式降级到内置模板
        prompts = build_default_registry(active.resolved_prompts_config_path())

        # 7) 宿主业务后端（commerce）：项目一白名单 Tool 的真实业务数据源。
        #    放在注册表之前，因为工具构造需要注入它的业务端口（依赖倒置：注入的是端口，不是实现）。
        commerce = await build_commerce_service(active, redis_client=redis_client)
        commerce_port = CommerceToolsAdapter(commerce)

        # 8) 审计日志 + 工具注册表 + 执行器（执行器负责把四级权限链写入审计）
        audit = AuditLog()
        registry = build_registry(memory, knowledge, commerce_port)
        executor = ToolExecutor(registry, max_concurrency=active.tool_max_concurrency, audit=audit)

        # 8) 配额：与限流是两条轴（总量/计费 vs 瞬时速率），各自独立
        quota = build_quota_manager(active, redis_client)

        # 9) 桥接层：鉴权 + 工具通道
        authenticator = ApiKeyAuthenticator(catalog)
        invoker: ToolInvoker = InProcessInvoker(executor)

        # 10) 子 Agent 运行时（共享 Harness，执行层唯一入口）
        harness = build_harness(
            settings=active,
            gateway=gateway,
            registry=registry,
            executor=executor,
            prompts=prompts,
        )

        # 11) 数据平台（数据治理与资产化门面，按请求装配流水线）
        dataplatform = DataPlatformService(active)

        # 11.5) 应用层（项目一）：编排/双端 Agent/HITL/实时语音。
        #       放在 harness 与 commerce 之后，因为它同时消费「执行内核」与「业务数据源」。
        agents = build_agents_service(
            settings=active,
            harness=harness,
            gateway=gateway,
            prompts=prompts,
            memory=memory,
            knowledge=knowledge,
            executor=executor,
            commerce=commerce,
        )

        # 11.6) 应用层（项目二）：运营内容工作流 / 多模态素材 / 直播 / 分析 / 自进化。
        #        只消费 infra（模型网关 + 检索 + 提示词中心），不依赖项目一，两条业务线互不耦合。
        ops = build_ops_service(
            settings=active,
            gateway=gateway,
            knowledge=knowledge,
            prompts=prompts,
        )

        # 12) 评测流水线：runner 直接跑共享 Harness（评测即线上链路，不另写一套流程）
        eval_pipeline = EvaluationPipeline(
            HarnessEvalRunner(harness=harness, gateway=gateway),
            limits=EvalLimits(
                accuracy_threshold=active.eval_accuracy_threshold,
                max_cost_usd=active.eval_max_cost_usd,
                max_latency_ms=active.eval_max_latency_ms,
            ),
            telemetry=telemetry,
        )

        # 13) MCP 服务集群（挂载路径在 api.py 完成）
        mcp_servers = build_mcp_servers(
            registry=registry, executor=executor, knowledge=knowledge, memory=memory, prompts=prompts
        )
        # 挂载路径带尾斜杠：Streamable HTTP 的会话端点位于子应用根路径下，
        # 不带尾斜杠会被 Starlette 以 307 重定向，客户端 POST 会丢失语义。
        mcp_paths = [f"/mcp/{name}/" for name in mcp_servers]

        logger.info(
            "container.ready",
            extra={
                "event": "startup",
                "model": active.provider_mode,
                "route": ",".join(sorted(catalog.current().models)),
                "quota_backend": quota.backend,
                "ratelimit_backend": ratelimit.backend,
                "budget_backend": budget.backend,
                "telemetry_backend": telemetry.backend,
                "degraded": redis_client is None,
                "reason": "redis_unavailable" if redis_client is None else "redis_connected",
            },
        )
        return cls(
            settings=active,
            catalog=catalog,
            gateway=gateway,
            registry=registry,
            executor=executor,
            invoker=invoker,
            knowledge=knowledge,
            memory=memory,
            prompts=prompts,
            quota=quota,
            authenticator=authenticator,
            harness=harness,
            dataplatform=dataplatform,
            commerce=commerce,
            mcp_servers=mcp_servers,
            mcp_paths=mcp_paths,
            telemetry=telemetry,
            ratelimit=ratelimit,
            audit=audit,
            budget=budget,
            eval_pipeline=eval_pipeline,
            redis_client=redis_client,
            agents=agents,
            ops=ops,
        )

    @staticmethod
    async def _try_connect_redis(settings: Settings) -> Any | None:
        """尝试连接 Redis。失败则**显式降级**并记录日志，不静默。"""
        if not settings.redis_url:
            return None
        try:
            import redis.asyncio as redis  # 延迟导入：未配置 Redis 时不需要该依赖

            client = redis.from_url(settings.redis_url, decode_responses=True)
            await client.ping()
            logger.info("redis.connected", extra={"event": "startup", "status": "ok", "degraded": False})
            return client
        except Exception as exc:  # noqa: BLE001 - 降级路径必须捕获全部异常
            logger.warning(
                "redis.unavailable",
                extra={
                    "event": "startup",
                    "status": "degraded",
                    "degraded": True,
                    "reason": f"{type(exc).__name__}",
                },
            )
            return None

    async def aclose(self) -> None:
        # 业务后端先关：它会先停后台 sweeper 再关连接池（顺序反了会在关闭窗口炸出假异常）
        await self.commerce.aclose()
        if self._redis_client is not None:
            await self._redis_client.aclose()
        await self.memory.dispose()
