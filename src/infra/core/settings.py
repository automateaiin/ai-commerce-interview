"""进程配置（环境变量驱动）。

约定：配置分两层。
- **基础设施参数**（超时、配额上限、日志级别、DB/Redis 地址）走环境变量，进本文件；
- **模型路由映射**（逻辑模型 → 物理端点）走 config/models.yaml，进 llm/catalog.py。

这样密钥永远不进仓库，而模型映射可以随业务调整。
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[3]

logger = get_logger("infra.settings")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- 基础 --------------------------------------------------------------- #
    app_name: str = "ecom-ai-infra"
    environment: str = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # -- 配置来源 ----------------------------------------------------------- #
    models_config_path: str = str(REPO_ROOT / "config" / "models.yaml")
    prompts_config_path: str = str(REPO_ROOT / "config" / "prompts.yaml")

    # -- 存储 --------------------------------------------------------------- #
    database_url: str = "sqlite+aiosqlite:///./data/infra.db"
    redis_url: str | None = None

    # -- 模型接入 ----------------------------------------------------------- #
    provider_mode: str = Field(default="mock", description="mock | openai_compat | auto")
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    llm_stream_enabled: bool = True

    # -- 工具执行 ----------------------------------------------------------- #
    tool_timeout_s: float = 8.0
    tool_max_concurrency: int = 8

    # -- 配额（预留-实扣） -------------------------------------------------- #
    quota_enabled: bool = True
    quota_requests_per_window: int = 600
    quota_tokens_per_window: int = 200_000
    quota_window_s: int = 3600
    quota_reservation_ttl_s: int = 120
    quota_default_max_output_tokens: int = 512

    # -- 限流（滑动窗口，与「配额」是两条不同的轴） -------------------------- #
    # 配额（quota）管的是「窗口内总量与计费」，限流（ratelimit）管的是「瞬时流量整形」：
    # 前者超限要能对账，后者超限只需要快速拒绝，因此两者的窗口与阈值独立配置。
    ratelimit_enabled: bool = True
    ratelimit_ip_limit: int = 600
    """单 IP 窗口内请求上限。默认给得比较宽：它是防刷入口的第一道闸，不是业务量控制。"""
    ratelimit_ip_window_s: int = 60
    ratelimit_service_limit: int = 600
    """单 Service（由凭证推导的业务线）窗口内请求上限。"""
    ratelimit_service_window_s: int = 60

    # -- 可观测（TelemetryPort） -------------------------------------------- #
    telemetry_backend: str = Field(default="console", description="console | otel（依赖缺失时自动降级到 console）")

    # -- 成本预算（单 Agent 日预算 + 单请求 Max Token + 超阈值切 Lite） ------ #
    budget_enabled: bool = True
    budget_daily_tokens_per_agent: int = 2_000_000
    """同一 Agent 的日 Token 预算。超过硬上限直接拒绝，避免单个 Agent 吃掉全部成本。"""
    budget_downgrade_ratio: float = 0.8
    """用掉日预算此比例后，后续请求自动降级到 Lite 模型（软阈值，不中断服务）。"""
    budget_lite_model: str = ""
    """降级目标逻辑模型；留空则用 routing.default。"""
    budget_lite_max_output_tokens: int = 512
    """降级后单请求输出上限，与 Max Token 控制配合，防止降级模型又被要求长输出。"""

    # -- 评测（Evaluation Pipeline） ---------------------------------------- #
    eval_accuracy_threshold: float = 0.7
    eval_max_cost_usd: float = 0.05
    """单用例成本上限，超过判为超预算（成本也是评测维度，不只是延迟）。"""
    eval_max_latency_ms: float = 5000.0

    # -- 子 Agent（工具调用循环）护栏 --------------------------------------- #
    agent_max_steps: int = 6
    agent_total_timeout_s: float = 60.0
    agent_token_budget: int = 20000

    # -- Harness 执行层护栏（项目二扩展） ---------------------------------- #
    harness_cost_budget_usd: float = 0.0
    """按钱熔断的预算上限；0 表示不单独设成本上限（只按 token 计）。"""
    harness_max_consecutive_errors: int = 3
    """连续失败熔断阈值：整轮工具全失败累计到此值即终止并交接。"""
    harness_repeat_window_steps: int = 3
    """打转检测时间窗（轮）：窗口内出现同参数调用即判定为原地打转。"""
    harness_repeat_exempt_tools: str = ""
    """合法轮询豁免工具名（逗号分隔），例如 get_order_status。"""
    harness_max_calls_per_turn: int = 4
    """单轮工具调用数量上限，防止模型一次派发一堆调用。"""
    harness_max_observation_chars: int = 1500
    """回填给模型的单条 observation 字符上限。"""

    # -- 数据平台（项目三） ------------------------------------------------- #
    data_chunk_max_tokens: int = 220
    """单个知识切片 token 上限。"""
    data_chunk_overlap_tokens: int = 40
    """切片重叠 token 数，避免语义在切口处断裂。"""
    data_chunk_time_window_s: int = 300
    """切片时间窗初筛（秒）：拿不到语义边界时才回退到它。"""
    data_embed_dim: int = 256
    """离线哈希向量维度（生产为真实 embedding 维度）。"""
    data_rerank_candidates: int = 20
    """进入重排的候选池大小。"""
    data_recall_top_k: int = 3
    """最终返回给上下文组装的证据条数。"""
    data_context_char_budget: int = 1200
    """拼进 Prompt 的证据总字符预算。"""
    data_export_version: str = "v1"
    """训练集导出批次版本号。"""

    # -- 数据平台：三级切片 / 重排 / 训练产物 --------------------------------- #
    data_chunk_mode: str = Field(default="semantic", description="semantic（语义边界）| hierarchical（L1/L2/L3 三级 + Auto-merging）")
    """切片模式。默认保持语义边界，切换为 hierarchical 才启用三级切片与向上合并。"""
    data_chunk_l1_window: int = 2
    """L1 句子滑动窗口长度（句数）。"""
    data_chunk_l2_group: int = 3
    """L2 段落聚合的 L1 数量。"""
    data_chunk_l3_group: int = 9
    """L3 章节聚合的 L1 数量。"""
    data_merge_ratio: float = 0.5
    """Auto-merging 阈值：同一父块下命中子块占比达到该值才合并回父块。"""
    data_rerank_backend: str = Field(default="lexical", description="lexical（本地词典，离线）| jina（远端，需 key）")
    data_jina_api_key: str | None = None
    """未配置时**显式降级**到本地词典重排，不影响检索可用性。"""
    data_jina_rerank_url: str = "https://api.jina.ai/v1/rerank"
    data_jina_rerank_model: str = "jina-reranker-v2-base-multilingual"
    data_training_base_model: str = "Qwen2.5-7B-Instruct"
    """训练 YAML 里写的基础模型名 —— 只是**产物配置**，本仓库不执行训练。"""

    # -- 宿主业务库（commerce，项目一白名单 Tool 的真实数据源） -------------- #
    commerce_database_url: str = "sqlite+aiosqlite:///./data/commerce.db"
    """业务库与基础设施库**分开**：infra 库存 trace/记忆/审计，业务库存商品/订单，
    两者备份、迁移、扩容的节奏不同，混在一张库里迟早互相拖累。"""
    commerce_seed_enabled: bool = True
    """启动时灌入演示数据。只在库为空时执行，重复启动幂等。"""
    commerce_seed_force: bool = False
    """强制重置演示数据（仅测试/冒烟用）。"""
    commerce_member_token_secret: str | None = None
    """会员 token 的签名密钥。未配置时**显式降级**为进程内随机密钥并打告警日志 —— 
    重启即失效，因此生产必须配；这里不硬编码任何默认密钥。"""
    commerce_member_token_ttl_s: int = 86400
    commerce_order_ttl_minutes: int = 30
    """未付款订单的存活时长，超时由 sweeper 取消并回滚库存。"""
    commerce_sweeper_interval_s: float = 60.0
    """订单超时清理周期（秒）；<= 0 表示显式禁用，不静默保留后台任务。"""
    commerce_order_rate_per_hour: int = 20
    """单会员每小时下单上限（业务限流）。与网关限流是两条轴：这条防的是**单个账号刷单**。"""

    # -- 记忆 --------------------------------------------------------------- #
    memory_context_token_budget: int = 2000
    memory_recent_turns: int = 12

    # -- 应用层（项目一 agents） -------------------------------------------- #
    # 会话回填复用 memory 的两项预算（此处不再重复定义），仅补编排自身需要的参数。
    agents_max_steps: int = 6
    """单次 Agent 运行的步数上限（与 harness 护栏同源，作为各角色 Agent 的默认值）。"""
    agents_max_handoffs: int = 2
    """Handoff 转派次数上限：没有上限的转派链会把一次请求变成一串无界模型调用。"""
    agents_max_tasks: int = 4
    """规划器给出的子任务数上限（模型可以提议，但不能无限扩张执行面）。"""
    agents_planner_model: str = ""
    """规划器使用的逻辑模型；留空则走网关默认路由。"""
    agents_grounding_enabled: bool = True
    """交易类回答的事实绑定闸门开关；关闭仅用于对照实验。"""
    agents_feedback_limit: int = 200
    """反馈收集器容量：跨请求存活但必须有界，满了丢最旧。"""
    agents_hitl_max_pending: int = 200
    """待确认检查点的容量上限。"""
    internal_service_token: str | None = None
    """内部服务间凭证（A2A / MCP 共用）。未配置时相关端点显式拒绝，不静默放行。"""
    agents_hitl_secret: str | None = None
    """HITL Resume Token 的签名密钥；未配置时回退到内部凭证，再回退到进程内随机值。"""

    # -- 应用层（项目二 ops：运营工作流 / 多模态 / 直播 / 归因 / 自进化） ---- #
    ops_max_attempts: int = 2
    """节点级失败重试上限（含首次执行）：Checkpointer 只对可重试错误重跑该节点。"""
    ops_structured_retries: int = 1
    """Structured Output 校验失败后的重试次数；仍不合法则回退到确定性兜底并记录原因。"""
    ops_parallel_max_concurrency: int = 4
    """并行生成的最大并发（asyncio.gather + 线程池），防止一次 fan-out 打满资源。"""
    ops_live_window_s: int = 90
    """长直播分段转录的时间窗（秒）：一次转录一段，避免整场直播一次性进模型。"""
    ops_live_overlap_s: int = 15
    """相邻转录窗口的重叠时长（秒）：有重叠才能修掉切点处的断句丢字。"""
    ops_live_batch_size: int = 3
    """直播分段分批分析的批大小（每批一次模型调用）。"""
    ops_asset_max_assets: int = 500
    """素材资产池容量上限（有界，超限淘汰最旧）。"""
    ops_selfevolve_max_events: int = 500
    """自进化事件池容量上限（跨请求存活但有界）。"""
    ops_analysis_model: str = "chat-pro"
    """复杂分析/策略节点使用的逻辑模型（选题、成分卖点、归因）。"""
    ops_copy_model: str = "chat-lite"
    """短文案/图片提示词节点使用的轻量逻辑模型（降本路径）。"""
    ops_asset_embed_version: str = "hash-256-v1"
    """素材向量版本号：与 embedding 实现绑定，便于新旧向量对账与重建。"""

    # -- 火山方舟 / OpenAI 兼容端点 ----------------------------------------- #
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_api_key: str | None = None

    def resolved_models_config_path(self) -> Path:
        path = Path(self.models_config_path)
        return path if path.is_absolute() else REPO_ROOT / path

    def resolved_prompts_config_path(self) -> Path:
        path = Path(self.prompts_config_path)
        return path if path.is_absolute() else REPO_ROOT / path

    def resolved_database_url(self) -> str:
        """把 sqlite 相对路径固定到仓库根，避免受启动目录影响。"""
        return _resolve_sqlite(self.database_url)

    def resolved_commerce_database_url(self) -> str:
        """业务库同样把相对 sqlite 路径钉到仓库根，两个库用同一套解析规则。"""
        return _resolve_sqlite(self.commerce_database_url)

    def resolved_internal_service_token(self) -> str:
        return self.internal_service_token or os.environ.get("INTERNAL_SERVICE_TOKEN", "")

    def resolved_agents_hitl_secret(self) -> str:
        """HITL 签名密钥：显式配置 > 内部凭证 > 进程内随机值（并告警，重启即失效）。"""
        if self.agents_hitl_secret:
            return self.agents_hitl_secret
        token = self.resolved_internal_service_token()
        if token:
            return token
        logger.warning(
            "hitl.secret_fallback",
            extra={
                "event": "startup",
                "status": "degraded",
                "reason": "HITL secret 未配置，使用进程内随机密钥（重启后旧 token 失效）",
            },
        )
        return secrets.token_hex(16)


def _resolve_sqlite(url: str) -> str:
    """把相对 sqlite 路径钉到仓库根，并**确保目录存在**。

    为什么要在这里建目录（而不是在容器启动时）：这是全仓唯一知道
    物理路径的地方，两个库（infra / commerce）都从这里解析出来，
    在别处再解析一次就是重复逻辑、迟早分叉。

    不建目录的后果是**全新 clone 直接起不来**：`data/` 不存在时
    SQLite 只会抛 `unable to open database file`，而它永远不会自己建目录。
    这里补上，才能保证「clone → 起服务」不需要任何人肉步骤。
    """
    prefix = "sqlite+aiosqlite:///./"
    if url.startswith(prefix):
        db_path = REPO_ROOT / url[len(prefix):]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path.as_posix()}"
    return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
