# 美妆电商 AI 应用工程 —— 面试项目总纲（AI 编码说明书）

> **给 AI 的指令**：本文件是本仓库的**唯一事实来源**。在写任何代码之前先通读本文件；
> 任何新增模块都必须满足第 4 章「工程化硬性门槛」，不符合的代码视为不合格。
> 本文件同时是面试讲稿的技术底稿：每个模块都要能被追问到「具体怎么做的」。

---

## 1. 业务背景

公司经营**高端美妆产品的海外电商业务**，三条业务线都要用大模型：

1. **消费者侧**：肤质测试、成分与功效咨询、护肤方案推荐、订单与售后；
2. **品牌运营侧**：经营分析、客服质检、商品卖点提炼、运营报告；
3. **内容生产侧**：种草文案、产品图、直播切片等营销素材。

如果不做统一基建，会出现四个具体问题：

1. **重复造轮子**：每个 Agent 各自实现模型调用、RAG、会话记忆、Prompt 管理，开发与维护成本高。
2. **能力无法复用**：工具实现标准不统一，形成新的壁垒，同一种能力被实现出多个版本。
3. **治理困难**：安全设置（鉴权、越权）、成本管理（Token 预算）、限流没有统一入口，标准不一容易出错。
4. **无法排障**：调用链路各不相同，出问题时无法定位到底慢在模型、检索、工具还是数据库。

所以要做的是**「应用层 + 一层可复用、可治理、可扩展的 AI 能力服务层 + 一条数据闭环」**，而不是一个聊天机器人。

## 2. 项目分层与四份材料

整个面试项目按「应用层 → 执行内核 → 平台层 → 数据层」组织，对外是**四个可独立讲述的子项目**，内部另有**两个共享底座**：

| 编号 | 名称 | 代码落位 | 定位 | 技术主线 |
| --- | --- | --- | --- | --- |
| **项目一** | AI 智能客服导购系统 | `src/agents` | 应用层：面向消费者与品牌运营的双端 Agent | Orchestrator 复杂度路由 → Customer/Operations Agent → 多模式编排 → Grounding/HITL → 实时语音 |
| **项目二** | AI 运营系统 | `src/ops` | 应用层：运营自动化 · 多模态素材 · 数据自进化 | 工作流图 + Checkpointer → 流式/并行生成 → 多模态知识化 → 直播处理 → 归因 → 自进化闭环 |
| **项目三** | AI 能力中台与统一调度网关 | `src/infra` | 平台层：统一 AI 公共能力与治理 | Provider 抽象 → 模型网关 → MCP → RAG/Memory/Prompt → 四级权限 → 成本预算 → 滑动窗口限流 → 可观测+评测 |
| **项目四** | AI 数据闭环与美妆知识工程平台 | `src/dataplatform` | 数据层：知识资产与训练数据 | 接入清洗 → 脱敏审核 → Session 构建 → 三级分块/检索 → 训练数据 → 回流评测 |
| 共享 | 执行内核 Harness | `src/harness` | 全仓**唯一**的 Agent Loop 与决策/护栏/轨迹/交接 | Decision 协议 → Loop Engine → 派发校验 → 三重护栏 → 观察回填 → 轨迹 → 交接包 |
| 共享 | 宿主业务后端 Commerce | `src/commerce` | 项目一的白名单 Tool 数据源（真实业务库） | 领域模型 → 原子扣库存 → 幂等下单 → Outbox → UoW |

**核心认知**：`infra` 提供能力，`harness` 组织执行，`agents`/`ops` 是消费者，`dataplatform` 提供知识与数据，
`commerce` 只被 `agents` 通过白名单 Tool 访问。**框架叫法对应关系**：简历中的「Microsoft Agent Framework WorkflowBuilder / LangGraph」
在本仓库是**语义等价的原生实现**（见第 8 章口径），不引入这两个框架依赖。

分层职责：

```
应用层(项目一/二)  agents（编排/护栏/HITL/语音） · ops（工作流/多模态/直播/自进化）
─────────────────────────────────────────────────────────────
执行内核(共享)     harness：Agent Loop · 决策校验 · 工具调度 · 状态 · 终止控制 · 轨迹 · 交接
─────────────────────────────────────────────────────────────
平台层(项目三)     infra：模型网关 · MCP · RAG · Memory · Prompt · 鉴权/权限 · 配额/限流 · 成本 · 可观测/评测
─────────────────────────────────────────────────────────────
数据层(项目四)     dataplatform：原始数据 · 治理审核 · 切片索引 · 训练集 · 评估集 · 回流
─────────────────────────────────────────────────────────────
宿主业务           commerce：商品 · 库存 · 购物车 · 订单 · 会员 · 优惠券 · 幂等 · Outbox
─────────────────────────────────────────────────────────────
资源层            外部模型 API · PostgreSQL/pgvector · Redis · 配置 · 日志
```

依赖方向（**单向，禁止倒置**）：

```
agents / ops   →   harness   →   infra
   agents      →   commerce  ──┘
   ops         →   dataplatform
   dataplatform→   infra
infra / harness / dataplatform  ✗→  agents / ops / commerce
```

## 3. 仓库结构与命名约定

```
面试项目/
├── PROJECT.md              # 本文件：AI 编码说明书
├── get-pip.py              # 无 pip 环境下的引导脚本
├── requirements.txt
├── pytest.ini
├── .env.example            # 环境变量样例（真实密钥一律走环境变量）
├── config/
│   ├── models.yaml         # 逻辑模型 → 物理端点映射（快照来源）
│   └── prompts.yaml        # 【阶段 1 新增】Prompt 模板中心（片段版本 + Agent Role 组装规则）
├── src/
│   ├── infra/              # 【项目三】平台层
│   │   ├── core/           # config / logging / tracing / errors / retry / context / metrics / audit
│   │   ├── llm/            # Provider 抽象 + 适配器 + Mock + 工厂路由 + 逻辑模型目录
│   │   ├── mcp/            # MCP Server（Streamable HTTP）与内部 MCP 会话
│   │   ├── tools/          # 工具定义、注册表、执行器、电商业务工具（白名单数据访问）
│   │   ├── memory/         # 会话消息 + 用户画像（UPSERT）
│   │   ├── rag/            # 知识检索
│   │   ├── prompts/        # registry（版本/角色组合）+ templates（YAML 契约，按 AgentRole 组装）
│   │   ├── quota/          # store（Token/请求配额，预留-实扣）+ budget（【阶段 1 新增】Agent 日预算与降级）
│   │   ├── ratelimit/      # 【阶段 1 新增】Redis ZSET + Lua 滑动窗口限流 + 内存降级 + fail-open
│   │   ├── telemetry/      # 【阶段 1 新增】TelemetryPort（Console 默认实现，OTel 可选适配）
│   │   ├── eval/           # 【阶段 1 新增】评测流水线（cases / evaluators / pipeline 六维打分）
│   │   ├── bridge/         # REST-to-MCP 桥接：鉴权、上下文注入、包装；数据平台路由
│   │   ├── static/         # 【阶段 7 新增】零构建演示前端：shop.html（官网页）/ admin.html（运营台）/ assets（app.css·app.js）
│   │   └── api.py          # 唯一组合根之外的 FastAPI 组装、中间件、异常处理器；并托管 `/`、`/admin`、`/assets`（均不入 OpenAPI）
│   ├── harness/            # 共享执行内核（已实现）
│   │   ├── decision.py     # 模型输出 → 归一化决策（FINAL / TOOL_CALLS）
│   │   ├── validate.py     # 派发前校验（存在/白名单/高风险/参数/数量/权限）
│   │   ├── policy.py       # 工具白名单与高风险动作确认策略
│   │   ├── guardrails.py   # 三重护栏（静态上限 / 打转检测 / 连续失败熔断）
│   │   ├── observation.py  # 工具结果 → 模型可消费观察（对齐 call_id、截断、压缩）
│   │   ├── state.py        # 运行状态与终止原因（对外契约）
│   │   ├── trace.py        # 结构化执行轨迹（可回放 / 可评估）
│   │   ├── handoff.py      # 终止即交接（交接包）
│   │   ├── engine.py       # 主循环（全仓唯一的 Agent Loop 实现）
│   │   └── factory.py      # 装配入口（配置 → 护栏参数）
│   ├── dataplatform/       # 【项目四】数据层
│   │   ├── schemas.py      # 数据契约：raw / standard / staging / article / chunk / sample / evidence
│   │   ├── ingest.py       # 按来源分解析器 + 时间/身份归一化（失败保留原始标识）
│   │   ├── dedupe.py       # 两级去重：消息级精确 + 语义单元级近重复（bigram Jaccard）
│   │   ├── scrub.py        # 脱敏规则集 + 占位符 + 命中类型统计
│   │   ├── filters.py      # 过滤规则链（每条带 rule_id / reason / 阶段）
│   │   ├── chunking.py     # 切片（→ 阶段 2 增加 L1/L2/L3 滑动窗口 + Auto-merging，保留语义边界模式）
│   │   ├── review.py       # 审核状态机（approve/reject/edit + before/after 留痕）
│   │   ├── routing.py      # 用途分流：事实→RAG / 风格→SFT；价格时效治理
│   │   ├── index.py        # 向量索引（过滤在前 ANN 在后 + 一致性核对 + delete）
│   │   ├── retrieval.py    # 混合召回 + 重排 + 预算内上下文组装
│   │   ├── rerank.py       # 【阶段 2 新增】RerankPort（本地默认 + Jina 可选）
│   │   ├── training/       # 【阶段 2 新增】DPO 偏好对构造 + LLaMA-Factory 数据集与 YAML 配置产物
│   │   ├── reflux.py       # 【阶段 2 新增】纠偏/审核/Tool/ HITL /直播 ASR 回流
│   │   ├── quiz.py         # 【阶段 2 新增】AI 美妆顾问自动出题 + 自动评分
│   │   ├── evaluation.py   # Recall@K / Hit Rate@K / MRR + 分组切分 + 泄漏自检
│   │   ├── export.py       # ShareGPT / Alpaca / OpenAI 导出 + 校验门禁 + 版本
│   │   ├── lineage.py      # 血缘有向图（ancestors 追溯 / descendants 影响面）
│   │   ├── pipeline.py     # 编排入口（含 stage 计时与分阶段统计）
│   │   ├── service.py      # 门面：按请求装配流水线并缓存最近一次产物
│   │   ├── fixtures.py     # 内置脏数据样例（离线可回归）
│   │   └── factory.py      # 由 Settings 装配
│   ├── commerce/           # 宿主业务后端（全层已写：domain / application / infrastructure / interface + service / factory / tools_port）
│   │   ├── domain/         # values / models / errors / services / repository / events / security / idempotency / ids
│   │   ├── application/    # ports / ai_hooks / events / usecases（catalog·cart·order·member·coupon）
│   │   ├── infrastructure/ # db(session+models) / repositories / cache / scheduler / seed / uow
│   │   ├── interface/      # deps / schemas / routers（/v1/shop/*、/v1/admin/*）
│   │   ├── service.py      # 门面
│   │   ├── factory.py      # 装配
│   │   └── tools_port.py   # 供 infra 白名单 Tool 消费的业务端口实现
│   ├── agents/             # 【项目一】AI 智能客服导购系统（阶段 4 已写全层）
│   │   ├── schemas.py            # 编排契约：AgentInvocation / SubTaskResult / RoutePlan / OrchestrationResult
│   │   ├── orchestrator.py       # 复杂度路由（规则优先 / 模型兜底 + 计划逐项校验）：简单查询直连 Tool，复杂跨域才进 Agent
│   │   ├── customer_agent.py     # Customer Agent：肤质测试/成分咨询/方案推荐/订单售后
│   │   ├── operations_agent.py   # Operations Agent：经营分析/质检/卖点/运营报告
│   │   ├── orchestration/        # Sequential / Fan-out-Fan-in / Handoff + OrchestrationMode Protocol + Registry
│   │   ├── grounding.py          # 交易类回答必须绑定 Tool Result 或知识来源（Grounding Gate）
│   │   ├── prompt_guard.py       # 分层 Prompt 治理 + fencing 清洗（防跨用户上下文污染）
│   │   ├── session.py            # Session Rehydrate + 摘要记忆（复用 infra/memory）
│   │   ├── a2a.py                # Specialist Agent A2A /message:send + session_id 透传
│   │   ├── hitl.py               # Stage→Approve→Apply + checkpoint + Resume Token
│   │   ├── realtime.py           # RTC→ASR→Orchestrator→RAG/Agent→LLM→TTS（端口 + Mock，事件流打通）
│   │   ├── retrieval_strategy.py # Step-Back / HyDE 查询重写
│   │   ├── feedback.py           # 纠偏/优质回复/失败案例 → Prompt 与 RAG few-shot 示例
│   │   ├── service.py            # 门面：一次装配所有子能力
│   │   ├── router.py             # /v1/agents/* 与 /v1/a2a/message:send
│   │   └── factory.py            # 装配
│   └── ops/                # 【项目二】AI 运营系统（阶段 5 新建）
│       ├── __init__.py
│       ├── schemas.py            # 契约：ContentBrief/NodeRecord/WorkflowResult/ContentAsset/AssetHit/Metric/Attribution/EvolutionCycle
│       ├── workflow/             # graph（节点+边+DAG 拓扑分层） + checkpointer（节点级持久化/回滚/重试/续跑/回放） + nodes.py + schemas.py
│       ├── streaming.py          # SSE + 事件级流式输出（客户端断开即取消后台 run）
│       ├── parallel.py           # asyncio.gather + run_in_executor 并行生成（实测墙钟加速比 + 失败隔离）
│       ├── multimodal.py         # rag_assets + OCR/VLM/ASR 端口 → retrieval_text（(source_table,source_id) UPSERT 幂等）
│       ├── live.py               # 直播分段转录 + 分批分析 + 重叠去重（FFmpeg 仅构造命令 + 降级声明）
│       ├── analytics.py          # 运营指标 + 归因 + 素材（爆款评分/标题/剪辑建议）
│       ├── selfevolve.py         # 回流 → 自动评测集/自动评分 → 模板与策略迭代（只登记不激活）
│       ├── service.py            # 门面：一次装配所有子能力
│       ├── router.py             # /v1/ops/* （独立 scope ops:rw）
│       └── factory.py            # 装配
├── scripts/
│   └── smoke.py            # 端到端冒烟脚本
└── tests/                  # pytest 用例
    ├── conftest.py         # 共享夹具：app_client（含 lifespan）与各项目演示 key
    ├── test_*.py           # 分层单测：infra / harness / dataplatform / commerce / agents / ops
    └── test_stage6_integration.py   # 阶段 6：架构守卫（依赖方向 / 唯一组合根 / 路由共存）+ 凭证隔离矩阵
```

约定：
- 所有业务代码必须**全异步**（`async def`），禁止在请求路径中调用阻塞 IO。
- 模块内部用**相对导入**，跨项目引用走 `src.infra.*` / `src.harness.*` / `src.dataplatform.*` 绝对导入。
- 每个包必须有 `__init__.py`，对外只暴露公共接口。
- 禁止在代码里硬编码密钥，一律 `os.environ` / `.env`。
- `config/models.yaml` 中**已有三条 api_key 的 scopes 不得改动**；新增能力只能另加 key。
- `src/infra/container.py` 是**唯一组合根**；`src/infra/api.py` 负责把各项目的 router 挂到同一应用。

## 4. 工程化硬性门槛（重点，AI 必须遵守）

以下 8 条是「工程化能力」的验收项。**每一条都要能在面试中被追问到具体实现**，因此不得用伪代码糊弄。

### 4.1 结构化日志（logging）
- 统一 JSON 结构化日志，字段固定：`ts / level / logger / trace_id / project_id / stage / duration_ms / status / event / msg`。
- 通过 `contextvar` 自动注入 `trace_id / project_id / session_id`，业务代码不需要手动传。
- **默认不记录完整用户隐私、密钥、未脱敏对话内容**；敏感字段走脱敏函数。
- 关键节点必须打点：请求入口、鉴权结果、权限判定、配额预留/结算、每次模型调用、每次工具调用、异常。
- 实现位置：`src/infra/core/logging.py`。

### 4.2 链路追踪（tracing）
- 入口生成或接收 `trace_id`（HTTP header `X-Trace-Id`），写入 `contextvar`；**缺失请求头时继承当前环境上下文**，避免内外层断链。
- 跨进程/跨服务调用把 `trace_id` 放进 HTTP header 透传；工具内部通过上下文取用。
- 提供 `stage()` 计时上下文管理器，记录每个阶段的 `start / duration_ms / status`。
- Trace 只做**请求关联**，不是身份凭证，不能当鉴权 token 用。
- 实现位置：`src/infra/core/tracing.py`；统一出口口径见 `src/infra/telemetry/`（阶段 1）。

### 4.3 异常体系（exception handling）
- 定义错误分类树：`InfraError` → `ValidationError / AuthError / ForbiddenError / NotFoundError / QuotaExceededError / UpstreamError(→TimeoutError / RateLimitError / ProviderError) / InternalError`。
- 每个错误携带：`code`（稳定错误码）、`http_status`、`retryable`（是否可重试）、`safe_message`（可暴露给模型/用户的摘要）、`detail`（仅进日志）。
- 统一异常处理器把内部异常映射为统一响应体 `{success, error:{code,message,trace_id}}`，**绝不把堆栈或密钥返回给调用方**。
- 实现位置：`src/infra/core/errors.py`。

### 4.4 重试机制（retry）
- 只重试 `retryable=True` 的错误（网络超时、限流、5xx）；参数错误、鉴权失败、业务拒绝**不重试**。
- 指数退避 + 随机抖动（避免重试风暴），带**重试预算**（次数上限）与整体 deadline。
- 非幂等操作必须显式声明幂等键后才允许重试。
- 每次重试记录 `event=retry`，包含 `attempt / error_code / sleep_ms`。
- 实现位置：`src/infra/core/retry.py`。

### 4.5 异步与并发控制（async）
- 全链路 `async`；模型、数据库、Redis 均使用异步客户端。
- 对外部依赖设置**超时**、**并发信号量**；同步阻塞逻辑用 `run_in_executor` 隔离。
- 独立只读调用允许并发（`asyncio.gather`），有依赖关系的调用必须串行。
- 流式响应必须支持客户端断开后的资源释放。

### 4.6 子 Agent 能力（sub-agent）
- 平台层通过**工具调用循环（Agent Loop）**为上层提供子 Agent 能力：模型决策 → 工具执行 → 观察回填 → 再决策。
- 必须有**迭代上限、总超时、Token 预算**三重护栏，防止无限循环与成本失控。
- 每个子步骤输出结构化结果（`text / tool_calls / usage / steps`），一次编排调用返回**完整结果契约**（含 token 用量用于成本统计）。
- 多子 Agent 分派时，任务依赖必须**显式声明**并把前序输出注入后序 prompt。
- 全仓**只有一个** Agent Loop 实现：`src/harness/engine.py`（原 `infra/agent` 已删除，避免「两套循环」）。项目一、项目二都复用它。
- 多 Agent 编排（Sequential / Fan-out-Fan-in / Handoff、Orchestrator 复杂度路由）在项目一 `src/agents/orchestration/`，以 `OrchestrationMode` Protocol + Registry 统一事件流。

### 4.7 配额与限流（quota / ratelimit）
- 配额区分**请求数配额**与 **Token 配额**：采用**预留-实扣两步结算**（请求前按输入长度与 `max_tokens` 估算预留，结束后按真实 `usage` 多退少补，失败/超时释放），并发下用原子操作避免超发，预留带过期时间防泄漏。实现位置 `src/infra/quota/`。
- 限流区分于配额：基于 **Redis ZSET + Lua 的滑动窗口**，按 IP / Service 维度控制；Redis 不可用时降级为内存实现，并 **fail-open**（保障核心客服链路可用）。实现位置 `src/infra/ratelimit/`（阶段 1）。
- 成本侧：采集 input/output token、模型、耗时、调用次数，按 Service/Agent 聚合；支持单 Agent 日预算与单请求 Max Token，**超阈值自动切换 Lite 模型**。

### 4.8 可观测与健康检查
- `/healthz`（存活）与 `/readyz`（依赖就绪：数据库、Redis、配置）。
- 每次请求记录：`trace_id / 耗时 / 状态 / 模型 / 工具 / token / 错误码`。
- **权限审计链**：`Service → Agent → Tool → Resource` 四级，Tool 调用记录 `caller / agent / tool / resource`，形成完整审计链。
- 提供轻量指标聚合（请求数、错误率、P95 延迟、Token 消耗）；错误率统计阈值必须包含 4xx（`status >= 400`），否则失败请求在监控里隐形。
- 统一 Trace 出口走 `src/infra/telemetry/`（默认 Console 实现，OTel 为可选适配）；不引入重型依赖。

## 5. 技术栈与运行方式

| 用途 | 选型 | 说明 |
| --- | --- | --- |
| Web 框架 | FastAPI + Uvicorn | 全异步 |
| 数据校验 | Pydantic v2 | 请求/响应/工具 Schema 统一 |
| 数据库 | SQLAlchemy 2.x async + aiosqlite / asyncpg | 本地默认 SQLite，生产切 PostgreSQL |
| 缓存/配额/限流 | Redis（可选） | 未配置 Redis 时降级为进程内原子实现 |
| 模型接入 | OpenAI 兼容协议（火山方舟 Ark） | 默认 MockProvider，凭据走环境变量 |
| 工具协议 | MCP（Streamable HTTP） | 服务可拆分部署；进程内也提供等价调用路径 |
| 配置 | YAML + 环境变量 | 不可变快照 + 原子切换 |
| 测试 | pytest + pytest-asyncio + httpx ASGITransport | 不依赖外网 |
| 编排 | **原生等价实现** | 不使用 LangGraph / Agent Framework 运行时依赖 |

运行：

```powershell
# 1) 依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 2) 配置
Copy-Item .env.example .env
# 3) 启动（MCP 端点需要 INTERNAL_SERVICE_TOKEN；非回环部署还需 MCP_ALLOWED_HOSTS）
.\.venv\Scripts\python.exe -m uvicorn src.infra.api:app --port 8000
# 4) 冒烟：进程内起应用，逐条断言端到端链路
.\.venv\Scripts\python.exe scripts\smoke.py
# 5) 单元/契约测试：覆盖异常分支与边界
.\.venv\Scripts\python.exe -m pytest
```

全部用例默认走 `PROVIDER_MODE=mock`，不需要任何真实密钥，也不出网。

## 6. 模块清单与验收标准

### 6.1 项目一：AI 智能客服导购系统（agents）

主线：**Orchestrator 复杂度路由 → 双端 Agent → 多模式编排 → 事实绑定 → 人机协同**。
核心认知：简单查询直连 Tool，复杂跨域任务才进 Agent 推理，**不按数据域堆叠 Agent**。

| 模块 | 必须做到 | 验收方式 |
| --- | --- | --- |
| Orchestrator 复杂度路由 | 判断任务是否跨域/多步：单域查询直连白名单 Tool（不花模型推理），跨域才分派给 Agent | 单域查询不产生 Agent 推理；跨域请求正确分派 |
| Customer Agent | 肤质测试 / 成分咨询 / 护肤方案推荐 / 订单与售后，复用 harness 循环 | 四类意图各有端到端用例 |
| Operations Agent | 经营分析 / 客服质检 / 商品卖点 / 运营报告 | 同上 |
| 多模式编排 | Sequential / Fan-out-Fan-in / Handoff 三种模式，统一 `OrchestrationMode` Protocol + Registry 与事件流 | 三种模式各自单测；Registry 按名分发 |
| Model-driven Agent Loop | 复用 `harness/engine.py`：ToolSelection→Execution→Observation→Replanning→Termination | 不新增第二套循环实现 |
| 双通道事实获取 | 美妆知识走 RAG；价格/库存/优惠/订单走白名单 Tool（Tool-based Data Access，**不做 Text-to-SQL**） | 价格类问题不经模型臆测 |
| Grounding / Guardrails | 交易类回答必须绑定 Tool Result 或知识来源，否则拒答或降级 | 无来源的交易类回答被拦 |
| Session Rehydrate + 摘要记忆 | 从 PostgreSQL 恢复历史；旧消息自动摘要注入 System Prompt 控制 token | 长对话 token 不随轮次线性膨胀 |
| A2A | 跨服务 Specialist Agent 走 `POST /v1/a2a/message:send`；同服务优先函数/Tool 调用 | 端点可测；session_id 正确透传 |
| HITL | 退款/改价/发券：Stage→Approve→Apply，checkpoint + hitl_request + Resume Token 恢复 | 未审批不落地；凭 token 可续跑 |
| Prompt 治理 | 分层 Prompt + fencing 清洗，防 Prompt Injection、越权 Tool Calling、跨用户上下文污染 | 注入样本被 fencing 隔离 |
| 实时语音 | RTC→ASR→Orchestrator→RAG/Agent→LLM→TTS 事件流打通（端口 + Mock 实现） | 全链路事件流单测；不宣称真实媒体服务器 |
| 查询重写 | Step-Back / HyDE 提升长尾问题召回 | 长尾问题召回有对比 |
| 反馈闭环 | 人工纠偏/优质回复/高频失败案例沉淀为 Prompt 与 RAG few-shot 示例 | 沉淀链路可测 |

**当前实测状态**：**已实现**（阶段 4 交付；pytest 214 全绿、smoke 109/109 通过）。

实测覆盖（对应上表逐条验收）：
- 复杂度路由：订单号/商品号/知识问题走**规则直连**（`llm_calls=0`，不产生 Agent 推理）；只有跨域任务才升级为 `kind=agent`（smoke 17.1-17.3）。
- 编排模式：Sequential（显式依赖注入 `notes`）/ Fan-out-Fan-in（按依赖分层并行 + fan-in 收敛）/ Handoff（交接包 + 转派次数上限）各有单测；`ModeRegistry` 按名分发。
- 双端 Agent：Customer / Operations 复用共享 `harness/engine.py` 的**唯一 Agent Loop**，未新增第二套循环。
- Grounding Gate：无来源的交易类断言被**拦截改写**（`stop_reason=grounding_blocked`、`enforced=true`），有交易断言但带明确不确定措辞时降级为 `transaction_claim_hedged` 放行；门禁关闭时保留原答（smoke 17.6）。
- HITL：Stage（登记 + HMAC Resume Token 绑定 checkpoint/action/args）→ Approve（**独立权限 `hitl:approve`** + token 校验）→ Apply（仅 approved 可执行，且以**原始请求者身份**执行）；无权限 403、篡改 token 403 均实测（smoke 17.7-17.9）。
- A2A：`/v1/a2a/message:send` 用内部共享令牌（`X-Internal-Token`，常量时间比较）鉴权，透传 `session_id`；未配置令牌直接拒绝、错误令牌 401（smoke 17.10）。
- 实时语音：端口 + Mock，文本轮与模拟上行语音轮的**事件流按序**可算各环节耗时（smoke 17.11-17.12）。
- 检索重写：none / step_back / hyde 三策略召回可对比（`delta` 给出命中数与 top_score 差值）；检索需 `tools:knowledge`，越权 403（smoke 17.13）。
- 反馈闭环：纠偏记录投影为 Prompt / RAG 两类 few-shot；收集器跨请求存活、只增不改、有界（smoke 17.14）。
- Session / Prompt Guard：会话记忆复用 `infra/memory` 的 token 预算与摘要；注入样本被 `<untrusted>` 围栏隔离、内容内围栏标记被转义（防提前闭合）。


### 6.2 项目二：AI 运营系统（ops）

主线：**工作流图 + Checkpointer → 流式/并行生成 → 多模态知识化 → 直播处理 → 归因 → 自进化**。

| 模块 | 必须做到 | 验收方式 |
| --- | --- | --- |
| 运营内容工作流 | 选题 / 成分卖点 / 文案 / 产品图 拆为子图（节点 + 边）；**Checkpointer 做节点级状态持久化、回滚、失败重试** | 中途失败可从上一节点续跑；节点状态可回放 |
| Structured Output | 各节点输出用 Pydantic 约束，不合法则重试或回退 | 非法输出被拒并记录原因 |
| 模型分工降本 | 轻量模型处理短文案、旗舰模型处理复杂分析与策略，逻辑模型名走 infra catalog | 两条路径可切换 |
| 流式生成 | SSE + 事件级流式输出 token | 客户端断开资源被释放 |
| 并行生成 | `asyncio.gather` + `run_in_executor` 并行生成产品图与文案 | 依赖关系正确的部分才并行 |
| 多模态知识化 | 产品图/详情页/教程视频/直播切片统一为 `rag_assets`，OCR/VLM/ASR → retrieval_text → Embedding；`(source_table, source_id)` UPSERT 幂等 | 文本查询能命中图片/视频；重复导入不产生重复资产 |
| 直播处理 | Video→ASR→LLM→FFmpeg→切片；长直播按时间窗分段转录，分批分析并做重叠去重；FFmpeg **仅构造命令与降级声明** | 分段/分批/去重用离线转录样例单测 |
| 运营分析 | 汇聚投放/客服/直播数据构建指标与归因，产出精彩片段、爆款评分、标题与剪辑建议 | 指标与归因可复算 |
| 自进化闭环 | 人工审核/线上反馈/客服纠偏/高频失败案例回流 → 自动评测集 + 自动评分 → 素材模板与生成策略迭代 | 回流到模板的链路可测 |

**当前实测状态**：**已实现**（阶段 5 交付）。`pytest` 258 条全绿（新增 `tests/test_ops_stage5.py` 44 条）；`scripts/smoke.py` 126 项全通过（新增第 18 段 17 项端到端断言）。

- 运营内容工作流（图 + Checkpointer）：选题 → 成分卖点 →（文案 ‖ 产品图）四节点成功，`models=['chat-lite','chat-pro']`，结构化作答降级留痕 4 处（smoke 18.2）。
- 节点级续跑：回滚 `copy` 后仅重跑该节点，其余节点复用已完成输出（`resumed_from=['image','selling_points','topic']`，`chat-lite` 增量调用 = 2）（smoke 18.4）。
- 回放留痕：每次尝试都记录（含 `rolled_back`），不是只留最终状态（`entries=6`）；未知 run 查询 → 404（smoke 18.4）。
- 成本可核算：编排用量按节点汇总（`tokens=1280 / cost=0.002112`）（smoke 18.3）。
- 流式生成：SSE 事件级推进（`run.start/node.start/token/run.done`）+ 回传 `X-Run-Id`；客户端提前断开 → 后台 run 取消并记 `aborted`（smoke 18.5-18.6）。
- 并行生成：`gather` + 线程池并行，墙钟 < 串行下界（`speedup≈3.96`，`wall=51ms / seq=203ms`），并发度受限；单点失败不取消同批其它单元（smoke 18.11）。
- 多模态知识化：同一 `(source_table, source_id)` 重复导入 `asset_id` 不变、计数不涨（`created=True→False`）；文本查询同时命中图片与页面素材（smoke 18.7）。
- 直播处理：分段转录 + 重叠去重（`trimmed_chars=44`）+ 分批分析 + 切片候选（`segments=3`）；FFmpeg 只构造命令并如实降级声明（`degraded=True reason=ffmpeg_not_installed`）（smoke 18.8）。
- 运营分析：CTR/CVR/ROAS 与明细一致可复算（`ctr=0.0389 roas=4.1053`），归因份额和为 1 且不经模型计算；素材建议有排序并给出标题与剪辑指令（smoke 18.9）。
- 自进化闭环：回流 → 自动评测集（`cases=3`）→ 评分 → 模板建议（`v1→v2` 只登记不激活，需人工审核）；`apply` 生效、`revert` 按版本顺序回滚（smoke 18.10）。

### 6.3 项目三：AI 能力中台与统一调度网关（infra）

主线：**Provider 抽象 → 模型网关 → MCP → RAG/Memory/Prompt → 鉴权权限 → 配额限流 → 成本 → 可观测+评测**。

| 模块 | 必须做到 | 验收方式 |
| --- | --- | --- |
| 配置快照 | 解析→校验→原子替换；校验失败保留旧配置；请求内只取一次快照 | 修改坏配置后 reload 抛错且旧配置仍可用 |
| LLM Provider | 三层抽象：统一出入参、统一响应结构（text/tool_calls/usage/finish_reason）、统一错误类型 | Mock 与真实适配器可互换，上层零改动 |
| 模型路由 | 逻辑名 `chat-lite/chat-pro/vision` → 物理端点；能力声明校验 | 缺能力时明确拒绝而非静默降级 |
| MockProvider | 可脚本化：第 N 步返回工具调用、第 N+1 步返回终答 | 离线覆盖多步循环与异常分支 |
| MCP 工具服务 | 工具含 name/description/input_schema/权限/超时/风险等级/幂等标记 | 服务端二次校验参数，不信任模型入参 |
| REST-to-MCP 桥接 | 鉴权（API Key→project_id/scopes）→ 注入 contextvar → 调用 → 统一包装 trace_id | 客户端传的 project_id 一律忽略 |
| 会话记忆 | `messages` 与 `user_profiles` 两张表；画像 UPSERT + 字段级合并；上下文按 token 预算组装 | 并发写不覆盖；跨项目查询被拒 |
| 配额 | 预留-实扣；失败释放；预留带过期 | 并发请求不超发 |
| 滑动窗口限流 | Redis ZSET + Lua 原子判定，IP/Service 维度；内存降级；fail-open | 超限被拒；Redis 故障链路不中断 |
| 四级权限链 | `Service→Agent→Tool→Resource` 全链传导，统一审计事件 | 缺任一维度权限即拒绝并留痕 |
| 成本 Metering | 按 Service/Agent 聚合 token/耗时/次数；日预算 + Max Token；超阈值切 Lite | 预算耗尽行为可测 |
| Telemetry | `TelemetryPort` 统一 Trace，Console 默认实现 | 每个模型/工具/A2A/RAG 调用都有 span |
| Evaluation Pipeline | 覆盖准确性 / Grounding / Tool Selection / Safety / 成本 / 延迟 | 六维度各有可运行用例 |
| 可观测 | 结构化日志 + trace 贯穿 + 阶段耗时 + 指标 | `/metrics` 能看到 token 与延迟 |

**当前实测状态**：既有能力（Provider/路由/MCP/桥接/记忆/配额/日志/追踪/指标）由 `pytest` 159 条 + `scripts/smoke.py` 83 项全绿覆盖。
**阶段 1 交付**（限流 / Telemetry / 四级权限链+审计 / Metering 聚合与预算降级 / YAML Prompt / Eval）已端到端验证：
`/v1/audit`（allow 与 deny 双向留痕、按 project 隔离）、`/v1/cost`（按 Service/Agent 聚合 + 当日预算水位）、
`/v1/prompts` 与 `/v1/prompts/{role}/preview`（`prompt:rw` 边界）、`/v1/eval/run`（七用例六维全通过、`eval:rw` 边界）、
Service 维度 `429 rate_limited`（与配额 `quota_exceeded` 区分）、预算软阈值降级与硬上限 429 均有断言。
未实测部分见第 8 章诚实边界（Redis ZSET 路径、OTel 适配器）。

### 6.4 项目四：AI 数据闭环与美妆知识工程平台（dataplatform）

主线：**Raw→Clean→Staging→Review→Approved→AI Dataset**，并由线上反馈回流驱动迭代。
核心认知：模型能力不能掩盖数据质量 —— 脏数据进来，只会被更快地放大。

| 模块 | 必须做到 | 验收方式 |
| --- | --- | --- |
| 接入与标准化 | 按来源分解析器（会话导出 / 商品知识 / 直播 ASR），统一 schema；时间吃 ISO/epoch/毫秒 | 三类来源混排解析；垃圾时间直接报错不猜 |
| 身份映射 | 映射失败标 `partial` 并保留原始标识（raw_speaker），**绝不猜一个身份出来** | 未映射发送者不会变成某个客服，且进人工审核 |
| 容错边界 | 单条脏记录不影响整批（失败计入 failed 计数并告警） | 一批里混坏记录，其余照常产出 |
| 两级去重 | **消息级精确**（session+时间+内容哈希）+ **语义单元级近重复**（bigram Jaccard，阈值 0.42，**仅用正文不用角色名**） | 精确重复删消息；近重复删**整轮**，不拆散问答对 |
| 脱敏 | 规则集（email / 身份证 / 银行卡 / 手机号 / api_key）+ 占位符 + **日志只记命中类型** + 导出门禁二次扫描 | 明文不进日志与训练集；19 位卡号不会被手机号规则二次命中 |
| 过滤规则链 | 每条规则带 `rule_id` / `reason` / 阶段；确定性问题直接剔除，不确定的转人工 | 空内容/系统通知/过短/广告/乱码各命中；可统计命中率 |
| Session 构建与预标注 | 时间窗 + 角色构建多轮美妆咨询 Session；预标注肤质分类 / 成分咨询 / 销售意图 / 质量分 / 敏感度 | 预标注字段与统计可核对 |
| 切片 | **L1/L2/L3 滑动窗口 + Auto-merging**（保留现有语义边界模式，可切换）；token 上限 220 + overlap 40 | 边界生效；超长正文按 token 切开且不越界 |
| 审核状态机 | pending / approved / rejected + reviewer + reason + **before/after 修改留痕**；驳回必须给理由 | 无理由驳回被拒；编辑后仍需再次通过才生效 |
| 用途分流 | 事实（成分/功效/适用）→ RAG；稳定的表达风格 → SFT；**价格时效治理**：价格条目只留 live_source，不进语义索引 | 价格条目不在索引里；SFT 样本强制一问一答 |
| 向量索引 | **元数据过滤在前、ANN 在后**；记录 HNSW 参数与 `embedding_version`；一致性核对 + delete 防孤儿 | 预过滤生效；下架内容主表与向量同步归零 |
| 检索与组装 | 混合召回（向量 0.6 + 关键词 0.4）→ 重排 → 预算内组装（去重 + 显式声明截断 + `[chunk_id]` 标注） | 检索块带来源标注；超预算显式告知被截断 |
| Rerank Port | `RerankPort` 抽象：本地词典重排为默认（离线可跑），Jina 为可选实现（需 key） | 不配 key 走本地；配 key 走远端 |
| 训练数据 | DPO 偏好对构造 + LLaMA-Factory **可直接消费的数据集与 YAML 配置**（不在仓库跑 GPU 训练） | 产物 schema 合法；配置可被 LLaMA-Factory 读取 |
| 评估 | Recall@K / Hit Rate@K / MRR + **分组切分防泄漏** + `leak_check` 自检 | 标准集达标；同组不跨 split；近同题跨集被检出 |
| 训练导出 | ShareGPT / Alpaca / OpenAI 三种结构转换 + 校验门禁（too_short / empty / role_order / duplicate / sensitive）+ 版本统计 | 每条不合规样本都被拒并给出原因 |
| 回流 | 客服纠偏 / 人工审核 / Tool 调用 / HITL / 直播 ASR 回流为训练与评测数据 | 各来源回流各有用例 |
| 自动出题评分 | AI 美妆顾问考核：自动出题 + 自动评分 | 出题与评分链路可测 |
| 血缘 | raw→standard→staging→article→chunk→vector→export 有向图；ancestors 反向追溯 + descendants 影响面 | 能从切片一路追回原始记录 |
| 对外接口 | `/v1/data/pipeline/run`、`/search`、`/evaluate`、`/export`、`/lineage/{node}`、`/training/export`、`/reflux`（GET/POST）、`/quiz`，需独立 `data:rw` scope | 会话凭证（llm:chat）被拒；未跑流水线查询返回 404 |

**当前实测状态**：项目四全部模块（含阶段 2 新增的三级分块 / RerankPort / training / reflux / quiz）
均有单元用例 + 冒烟断言覆盖；本次实测 `pytest` **159 条**、`scripts/smoke.py` **83 项**全绿。
`pipeline/run` 为唯一写接口，其余只读消费「最近一次 run 的产物」，且**每次请求都新装一条流水线**
（避免跨请求脏状态与索引累积）。

阶段 2 交付要点与实测口径：
- **三级切片**：`semantic`（默认，语义边界，行为未变）/ `hierarchical`（L1 句子滑动窗口 → L2 段落 → L3 章节）双模式可配；
  超限节点用同一套 token 上限 + overlap 再切，且**首片保留原 node_id**（否则子块 parent 链断裂，Auto-merging 永远合不上）。
- **Auto-merging**：检索单元只有 L1（父块仅作合并上下文，避免长块向量系统性压过精准小块）；
  同父块下命中子块占比 ≥ `DATA_MERGE_RATIO`（默认 0.5）即合并回父块，冒烟实测合并 2 个子块为 1 个 L2。
- **RerankPort**：本地词典重排为默认（离线可跑，实体命中 +0.25 / 缺实体 −0.15）；
  `DATA_RERANK_BACKEND=jina` 但未配 key 时**显式降级**并记录原因，不静默换后端。
- **training**：DPO 偏好对来自审核编辑留痕（before/after）与回流改写；产出 LLaMA-Factory 可直接消费的
  ShareGPT SFT 数据集 + `ranking: true` 的 DPO 数据集 + `dataset_info.json` + `stage: dpo` 训练 YAML；
  偏好对与 SFT 样本共用同一套导出门禁（too_short / duplicate / sensitive）。
- **reflux**：五来源（客服纠偏 / 人工审核 / Tool 调用 / HITL / 直播 ASR）归一化为
  「提问 / 回答 / 否定回答 / 金标切片 / 分组」，再分别投影为训练样本、DPO 偏好对与带 `group` 的评测金标；
  脏事件被拒并记录 `missing` 原因。收集器**刻意跨请求存活**（回流是多次事件累积出来的），且只增不改、有界。
- **quiz**：从已审核知识条目自动生成事实型 / 边界型 / 拒答型三类题目（拒答型来自 `price_freshness` 条目）；
  自动评分为**启发式**（证据一致性 0.6 + 金标要点覆盖 0.4；拒答型必须命中「引导官方渠道」策略），用于粗筛与回归防退化。

### 6.5 共享：执行内核（harness）

| 模块 | 必须做到 | 验收方式 |
| --- | --- | --- |
| Decision 协议 | 供应商差异收敛成 FINAL / TOOL_CALLS 两类；文本+工具调用混排以工具调用为准，文本记为前言 | 纯文本、纯调用、混排、空输出四种输入各有单测 |
| 派发前校验 | 四道闸门：工具存在且在可见集 / 项目白名单 / 高风险确认 / 参数 Schema+每轮数量+权限；**可恢复问题转 observation 回填**，终止性问题才中断 | 未知工具/参数非法→模型自纠错后仍能完成；越权/高风险→终止并交接 |
| 执行策略（policy） | 工具白名单与高风险确认独立成层，不写进工具实现 | 高风险动作未确认时不执行、直接交接 |
| 三重护栏 | ① 静态上限（轮数/总时长/token/成本）② 打转检测（参数指纹+时间窗+合法轮询豁免）③ 连续失败熔断 | 每项单独用例；`max_steps` 支持按任务收紧（**本次生效值须真正传入护栏**） |
| 观察回填（observation） | 对齐 `tool_call_id`；超大结果分页且**显式声明被截断**；参数错误带 issues；历史按「轮」保留并压缩（不产生孤儿 tool 消息） | 轨迹里能看到截断标记与压缩标记 |
| 结构化执行轨迹 | 记录每轮决策/调用/错误，**只记参数指纹不记参数原文**；可回放/可评估 | `trace.digest()` 给出 rounds/stop_reason/error_distribution/tokens |
| 终止即交接（handoff） | 任何非正常结束都带：reason + summary + completed_steps + tool_results + failure + next_actions | 预算耗尽/熔断/高风险/越权四条路径均有交接包 |
| 循环不变量 | 一轮 = 一次模型决策 + 一批工具执行；护栏放在轮首（超限请求不再花一次模型调用） | 单测覆盖 |
| 响应契约 | `/v1/agent/run` 保持 `success/stop_reason/steps/tool_calls/usage/transcript/answer`，并附加 `handoff` 与 `trace` | 既有桥接测试与冒烟断言不变 |

**当前实测状态**：用例与冒烟中的 5 项 Harness 检查（未知工具自纠错、连续失败熔断、高风险确认门、确认后执行、轨迹摘要）全部通过。

### 6.6 共享：宿主业务后端（commerce）

定位：项目一白名单 Tool 的**真实业务数据源**（商品读、库存读、购物车、下单幂等、订单状态/取消/退款、会员、优惠券）。
只做**项目一真正需要的用例**，不做 CMS / 报表 / 后台管理页面。

| 模块 | 必须做到 | 验收方式 |
| --- | --- | --- |
| 领域模型 | 商品/色号/SKU/库存/批次/购物车/订单/会员/优惠券分离；**库存独立成表**以便单条条件 UPDATE 原子扣减 | 领域不变量单测 |
| 原子扣库存 | 单 SQL 条件 UPDATE（`available >= qty`），带版本号；**不变量 `available >= 0`** | 并发扣减不超卖 |
| 下单幂等 | `(project_id, idempotency_key)` 唯一约束兜住并发重放 | 重放返回同一订单 |
| Outbox | 事件与业务写在**同一事务**提交，避免两步之间崩溃丢事件 | 事务内写入可断言 |
| UoW | 请求级工作单元，提交/回滚边界清晰 | 回滚后无残留写 |
| 接口 | `/v1/shop/*`、`/v1/admin/*` 只读为主；写操作经白名单 Tool | 路由挂载与鉴权正确 |
| 与 infra 对接 | `src/infra/tools/commerce.py` 通过 `CommercePort` Protocol 访问，infra 不依赖 commerce 实现 | 既有 SKU/ORD 断言不变 |

**当前实测状态**：`domain` / `application` / `infrastructure` / `interface` + `service` / `factory` / `tools_port` 与接线**已交付**（阶段 3）。`pytest` 173 全绿（含 21 条 commerce 专项：领域不变量、12 并发抢 4 件**不超卖且余量精确为 0**、幂等回放/同键不同体被拒、UoW 回滚无残留写、取消/超时清理退库存、双凭证三态 401/403/200、工具契约）；`smoke` 94/94（含第 16 段 commerce 端到端：REST 下单幂等、Outbox 按 order_id 匹配、付款/取消状态迁移）。既有 SKU/ORD 工具断言**未放宽、未改动**。

### 6.7 统一验证（阶段 6）

定位：前六节各自证明「**一个项目能跑**」，本节证明「四个项目其实是**同一个系统**」——
路由共存于一次装机、鉴权与错误信封跨项目一致、排他权限互不串门、`trace_id` 贯穿全部入口。
这些性质单项目测试**永远测不到**（每个项目自己的用例都只会走自己的凭证和自己的信封），
所以必须由跨项目用例兜住。

| 不变量 | 必须做到 | 验收方式 |
| --- | --- | --- |
| 分层依赖方向 | `infra / harness / dataplatform` 不得反向 import 应用层与业务库（`agents / ops / commerce`）；`commerce`、`agents`、`ops` 之间也不得互相反向引用 | 静态扫描源码 import：违规即失败（`api.py` / `container.py` 为显式例外） |
| 唯一组合根 | `build_agents_service` / `build_ops_service` / `build_commerce_service` 只允许在自己 `factory.py` 与 `container.py` 被调用 | 静态扫描：装配点溢出组合根即失败 |
| 路由共存 | 四个项目与共享底座的六个前缀挂载在**同一个 `create_app()`** 上 | 断言 OpenAPI 路径表含 `/v1/tools`、`/v1/data`、`/v1/agents`、`/v1/shop`、`/v1/admin`、`/v1/ops` |
| 鉴权契约一致 | 各项目受保护端点无凭证一律 401，且错误信封形状相同（`success=false` + `error.code=unauthenticated` + `trace_id`） | 五个端点匿名请求逐条校验信封字段 |
| 排他 scope 隔离 | `ops:rw` / `data:rw` / `commerce:rw` / `eval:rw` 四行「匿名 401 / 越权（客服凭证）403 / 属主 200」三态全对 | 凭证隔离矩阵；任意一行不符即失败 |
| trace 贯穿 | 显式传入的 `X-Trace-Id` 原样回传；未传时生成 `tr_` 前缀，跨项目同一套追踪 | 对 ops 与 data 两个项目各发一次请求比对响应头 |

**当前实测状态**：**已实现**（阶段 6 交付）。新增 `tests/test_stage6_integration.py` 5 条用例 + `scripts/smoke.py` 第 19 段 4 项端到端断言；
全量 `pytest` 263 条全绿、`smoke` 130/130 通过。

- 分层依赖方向、唯一组合根、路由共存均**静态/契约级**验证通过（扫描 `src/` 全量 import 与 OpenAPI 路径表）。
- 凭证隔离矩阵四行全部符合 `401 / 403 / 200`；错误信封在 ops/data/admin/agents/eval 五个端点形状一致（smoke 19.2-19.3）。
- trace 贯穿：`tr_smoke_unified_001` 被原样回传；未传时两个项目都会生成 `tr_` 前缀（smoke 19.4）。

### 6.8 演示前端（阶段 7）

定位：把四个项目从「Swagger 里能调」变成「**能当着面试官点开看**」。
刻意采取**零构建**方案（手写 CSS + 原生 `fetch`，无 npm / 无打包器）：多一条构建链就多一处
「clone 下来跑不起来」的风险，而演示页的全部价值就是「一条命令起服务即可打开」。

| 页面 | 入口 | 凭证 | 调用的接口 |
| --- | --- | --- | --- |
| 公司官网 | `GET /`（`shop.html`） | 浏览：无凭证；交易：会员 token（`Authorization: Bearer`）；导购客服：项目 API Key | `/v1/shop/products`、`/v1/shop/products/{id}`、`/v1/shop/members/*`、`/v1/shop/cart*`、`/v1/shop/orders*`、`/v1/shop/coupons/*`、`/v1/agents/orchestrate` |
| 电商运营台 | `GET /admin`（`admin.html`） | 项目 API Key（`X-API-Key`，可切换演示 401/403/200） | `/v1/admin/orders*`、`/v1/admin/products*`、`/v1/admin/stock/*`、`/v1/admin/coupons/templates`、`/v1/ops/*`、`/v1/cost`、`/v1/quota`、`/v1/audit`、`/v1/prompts`、`/v1/eval/run` |

前端不做「另一种真相」——所有数字、状态、成本、trace 都直接来自后端响应，
每个面板都附**原始响应折叠块**。这是刻意的：演示页如果替后端把数据美化一遍，
面试时反而答不出「这个值到底从哪来」。

| 不变量 | 必须做到 | 验收方式 |
| --- | --- | --- |
| 页面不污染 API 契约 | `/`、`/admin`、`/assets`、`/favicon.ico` 一律 `include_in_schema=False` | OpenAPI 路径断言仍只含六个业务前缀 |
| 权限面不因前端而放宽 | 前端只是换成浏览器发请求，后端 scope 校验一行未改；缺 scope 时页面给出「该切哪把钥匙」的提示而非静默失败 | 凭证矩阵页现场打 401 → 403 → 200 |
| 面向上架运营可复原 | 后台商品列表必须能看到**非在售**商品（否则下架后无法再上架） | `/v1/admin/products` 返回全状态，买家侧 `/v1/shop/products` 仍只有 `ON_SALE` |
| 运营视角订单可见 | 买家侧 `list_by_member` 只看自己的，运营侧需要跨会员视图 | `/v1/admin/orders`（可按状态过滤）与 `/v1/shop/orders` 分离 |

**当前实测状态**：**已交付**。新增 `GET /v1/admin/orders`（跨会员、按状态过滤）与
`GET /v1/admin/products`（全状态）两个只读端点补齐运营台缺口（`repository.list_all` → 仓储 → 用例 → 路由全链）；
`/`、`/admin`、`/assets/*`、`/favicon.ico` 实测返回 200/204；
两个页面均已在真实浏览器中走通渲染与交互（浏览 → 详情 → 加购 → 下单 → 查询订单、AI 导购对话含执行详情、
运营台订单列表与发货、AI 运营工作流跑通并产出文案、凭证 401/403/200 三态矩阵），浏览器控制台**零 JS 报错**。
全量 `pytest` 263 条全绿、`smoke` 132/132 通过（第 16 段新增 2 条运营台只读端点断言）。

## 7. 编码规范

- 类型注解齐全，公共函数必须有 docstring（**工具函数 docstring 直接决定模型调用准确率**，要写清用途、参数含义、返回结构）。
- 常量集中定义，禁止魔法数字；超时、重试、上限一律可配置。
- 所有外部调用（LLM / DB / MCP / Redis）必须经过超时 + 异常归一。
- 日志用 `logger.info("event_name", extra={...})` 风格的结构化字段，禁止 f-string 拼接正文。
- 不在循环里做同步阻塞、不在请求路径里读大文件。
- 测试优先覆盖**异常分支与边界**（未知工具、参数非法、超时、超配额、越权，MCP 挂载路径与 Host 白名单）。

## 8. 诚实边界（面试时必须如实说明）

| 能力 | 当前状态 | 面试口径 |
| --- | --- | --- |
| Provider 抽象 / MockProvider | 已实现 | 直接讲，可上代码 |
| 逻辑模型映射 | 已实现 | 讲快照切换 |
| 配置热更新 | 实现了原子重载接口，**未挂文件监听** | 「设计为快照切换，监听未接」 |
| REST-to-MCP 桥接 | 已实现 | 讲 contextvar 注入，project_id 由凭证推导 |
| MCP Streamable HTTP | 已实测：真实 uvicorn 下 initialize / tools/list / tools/call 全通 | 可讲；但要说清它是**进程内挂载**，尚未做多副本部署与网关鉴权联调 |
| Token 配额 | 已实现（内存/Redis 双实现） | 讲预留-结算；**Redis 路径未在本机实测**，线上并发数不报 |
| 滑动窗口限流 | **已实现并冒烟验证**：ZSET + Lua 原子脚本（淘汰/计数/写入一次 `EVAL`）、IP/Service 双维度、内存降级、Redis 故障 fail-open（拒绝时 429 `rate_limited` + `Retry-After`） | 可讲；**Redis ZSET 路径未在本机实测**（无 Redis 环境），只讲 Lua 原子性与降级策略对应代码 |
| 四级权限链 | **已实现**：`Service→Agent→Tool→Resource` 全链写入统一审计事件；资源由参数推导、拒绝与失败同样留痕、`/v1/audit` 按 project 隔离 | 可讲；审计为进程内有界环形缓冲，生产应落库加索引 |
| OTel 统一 Trace | **已实现 `TelemetryPort`**（Console 默认，`span/event/counter/observe`）；OTel 适配器为延迟导入 + `probe()`，依赖缺失时显式降级 | 如实说「未接 collector，OTel 适配器未实测」 |
| Evaluation Pipeline | **已实现**：七条内置离线用例（Mock 脚本化）+ 六维打分 + 分组报告；runner 复用共享 Harness（评测即线上链路） | 可讲；**不可**宣称评测覆盖了真实流量 |
| YAML Prompt Template Center | **已实现**：`config/prompts.yaml` 为单一事实来源，片段可版本化（同 name 多版、单 active），Role 声明 base + scenario 组装顺序；引用交叉校验在启动期失败 | 可讲；改 Prompt 是「可回滚的发布」 |
| 单 Agent 日预算 + 降级 | **已实现并冒烟验证**：软阈值（默认 0.8）自动切 Lite 并收紧 Max Token，硬上限直接 429；UTC 自然日切分 | 可讲；Mock 成本接近 0，未打出真实金额级路径 |
| 子 Agent 护栏 | 已实现并有单测（迭代上限/总超时/Token 预算/重复调用检测） | 直接讲，可上代码 |
| Harness 执行层 | 已实现并单测：决策归一 / 派发前校验 / 三重护栏 / 观察回填与压缩 / 结构化轨迹 / 终止即交接 | 直接讲，可上代码与轨迹摘要示例 |
| 高风险动作确认门 | 已实现：`create_order` 标为 high_risk，未确认时**不执行、直接终止并交接** | 讲「有副作用的动作不能由模型自己拍板」 |
| 按钱熔断（成本预算） | 已实现但 **Mock 成本恒为 0，未打出真实路径** | 只讲设计与单测，不说线上跑过 |
| 编排框架（Agent Framework / LangGraph） | **不引入依赖，原生等价实现**（DAG + 自研 Checkpointer + OrchestrationMode Registry） | 讲「语义等价，生产可平滑替换为 LangGraph」 |
| 多 Agent 编排 / A2A / HITL | **已实现并单测 + 冒烟**（阶段 4）：三种模式统一 Protocol+Registry、复杂度路由规则优先、A2A 内部令牌、HITL 三步链路 | 直接讲，可上代码 |
| 编排模式（项目一） | Sequential / Fan-out-Fan-in / Handoff 为**原生等价实现**（非 LangGraph），仅覆盖「依赖注入 / 分层并行 fan-in / 交接包」三类语义 | 讲「语义等价，生产可平滑替换为 LangGraph」 |
| Grounding Gate | **已实现**：确定性判定（正则 + 证据存在性），不引入第二个模型；只拦「有具体值的交易断言 + 无本次 Tool 证据」，带明确不确定措辞时放行 | 讲保守判定与为什么不用模型判模型 |
| HITL 存储与 Resume Token | Stage/Approve/Apply 三步 + HMAC 绑定 (checkpoint_id, action, args)；**存储为进程内有界字典**，Token 密钥未配置时回退内部令牌或进程内随机密钥 | 讲机制；生产应落库并接 KMS |
| A2A 内部令牌 | **单一共享令牌**（常量时间比较）只证明「调用方是受信内网服务」；**未做逐服务能力注册与最小权限下发**，调用方声明的 scopes 需由内网网关约束 | 如实说「演示级，未做服务注册中心」 |
| 实时语音 RTC/ASR/TTS | **端口 + Mock，未接真实媒体服务器**：`RTCSession`/`ASRPort`/`TTSPort` 三 Protocol，Mock 把语音退化为帧，事件流打通；不做回声消除 / VAD 断句 / 真流式增量识别 | 如实说「无真实媒体服务器，未做编解码」 |
| 检索重写（Step-Back / HyDE） | **离线启发式实现**（规则改写 + 模板假设），不额外烧模型调用 | 讲流程正确性与可对比性，**不宣称重写质量**；接真网关后可走模型生成 |
| 反馈闭环 | **已实现**：两类投影（Prompt / RAG few-shot）+ 只增不改 + 有界 deque；单机内存实现 | 讲闭环最小可验证性；生产应落库 |
| 会话记忆 | 复用 `infra/memory` 的 token 预算与摘要策略；**未做跨实例共享**（单机内存 + 库内历史） | 讲「长对话 token 不线性膨胀」 |
| HITL 审批凭证 | 由 `config/models.yaml` 新增的 `sk-project-hitl-demo` **演示 key** 承载（`hitl:approve`），与客服凭证分离，未改动既有 key 的 scopes | 直接讲「确认是独立权限」；演示 key 非生产密钥 |
| 项目二 运营工作流 / Checkpointer | **已实现并单测 + 冒烟**（阶段 5）：DAG 拓扑分层 + 节点级持久化 / 回滚（连带下游）/ 失败重试 / 跨请求续跑 / 回放（记录每一次尝试）；**进程内有界**（`max_runs=200`，超限淘汰最旧） | 直接讲，可上代码；如实说「生产应把 records 落表/Redis，用 run_id + node 幂等 UPSERT」 |
| 流式生成（SSE） | **已实现**：事件级推进 + `X-Run-Id` 回传 + 客户端断开即取消后台 run 并记 `aborted`；**不参与预算预扣/重试**（流式路径不上配额预扣） | 讲「断开真的止损」；不多报「流式也走成本预算」 |
| 并行生成实测 | `asyncio.gather` + `run_in_executor` 线程池；**加速比仅对本地阻塞步骤（sleep）实测**（`speedup≈3.96`），非真实模型调用耗时 | 讲「依赖关系正确的部分才并行」+ 失败隔离；不把本地基准当线上性能 |
| 多模态素材知识化 | **已实现**：OCR/VLM/ASR **端口 + 离线等价实现**（`OfflineOcr/Vlm/Asr`，确定性、可复现）；向量为**进程内哈希向量**（`HashingEmbedder`，dim=256），未落 pgvector | 讲「文本查询能命中图片/页面」是靠 retrieval_text 里的**同词命中**，**不是真正的图文跨模态对齐**；换真实云服务只需换端口 |
| 直播处理 / FFmpeg | **已实现**：分段转录 + 分批分析 + 重叠去重（精确裁剪为主、整段近重复丢弃阈值为 **0.85**，兜底）；**FFmpeg 仅构造命令并如实降级声明**（缺二进制 → `degraded=true`），不执行、不产出真实视频 | 讲「宁可少删不可错删」与降级策略；**不宣称真实转码** |
| 运营分析与自进化 | **已实现**：指标（CTR/CVR/ROAS）与归因份额为**确定性计算**（可复算、可解释）；素材评分与自进化评分为**启发式且权重显式写死**；模板迭代**只登记不激活**，需人工 `apply` | 讲「指标/归因不用 LLM，LLM 只负责表达」；**不宣称启发式评分等于 LLM-as-judge** |
| `ops:rw` 权限 | 由 `config/models.yaml` 新增的 `sk-project-ops-demo` **演示 key** 承载（`llm:chat` + `tools:knowledge` + `ops:rw`），未改动既有 key 的 scopes | 直接讲「独立 scope 不复用 llm:chat」；演示 key 非生产密钥 |
| 统一验证（阶段 6） | **已实现**：架构守卫（全量源码 import 静态扫描 + OpenAPI 路由表）+ 凭证隔离矩阵 + 错误信封一致性 | 矩阵取**四个排他 scope 的代表端点**（抽样，非全端点穷举）；讲「跨项目不变量由跨项目用例兜住」，不夸大为全量形式化验证 |
| 数据平台（项目四） | 已实现并单测：接入标准化 / 两级去重 / 脱敏 / 过滤链 / 语义+三级切片 / 审核留痕 / 用途分流 / 索引一致性 / 混合检索组装（含 Auto-merging）/ 召回评估 / 三格式导出（带门禁）/ 血缘追溯 / 回流 / 出题评分 / 训练产物 | 直接讲，可上代码；接口经冒烟逐条断言 |
| 三级分块 / Rerank / training / reflux / quiz | **已实现并单测 + 冒烟**（阶段 2）。三级切片与 RerankPort 为**离线等价实现**：L1 为句子滑动窗口、重排为本地词典打分（非学习型 cross-encoder）；`hierarchical` 模式下只有 L1 参与召回、L2/L3 只作合并上下文 | 讲机制与权衡，**不宣称语义质量**；Jina 未配 key 时走本地要主动说明 |
| LLaMA-Factory / QLoRA / DPO 训练 | **只做到数据准备与验证链路**：产出可直接消费的数据集 + `dataset_info.json` + 训练 YAML，门禁逐条给原因 | 「数据侧全通，**训练在外部环境执行**；本仓库不装 LLaMA-Factory、不跑 GPU」 |
| 自动出题 / 自动评分 | **已实现**：三类题型（事实/边界/拒答）+ 启发式评分（证据一致性 + 要点覆盖 + 拒答策略） | 明确说这是**启发式打分，不是 LLM-as-judge**；升级路径为接 judge 模型 + 人工抽检校准 |
| 回流收集器 | **已实现**：五来源归一化 + 去重门禁 + 训练/评测双投影；**跨请求持久（刻意设计）** | 讲清「为什么它必须跨请求存活」；单机内存实现，生产应落库 |
| 向量检索（数据层） | 离线为**进程内余弦**（哈希向量），HNSW 参数只记录未落库 | 讲清「过滤在前 ANN 在后」的接口与一致性核对，**不说跑过 pgvector 线上索引** |
| Embedding | 默认 `HashingEmbedder`（dim=256）离线确定性实现 | 讲抽象与 `embedding_version` 对账，不宣称语义质量 |
| 近重复度量 | 以 bigram Jaccard 为准（SimHash 实测区分度不足，已弃用为主判据）；**仅用正文，不含角色名** | 可讲实测数据；**生产用 SimHash LSH 分桶替代 O(n²)** 属设计规划 |
| 性能数字（RAG TTFT 15s→2s / 成本 -40% / TTFT -60% / 生成耗时 -70% / 命中率 72%→89% / 盲评 2.1→4.3） | **均无实测** | **必须先用 Mock/离线基准脚本产出真实差值**，否则一律改「设计目标」 |
| 2.5GB → 30MB（FFmpeg.wasm） | 未做 | 不做，或明确「未实测」 |
| 重试机制 | 已实现（分类 + 指数退避 + 抖动 + 总预算），5xx/429/超时可重试 | 讲「只重试可重试错误」，并说明非幂等写操作默认不重试 |
| 分布式限流 / 灰度 | 见「滑动窗口限流」 | 不说，或明确「规划中」 |
| 宿主业务后端（commerce） | **已实现并单测 + 冒烟**（阶段 3）：领域模型 / 原子扣库存 / 幂等下单 / 事务性 Outbox / UoW / `/v1/shop/*` 与 `/v1/admin/*` / `CommercePort` 适配器与接线 | 直接讲，可上代码；「不超卖」有 12 并发抢 4 件、余量精确为 0 的实测用例 |
| commerce 存储与并发 | SQLite（`WAL` + `busy_timeout` + `BEGIN IMMEDIATE`）下**写请求严格串行排队**；原子扣库存为单语句条件 UPDATE | 讲「单机串行化换来正确性，读也走 IMMEDIATE 是取舍」；**未做多副本/分库压测**，线上并发容量不报 |
| 会员 token（`MemberTokenSigner`） | HMAC 无状态签章；**未配置 secret 时降级为进程内随机密钥**（多副本重启即失效） | 生产必须注入固定 secret 或走 KMS；**不宣称无状态集群已联调** |
| commerce AI 钩子 | 下单后 `EventRouter.dispatch` 触发的 AI 钩子当前为 **`NoopHook`**（只落 Outbox 事件，不接真实推荐/知识抽取） | 讲「事件与业务同事务、钩子在 commit 之后」的时序；钩子实现本身为占位 |
| `commerce:rw` 权限 | 由 `config/models.yaml` 新增的 `sk-project-shopadmin-demo` **演示 key** 承载，未改动既有 key 的 scopes | 直接讲四级权限链复用；演示 key 非生产密钥 |
| 演示前端（`/` 与 `/admin`） | **已实现并在真实浏览器实测**：零构建静态页（手写 CSS + 原生 fetch），后端直接托管；渲染、加购下单、AI 导购对话、运营台订单/工作流/凭证矩阵均已走通，控制台零报错 | 直接讲「展示层不另立真相」；**诚实说明：页面把演示用的明文 API Key 放在浏览器 localStorage**，只为免去搭一个 BFF —— 真实系统里运营台凭证必须由服务端会话持有，前端拿不到明文 key |
| 前端未做的部分 | 无打包器 / 无组件库 / 无 SSR / 无 i18n；响应式只做了栅格自适应，**未做移动端专项适配与无障碍审计**；页面**没有任何前端路由守卫**，权限判定 100% 在后端 | 讲「前端不承担鉴权」，不宣称有前端安全边界 |
| 运营台成本口径 | `/v1/cost` 按当前凭证的 service 前缀聚合，**换凭证后看到的成本数字会变**（这是项目隔离而不是 bug） | 讲隔离设计；不把某一凭证下的累计值说成「平台总成本」 |

**规则**：只讲真正实现过的功能；性能数字必须有实测依据，没有就定性描述。

## 9. 交付分期（每阶段结束必须 pytest + smoke 全绿）

| 阶段 | 内容 | 状态 | 风险 |
| --- | --- | --- | --- |
| 0 | 口径与结构：重写本文件、建 `agents`/`ops` 骨架、作废旧方案文档 | ✅ 已完成 | 低 |
| 1 | 项目三补齐：限流 / Telemetry / 四级权限 / Metering 聚合 / YAML Prompt 中心 / Eval 骨架 | ✅ 已完成 | 低（改造已有） |
| 2 | 项目四补齐：三级分块 + Auto-merging / RerankPort / training / reflux / quiz | ✅ 已完成（pytest 159 / smoke 83） | 低（改造已有） |
| 3 | `commerce` 收窄补齐：usecases / interface / service / factory / tools_port / 接线 | ✅ 已完成（pytest 173 / smoke 94） | 中（新增接线） |
| 4 | 项目一 `agents`：Orchestrator / 双端 Agent / 三种编排模式 / Grounding / HITL / A2A / 实时语音 | ✅ 已完成（pytest 214 / smoke 109） | 高（全新） |
| 5 | 项目二 `ops`：工作流 + Checkpointer / 流式 / 并行 / 多模态 / 直播 / 归因 / 自进化 | ✅ 已完成（pytest 258 / smoke 126） | 高（全新） |
| 6 | 统一验证与文档：扩充 tests + smoke，回填本章验收状态 | ✅ 已完成（pytest 263 / smoke 130） | 中 |
| 7 | 演示前端：官网页 + 运营台（零构建静态页）+ 补齐运营台只读端点 + 浏览器实测 | ✅ 已完成（pytest 263 / smoke 132） | 低（纯展示层） |

**回归红线**：任一阶段结束，`pytest` 与 `scripts/smoke.py` 必须全绿；**已有断言不得放宽**。
新增断言必须带不变量（如 `available >= 0`、`success_count × qty ≤ 初始库存`），不做表演式测试。
