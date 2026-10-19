# 美妆出海电商 AI 应用工程

面向 **高端美妆出海电商** 的 AI 应用工程仓库：把「模型调用 / MCP 工具 / RAG / 记忆 / 配额与限流 / 可观测 / 数据闭环」收敛成一层**可复用、可治理、可扩展**的平台能力，上面跑四个可独立讲述的子项目，并由同一条启动命令直接托管两个可点开的演示页面。

不是聊天机器人 demo —— 关注的是「多线路复用同一套 AI 基建」之后必然出现的四类问题：重复造轮子、能力无法复用、治理（鉴权/成本/限流）没有统一入口、出问题无法定位。

> **完整编码说明书（唯一事实来源）：[`PROJECT.md`](./PROJECT.md)**
> 本 README 只做入口。技术细节、模块验收标准、逐条**诚实边界**一律以 PROJECT.md 为准。

---

## 一条命令跑起来

```powershell
# 1) 依赖（Python 3.11+）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 配置（可选；不配则全部走内置默认值）
Copy-Item .env.example .env

# 3) 启动
.\.venv\Scripts\python.exe -m uvicorn src.infra.api:app --port 8000
```

默认 `PROVIDER_MODE=mock`：**不需要任何真实密钥、不联网**即可完整跑通全链路（含 Agent 编排、工具调用、RAG、配额、审计）。要切真实模型，在 `.env` 里设 `PROVIDER_MODE=ark` 并填 `ARK_BASE_URL` / `ARK_API_KEY`。

打开 <http://127.0.0.1:8000/>。

| 入口 | 内容 | 说明 |
| --- | --- | --- |
| `/` | **公司官网**（`shop.html`） | 商品筛选与详情、会员登录、购物车、下单、我的订单、优惠券 + **AI 导购客服**悬浮窗（可看到路由/工具调用/证据/成本/trace） |
| `/admin` | **电商运营台**（`admin.html`） | KPI、跨会员订单与发货、商品上下架、库存、优惠券、**AI 运营**（工作流 + SSE 流式）、**AI 成本**（成本/配额/审计/评测）+ **凭证 401→403→200 三态演示** |
| `/docs` · `/redoc` · `/openapi.json` | API 文档 | 只包含六个业务前缀（演示页路由刻意 `include_in_schema=False`，不污染 API 契约） |
| `/healthz` · `/readyz` · `/metrics` | 探针与指标 | 存活 / 就绪（校验依赖真实可用）/ 指标 |
| `/mcp/{commerce,knowledge,memory,prompt}/` | MCP 工具端点 | Streamable HTTP；需 `INTERNAL_SERVICE_TOKEN`，**注意尾斜杠** |

前端是**零构建**的：手写 CSS + 原生 `fetch`，由 FastAPI 同一进程直接托管。刻意不引入 npm / 打包器 —— 多一条构建链就多一处「clone 下来跑不起来」的风险，而演示页的全部价值就是「起服务即可打开」。

---

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest          # 263 条：单测 + 契约 + 架构守卫
.\.venv\Scripts\python.exe scripts\smoke.py   # 132 项：端到端冒烟，进程内起应用
```

两者都**进程内起应用、不出网、不需要任何密钥**，`mock` 模式下全绿。

---

## 四个子项目 + 两个共享底座

| 编号 | 名称 | 代码落位 | 定位 | 技术主线 |
| --- | --- | --- | --- | --- |
| 项目一 | AI 智能客服导购系统 | `src/agents` | 应用层 | Orchestrator 复杂度路由 → 双端 Agent → 三种编排模式 → Grounding 门禁 / HITL → A2A → 实时语音 |
| 项目二 | AI 运营系统 | `src/ops` | 应用层 | 工作流 DAG + Checkpointer → SSE 流式 / 并行生成 → 多模态知识化 → 直播处理 → 归因 → 自进化闭环 |
| 项目三 | AI 能力中台与统一调度网关 | `src/infra` | 平台层 | Provider 抽象 → 模型网关 → MCP → RAG / Memory / Prompt → 四级权限 → 成本预算 → 滑动窗口限流 → 可观测 + 评测 |
| 项目四 | AI 数据闭环与美妆知识工程平台 | `src/dataplatform` | 数据层 | 接入清洗 → 脱敏审核 → Session 构建 → 三级切片 / 混合检索 → 训练数据 → 回流评测 |
| 共享 | 执行内核 Harness | `src/harness` | 全仓唯一的 Agent Loop | 决策协议 → Loop Engine → 派发校验 → 三重护栏 → 观察回填 → 轨迹 → 交接包 |
| 共享 | 宿主业务后端 Commerce | `src/commerce` | 项目一的白名单 Tool 数据源 | 领域模型 → 原子扣库存 → 幂等下单 → 事务性 Outbox → UoW |

分层职责与依赖方向（**单向，禁止倒置**）：

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

依赖方向：
agents / ops  →  harness  →  infra
      agents  →  commerce  ┘
      ops     →  dataplatform → infra
infra / harness / dataplatform  ✗→  agents / ops / commerce   （由架构守卫测试强制）
```

唯一组合根是 `src/infra/container.py`：四个项目挂在同一个 FastAPI 应用里，模块间不互相 import 对方的具体实现。

---

## 演示凭证

演示页把 API Key 放在浏览器 `localStorage` 里（详见下方「诚实边界」）。三把主凭证：

| 凭证 | scopes | 用途 |
| --- | --- | --- |
| `sk-project-service-demo` | `tools:read`, `tools:knowledge`, `llm:chat`, `memory:rw` | 官网的 **AI 导购客服**（`/v1/agents/orchestrate`） |
| `sk-project-shopadmin-demo` | `commerce:rw`, `tools:read` | 运营台的**电商业务**（`/v1/admin/*`） |
| `sk-project-ops-demo` | `llm:chat`, `tools:knowledge`, `ops:rw` | 运营台的 **AI 运营**（`/v1/ops/*` + 成本/配额/审计/评测） |

另有 `sk-project-content-demo`（`data:rw`）、`sk-project-eval-demo`（`eval:rw` + `prompt:rw`）、`sk-project-hitl-demo`（`hitl:approve`）、`sk-project-sales-demo`（`tools:write`），完整定义见 [`config/models.yaml`](./config/models.yaml)。

权限刻意按能力拆开：客服线的 key **不能**上架商品、不能批自己发起的高风险动作、也看不到全公司成本 —— 越权面在凭证设计阶段就被切断，而不是靠运行时 `if` 判断。

---

## 目录结构

```
src/
  agents/         项目一：编排、双端 Agent、HITL、A2A、实时语音
  ops/            项目二：内容工作流、多模态素材、直播、分析、自进化
  infra/          项目三：模型网关、MCP、RAG/Memory/Prompt、鉴权、配额、可观测
    static/       演示前端（shop.html / admin.html / assets）
  dataplatform/   项目四：清洗、切片、检索、训练数据、回流、评测
  harness/        共享：Agent 执行内核
  commerce/       共享：宿主电商业务后端
config/           models.yaml（模型路由与凭证）、prompts.yaml（提示词模板中心）
scripts/          smoke.py 端到端冒烟
tests/            分层单测 + 架构守卫
PROJECT.md        唯一事实来源
```

---

## 诚实边界（面试先看这一节）

这个仓库刻意把「实现了什么」和「没做什么」分得很清。完整清单见 [`PROJECT.md` 第 8 章](./PROJECT.md)，高频几条：

- **编排框架是原生等价实现**：DAG + 自研 Checkpointer + 模式注册表，**不引入** LangGraph / Microsoft Agent Framework 运行时依赖；语义等价，生产可平滑替换。
- **默认 Provider 是 Mock**：`ARK_API_KEY` 未配置时走 MockProvider。所以**成本、预算、金额级熔断没有打过真实路径**（Mock 成本恒为 0），只讲设计与单测，不宣称线上跑过。
- **向量与检索是离线等价实现**：默认 `HashingEmbedder`（dim=256），未落 pgvector；三级切片中**只有 L1 参与召回**，L2/L3 仅作合并上下文；跨模态命中的是 `retrieval_text` 里的**同词命中**，不是真正的图文对齐。
- **训练在外部环境执行**：本仓库只产出可直接消费的数据集 + `dataset_info.json` + 训练 YAML，**不装 LLaMA-Factory、不跑 GPU**。
- **外部重型依赖全部走「端口 + 离线等价实现」**：OTel（未接 collector）、OCR/VLM/ASR、实时语音 RTC、FFmpeg（只构造命令并如实声明 `degraded`）、Rerank 与检索重写（本地启发式）。换真实服务只需替换端口实现。
- **存储是单机的**：默认 SQLite（WAL + `BEGIN IMMEDIATE`，写请求串行排队）；Redis 未配置时配额/限流/缓存降级为进程内实现，**Redis 路径未在本机实测**。审计与 HITL 存储是进程内有界缓冲，生产应落库。
- **演示前端的 key 是明文**：为了不搭 BFF，页面把演示 key 放在 `localStorage`；页面**没有任何前端路由守卫**，权限判定 100% 在后端。真实系统里运营台凭证必须由服务端会话持有。
- **性能数字一律未实测**：仓库里没有「TTFT 降 60%」「成本降 40%」这类结论 —— 第 8 章把每个数字都标为「无实测 / 设计目标」。要讲必须先用自己的基准脚本跑出真实差值。

**规则**：只讲真正实现过的功能；性能数字必须有实测依据，没有就定性描述。

---

## 技术栈

| 用途 | 选型 |
| --- | --- |
| Web / 校验 | FastAPI + Uvicorn（全异步）· Pydantic v2 |
| 数据库 | SQLAlchemy 2.x async + aiosqlite（本地）/ asyncpg（生产） |
| 缓存 / 配额 / 限流 | Redis（可选，未配置则进程内原子实现） |
| 模型接入 | OpenAI 兼容协议（默认 Mock，可切火山方舟 Ark） |
| 工具协议 | MCP（Streamable HTTP） |
| 配置 | YAML + 环境变量（不可变快照 + 原子切换） |
| 测试 | pytest + pytest-asyncio + httpx ASGITransport |
| 编排 | 原生等价实现（不依赖 LangGraph / Agent Framework 运行时） |
| 前端 | 零构建静态页（手写 CSS + 原生 fetch，无 npm） |
