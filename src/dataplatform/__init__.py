"""项目三：AI 电商数据与模型平台（数据层）。

职责边界（面试要能一句话讲清）：
- 平台层(infra)提供能力，执行层(harness)组织执行，**数据层为它们提供知识与训练数据**；
- 本层只做「数据」这件事：接入 → 治理 → 审核 → 资产化 → 检索 → 评估 → 导出；
- 依赖方向：dataplatform → infra 的**基础原语**（日志/异常/trace/指标/embedding 抽象）。
  反向不成立：infra 不 import 本包，两者通过 artifact（结构化数据）解耦 ——
  infra/rag 的 `knowledge=` 注入点就是消费本包产物的接缝，因此不会形成循环依赖。

一条主线的落点（原始 → 可信）：
    raw → standard → (dedupe/scrub/filter) → staging → review(approved)
        → routing → knowledge_articles → knowledge_chunks → vector index
        → retrieval/rerank → context → answer → evaluation → export(SFT)

模块清单：
    schemas   统一数据契约（每层都有明确的「准入边界」）
    ingest    按来源分解析器，映射成统一字段（映射失败保留原始标识，不猜身份）
    dedupe    精确哈希 + 近重复检测（两件事，缺一不可）
    scrub     脱敏规则集 + 导出门禁（双保险）
    filters   过滤规则链，每条带原因，可统计误删率
    chunking  语义边界优先，fallback 时间窗；token 上限 + overlap
    review    人工审核状态机（pending/approved/rejected，留修改前后）
    routing   知识 / 训练分流；价格与时效内容隔离
    index     向量索引（离线内存实现，语义对齐 pgvector + HNSW + 预过滤）
    retrieval 元数据预过滤 → 混合召回 → 重排 → 上下文组装
    evaluation Recall@K / Hit@K / MRR + 按会话分组切分防泄漏
    export    ShareGPT / Alpaca / OpenAI JSONL + 校验门禁 + 版本统计
    lineage   raw→staging→article→chunk→vector→export 的可追溯链
    pipeline  编排入口（串起全部阶段，带 stage 计时与指标）
    service   门面：按请求装配流水线并缓存最近一次产物
    factory   由 Settings 装配
"""

from __future__ import annotations

__all__ = [
    "schemas",
    "ingest",
    "dedupe",
    "scrub",
    "filters",
    "chunking",
    "review",
    "routing",
    "index",
    "retrieval",
    "evaluation",
    "export",
    "lineage",
    "pipeline",
    "service",
    "factory",
]
