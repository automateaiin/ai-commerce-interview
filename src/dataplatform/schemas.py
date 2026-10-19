"""统一数据契约。

面试要点：**每一层都有明确的准入边界，层与层之间靠结构化数据交接**。
如果只保留「最终结果」，一旦出现误删/污染就无法定位；所以 raw 层原样保留，
staging 层记录审核前后，approved 之后才允许进入知识/训练资产。

关键字段设计的「为什么」：
- `source` / `source_id`：可追溯的最小单元，出了错误知识要能回到源头；
- `parse_status`：解析失败时**保留原始标识并标记状态**，绝不去猜发送人身份；
- `valid_from` / `valid_to` / `version` / `status`：过期知识与冲突知识的处理依据；
- `rejected_reason`：过滤是「有原因、可统计」的，而不是静默丢弃。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def speaker_role(speaker: str | None) -> str:
    """把 speaker 映射成对话角色：customer/user → user，其余（客服/知识库）→ assistant。

    为什么单独抽一个函数：`speaker` 是「谁说的」（customer_bob / cs_agent_alice），
    `role` 是「在对话里扮演什么」（提问者/回答者）。切片要按角色判断语义边界，
    训练样本要按角色排 user/assistant，两处都依赖它 —— 写死字符串比较必然出错。
    """
    if not speaker:
        return "assistant"
    lowered = speaker.strip().lower()
    if lowered.startswith("customer") or lowered in ("user", "buyer", "客户"):
        return "user"
    return "assistant"


class SourceKind(str, Enum):
    CHAT_EXPORT = "chat_export"
    PRODUCT_KNOWLEDGE = "product_knowledge"
    LIVE_ASR = "live_asr"


class ParseStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class KnowledgeStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class SamplePurpose(str, Enum):
    KNOWLEDGE = "knowledge"
    SFT_STYLE = "sft_style"
    EXCLUDED = "excluded"


@dataclass(slots=True)
class RawRecord:
    """原始记录：**尽量不做破坏性修改**，清洗规则升级后可重跑整条流水线。"""

    source: SourceKind
    source_id: str
    payload: dict[str, Any]
    ingested_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class StandardMessage:
    """标准化消息（统一 schema：session_id / speaker / ts / content / source_id）。"""

    source: SourceKind
    source_id: str
    session_id: str
    speaker: str | None
    ts: datetime
    content: str
    parse_status: ParseStatus = ParseStatus.OK
    raw_speaker: str | None = None
    """映射失败时保留原始标识，用于后续人工核对，而不是猜身份。"""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        from .dedupe import content_hash

        return content_hash(self.content)


@dataclass(slots=True)
class StagingConversation:
    """待审核会话块。可信数据的**唯一入口**。"""

    staging_id: str
    source: SourceKind
    session_id: str
    messages: list[StandardMessage]
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: str | None = None
    reason: str | None = None
    before: str | None = None
    after: str | None = None
    reviewed_at: datetime | None = None
    product_id: str | None = None
    scene: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """会话正文（审核人看到的、下游使用的文本）。编辑过则取 after。

        **刻意不含说话人前缀**：这段文本会参与近重复相似度计算，
        若把 `customer_bob:` / `cs_agent_alice:` 这类角色名也拼进去，
        所有「客户-客服」对话都会因共享角色名而相似度虚高 ——
        实测会把内容完全不同的两轮问答误判成近重复。
        """
        return self.after if self.after is not None else "\n".join(m.content for m in self.messages)


@dataclass(slots=True)
class FilterDecision:
    """过滤结果：保留 or 拒绝，**拒绝必须带可解释的原因**。"""

    kept: bool
    rule_id: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class KnowledgeArticle:
    """面向业务人员理解的结构化知识（商品知识 / 服务规则 / 销售话术）。"""

    article_id: str
    title: str
    scene: str
    question: str
    answer: str
    conditions: list[str] = field(default_factory=list)
    confidence: float = 0.8
    source_staging_ids: list[str] = field(default_factory=list)
    product_id: str | None = None
    market: str | None = None
    tags: list[str] = field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    version: int = 1
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    price_freshness: bool = False
    """价格/促销类知识：具有强时效性，**不允许作为长期语义知识**被检索。"""
    live_source: str | None = None
    """需要实时报价时应查询的可信业务接口（回答前调用，而不是读历史聊天）。"""


@dataclass(slots=True)
class KnowledgeChunk:
    """检索粒度切片。保留 article_id / chunk_index / metadata / version / 有效期。

    `level` / `parent_id` 用于三级切片与 Auto-merging：L1 参与检索，
    命中多个同父子块时可在组装阶段合并回父块（L2/L3）获得完整上下文。
    语义模式（默认）下 level 恒为 1、parent_id 为空，因此不影响既有检索行为。
    """

    chunk_id: str
    article_id: str
    chunk_index: int
    text: str
    product_id: str | None = None
    scene: str | None = None
    market: str | None = None
    tags: list[str] = field(default_factory=list)
    version: int = 1
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    embedding_version: str | None = None
    level: int = 1
    parent_id: str | None = None
    score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(slots=True)
class TrainingSample:
    """SFT 候选样本（学习表达风格与处理模式，不承载易变事实）。"""

    sample_id: str
    messages: list[dict[str, str]]
    purpose: SamplePurpose
    source_staging_ids: list[str] = field(default_factory=list)
    scene: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Dataset:
    """一次流水线的完整产物：层与层都有，便于追溯与重跑。"""

    raw: list[RawRecord] = field(default_factory=list)
    messages: list[StandardMessage] = field(default_factory=list)
    staging: list[StagingConversation] = field(default_factory=list)
    articles: list[KnowledgeArticle] = field(default_factory=list)
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    sft_samples: list[TrainingSample] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Evidence:
    """检索产出的证据片段（带来源与 chunk id，方便审计与答案引用）。"""

    chunk_id: str
    article_id: str
    text: str
    source: str
    score: float
    valid_to: datetime | None = None


@dataclass(slots=True)
class AssembledContext:
    """上下文组装结果：按相关性/来源排序、去重、限制总长度、标注来源。"""

    blocks: list[Evidence]
    total_chars: int
    dropped_duplicates: int
    truncated: bool = False

    @property
    def has_evidence(self) -> bool:
        return bool(self.blocks)

    def render(self) -> str:
        """渲染成给模型看的证据文本，每段带 [chunk_id] 以便引用与审计。"""
        return "\n\n".join(f"[{item.chunk_id}] {item.text}" for item in self.blocks)


@dataclass(slots=True)
class LineageNode:
    """血缘节点：任一环节都能回答「它从哪来、到哪去」。"""

    node_id: str
    kind: str
    parents: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "kind": self.kind, "parents": list(self.parents), **self.attributes}
