"""项目二（ops）跨模块数据契约。

为什么这些契约要集中在一个文件（而不是各模块自定义 dict）：
- 工作流节点、多模态素材、直播处理、运营归因、自进化闭环是**同一条业务链**：
  内容生产出来的素材会变成资产、直播切片会变成资产、运营指标又反过来决定生产哪类素材。
  契约一旦分家，字段口径就会在模块边界处漂移（`asset_id` vs `id`，`score` vs `viral_score`）。
- 用 Pydantic 约束而不是自由 dict：非法输出能在**产生它的模块内**被拒，
  而不是等到下游消费时才炸；这也让「结构化输出」成为整条链的默认形态。

诚实边界：本文件只描述契约；向量化一律走 `infra.rag.Embedder` 的离线哈希实现，
素材向量只索引在进程内（生产应落 pgvector），见 PROJECT.md 第 8 章。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# 内容生产：选题 / 卖点 / 文案 / 产品图
# --------------------------------------------------------------------------- #
class ContentBrief(BaseModel):
    """一次内容生产的需求单。它是所有节点的公共输入，也是可复盘的最小上下文。"""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    product_name: str = Field(default="", max_length=200)
    market: str = Field(default="global", max_length=32)
    audience: str = Field(default="敏感肌人群", max_length=200)
    channel: str = Field(default="social", max_length=32)
    keywords: list[str] = Field(default_factory=list, max_length=20)
    tone: str = Field(default="专业亲和", max_length=64)

    @field_validator("keywords")
    @classmethod
    def _clean_keywords(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class AssetModality(str, Enum):
    """素材模态。多模态知识化的关键就是「四种模态进同一张资产表」。"""

    IMAGE = "image"
    PAGE = "page"
    VIDEO = "video"
    LIVE_CLIP = "live_clip"


class ContentAsset(BaseModel):
    """多模态素材资产（rag_assets）。

    幂等键是 `(source_table, source_id)`：同一条业务记录重复导入必须命中原资产，
    否则每一次重跑流水线都会给检索库塞进一份近重复副本，召回质量会被慢慢稀释。
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    source_table: str
    source_id: str
    modality: AssetModality
    retrieval_text: str
    title: str = ""
    product_id: str | None = None
    market: str | None = None
    tags: list[str] = Field(default_factory=list)
    extractor: str = ""
    embed_version: str = ""
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_payload(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "source": {"table": self.source_table, "id": self.source_id},
            "modality": self.modality.value,
            "title": self.title,
            "retrieval_text": self.retrieval_text,
            "product_id": self.product_id,
            "market": self.market,
            "tags": list(self.tags),
            "extractor": self.extractor,
            "embed_version": self.embed_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class AssetHit(BaseModel):
    asset: ContentAsset
    score: float

    def to_payload(self) -> dict[str, Any]:
        return {**self.asset.to_payload(), "score": round(self.score, 4)}


# --------------------------------------------------------------------------- #
# 直播处理
# --------------------------------------------------------------------------- #
class LiveSegment(BaseModel):
    """按时间窗切出的一段转写。带 overlap 是为了修掉切点处的断句丢字。"""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    text: str
    source: str = "asr"

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "text": self.text,
            "source": self.source,
        }


class LiveClip(BaseModel):
    """一个候选切片（精彩片段）：只描述「要剪哪一段」，不真的转码。"""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    reason: str
    score: float = 0.0
    text: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.end_s - self.start_s, 2),
            "reason": self.reason,
            "score": round(self.score, 4),
            "text": self.text,
        }


class BatchAnalysis(BaseModel):
    """一批分段的分析结果（一次模型调用的产物，结构与单次调用对齐便于成本归集）。"""

    model_config = ConfigDict(extra="forbid")

    batch_index: int = Field(ge=0)
    segment_indexes: list[int]
    summary: str
    selling_points: list[str] = Field(default_factory=list)
    highlight_index: int | None = None
    logical_model: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "segments": list(self.segment_indexes),
            "summary": self.summary,
            "selling_points": list(self.selling_points),
            "highlight_index": self.highlight_index,
            "logical_model": self.logical_model,
        }


class FfmpegCommand(BaseModel):
    """FFmpeg 命令**描述**（不是执行结果）。

    诚实边界：本仓库不调用 FFmpeg 二进制，只构造命令并声明可用性；
    缺少二进制时 `degraded=true` 并给出原因，绝不假装切片已经产出。
    """

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    command: str
    output: str

    def to_payload(self) -> dict[str, Any]:
        return {"clip_id": self.clip_id, "command": self.command, "output": self.output}


# --------------------------------------------------------------------------- #
# 运营分析
# --------------------------------------------------------------------------- #
class Metric(BaseModel):
    key: str
    label: str
    value: float
    unit: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "value": round(self.value, 4), "unit": self.unit}


class Attribution(BaseModel):
    """一条归因结论：必须挂到具体指标与可复算的增量上，不允许「感觉更好」。"""

    channel: str
    metric_key: str
    delta: float
    share: float
    basis: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "metric": self.metric_key,
            "delta": round(self.delta, 4),
            "share": round(self.share, 4),
            "basis": self.basis,
        }


class MaterialSuggestion(BaseModel):
    """素材侧的可执行建议：爆款评分 + 标题 + 剪辑建议，供内容工作流作为下一轮 brief。"""

    source_id: str
    viral_score: float
    title: str
    editing: list[str] = Field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "viral_score": round(self.viral_score, 4),
            "title": self.title,
            "editing": list(self.editing),
        }


# --------------------------------------------------------------------------- #
# 自进化闭环
# --------------------------------------------------------------------------- #
class EvalCase(BaseModel):
    """自动生成的评测题（来自回流），带 group 以便分组切分防泄漏。"""

    case_id: str
    question: str
    gold_answer: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    group: str = ""
    case_type: str = "fact"
    source: str = "reflux"


class TemplatePatch(BaseModel):
    """一次「模板迭代」建议：指向具体片段与版本，可回滚，不自动发布。"""

    template: str
    from_version: str
    to_version: str
    reason: str
    suggested_text: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
            "suggested_text": self.suggested_text,
        }


class StrategyIteration(BaseModel):
    """生成策略迭代：影响的是「怎么生成」，而不是「哪句话」。"""

    name: str
    before: Any
    after: Any
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "before": self.before, "after": self.after, "reason": self.reason}


__all__ = [
    "AssetHit",
    "AssetModality",
    "Attribution",
    "BatchAnalysis",
    "ContentAsset",
    "ContentBrief",
    "EvalCase",
    "FfmpegCommand",
    "LiveClip",
    "LiveSegment",
    "MaterialSuggestion",
    "Metric",
    "StrategyIteration",
    "TemplatePatch",
]
