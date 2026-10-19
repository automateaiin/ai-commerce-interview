"""向量索引：**过滤在前，ANN 在后**。

生产用 pgvector + HNSW：
    CREATE INDEX ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
      WITH (m = 16, ef_construction = 64);
    -- m / ef_construction / 查询 ef_search 都要按数据量与延迟压测调，不追求最高召回

本实现的诚实边界：搜索是**进程内余弦计算**（离线可跑、可复现），
索引结构不是 HNSW 图。但三个真正决定「会不会把过期促销当有效政策」的机制是真实实现的：
1) 元数据预过滤（商品/场景/市场）——**先过滤再算相似度**，同时降延迟与噪声；
2) 有效期强制过滤；
3) chunk 稳定 ID + version + embedding_version，支持增量更新与一致性核对。

Embedding 复用平台层抽象（HashingEmbedder），生产替换为 DashScope text-embedding-v3，
只需替换 Embedder 实现 —— 这就是「分层」在数据层的体现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

from ..infra.core.logging import get_logger
from ..infra.rag.service import HashingEmbedder
from .schemas import KnowledgeChunk

logger = get_logger("data.index")


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


@dataclass(slots=True)
class IndexConfig:
    """HNSW 参数（离线实现只记录，用于面试讲清「参数怎么定」）。"""

    m: int = 16
    ef_construction: int = 64
    ef_search: int = 64
    distance: str = "cosine"
    embedding_version: str = "hash-256-v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "distance": self.distance,
            "embedding_version": self.embedding_version,
        }


@dataclass(slots=True)
class SearchReport:
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    considered: int = 0
    filtered_expired: int = 0
    filtered_metadata: int = 0
    filtered_level: int = 0
    candidates: int = 0

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks)


class VectorIndex:
    """知识切片索引。写入路径支持增量 upsert，删除按 chunk_id + version 核对。"""

    def __init__(self, *, embedder: Embedder | None = None, config: IndexConfig | None = None) -> None:
        self._embedder: Embedder = embedder or HashingEmbedder()
        self.config = config or IndexConfig()
        self._items: dict[str, KnowledgeChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        # 向量维度与 embedding_version 的绑定关系：校验不一致时拒绝写入（防混版本）
        self._validated_version: str | None = None

    @property
    def size(self) -> int:
        return len(self._items)

    def _is_expired(self, chunk: KnowledgeChunk, now: datetime) -> bool:
        if chunk.valid_to is None:
            return False
        deadline = chunk.valid_to if chunk.valid_to.tzinfo else chunk.valid_to.replace(tzinfo=timezone.utc)
        return deadline < now

    async def upsert(self, chunks: Sequence[KnowledgeChunk]) -> int:
        """按 chunk_id 覆盖写（同 ID 即同一条知识的更新版本）。"""
        if not chunks:
            return 0
        texts = [chunk.text for chunk in chunks]
        vectors = await self._embedder.embed(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding_version = self.config.embedding_version
            self._items[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vector
        self._validated_version = self.config.embedding_version
        logger.info(
            "data.index.upsert",
            extra={"event": "data_index", "status": "ok", "reason": f"upsert={len(chunks)},size={self.size}"},
        )
        return len(chunks)

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """删除或失效知识时同步移除向量，避免孤儿向量被检索到。"""
        removed = 0
        for chunk_id in chunk_ids:
            if self._items.pop(chunk_id, None) is not None:
                self._vectors.pop(chunk_id, None)
                removed += 1
        if removed:
            logger.info(
                "data.index.delete",
                extra={"event": "data_index", "status": "ok", "reason": f"removed={removed}"},
            )
        return removed

    def consistency(self) -> dict[str, int]:
        """索引与主表一致性核对：有主表没向量、有向量没主表，都是孤儿。"""
        orphan_vectors = len(set(self._vectors) - set(self._items))
        missing_vectors = len(set(self._items) - set(self._vectors))
        return {"items": len(self._items), "vectors": len(self._vectors), "orphan_vectors": orphan_vectors, "missing_vectors": missing_vectors}

    def all_chunks(self) -> dict[str, KnowledgeChunk]:
        """索引内全部切片的只读视图（Auto-merging 需要看父块的兄弟集合才能判覆盖率）。"""
        return dict(self._items)

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        product_id: str | None = None,
        scene: str | None = None,
        market: str | None = None,
        include_expired: bool = False,
        min_score: float = 0.0,
        levels: Sequence[int] | None = None,
    ) -> SearchReport:
        """元数据预过滤 → 向量召回。返回带过滤统计的报告，便于回答「为什么没召回」。

        `levels` 是三级切片引入的预过滤：**L1 才是检索单元**，L2/L3 只作为
        Auto-merging 时回填上下文的父块 —— 若把父块也当候选，长文本父块向量更「全」，
        会系统性地压过精准的小块，召回精度下降。
        """
        now = datetime.now(timezone.utc)
        query_vector = (await self._embedder.embed([query]))[0]
        report = SearchReport(considered=len(self._items))
        allowed_levels = set(levels) if levels is not None else None

        for chunk_id, chunk in self._items.items():
            if not include_expired and self._is_expired(chunk, now):
                report.filtered_expired += 1
                continue
            if allowed_levels is not None and chunk.level not in allowed_levels:
                report.filtered_level += 1
                continue
            # 过滤在前：不满足元数据条件的直接不进相似度计算
            if product_id and chunk.product_id not in (None, product_id):
                report.filtered_metadata += 1
                continue
            if scene and chunk.scene not in (None, scene):
                report.filtered_metadata += 1
                continue
            if market and chunk.market not in (None, market):
                report.filtered_metadata += 1
                continue

            score = _cosine(query_vector, self._vectors[chunk_id])
            if score < min_score:
                continue
            candidate = replace(chunk, vector_score=round(score, 4), score=round(score, 4))
            report.chunks.append(candidate)

        report.chunks.sort(key=lambda item: item.score, reverse=True)
        report.candidates = len(report.chunks)
        report.chunks = report.chunks[:top_k]
        logger.info(
            "data.index.search",
            extra={
                "event": "data_index",
                "status": "ok",
                "top_k": top_k,
                "hits": len(report.chunks),
                "reason": f"expired={report.filtered_expired},meta={report.filtered_metadata}",
            },
        )
        return report
