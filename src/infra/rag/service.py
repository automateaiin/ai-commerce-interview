"""RAG 知识检索服务。

检索链路（面试要能逐步讲）：查询预处理 → 向量召回 → 关键词/元数据过滤 → 重排 → 上下文组装。

本实现的诚实边界：
- Embedding 用**确定性哈希向量**（离线可跑、无外部依赖），生产替换为
  DashScope text-embedding-v3 / 其它 embedding 服务，只需替换 Embedder 实现；
- 向量检索用进程内余弦计算，生产替换为 pgvector（HNSW 索引）；
- 但「有效期过滤」「元数据过滤」「向量+关键词混合 + 重排」这三层是真实实现的，
  它们才是决定「会不会把过期促销当有效政策返回」的关键。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..data.demo_data import KNOWLEDGE

logger = get_logger("infra.rag")


class Embedder(Protocol):
    """向量化接口。切换 embedding 模型时只改这一层。"""

    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """确定性哈希向量：离线测试用，结果可复现。

    做法：把词元哈希到固定维度并累加，再 L2 归一化。
    它与真实 embedding 的差别是「无语义泛化能力」，
    所以只在向量召回阶段需要泛化时才会暴露差距 —— 这正是需要替换的点。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = text.lower()
        tokens: list[str] = []
        # 英文按空白切分；中文按 2-gram 切分，让中文也能形成可比较的向量
        buffer: list[str] = []
        for char in lowered:
            if char.isascii() and char.isalnum():
                buffer.append(char)
            else:
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                if "\u4e00" <= char <= "\u9fff":
                    tokens.append(char)
                    tokens.append(char + "\u3000")  # 单字增强
        if buffer:
            tokens.append("".join(buffer))
        tokens.extend(lowered[index : index + 2] for index in range(len(lowered) - 1) if not lowered[index].isspace())
        return tokens or [lowered]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass(slots=True)
class KnowledgeChunk:
    chunk_id: str
    article_id: str
    title: str
    text: str
    source: str
    product_id: str | None
    market: str | None
    tags: list[str] = field(default_factory=list)
    valid_to: str | None = None
    score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[KnowledgeChunk]
    considered: int
    filtered_expired: int
    filtered_metadata: int

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class KnowledgeService:
    """知识检索服务。索引在内存中，写入路径留了 rebuild 接口。"""

    def __init__(self, embedder: Embedder | None = None, *, knowledge: Sequence[dict[str, Any]] | None = None) -> None:
        self._embedder = embedder or HashingEmbedder()
        self._items: list[dict[str, Any]] = list(knowledge if knowledge is not None else KNOWLEDGE)
        self._vectors: list[list[float]] = []
        self._indexed = False

    @property
    def size(self) -> int:
        return len(self._items)

    async def build_index(self) -> int:
        texts = [f"{item['title']} {item['text']} {' '.join(item.get('tags') or [])}" for item in self._items]
        self._vectors = await self._embedder.embed(texts)
        self._indexed = True
        logger.info(
            "rag.index_built",
            extra={"event": "rag_index", "changed_keys": self.size, "reason": f"embedder={type(self._embedder).__name__}"},
        )
        return self.size

    @staticmethod
    def _is_expired(item: dict[str, Any], now: datetime) -> bool:
        valid_to = item.get("valid_to")
        if not valid_to:
            return False
        try:
            deadline = datetime.fromisoformat(str(valid_to).replace("Z", "+00:00"))
        except ValueError:
            return True  # 解析不了的有效期，按「不可信」处理，宁可不返回
        return deadline < now

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        product_id: str | None = None,
        market: str | None = None,
        include_expired: bool = False,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """混合检索：向量相似度 + 关键词命中 + 元数据过滤 + 有效期过滤。"""
        if not query.strip():
            raise ValidationError("检索 query 不能为空")

        if not self._indexed:
            await self.build_index()

        now = datetime.now(timezone.utc)
        query_vector = (await self._embedder.embed([query]))[0]
        query_terms = {term for term in HashingEmbedder._tokens(query) if len(term) >= 2}

        scored: list[KnowledgeChunk] = []
        filtered_expired = 0
        filtered_metadata = 0

        for item, vector in zip(self._items, self._vectors):
            if not include_expired and self._is_expired(item, now):
                filtered_expired += 1
                continue
            if product_id and item.get("product_id") not in (None, product_id):
                filtered_metadata += 1
                continue
            if market and item.get("market") not in (None, market):
                filtered_metadata += 1
                continue

            vector_score = _cosine(query_vector, vector)
            haystack = f"{item['title']} {item['text']} {' '.join(item.get('tags') or [])}".lower()
            hit_terms = {term for term in query_terms if term in haystack}
            keyword_score = len(hit_terms) / max(len(query_terms), 1)
            # 混合打分：纯向量会漏掉型号/成分这类专有名词，纯关键词缺泛化
            score = 0.6 * vector_score + 0.4 * keyword_score
            if score < min_score:
                continue
            scored.append(
                KnowledgeChunk(
                    chunk_id=item["chunk_id"],
                    article_id=item["article_id"],
                    title=item["title"],
                    text=item["text"],
                    source=item["source"],
                    product_id=item.get("product_id"),
                    market=item.get("market"),
                    tags=list(item.get("tags") or []),
                    valid_to=item.get("valid_to"),
                    score=round(score, 4),
                    vector_score=round(vector_score, 4),
                    keyword_score=round(keyword_score, 4),
                )
            )

        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        result = RetrievalResult(
            chunks=scored[:top_k],
            considered=len(self._items),
            filtered_expired=filtered_expired,
            filtered_metadata=filtered_metadata,
        )
        logger.info(
            "rag.search",
            extra={
                "event": "rag_search",
                "hits": len(result.chunks),
                "top_k": top_k,
                "reason": f"expired_filtered={filtered_expired},meta_filtered={filtered_metadata}",
            },
        )
        return result
