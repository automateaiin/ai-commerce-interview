"""Step-Back / HyDE 查询重写。

长尾问题的典型失败形态：用户用**具体词**提问，但知识库里存的是**通用表述**
（"SKU-1001 那瓶到底几滴" vs "精华的推荐用量与叠加顺序"），
直接把原问句拿去检索，向量相似度会偏低 —— 召回差，不等于知识没有。

两种重写策略解决的是同一个问题、方向不同：
- **Step-Back**：先把具体问题**抽象**成上位问题，用更通用的表述去检索（提高召回）；
- **HyDE**：先**假设一个答案**，用假设答案去检索（答案与文档在向量空间里更接近，提高精度）。

诚实边界：默认实现是**离线启发式**（规则改写 + 模板假设），不额外烧一次模型调用；
接上真实网关后才走模型生成。因此这里关心的是**流程正确性与可对比性**，不是重写质量本身。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage
from src.infra.rag.service import KnowledgeService

logger = get_logger("agents.retrieval")

STRATEGIES = ("none", "step_back", "hyde")

_ID_TOKENS = re.compile(r"\b(?:SKU|ORD)-\d+\b", re.IGNORECASE)
_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_HYDE_TEMPLATE = "关于「{topic}」，要点包括：核心成分与浓度、适用肤质、使用频率与顺序、以及需要注意的禁忌。"


@dataclass(slots=True)
class RetrievalOutcome:
    strategy: str
    query: str
    rewritten: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    has_evidence: bool = False
    top_score: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query": self.query,
            "rewritten": self.rewritten,
            "hits": len(self.chunks),
            "has_evidence": self.has_evidence,
            "top_score": round(self.top_score, 4),
        }


class QueryRewriter:
    def __init__(self, *, knowledge: KnowledgeService, gateway: Any | None = None, model: str | None = None) -> None:
        self._knowledge = knowledge
        self._gateway = gateway
        self._model = model

    # -- 重写 --------------------------------------------------------------- #
    def step_back(self, query: str) -> str:
        """抽象成上位问题：抹掉具体标识与数字，换成类目级表述。"""
        topic = self._topic(query)
        return f"关于{topic}的通用原理、适用条件与常见注意事项是什么？"

    def hyde(self, query: str) -> str:
        """假设答案：不追求正确，只求「像知识库里的答案那样」的用词分布。"""
        return _HYDE_TEMPLATE.format(topic=self._topic(query))

    @staticmethod
    def _topic(query: str) -> str:
        cleaned = _ID_TOKENS.sub("", query)
        cleaned = _DIGITS.sub("", cleaned)
        cleaned = re.sub(r"[？?。.!！,，;；]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or "该类商品"

    def rewrite(self, query: str, strategy: str) -> str:
        if strategy == "step_back":
            return self.step_back(query)
        if strategy == "hyde":
            return self.hyde(query)
        return query

    # -- 检索与对比 --------------------------------------------------------- #
    async def retrieve(
        self,
        query: str,
        *,
        strategy: str = "none",
        top_k: int = 3,
        product_id: str | None = None,
        market: str | None = None,
    ) -> RetrievalOutcome:
        if strategy not in STRATEGIES:
            raise ValueError(f"未知检索策略: {strategy}")
        rewritten = self.rewrite(query, strategy)
        async with stage("agents.retrieval"):
            result = await self._knowledge.search(rewritten, top_k=top_k, product_id=product_id, market=market)
        chunks = [
            {"chunk_id": chunk.chunk_id, "title": chunk.title, "score": round(chunk.score, 4)}
            for chunk in result.chunks
        ]
        return RetrievalOutcome(
            strategy=strategy,
            query=query,
            rewritten=rewritten,
            chunks=chunks,
            has_evidence=result.has_evidence,
            top_score=chunks[0]["score"] if chunks else 0.0,
        )

    async def compare(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        """同一问题跑三种策略，给出可对比的召回结果 —— 「召回有对比」不能只靠感觉。"""
        outcomes = [
            await self.retrieve(query, strategy=strategy, top_k=top_k) for strategy in STRATEGIES
        ]
        baseline = outcomes[0]
        payload = {"query": query, "strategies": [item.to_payload() for item in outcomes]}
        payload["delta"] = {
            item.strategy: {
                "hits": len(item.chunks) - len(baseline.chunks),
                "top_score": round(item.top_score - baseline.top_score, 4),
            }
            for item in outcomes[1:]
        }
        logger.info(
            "agents.retrieval.compared",
            extra={"event": "retrieval", "reason": "compare", "status": "ok"},
        )
        return payload


__all__ = ["QueryRewriter", "RetrievalOutcome", "STRATEGIES"]
