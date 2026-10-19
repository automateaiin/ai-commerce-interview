"""检索链路：元数据预过滤 → 混合召回 → 重排 → 上下文组装。

面试重点（判断「检索错还是生成错」的分界线）：
先看候选里**有没有标准证据**。没有 → 检索问题（切片 / embedding / 过滤 / 召回参数）；
有但答错 → 生成问题（上下文组织 / Prompt / 模型遵循证据的能力）。
所以本模块把「候选、分数、来源」全部结构化返回，而不是只回最终文本。

三个真实机制：
1) **混合召回**：型号、SKU、成分名这类专有名词向量未必命中（向量擅长语义泛化，
   不擅长精确串匹配），所以同时走关键词召回再合并候选；
2) **重排**：对 (query, doc) 做更细粒度打分 —— 向量高但**缺少 query 实体**的片段会被压下去
   （这类片段正是「表面相似但不能回答问题」的典型）；
3) **上下文组装**：排序 + 去重 + 预算 + **标注来源/chunk id**，保证答案可引用、可审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..infra.core.errors import ValidationError
from ..infra.core.logging import get_logger
from ..infra.core.tracing import stage
from .chunking import ChunkNode, auto_merge
from .index import SearchReport, VectorIndex
from .rerank import RerankPort, terms
from .schemas import AssembledContext, Evidence, KnowledgeChunk

logger = get_logger("data.retrieval")

VECTOR_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4


@dataclass(slots=True)
class RetrievalReport:
    evidence: list[Evidence] = field(default_factory=list)
    candidates: int = 0
    filtered_expired: int = 0
    filtered_metadata: int = 0
    reranked: int = 0
    merged_children: int = 0
    """Auto-merging 吸收的子块数量（0 表示未触发合并或本就是语义切片模式）。"""
    reasons: list[str] = field(default_factory=list)
    rerank_backend: str = "lexical"
    context: AssembledContext | None = None

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


def _keyword_score(query_terms: set[str], chunk: KnowledgeChunk) -> float:
    haystack = " ".join([chunk.text, " ".join(chunk.tags), chunk.scene or "", chunk.product_id or ""]).lower()
    if not query_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in haystack)
    return round(hits / len(query_terms), 4)


class Retriever:
    """一次检索调用返回「证据 + 分数 + 来源 + 过滤统计」，便于分层定位问题。"""

    def __init__(
        self,
        index: VectorIndex,
        *,
        rerank_candidates: int = 20,
        context_char_budget: int = 1200,
        reranker: RerankPort | None = None,
        merge_ratio: float = 0.5,
    ) -> None:
        self._index = index
        self._rerank_candidates = rerank_candidates
        self._budget = context_char_budget
        # 重排走端口：默认本地词典实现（离线可跑），配了 key 才换成远端
        from .rerank import LexicalReranker

        self._reranker: RerankPort = reranker or LexicalReranker()
        self._merge_ratio = merge_ratio

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        product_id: str | None = None,
        scene: str | None = None,
        market: str | None = None,
        include_expired: bool = False,
    ) -> RetrievalReport:
        if not query.strip():
            raise ValidationError("检索 query 不能为空")

        with stage("data.retrieval", top_k=top_k):
            # 1) 向量召回（预过滤在索引内完成），候选池放大给重排用
            recalled: SearchReport = await self._index.search(
                query,
                top_k=self._rerank_candidates,
                product_id=product_id,
                scene=scene,
                market=market,
                include_expired=include_expired,
                levels=(1,),  # 只召回 L1（三级切片的检索单元）；L2/L3 仅用于向上合并
            )
            # 2) 混合打分：向量 + 关键词
            query_terms = terms(query)
            merged: dict[str, KnowledgeChunk] = {}
            for chunk in recalled.chunks:
                keyword = _keyword_score(query_terms, chunk)
                merged[chunk.chunk_id] = _stamp(chunk, vector=chunk.vector_score, keyword=keyword)

            report = RetrievalReport(
                candidates=len(merged),
                filtered_expired=recalled.filtered_expired,
                filtered_metadata=recalled.filtered_metadata,
            )

            # 3) 重排（走 RerankPort）：实体缺失的片段被压分
            outcome = await self._reranker.score(query, list(merged.values()))
            report.reasons.extend(outcome.reasons)
            report.rerank_backend = outcome.backend
            reranked = [
                _stamp(chunk, rerank=outcome.scores.get(chunk.chunk_id, chunk.score))
                for chunk in merged.values()
            ]
            reranked.sort(key=lambda item: item.rerank_score, reverse=True)
            report.reranked = len(reranked)

            # 4) 取 top_k → Auto-merging（L1 命中够密就并回 L2/L3，给模型完整上下文）→ Evidence
            selected = reranked[:top_k]
            final, merged_children, merge_reasons = self._auto_merge(selected)
            report.merged_children = merged_children
            report.reasons.extend(merge_reasons)
            report.evidence = [
                Evidence(
                    chunk_id=chunk.chunk_id,
                    article_id=chunk.article_id,
                    text=chunk.text,
                    source=f"kb:{chunk.article_id}#v{chunk.version}",
                    score=round(chunk.rerank_score, 4),
                    valid_to=chunk.valid_to,
                )
                for chunk in final
            ]

        logger.info(
            "data.retrieval.done",
            extra={
                "event": "data_retrieval",
                "status": "ok",
                "hits": len(report.evidence),
                "top_k": top_k,
                "reason": f"candidates={report.candidates},expired={report.filtered_expired},meta={report.filtered_metadata},merged={report.merged_children}",
            },
        )
        return report

    def _auto_merge(self, selected: Sequence[KnowledgeChunk]) -> tuple[list[KnowledgeChunk], int, list[str]]:
        """三级切片的向上合并：命中同一父块下的子块够多时，用父块替换它们。

        语义切片模式（默认）下所有 chunk 都没有 parent_id，这里直接原样返回 —— 
        这也是「两种模式可切换」且不改变既有检索行为的原因。
        """
        if not selected:
            return [], 0, []
        all_chunks = self._index.all_chunks()
        if not any(chunk.parent_id for chunk in all_chunks.values()):
            return list(selected), 0, []

        nodes = [
            ChunkNode(node_id=chunk.chunk_id, level=chunk.level, text=chunk.text, parent_id=chunk.parent_id, index=chunk.chunk_index)
            for chunk in all_chunks.values()
        ]
        by_id = {chunk.chunk_id: chunk for chunk in selected}
        merged_nodes, reasons = auto_merge(nodes, [chunk.chunk_id for chunk in selected], ratio=self._merge_ratio)
        if len(merged_nodes) == len(selected):
            return list(selected), 0, []

        final: list[KnowledgeChunk] = []
        for node in merged_nodes:
            child = by_id.get(node.node_id)
            if child is not None:
                final.append(child)
                continue
            parent = all_chunks.get(node.node_id)
            if parent is None:
                continue
            best = max(
                (chunk for chunk in selected if chunk.parent_id == node.node_id),
                key=lambda chunk: chunk.rerank_score,
                default=None,
            )
            if best is None:
                final.append(parent)
                continue
            # 父块本身没有召回分，用命中最强子块的分作为它的排序依据（否则会被排到末尾）
            final.append(
                replace(parent, score=best.score, rerank_score=best.rerank_score, vector_score=best.vector_score, keyword_score=best.keyword_score)
            )
        final.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
        return final, len(selected) - sum(1 for chunk in final if chunk.chunk_id in by_id), reasons

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        product_id: str | None = None,
        scene: str | None = None,
        market: str | None = None,
    ) -> AssembledContext:
        """检索 + 组装一步到位，返回可直接注入 Prompt 的上下文。"""
        report = await self.search(query, top_k=top_k, product_id=product_id, scene=scene, market=market)
        report.context = self.assemble(report.evidence)
        return report.context

    def assemble(self, evidence: Sequence[Evidence]) -> AssembledContext:
        """排序 + 去重 + 预算 + 标注来源。被截断时**显式声明**，避免模型以为「只有这么多」。"""
        blocks: list[Evidence] = []
        seen_texts: set[str] = set()
        dropped = 0
        total = 0
        truncated = False
        for item in sorted(evidence, key=lambda entry: entry.score, reverse=True):
            fingerprint = " ".join(item.text.split()).lower()
            if fingerprint in seen_texts:
                dropped += 1
                continue
            seen_texts.add(fingerprint)
            if total + len(item.text) > self._budget:
                truncated = True
                break
            blocks.append(item)
            total += len(item.text)
        return AssembledContext(blocks=blocks, total_chars=total, dropped_duplicates=dropped, truncated=truncated)

    def stats(self) -> dict[str, Any]:
        return {
            "rerank_candidates": self._rerank_candidates,
            "rerank_backend": self._reranker.backend,
            "merge_ratio": self._merge_ratio,
            "context_char_budget": self._budget,
            "index": self._index.config.as_dict(),
        }


def _stamp(chunk: KnowledgeChunk, *, vector: float | None = None, keyword: float | None = None, rerank: float | None = None) -> KnowledgeChunk:
    from dataclasses import replace

    changes: dict[str, Any] = {}
    if vector is not None:
        changes["vector_score"] = vector
    if keyword is not None:
        changes["keyword_score"] = keyword
        changes["score"] = round(VECTOR_WEIGHT * (vector if vector is not None else chunk.vector_score) + KEYWORD_WEIGHT * keyword, 4)
    if rerank is not None:
        changes["rerank_score"] = rerank
    return replace(chunk, **changes)
