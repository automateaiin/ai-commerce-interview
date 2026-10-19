"""重排端口（RerankPort）：把「用哪个模型重排」与「检索主流程」解耦。

面试要点（为什么要抽端口，而不是在 retrieval 里 if backend）：
- 重排是**可选增强**：本地词典重排离线可跑，远端重排（Jina / Cohere）需要 key 与网络；
- 抽象成端口后，检索链路只依赖 `score(query, chunks)` 这一个契约，
  **换实现不改主流程**，缺 key 时也能显式降级回本地，而不是让整条检索挂掉；
- 诚实边界：本地词典重排不是学习型 cross-encoder，它只做**实体对齐 + 词元覆盖**
  的结构化打分，因此必须在面试里说清「离线等价实现」，不宣称语义质量。

打分口径（与阶段 1 之前的内联重排在数值上保持一致，避免回归漂移）：
    最终分 = 混合分（向量 0.6 + 关键词 0.4）+ 实体命中奖励 0.25 − 缺实体惩罚 0.15
本地实现刻意保留「缺实体降权」——向量高但**不含 query 专有名词（型号/SKU/成分）**的
片段，正是「表面相似但答不了问题」的典型，压下去能显著提升 Top-K 精度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..infra.core.logging import get_logger
from .schemas import KnowledgeChunk

logger = get_logger("data.rerank")

ENTITY_BONUS = 0.25
MISSING_ENTITY_PENALTY = 0.15

_ENTITY_PATTERN = re.compile(r"[A-Za-z]{1,6}[-_]?\d{2,6}|\b\d{3,}\b")


def terms(text: str) -> set[str]:
    """词元集合：空白切分的词（≥2 字） + 字符 bigram（中文没有空格，bigram 是必要的）。"""
    lowered = " ".join(text.split()).lower()
    tokens = {token for token in lowered.split() if len(token) >= 2}
    tokens.update(lowered[i : i + 2] for i in range(len(lowered) - 1) if not lowered[i].isspace())
    return tokens


def entity_terms(text: str) -> set[str]:
    """实体词：型号 / SKU / 成分名等专有名词（精确串匹配比语义相似更可靠）。"""
    return {token.lower() for token in _ENTITY_PATTERN.findall(text)}


@dataclass(slots=True)
class RerankOutcome:
    """重排结果：chunk_id → 重排分，附带可解释原因与生效后端。"""

    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    backend: str = "lexical"


class RerankPort(Protocol):
    """重排契约：输入 query 与候选切片，输出每个候选的重排分。"""

    backend: str

    async def score(self, query: str, chunks: Sequence[KnowledgeChunk]) -> RerankOutcome: ...


class LexicalReranker:
    """默认实现：本地词典重排。无需 key、无网络，离线可跑且结果可复现。"""

    backend = "lexical"

    def __init__(
        self,
        *,
        entity_bonus: float = ENTITY_BONUS,
        missing_entity_penalty: float = MISSING_ENTITY_PENALTY,
    ) -> None:
        self._bonus = entity_bonus
        self._penalty = missing_entity_penalty

    async def score(self, query: str, chunks: Sequence[KnowledgeChunk]) -> RerankOutcome:
        entities = entity_terms(query)
        outcome = RerankOutcome(backend=self.backend)
        for chunk in chunks:
            lowered = chunk.text.lower()
            hit = bool(entities) and any(term in lowered for term in entities)
            bonus = self._bonus if hit else 0.0
            penalty = 0.0
            if entities and not hit:
                penalty = self._penalty
                outcome.reasons.append(f"{chunk.chunk_id}: 缺少 query 实体，重排降权")
            outcome.scores[chunk.chunk_id] = round(chunk.score + bonus - penalty, 4)
        return outcome


class JinaReranker:
    """可选实现：远端 Jina Rerank。需 key + 网络；未配置时由工厂显式降级到本地。

    参数与返回结构按 Jina `/v1/rerank` 契约：请求 `{model, query, documents, top_n}`，
    响应 `{results:[{index, relevance_score}]}`。缺依赖（httpx）或调用失败都写日志降级，
    但**降级由调用方（工厂/检索）决定**，这里只负责如实抛错，不静默换后端。
    """

    backend = "jina"

    def __init__(self, *, api_key: str, url: str, model: str = "jina-reranker-v2-base-multilingual", timeout_s: float = 8.0) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._timeout = timeout_s

    def probe(self) -> bool:
        """探针：依赖是否可用。仅检查导入，不发真实请求（避免启动期出网）。"""
        try:
            import httpx  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    async def score(self, query: str, chunks: Sequence[KnowledgeChunk]) -> RerankOutcome:
        if not chunks:
            return RerankOutcome(backend=self.backend)
        import httpx

        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": [chunk.text for chunk in chunks],
            "top_n": len(chunks),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        outcome = RerankOutcome(backend=self.backend, reasons=[f"远端重排生效（{self._model}）"])
        for item in data.get("results", []):
            position = int(item.get("index", -1))
            if 0 <= position < len(chunks):
                outcome.scores[chunks[position].chunk_id] = round(float(item.get("relevance_score", 0.0)), 4)
        return outcome


def build_reranker(settings: Any) -> RerankPort:
    """由配置装配重排器：未配 key 时**显式降级**到本地实现，并记录降级原因。"""
    backend = (getattr(settings, "data_rerank_backend", "lexical") or "lexical").lower()
    api_key = getattr(settings, "data_jina_api_key", None)
    if backend != "jina":
        return LexicalReranker()
    if not api_key:
        logger.warning(
            "data.rerank.degraded",
            extra={"event": "data_rerank", "status": "degraded", "degraded": True, "reason": "missing jina api key"},
        )
        return LexicalReranker()
    reranker = JinaReranker(
        api_key=api_key,
        url=str(getattr(settings, "data_jina_rerank_url", "https://api.jina.ai/v1/rerank")),
        model=str(getattr(settings, "data_jina_rerank_model", "jina-reranker-v2-base-multilingual")),
    )
    if not reranker.probe():
        logger.warning(
            "data.rerank.degraded",
            extra={"event": "data_rerank", "status": "degraded", "degraded": True, "reason": "httpx unavailable"},
        )
        return LexicalReranker()
    return reranker


__all__ = [
    "ENTITY_BONUS",
    "MISSING_ENTITY_PENALTY",
    "JinaReranker",
    "LexicalReranker",
    "RerankOutcome",
    "RerankPort",
    "build_reranker",
    "entity_terms",
    "terms",
]
