"""评估：分层指标 + 分组切分防泄漏。

两个必须讲清的点：
1) **检索错还是生成错要分开评估**：没召回正确知识属检索问题（看 Recall@K / MRR），
   召回却答错属生成问题（看答案正确性 / 证据一致性 / 拒答）。只看最终主观效果无法指导优化。
2) **分组切分防数据泄漏**：同一会话/客户的近重复样本若跨训练集与测试集，
   离线指标会虚高。所以切分按 session（分组键）进行，切完再跑一次近重复交叉检查。

诚实边界：本实现的「答案正确性」是**结构化的证据一致性判定**（回答是否覆盖 gold 证据），
不是人工/模型评审；真实评估集需要人工标注。指标口径必须与样本规模一起报，不能只报百分比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..infra.core.logging import get_logger
from ..infra.core.tracing import stage
from .dedupe import jaccard
from .retrieval import Retriever

logger = get_logger("data.evaluation")


@dataclass(slots=True)
class GoldenItem:
    """带标准证据标注的问题。group 用于分组切分（同一会话/客户必须同组）。"""

    question: str
    gold_chunk_ids: list[str]
    group: str
    product_id: str | None = None
    scene: str | None = None
    expected_answer: str | None = None


@dataclass(slots=True)
class EvalMetrics:
    k: int
    total: int
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    retrieval_errors: list[str] = field(default_factory=list)
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "total": self.total,
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mrr": self.mrr,
            "retrieval_error_samples": self.retrieval_errors[:5],
        }


def grouped_split(items: Sequence[GoldenItem], *, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, list[GoldenItem]]:
    """按 group 整体切分：同一会话/客户的样本绝不跨集合（这是防泄漏的第一道）。"""
    groups: dict[str, list[GoldenItem]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item)
    ordered = sorted(groups)
    total = len(ordered)
    if total == 0:
        return {"train": [], "valid": [], "test": []}
    train_end = max(1, int(total * ratios[0]))
    valid_end = min(max(train_end + 1, int(total * (ratios[0] + ratios[1]))), total)
    return {
        "train": [item for group in ordered[:train_end] for item in groups[group]],
        "valid": [item for group in ordered[train_end:valid_end] for item in groups[group]],
        # 若分组太少导致 test 为空，就保持为空并如实报告，绝不把 valid 复制进 test（那是自欺的泄漏）
        "test": [item for group in ordered[valid_end:] for item in groups[group]],
    }


def leak_check(splits: dict[str, list[GoldenItem]], *, threshold: float = 0.42) -> list[str]:
    """切分后再查一次近重复：跨集合的近重复样本会导致评估虚高。"""
    leaks: list[str] = []
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if any(
                jaccard(l_item.question, r_item.question) >= threshold
                for l_item in splits[left]
                for r_item in splits[right]
            ):
                leaks.append(f"{left}~{right}")
    return leaks


class Evaluator:
    """在固定测试集上算 Recall@K / Hit Rate@K / MRR，并保留逐条明细用于归因。"""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def evaluate(self, items: Sequence[GoldenItem], *, k: int = 3) -> EvalMetrics:
        hits_with_gold = 0
        reciprocal_sum = 0.0
        per_query: list[dict[str, Any]] = []
        errors: list[str] = []

        with stage("data.evaluation", samples=len(items), k=k):
            for item in items:
                report = await self._retriever.search(
                    item.question, top_k=k, product_id=item.product_id, scene=item.scene
                )
                retrieved = [evidence.chunk_id for evidence in report.evidence]
                rank = next(
                    (index + 1 for index, chunk_id in enumerate(retrieved) if chunk_id in item.gold_chunk_ids),
                    None,
                )
                if rank is not None:
                    hits_with_gold += 1
                    reciprocal_sum += 1.0 / rank
                else:
                    errors.append(item.question)
                per_query.append(
                    {
                        "question": item.question,
                        "retrieved": retrieved,
                        "gold": list(item.gold_chunk_ids),
                        "rank": rank,
                        # 候选里有没有 gold → 判定检索问题还是生成问题的分界
                        "retrieval_error": rank is None,
                    }
                )

        total = max(len(items), 1)
        metrics = EvalMetrics(
            k=k,
            total=len(items),
            recall_at_k=round(hits_with_gold / total, 4),
            hit_rate_at_k=round(hits_with_gold / total, 4),
            mrr=round(reciprocal_sum / total, 4),
            retrieval_errors=errors,
            per_query=per_query,
        )
        logger.info(
            "data.evaluation.done",
            extra={
                "event": "data_evaluation",
                "status": "ok",
                "top_k": k,
                "reason": f"recall@{k}={metrics.recall_at_k},mrr={metrics.mrr}",
            },
        )
        return metrics

    @staticmethod
    def evidence_consistency(answer: str, gold_texts: Sequence[str]) -> float:
        """生成侧的一致性判定：回答覆盖了多少 gold 证据的关键词元（结构化、可复现）。"""
        if not gold_texts:
            return 0.0
        answer_lower = answer.lower()
        covered = 0
        for text in gold_texts:
            tokens = {token for token in text.lower().split() if len(token) >= 2}
            if not tokens:
                continue
            ratio = sum(1 for token in tokens if token in answer_lower) / len(tokens)
            if ratio >= 0.3:
                covered += 1
        return round(covered / len(gold_texts), 4)
