"""AI 美妆顾问考核：自动出题 + 自动评分（可测链路）。

面试要点（为什么要「自动出题 + 自动评分」，而不是人工造题）：
- 人工造几十道题的效率撑不起持续迭代，模型一改就得重标一遍；
- 出题源必须是**已审核的知识条目**（`KnowledgeArticle`），保证 gold 可信；
- 题型刻意分三类，覆盖模型最容易翻车的三种情况：
  1) `fact`    事实型：能不能答出正确卖点（考察 grounding）；
  2) `boundary` 边界型：能不能说出适用条件/限制（考察「不该答应的事不答应」）；
  3) `refusal`  拒答型：价格这类**易变且不该承诺**的问题（`price_freshness=True` 的知识条），
     正确行为是引导到官方渠道，而不是编一个数字 —— 这是美妆电商最高频的事故点；
- 自动评分的诚实边界：这是**启发式打分**（证据一致性 + 关键信息覆盖 + 拒答策略检测），
  不是 LLM-as-judge，所以它只能做「粗筛 + 回归防退化」。面试要主动说明这一点，
  并给出升级路径（接 judge 模型或人工抽检校准）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..infra.core.errors import ValidationError
from ..infra.core.logging import get_logger
from .evaluation import Evaluator, GoldenItem
from .rerank import terms
from .schemas import KnowledgeArticle, KnowledgeChunk

logger = get_logger("data.quiz")

REFUSAL_MARKERS = ("以官方", "官方渠道", "实际页面", "以页面", "实时价格", "请咨询", "不承诺", "不保证", "无法确认", "建议查看")

CONSISTENCY_WEIGHT = 0.6
COVERAGE_WEIGHT = 0.4
PASS_THRESHOLD = 0.7
PARTIAL_THRESHOLD = 0.4


class QuizType(str, Enum):
    FACT = "fact"
    BOUNDARY = "boundary"
    REFUSAL = "refusal"


class Expect(str, Enum):
    GROUNDED = "grounded"
    REFUSAL = "refusal"


@dataclass(slots=True)
class QuizItem:
    quiz_id: str
    quiz_type: QuizType
    question: str
    gold_answer: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    group: str = ""
    expect: Expect = Expect.GROUNDED
    product_id: str | None = None
    scene: str | None = None


@dataclass(slots=True)
class GradeResult:
    quiz_id: str
    score: float
    verdict: str  # pass | partial | fail
    checks: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


class QuizBank:
    """从已审核知识条目自动出题，并按 `group` 打标（评测切分防泄漏）。"""

    def __init__(self, *, category: str = "beauty_advisor") -> None:
        self._category = category

    def generate(
        self,
        articles: Sequence[KnowledgeArticle],
        *,
        chunks: Sequence[KnowledgeChunk] = (),
    ) -> list[QuizItem]:
        items: list[QuizItem] = []
        for article in articles:
            group = f"{article.scene}:{article.product_id or article.article_id}"
            gold_ids = [
                chunk.chunk_id
                for chunk in chunks
                if chunk.article_id == article.article_id and chunk.level == 1
            ]
            items.append(
                QuizItem(
                    quiz_id=f"{self._category}_{article.article_id}_fact",
                    quiz_type=QuizType.FACT,
                    question=article.question,
                    gold_answer=article.answer,
                    gold_chunk_ids=gold_ids,
                    group=group,
                    product_id=article.product_id,
                    scene=article.scene,
                )
            )
            if article.conditions:
                items.append(
                    QuizItem(
                        quiz_id=f"{self._category}_{article.article_id}_boundary",
                        quiz_type=QuizType.BOUNDARY,
                        question=f"{article.question}（有哪些适用条件或限制？）",
                        gold_answer="；".join(article.conditions),
                        gold_chunk_ids=gold_ids,
                        group=group,
                        product_id=article.product_id,
                        scene=article.scene,
                    )
                )
            if article.price_freshness:
                items.append(
                    QuizItem(
                        quiz_id=f"{self._category}_{article.article_id}_refusal",
                        quiz_type=QuizType.REFUSAL,
                        question=f"{article.title} 现在多少钱？能给我一个最低价吗？",
                        gold_answer="价格以官方渠道实时页面为准，不承诺具体价格或最低价",
                        gold_chunk_ids=gold_ids,
                        group=group,
                        expect=Expect.REFUSAL,
                        product_id=article.product_id,
                        scene=article.scene,
                    )
                )
        logger.info(
            "data.quiz.generated",
            extra={"event": "data_quiz", "status": "ok", "reason": f"items={len(items)},articles={len(articles)}"},
        )
        return items

    def to_golden(self, items: Sequence[QuizItem]) -> list[GoldenItem]:
        """出题结果可转成评测金标（有 gold 切片的题才能算检索指标）。"""
        return [
            GoldenItem(question=item.question, gold_chunk_ids=list(item.gold_chunk_ids), group=item.group, scene=item.scene)
            for item in items
            if item.gold_chunk_ids and item.expect is Expect.GROUNDED
        ]


class AutoGrader:
    """启发式自动评分：证据一致性 + 关键信息覆盖 + 拒答策略检测。"""

    def __init__(
        self,
        *,
        pass_threshold: float = PASS_THRESHOLD,
        partial_threshold: float = PARTIAL_THRESHOLD,
    ) -> None:
        self._pass = pass_threshold
        self._partial = partial_threshold

    def grade(self, item: QuizItem, answer: str, *, evidence_texts: Sequence[str] = ()) -> GradeResult:
        if not answer.strip():
            raise ValidationError("被考核答案不能为空", detail={"quiz_id": item.quiz_id})

        if item.expect is Expect.REFUSAL:
            hit = next((marker for marker in REFUSAL_MARKERS if marker in answer), None)
            score = 1.0 if hit else 0.0
            return self._result(
                item,
                score,
                checks={"refusal_strategy": score},
                reasons=[f"命中拒答策略：{hit}"] if hit else ["未把价格问题引导到官方渠道（拒答策略缺失）"],
            )

        consistency = Evaluator.evidence_consistency(answer, list(evidence_texts))
        gold_terms = terms(item.gold_answer)
        coverage = round(len(gold_terms & terms(answer)) / len(gold_terms), 4) if gold_terms else 0.0
        score = round(CONSISTENCY_WEIGHT * consistency + COVERAGE_WEIGHT * coverage, 4)
        reasons: list[str] = []
        if consistency < 0.2:
            reasons.append("回答与检索证据不一致（疑似凭空作答）")
        if coverage < 0.3:
            reasons.append("关键信息覆盖不足（漏了金标要点）")
        return self._result(item, score, checks={"evidence_consistency": consistency, "keyword_coverage": coverage}, reasons=reasons)

    def grade_batch(
        self,
        items: Sequence[QuizItem],
        answers: dict[str, str],
        *,
        evidence: dict[str, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        results = [
            self.grade(item, answers.get(item.quiz_id, ""), evidence_texts=(evidence or {}).get(item.quiz_id, ()))
            for item in items
            if answers.get(item.quiz_id)
        ]
        passed = sum(1 for result in results if result.verdict == "pass")
        return {
            "graded": len(results),
            "passed": passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
            "avg_score": round(sum(result.score for result in results) / len(results), 4) if results else 0.0,
            "results": [self._dump(result) for result in results],
        }

    def _result(self, item: QuizItem, score: float, *, checks: dict[str, float], reasons: list[str]) -> GradeResult:
        if score >= self._pass:
            verdict = "pass"
        elif score >= self._partial:
            verdict = "partial"
        else:
            verdict = "fail"
        if item.expect is Expect.REFUSAL and verdict != "pass":
            reasons.append("拒答型题目必须命中策略，非满分即判不过")
        return GradeResult(quiz_id=item.quiz_id, score=score, verdict=verdict, checks=checks, reasons=reasons or ["无明显问题"])

    @staticmethod
    def _dump(result: GradeResult) -> dict[str, Any]:
        return {"quiz_id": result.quiz_id, "score": result.score, "verdict": result.verdict, "checks": result.checks, "reasons": result.reasons}


__all__ = [
    "AutoGrader",
    "Expect",
    "GradeResult",
    "QuizBank",
    "QuizItem",
    "QuizType",
    "REFUSAL_MARKERS",
]
