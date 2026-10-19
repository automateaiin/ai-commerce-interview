"""反馈闭环：纠偏 / 优质回复 / 失败案例 → Prompt 与 RAG 的 few-shot 示例。

「自进化」如果只写在 PPT 上就是空话。这里把它收成一个**最小可验证的闭环**：

    线上一条反馈（用户纠偏 / 人工标记优质 / 工具失败案例）
      → 结构化记录（kind + 场景 + 问题 + 原答案 + 纠正后答案）
      → 投影成两类可直接消费的产物：
          1) Prompt few-shot：系统提示里给「这类场景，正确回答长这样」；
          2) RAG few-shot：作为示例片段参与检索，让相似问题直接命中「已被纠正过的答案」。

两条纪律：
- **收集器跨请求存活**（反馈本身就是多轮累积的产物），但它**只增不改、且有界**，
  满了就丢最旧的 —— 否则会变成一块只涨不降的脏内存；
- 收集器**只保存已脱敏的结构化字段**，不落原始隐私原文（脱敏由数据层负责，这里不重复实现）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.infra.core.logging import get_logger

logger = get_logger("agents.feedback")

KINDS = ("correction", "preferred", "failure")


@dataclass(slots=True)
class FeedbackRecord:
    seq: int
    kind: str
    question: str
    answer: str | None = None
    corrected: str | None = None
    agent: str = "customer"
    scenario: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "agent": self.agent,
            "scenario": self.scenario,
            "question": self.question,
            "answer": self.answer,
            "corrected": self.corrected,
            "tags": self.tags,
        }


class FeedbackCollector:
    def __init__(self, *, limit: int = 200) -> None:
        self._items: deque[FeedbackRecord] = deque(maxlen=limit)
        self._seq = 0

    def record(
        self,
        *,
        kind: str,
        question: str,
        answer: str | None = None,
        corrected: str | None = None,
        agent: str = "customer",
        scenario: str | None = None,
        tags: Iterable[str] = (),
    ) -> FeedbackRecord:
        if kind not in KINDS:
            raise ValueError(f"未知反馈类型: {kind}，可选 {KINDS}")
        self._seq += 1
        record = FeedbackRecord(
            seq=self._seq,
            kind=kind,
            question=question.strip(),
            answer=answer,
            corrected=corrected,
            agent=agent,
            scenario=scenario,
            tags=[str(tag) for tag in tags],
        )
        self._items.append(record)
        logger.info(
            "agents.feedback.recorded",
            extra={"event": "feedback", "reason": kind, "status": "ok"},
        )
        return record

    # -- 投影 1：Prompt few-shot ------------------------------------------- #
    def prompt_examples(self, *, agent: str | None = None, limit: int = 3) -> list[dict[str, str]]:
        """给系统提示用的示例：优先用「纠偏后」的正确版本。"""
        picked = [item for item in reversed(self._items) if item.corrected and (agent is None or item.agent == agent)]
        return [{"question": item.question, "answer": item.corrected or ""} for item in picked[:limit]]

    # -- 投影 2：RAG few-shot ---------------------------------------------- #
    def rag_examples(self, *, agent: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
        """给检索用的示例片段：相似问题进来时可直接命中「已被纠正过的答案」。"""
        picked = [
            item
            for item in reversed(self._items)
            if (item.corrected or item.kind == "preferred") and (agent is None or item.agent == agent)
        ]
        return [
            {
                "chunk_id": f"feedback_{item.seq}",
                "text": f"问：{item.question}\n答：{item.corrected or item.answer or ''}",
                "source": "feedback",
                "agent": item.agent,
                "kind": item.kind,
            }
            for item in picked[:limit]
        ]

    def prompt_block(self, *, agent: str | None = None, limit: int = 3) -> str:
        """渲染成可注入系统提示的文本块；没有反馈时返回空串（不污染 Prompt）。"""
        examples = self.prompt_examples(agent=agent, limit=limit)
        if not examples:
            return ""
        lines = [f"- 问：{item['question']}\n  答：{item['answer']}" for item in examples]
        return "\n\n[已纠正示例（供参考，不要照抄无关内容）]\n" + "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {kind: 0 for kind in KINDS}
        for item in self._items:
            counts[item.kind] += 1
        return {"total": len(self._items), "by_kind": counts, "capacity": self._items.maxlen}

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return [item.to_payload() for item in list(self._items)[-limit:]]


__all__ = ["FeedbackCollector", "FeedbackRecord", "KINDS"]
