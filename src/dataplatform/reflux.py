"""数据回流：把五个真实业务来源沉淀为训练数据与评测数据。

面试要点（为什么说「回流」而不是「采集」）：
- 采集是「把线上数据拉下来」；回流是**把业务动作变成模型改进信号**，闭环才是价值；
- 五个来源各自携带不同的信号，缺一个闭环就缺一环：
  1) 客服纠偏（correction）：人工改过的答案 → 最真实的偏好对（chosen/rejected）；
  2) 人工审核（review）：审计留痕的 before/after → 同上一路，但带审核人与驳回原因；
  3) Tool 调用（tool_call）：工具真实返回值 → 训练模型「怎么用工具结果作答」，也是评测金标来源；
  4) HITL（人机协同）：人工确认/驳回 → 通过的做金标，驳回且改写过的做偏好对；
  5) 直播 ASR：主播口播的讲解 → 口语化话术样本，补「表达风格」而非事实；
- **去重与分组**：回流数据必须带 `group`（场景+商品），否则评测切分时同一商品的知识
  会同时出现在训练集与测试集里 → 指标虚高（泄漏）。这是 `grouped_split` 的输入前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..infra.core.errors import ValidationError
from ..infra.core.logging import get_logger
from .evaluation import GoldenItem
from .schemas import SamplePurpose, TrainingSample
from .training.dpo import DpoPair

logger = get_logger("data.reflux")


class RefluxSource(str, Enum):
    CORRECTION = "correction"
    REVIEW = "review"
    TOOL_CALL = "tool_call"
    HITL = "hitl"
    LIVE_ASR = "live_asr"


_REQUIRED_FIELDS = {
    RefluxSource.CORRECTION: ("question", "corrected"),
    RefluxSource.REVIEW: ("question", "after"),
    RefluxSource.TOOL_CALL: ("question", "tool_name"),
    RefluxSource.HITL: ("question", "answer"),
    RefluxSource.LIVE_ASR: ("question", "answer"),
}


@dataclass(slots=True)
class RefluxRecord:
    """归一化后的回流记录：统一「提问 / 回答 / 否定回答 / 金标切片 / 分组」。"""

    record_id: str
    source: RefluxSource
    question: str
    answer: str | None = None
    rejected: str | None = None
    gold_chunk_ids: list[str] = field(default_factory=list)
    group: str = ""
    scene: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def can_train(self) -> bool:
        return bool(self.answer)

    @property
    def can_evaluate(self) -> bool:
        return bool(self.question) and bool(self.gold_chunk_ids)


class RefluxCollector:
    """回流收集器：把五来源事件归一化，再分别投影成训练集与评测集。"""

    def __init__(self, *, max_records: int = 2000) -> None:
        self._records: list[RefluxRecord] = []
        self._max_records = max_records
        self._dropped: list[str] = []

    # -- 采集 -------------------------------------------------------------- #
    def add(self, source: RefluxSource | str, **payload: Any) -> RefluxRecord | None:
        """写入一条业务事件。字段不全的脏数据会被拒绝并记录原因，不静默吞掉。"""
        try:
            source = RefluxSource(source)
        except ValueError as exc:
            raise ValidationError(f"未知回流来源: {source}", detail={"supported": [s.value for s in RefluxSource]}) from exc

        missing = [name for name in _REQUIRED_FIELDS[source] if not str(payload.get(name) or "").strip()]
        if missing:
            self._dropped.append(f"{source.value}:missing:{','.join(missing)}")
            logger.warning(
                "data.reflux.dropped",
                extra={"event": "data_reflux", "status": "skipped", "degraded": True, "reason": f"missing={','.join(missing)}"},
            )
            return None

        record = self._normalize(source, payload)
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]
        return record

    def add_many(self, source: RefluxSource | str, payloads: Sequence[dict[str, Any]]) -> list[RefluxRecord]:
        return [record for record in (self.add(source, **payload) for payload in payloads) if record is not None]

    def records(self) -> list[RefluxRecord]:
        return list(self._records)

    def _normalize(self, source: RefluxSource, payload: dict[str, Any]) -> RefluxRecord:
        scene = payload.get("scene")
        product_id = payload.get("product_id")
        group = str(payload.get("group") or f"{scene or 'unknown'}:{product_id or payload.get('article_id') or 'general'}")
        record_id = str(payload.get("id") or f"{source.value}_{len(self._records):04d}")
        chunk_ids = [str(cid) for cid in payload.get("chunk_ids") or []]

        if source is RefluxSource.CORRECTION:
            return RefluxRecord(
                record_id=record_id,
                source=source,
                question=str(payload["question"]),
                answer=str(payload["corrected"]),
                rejected=str(payload["original"]) if payload.get("original") else None,
                gold_chunk_ids=chunk_ids,
                group=group,
                scene=scene,
                meta={"operator": payload.get("operator")},
            )
        if source is RefluxSource.REVIEW:
            rejected = str(payload["before"]) if payload.get("before") else None
            if rejected and rejected.strip() == str(payload["after"]).strip():
                rejected = None  # 审核没改动 → 没有偏好信号，只作为正向样本
            return RefluxRecord(
                record_id=record_id,
                source=source,
                question=str(payload["question"]),
                answer=str(payload["after"]),
                rejected=rejected,
                gold_chunk_ids=chunk_ids,
                group=group,
                scene=scene,
                meta={"reviewer": payload.get("reviewer"), "reason": payload.get("reason")},
            )
        if source is RefluxSource.TOOL_CALL:
            rendered = f"调用工具 {payload['tool_name']}（参数：{payload.get('params') or {}}）结果：{payload.get('result')}"
            return RefluxRecord(
                record_id=record_id,
                source=source,
                question=str(payload["question"]),
                answer=rendered,
                gold_chunk_ids=chunk_ids,
                group=group,
                scene=scene,
                meta={"tool": payload["tool_name"], "latency_ms": payload.get("latency_ms")},
            )
        if source is RefluxSource.HITL:
            approved = bool(payload.get("approved", True))
            corrected = payload.get("corrected")
            return RefluxRecord(
                record_id=record_id,
                source=source,
                question=str(payload["question"]),
                answer=str(corrected if corrected else payload["answer"]),
                rejected=str(payload["answer"]) if (not approved and corrected) else None,
                gold_chunk_ids=chunk_ids,
                group=group,
                scene=scene,
                meta={"approved": approved, "operator": payload.get("operator")},
            )
        # LIVE_ASR：口语化话术，只作为风格样本，不承载易变事实
        return RefluxRecord(
            record_id=record_id,
            source=source,
            question=str(payload["question"]),
            answer=str(payload["answer"]),
            gold_chunk_ids=chunk_ids,
            group=group,
            scene=scene,
            meta={"asr_confidence": payload.get("asr_confidence"), "live_id": payload.get("live_id")},
        )

    # -- 投影：训练 / 评测 -------------------------------------------------- #
    def to_dpo_pairs(self) -> list[DpoPair]:
        """只有「有否定回答」的记录才能构成偏好对（其余只是正向样本）。"""
        pairs: list[DpoPair] = []
        for record in self._records:
            if not record.answer or not record.rejected:
                continue
            if record.rejected.strip() == record.answer.strip():
                continue
            pairs.append(
                DpoPair(
                    pair_id=f"reflux_{record.record_id}",
                    prompt=record.question,
                    chosen=record.answer,
                    rejected=record.rejected,
                    source=f"reflux_{record.source.value}",
                    source_ids=[record.record_id],
                    scene=record.scene,
                    meta={"group": record.group, **record.meta},
                )
            )
        return pairs

    def to_training(self) -> list[TrainingSample]:
        """回流 → SFT 样本：口语化话术与工具结果表述统一进风格样本集。"""
        samples: list[TrainingSample] = []
        for record in self._records:
            if not record.can_train:
                continue
            samples.append(
                TrainingSample(
                    sample_id=f"reflux_{record.record_id}",
                    messages=[
                        {"role": "user", "content": record.question},
                        {"role": "assistant", "content": str(record.answer)},
                    ],
                    purpose=SamplePurpose.SFT_STYLE,
                    source_staging_ids=[record.record_id],
                    scene=record.scene,
                    meta={"reflux_source": record.source.value, "group": record.group, **record.meta},
                )
            )
        return samples

    def to_golden(self) -> list[GoldenItem]:
        """回流 → 评测金标：带 group，供 `grouped_split` 做防泄漏切分。"""
        return [
            GoldenItem(
                question=record.question,
                gold_chunk_ids=list(record.gold_chunk_ids),
                group=record.group,
                scene=record.scene,
            )
            for record in self._records
            if record.can_evaluate
        ]

    def report(self) -> dict[str, Any]:
        by_source = {source.value: 0 for source in RefluxSource}
        for record in self._records:
            by_source[record.source.value] += 1
        return {
            "records": len(self._records),
            "by_source": by_source,
            "trainable": sum(1 for record in self._records if record.can_train),
            "evaluable": sum(1 for record in self._records if record.can_evaluate),
            "dpo_pairs": len(self.to_dpo_pairs()),
            "dropped": list(self._dropped),
        }


__all__ = ["RefluxCollector", "RefluxRecord", "RefluxSource"]
