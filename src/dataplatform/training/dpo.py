"""DPO 偏好对构造。

面试要点（为什么 DPO 的数据比 SFT 更难攒）：
- SFT 只需要「一个还算好的回答」；DPO 需要**同一 prompt 下的 chosen / rejected 对比**，
  而线上能自然产生这种对比的场景只有一个 —— **人改了机器/客服的答案**：
    * 审核编辑（review edit）：原始回答 = rejected，改后回答 = chosen，天然成对；
    * 人工纠偏（correction）：客服或运营手工重写了一条回答；
- 所以偏好数据不是「再标一遍」，而是**把已经在发生的纠正动作沉淀下来**。
  这也是「数据回流」章节的落点：纠偏动作本身就是最有价值的训练信号。
- 反面清单：绝不能拿「两条随机回答」凑对（没有偏好依据的对比会教坏模型）。

本模块只做构造与校验（成对性、非空、chosen ≠ rejected），不做训练。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ...infra.core.logging import get_logger
from ..schemas import StagingConversation, speaker_role

logger = get_logger("data.training.dpo")


@dataclass(slots=True)
class DpoPair:
    """一条偏好对：同一 prompt 下的采纳回答与否定回答。"""

    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    source: str  # review_edit | correction
    source_ids: list[str] = field(default_factory=list)
    scene: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def split_qa(item: StagingConversation) -> tuple[str, str]:
    """把一轮会话拆成（提问, 回答）：问答口径与 routing 的 SFT 构造保持一致。"""
    question = "\n".join(
        message.content for message in item.messages if speaker_role(message.speaker) == "user"
    )
    answer = "\n".join(
        message.content for message in item.messages if speaker_role(message.speaker) == "assistant"
    )
    return question, answer


class DpoBuilder:
    """从审核留痕与纠偏记录构造偏好对，并给出**逐条拒绝原因**。"""

    def __init__(self, *, min_chars: int = 4) -> None:
        self._min_chars = min_chars

    def from_reviews(self, items: Sequence[StagingConversation]) -> list[DpoPair]:
        """审核编辑留痕：`before`（原始回答）作 rejected，`after`（改后回答）作 chosen。"""
        pairs: list[DpoPair] = []
        for item in items:
            if item.after is None:
                continue
            question, original = split_qa(item)
            prompt = question or (item.before or "")
            pair = self._make(
                pair_id=f"dpo_rev_{item.staging_id}",
                prompt=prompt,
                chosen=item.after,
                rejected=original,
                source="review_edit",
                source_ids=[item.staging_id],
                scene=item.scene,
                meta={"reviewer": item.reviewer, "reason": item.reason},
            )
            if pair is not None:
                pairs.append(pair)
        return pairs

    def from_corrections(self, records: Sequence[dict[str, Any]]) -> list[DpoPair]:
        """人工纠偏记录：`{prompt, chosen, rejected, id?, scene?}` —— 客服/运营手工重写的回答。"""
        pairs: list[DpoPair] = []
        for record in records:
            pair = self._make(
                pair_id=f"dpo_cor_{record.get('id') or abs(hash(str(record))) % 10**6}",
                prompt=str(record.get("prompt") or ""),
                chosen=str(record.get("chosen") or ""),
                rejected=str(record.get("rejected") or ""),
                source="correction",
                source_ids=[str(record["id"])] if record.get("id") else [],
                scene=record.get("scene"),
                meta={"operator": record.get("operator")},
            )
            if pair is not None:
                pairs.append(pair)
        return pairs

    def build(
        self,
        *,
        reviews: Sequence[StagingConversation] = (),
        corrections: Sequence[dict[str, Any]] = (),
    ) -> list[DpoPair]:
        return [*self.from_reviews(reviews), *self.from_corrections(corrections)]

    def _make(
        self,
        *,
        pair_id: str,
        prompt: str,
        chosen: str,
        rejected: str,
        source: str,
        source_ids: list[str],
        scene: str | None,
        meta: dict[str, Any],
    ) -> DpoPair | None:
        """成对性校验：任一字段为空、chosen 与 rejected 相同 → 这对没有偏好信息，直接丢弃。"""
        if len(prompt.strip()) < self._min_chars:
            return None
        if not chosen.strip() or not rejected.strip():
            return None
        if chosen.strip() == rejected.strip():
            logger.info(
                "data.training.dpo.dropped",
                extra={"event": "data_training", "status": "skipped", "reason": "chosen_equals_rejected"},
            )
            return None
        return DpoPair(
            pair_id=pair_id,
            prompt=prompt.strip(),
            chosen=chosen.strip(),
            rejected=rejected.strip(),
            source=source,
            source_ids=list(source_ids),
            scene=scene,
            meta=meta,
        )


__all__ = ["DpoBuilder", "DpoPair", "split_qa"]
