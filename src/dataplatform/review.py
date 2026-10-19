"""人工审核：可信数据的唯一入口。

面试要点（「为什么机器过滤了还要人工审核」）：
规则能稳定过滤格式错误和明显噪声，但**很难稳定判断客服回答是否符合品牌口径、
商品事实与促销政策**。审核层把机器初筛与人工业务判断结合，是控制「错误知识进入线上」的闸门。

三件事必须留痕：
- 审核状态（pending/approved/rejected）—— 决定这条数据有没有资格流向知识或训练；
- **修改前后内容（before/after）** —— 审核不是只看一眼，编辑才是常态；
- 审核人 + 理由 + 时间 —— 出错时能回答「谁在什么时候因为什么放行的」。
"""

from __future__ import annotations

from ..infra.core.errors import NotFoundError, ValidationError
from ..infra.core.logging import get_logger
from .schemas import ReviewStatus, StagingConversation, utcnow

logger = get_logger("data.review")


class ReviewQueue:
    """staging 队列与状态机。审核动作全部留痕，且不允许跨状态乱跳。"""

    def __init__(self, items: list[StagingConversation] | None = None) -> None:
        self._items: dict[str, StagingConversation] = {item.staging_id: item for item in (items or [])}

    def add(self, item: StagingConversation) -> StagingConversation:
        self._items[item.staging_id] = item
        return item

    def add_many(self, items: list[StagingConversation]) -> int:
        for item in items:
            self.add(item)
        return len(items)

    def get(self, staging_id: str) -> StagingConversation:
        item = self._items.get(staging_id)
        if item is None:
            raise NotFoundError(f"staging 记录不存在: {staging_id}", detail={"staging_id": staging_id})
        return item

    def all(self) -> list[StagingConversation]:
        return list(self._items.values())

    def pending(self) -> list[StagingConversation]:
        return [item for item in self._items.values() if item.status is ReviewStatus.PENDING]

    def approved(self) -> list[StagingConversation]:
        return [item for item in self._items.values() if item.status is ReviewStatus.APPROVED]

    def rejected(self) -> list[StagingConversation]:
        return [item for item in self._items.values() if item.status is ReviewStatus.REJECTED]

    def approve(self, staging_id: str, *, reviewer: str, note: str = "") -> StagingConversation:
        return self._transition(staging_id, ReviewStatus.APPROVED, reviewer=reviewer, reason=note)

    def reject(self, staging_id: str, *, reviewer: str, reason: str) -> StagingConversation:
        if not reason.strip():
            # 驳回必须给理由：没有理由是「说不清为什么不要」，后续无法复盘误杀
            raise ValidationError("驳回必须填写理由", detail={"staging_id": staging_id})
        return self._transition(staging_id, ReviewStatus.REJECTED, reviewer=reviewer, reason=reason)

    def edit(self, staging_id: str, *, reviewer: str, new_text: str, reason: str = "") -> StagingConversation:
        """编辑后仍需再次通过才生效 —— 编辑本身不等于放行。"""
        item = self.get(staging_id)
        if not new_text.strip():
            raise ValidationError("编辑内容不能为空", detail={"staging_id": staging_id})
        item.before = item.before if item.before is not None else item.text
        item.after = new_text
        item.reviewer = reviewer
        item.reason = reason
        item.status = ReviewStatus.PENDING
        logger.info(
            "data.review.edited",
            extra={"event": "data_review", "status": "ok", "reason": staging_id},
        )
        return item

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self._items),
            "pending": len(self.pending()),
            "approved": len(self.approved()),
            "rejected": len(self.rejected()),
        }

    def _transition(self, staging_id: str, status: ReviewStatus, *, reviewer: str, reason: str) -> StagingConversation:
        item = self.get(staging_id)
        item.status = status
        item.reviewer = reviewer
        item.reason = reason or item.reason
        item.reviewed_at = utcnow()
        logger.info(
            "data.review.transition",
            extra={"event": "data_review", "status": status.value, "reason": staging_id},
        )
        return item
