"""Human-in-the-Loop：Stage → Approve → Apply（人工确认高风险动作）。

高风险动作（真实下单/退款/改价/发券）不能由模型自己拍板，但也不能简单「拒绝执行」——
业务要的是「**先登记、再让人确认、确认后原样执行**」这条链路。

三步分离的理由（面试重点）：
- **Stage（登记）**：把「要做什么」冻结成一个检查点（动作名 + 参数 + 原始身份），并发一张
  绑定了该内容的 **Resume Token**。此后即使会话断了，也能凭 token 找回同一件事；
- **Approve（确认）**：确认是一个**独立权限**（`hitl:approve`），并且要校验 token 与检查点内容
  完全一致 —— 否则确认 A 事后被掉包成 B 就是提权漏洞；
- **Apply（执行）**：只有 approved 状态才允许执行，并且执行时**用回原始请求者的身份**，
  而不是审批人的身份。用审批人身份执行会把「谁下的单」记错，审计与售后都会跟着错。

Token 用 HMAC 绑定 (checkpoint_id, action, 参数) —— 改动任何一项都会导致校验失败。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from src.infra.core.errors import ForbiddenError, NotFoundError, ValidationError
from src.infra.core.logging import get_logger
from src.infra.tools.base import ToolContext, ToolExecutor

logger = get_logger("agents.hitl")

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_APPLIED = "applied"


@dataclass(slots=True)
class HITLPending:
    checkpoint_id: str
    action: str
    args: dict[str, Any]
    reason: str
    resume_token: str
    requested_by: str | None
    identity: dict[str, Any]
    status: str = STATUS_PENDING
    approver: str | None = None
    note: str | None = None
    applied: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)

    def to_payload(self, *, include_token: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint_id": self.checkpoint_id,
            "action": self.action,
            "args": self.args,
            "reason": self.reason,
            "status": self.status,
            "requested_by": self.requested_by,
            "approver": self.approver,
            "applied": self.applied,
        }
        if include_token:
            payload["resume_token"] = self.resume_token
        return payload


class HITLStore:
    """检查点存储。跨请求存活，但**有界且只增不改**（状态迁移只改自身状态字段）。"""

    def __init__(self, *, max_pending: int = 200) -> None:
        self._items: dict[str, HITLPending] = {}
        self._order: list[str] = []
        self._max = max_pending

    def put(self, record: HITLPending) -> HITLPending:
        self._items[record.checkpoint_id] = record
        self._order.append(record.checkpoint_id)
        while len(self._order) > self._max:
            oldest = self._order.pop(0)
            self._items.pop(oldest, None)
        return record

    def get(self, checkpoint_id: str) -> HITLPending | None:
        return self._items.get(checkpoint_id)

    def list(self, *, status: str | None = None) -> list[HITLPending]:
        items = [self._items[key] for key in self._order if key in self._items]
        if status:
            items = [item for item in items if item.status == status]
        return items


class HITLManager:
    def __init__(
        self,
        *,
        executor: ToolExecutor,
        store: HITLStore,
        secret: str,
        high_risk_tools: set[str],
        approve_scope: str = "hitl:approve",
    ) -> None:
        self._executor = executor
        self._store = store
        self._secret = secret.encode()
        self._high_risk = set(high_risk_tools)
        self._approve_scope = approve_scope

    # -- Stage -------------------------------------------------------------- #
    def stage(
        self,
        *,
        action: str,
        args: dict[str, Any],
        ctx: ToolContext,
        reason: str = "",
    ) -> HITLPending:
        if action not in self._high_risk:
            raise ValidationError(f"动作 {action} 不属于需要人工确认的高风险动作", detail={"action": action})
        checkpoint_id = f"hitl_{secrets.token_hex(6)}"
        record = HITLPending(
            checkpoint_id=checkpoint_id,
            action=action,
            args=dict(args),
            reason=reason or "高风险动作待人工确认",
            resume_token=self._sign(checkpoint_id, action, args),
            requested_by=ctx.agent,
            identity={
                "project_id": ctx.project_id,
                "session_id": ctx.session_id,
                "user_id": ctx.user_id,
                "scopes": sorted(ctx.scopes),
            },
        )
        self._store.put(record)
        logger.info(
            "agents.hitl.staged",
            extra={"event": "hitl", "status": "pending", "reason": action},
        )
        return record

    # -- Approve ------------------------------------------------------------ #
    def approve(
        self,
        *,
        checkpoint_id: str,
        token: str,
        ctx: ToolContext,
        note: str | None = None,
    ) -> HITLPending:
        record = self._require(checkpoint_id)
        if not ctx.has_scope(self._approve_scope):
            raise ForbiddenError(f"确认高风险动作需要 {self._approve_scope} 权限", detail={"checkpoint": checkpoint_id})
        if not self._verify(record, token):
            raise ForbiddenError("Resume Token 校验失败或检查点内容已被改动", detail={"checkpoint": checkpoint_id})
        if record.status != STATUS_PENDING:
            raise ValidationError(f"检查点当前状态为 {record.status}，不能重复确认", detail={"checkpoint": checkpoint_id})
        record.status = STATUS_APPROVED
        record.approver = ctx.user_id or ctx.agent or "unknown"
        record.note = note
        logger.info("agents.hitl.approved", extra={"event": "hitl", "status": "approved", "reason": record.action})
        return record

    def reject(self, *, checkpoint_id: str, ctx: ToolContext, note: str | None = None) -> HITLPending:
        record = self._require(checkpoint_id)
        if not ctx.has_scope(self._approve_scope):
            raise ForbiddenError(f"处置高风险动作需要 {self._approve_scope} 权限", detail={"checkpoint": checkpoint_id})
        if record.status != STATUS_PENDING:
            raise ValidationError(f"检查点当前状态为 {record.status}，不能处置", detail={"checkpoint": checkpoint_id})
        record.status = STATUS_REJECTED
        record.approver = ctx.user_id or ctx.agent or "unknown"
        record.note = note
        return record

    # -- Apply -------------------------------------------------------------- #
    async def apply(self, *, checkpoint_id: str, ctx: ToolContext) -> dict[str, Any]:
        record = self._require(checkpoint_id)
        if record.status != STATUS_APPROVED:
            raise ValidationError(
                f"检查点状态为 {record.status}，只有已确认（approved）才能执行", detail={"checkpoint": checkpoint_id}
            )
        # 用**原始请求者身份**执行，而不是审批人身份：下单归属不能因为审批而改变
        identity = record.identity
        exec_ctx = ToolContext(
            trace_id=ctx.trace_id,
            project_id=identity.get("project_id") or ctx.project_id,
            session_id=identity.get("session_id"),
            user_id=identity.get("user_id"),
            scopes=frozenset(identity.get("scopes") or []),
            service=ctx.service,
            agent=record.requested_by,
        )
        result = await self._executor.execute(record.action, dict(record.args), exec_ctx)
        record.applied = {
            "ok": result.ok,
            "tool": record.action,
            "data": result.data,
            "error_code": result.error_code,
            "error_message": result.error_message,
        }
        record.status = STATUS_APPLIED if result.ok else STATUS_APPROVED
        logger.info(
            "agents.hitl.applied",
            extra={
                "event": "hitl",
                "status": "ok" if result.ok else "error",
                "reason": record.action,
            },
        )
        return record.to_payload()

    def resume(self, *, checkpoint_id: str, token: str) -> HITLPending:
        """凭 Resume Token 找回同一件事（会话断了也能续上）。"""
        record = self._require(checkpoint_id)
        if not self._verify(record, token):
            raise ForbiddenError("Resume Token 校验失败", detail={"checkpoint": checkpoint_id})
        return record

    # -- 内部 -------------------------------------------------------------- #
    def _require(self, checkpoint_id: str) -> HITLPending:
        record = self._store.get(checkpoint_id)
        if record is None:
            raise NotFoundError("检查点不存在或已过期", detail={"checkpoint": checkpoint_id})
        return record

    def _sign(self, checkpoint_id: str, action: str, args: dict[str, Any]) -> str:
        payload = f"{checkpoint_id}|{action}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        return hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()

    def _verify(self, record: HITLPending, token: str) -> bool:
        expected = self._sign(record.checkpoint_id, record.action, record.args)
        return hmac.compare_digest(expected, token or "")


__all__ = ["HITLManager", "HITLStore", "HITLPending", "STATUS_APPLIED", "STATUS_APPROVED", "STATUS_PENDING"]
