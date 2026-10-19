"""统一审计事件：把权限链的四级落到可追溯的记录上。

权限链 `Service → Agent → Tool → Resource`（PROJECT.md 4.x）里，前三级是「谁在调」，
第四级是「调了哪份数据」。只记工具名是不够的 —— 出事复盘时要能回答
「内容运营线（Service）的 operations_agent（Agent）通过 create_order（Tool）动了 SKU-1001（Resource）」。

三个设计点：
1) **资源由参数推导，不由调用方声明**：调用方声明等于让被审计者自己写审计记录；
2) **拒绝也要留痕**：只记成功调用，越权尝试就完全隐形 —— 而越权尝试恰恰是最该看的；
3) **审计不是日志**：它是结构化、可按 project 查询的环形记录，供复盘与合规检查，
   而日志是运维视角的流水。
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from .logging import get_logger, scrub
from .metrics import metrics

logger = get_logger("infra.audit")

# 参数中作为「资源主体」的键（无 *_id 命中时的兜底），按此顺序取第一个非空值
_PRIMARY_ARG_KEYS: tuple[str, ...] = ("query", "keyword", "sku", "code", "topic")

# 同一实体的不同叫法归一，避免审计里出现 sku / product 两套命名
_KIND_ALIASES: dict[str, str] = {
    "sku": "product",
    "order_no": "order",
    "order": "order",
    "product": "product",
    "user": "user",
    "session": "session",
    "chunk": "knowledge",
    "article": "knowledge",
    "knowledge": "knowledge",
}


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """一次工具调用的审计记录（不可变，可长期留存）。"""

    audit_id: str
    ts: float
    trace_id: str | None
    service: str
    agent: str
    caller: str
    project_id: str | None
    tool: str
    resource: str
    decision: str
    """allow（放行执行）| deny（权限拒绝）| error（执行失败）。"""
    reason: str | None = None
    risk: str | None = None
    duration_ms: float = 0.0
    error_code: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "ts": self.ts,
            "trace_id": self.trace_id,
            "service": self.service,
            "agent": self.agent,
            "caller": self.caller,
            "project_id": self.project_id,
            "tool": self.tool,
            "resource": self.resource,
            "decision": self.decision,
            "reason": self.reason,
            "risk": self.risk,
            "duration_ms": round(self.duration_ms, 2),
            "error_code": self.error_code,
        }


def _normalize_kind(raw: str) -> str:
    key = raw.strip().lower()
    return _KIND_ALIASES.get(key, key)


def derive_resource(tool: str, args: Mapping[str, Any] | None) -> str:
    """从工具名与参数推导资源标识，形如 `order:ORD-90001`。

    - 优先取参数里的 `*_id`（多个时按字母序取第一个，保证同一调用推导结果稳定）；
    - 其次取业务主体字段（query 等），截断并脱敏后作为资源名；
    - 两者都没有则返回 `tool:*`，表示该调用不针对某个具体资源。
    """
    payload = dict(args or {})
    id_keys = sorted(key for key in payload if key.endswith("_id") and payload[key] not in (None, ""))
    if id_keys:
        key = id_keys[0]
        kind = _normalize_kind(key[:-3])
        return f"{kind}:{scrub(str(payload[key]))}"

    for key in _PRIMARY_ARG_KEYS:
        value = payload.get(key)
        if value not in (None, ""):
            text = scrub(str(value))
            return f"{tool}:{text[:48]}"
    return f"{tool}:*"


class AuditLog:
    """审计记录容器（进程内环形缓冲）。

    为什么用有界环形缓冲而不是无界列表：审计是在线写入，无界增长会变成内存泄漏；
    生产环境应落库（带 project_id 索引）并设置保留期，这里的区别只是存储介质。
    """

    def __init__(self, *, max_entries: int = 2000) -> None:
        self._entries: deque[AuditEvent] = deque(maxlen=max_entries)

    def record(
        self,
        *,
        tool: str,
        resource: str,
        decision: str,
        service: str | None = None,
        agent: str | None = None,
        caller: str | None = None,
        project_id: str | None = None,
        trace_id: str | None = None,
        reason: str | None = None,
        risk: str | None = None,
        duration_ms: float = 0.0,
        error_code: str | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            ts=time.time(),
            trace_id=trace_id,
            service=service or "unknown",
            agent=agent or "unknown",
            caller=caller or "unknown",
            project_id=project_id,
            tool=tool,
            resource=resource,
            decision=decision,
            reason=reason,
            risk=risk,
            duration_ms=duration_ms,
            error_code=error_code,
        )
        self._entries.append(event)
        metrics.incr("audit", f"{decision}")
        logger.info(
            "audit.tool_call",
            extra={
                "event": "audit",
                "status": "ok" if decision == "allow" else "error",
                "audit_id": event.audit_id,
                "service": event.service,
                "agent": event.agent,
                "caller": event.caller,
                "tool": tool,
                "resource": resource,
                "decision": decision,
                "reason": reason,
                "risk": risk,
                "error_code": error_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return event

    def query(self, *, project_id: str | None, limit: int = 50, decision: str | None = None) -> list[AuditEvent]:
        """按项目维度取最近记录。limit 同时限制扫描量，避免大缓冲下的全量遍历。"""
        matched: list[AuditEvent] = []
        for event in reversed(self._entries):
            if project_id is not None and event.project_id != project_id:
                continue
            if decision is not None and event.decision != decision:
                continue
            matched.append(event)
            if len(matched) >= limit:
                break
        return matched

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for event in self._entries:
            counts[event.decision] = counts.get(event.decision, 0) + 1
        return {"total": len(self._entries), "by_decision": counts}


__all__ = ["AuditEvent", "AuditLog", "derive_resource"]
