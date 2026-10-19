"""Grounding Gate（事实绑定闸门）。

要解决的具体问题：模型会**凭记忆编造价格、库存、优惠与订单状态**。
这类信息是「会随时间变化、且直接影响交易」的，一旦编错就是真实的商业损失，
所以不能靠「在 Prompt 里叮嘱一句」来兜底 —— 必须有一道**可执行的闸门**：

    交易类结论（价格/库存/优惠/订单）必须绑定本次调用中真实取到的 Tool Result；
    没有来源就直接拒答或降级，而不是把模型的说法原样放行。

判定刻意做得**保守且确定**（正则 + 证据存在性），不引入第二个模型：
闸门本身如果也要靠模型判断，那它就和被检查的对象一样不可靠。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from src.infra.core.logging import get_logger

from .schemas import GroundingVerdict

logger = get_logger("agents.grounding")

TRANSACTION_INTENTS = frozenset({"price", "stock", "order_status", "coupon"})

_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("price", re.compile(r"(?:[$￥€]|\bUSD\b|\b\d+(?:\.\d+)?\s*(?:美元|美金|元|刀))", re.IGNORECASE)),
    ("stock", re.compile(r"(库存|有货|无货|现货|售罄|仅剩|剩余\s*\d+)")),
    ("coupon", re.compile(r"(优惠券|折扣|满减|立减|coupon)", re.IGNORECASE)),
    ("order_status", re.compile(r"\bORD-\d+\b", re.IGNORECASE)),
)

_ORDER_STATE_WORDS = re.compile(r"(已发货|已付款|待付款|已取消|退款中|已签收|运输中)")

# 明确的「拒答/不确定」措辞：这类回答本身没有断言事实，不应被判为需要来源
_HEDGE_MARKERS = ("无法确认", "无法提供", "不能确认", "以页面为准", "建议联系人工", "请联系人工", "需要查询")

_REFUSAL_TEMPLATE = (
    "关于{subject}的问题必须以实时查询结果为准。当前这轮对话没有取到可信来源，"
    "我不能给出具体数字；请稍后重试，或联系人工客服核实。"
)

_SUBJECTS = {"price": "价格", "stock": "库存", "coupon": "优惠", "order_status": "订单状态"}


class GroundingGate:
    """交易类回答的事实绑定闸门。"""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(
        self,
        *,
        answer: str | None,
        tool_results: Sequence[dict[str, Any]] = (),
        knowledge_chunks: Sequence[dict[str, Any]] = (),
        intent: str | None = None,
    ) -> GroundingVerdict:
        """返回判定 +（必要时）改写后的回答。

        `enforced=True` 表示原文被替换成拒答模板 —— 这是唯一一种「拦截」形态，
        不做「加一句免责声明后照发」的软化处理，因为数字已经说出去了。
        """
        if not self.enabled:
            return GroundingVerdict(ok=True, sanitized_answer=answer, reasons=["gate_disabled"])
        if not answer:
            return GroundingVerdict(ok=True, sanitized_answer=answer, reasons=[])

        has_tool_evidence = any(item.get("ok") for item in tool_results)
        has_knowledge = bool(knowledge_chunks)
        claimed = self._claims(answer)
        hedged = any(marker in answer for marker in _HEDGE_MARKERS)

        reasons: list[str] = []
        if claimed and not has_tool_evidence:
            if hedged:
                reasons.append("transaction_claim_hedged")
            else:
                return self._enforce(claimed[0], claimed, intent)

        if not has_tool_evidence and not has_knowledge and intent in TRANSACTION_INTENTS and not hedged:
            # 连意图都判成交易类，却没有取到任何证据：按无来源处理
            return self._enforce(intent, claimed or [intent], intent)

        return GroundingVerdict(
            ok=True,
            sanitized_answer=answer,
            reasons=reasons or ["grounded"],
        )

    @staticmethod
    def _claims(answer: str) -> list[str]:
        """回答里出现的交易类断言（有具体值才算断言，纯文字提及不算）。"""
        claims = [name for name, pattern in _CLAIM_PATTERNS if pattern.search(answer)]
        if _ORDER_STATE_WORDS.search(answer):
            claims.append("order_status")
        return sorted(set(claims))

    @staticmethod
    def _enforce(claim: str, claims: list[str], intent: str | None) -> GroundingVerdict:
        subject = _SUBJECTS.get(claim, _SUBJECTS.get(intent or "", claim))
        logger.warning(
            "agents.grounding_blocked",
            extra={
                "event": "grounding",
                "status": "error",
                "reason": "transaction_claim_without_evidence",
                "changed_keys": ",".join(claims),
            },
        )
        return GroundingVerdict(
            ok=False,
            reasons=[f"no_evidence_for:{item}" for item in claims],
            sanitized_answer=_REFUSAL_TEMPLATE.format(subject=subject),
            enforced=True,
        )


__all__ = ["GroundingGate", "TRANSACTION_INTENTS"]
