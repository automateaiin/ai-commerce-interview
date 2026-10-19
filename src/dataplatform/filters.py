"""过滤：**规则链 + 每条带原因 + 可统计误删率**。

面试会追问「为什么不用一个模型判质量」：
规则透明、可回归、成本低、误判可定位；不确定的宁可进待审核队列，
也不自动采信或直接删除。文档里说的「九级过滤」不是重点，
重点是每条规则**可解释、可测试、可统计**，并允许规则迭代（误删率高就调阈值）。

这里还刻意把「格式无效」与「业务不合格」分开：前者在入库前就应剔除，
后者必须交给人工审核 —— 混在一起会导致真正的噪声穿透或好样本被误杀。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..infra.core.logging import get_logger
from .schemas import FilterDecision, StandardMessage

logger = get_logger("data.filters")

_SYSTEM_HINTS = ("已加入群聊", "撤回了一条消息", "对方正在输入", "系统通知", "你已添加")
_AD_HINTS = ("加微信", "代购", "本店最低价", "私聊下单", "扫码进群")
_MOJIBAKE = re.compile(r"[\ufffd]|(?:Ã.|Â.){2,}")


@dataclass(slots=True)
class Rule:
    """一条过滤规则：id + 原因 + 判定函数。id 用于统计每类规则的命中量。"""

    rule_id: str
    reason: str
    predicate: Callable[[StandardMessage], bool]
    stage: str = "pre"  # pre=入库前剔除；review=交给人工审核判断


def _is_empty(message: StandardMessage) -> bool:
    return not message.content.strip()


def _is_system(message: StandardMessage) -> bool:
    return any(hint in message.content for hint in _SYSTEM_HINTS)


def _is_too_short(message: StandardMessage) -> bool:
    return len(message.content.strip()) < 4 and not message.meta.get("question")


def _is_ad_noise(message: StandardMessage) -> bool:
    return sum(1 for hint in _AD_HINTS if hint in message.content) >= 2


def _is_mojibake(message: StandardMessage) -> bool:
    return bool(_MOJIBAKE.search(message.content))


def _needs_review(message: StandardMessage) -> bool:
    """业务可疑但不明显无效：交人工审核，而不是直接删。"""
    return message.parse_status.value == "partial" or "**" in message.content


DEFAULT_RULES: list[Rule] = [
    Rule("empty_content", "内容为空", _is_empty),
    Rule("system_notice", "系统通知/撤回消息，非业务问答", _is_system),
    Rule("too_short", "无意义短文本", _is_too_short),
    Rule("ad_noise", "广告噪声（多条营销话术）", _is_ad_noise),
    Rule("mojibake", "乱码", _is_mojibake),
    Rule("needs_review", "身份映射失败或内容可疑，转人工判断", _needs_review, stage="review"),
]


@dataclass(slots=True)
class FilterReport:
    kept: list[StandardMessage] = field(default_factory=list)
    rejected: list[tuple[StandardMessage, FilterDecision]] = field(default_factory=list)
    review_flagged: list[tuple[StandardMessage, FilterDecision]] = field(default_factory=list)
    rule_hits: dict[str, int] = field(default_factory=dict)

    def hit_rate(self, rule_id: str, total: int) -> float:
        """单条规则的命中率，用于观察「是否过严导致误杀」。"""
        return round(self.rule_hits.get(rule_id, 0) / max(total, 1), 4)


class FilterChain:
    """规则链。命中即记录原因并短路 —— 顺序即优先级。"""

    def __init__(self, rules: Sequence[Rule] | None = None) -> None:
        self._rules = list(rules or DEFAULT_RULES)

    def run(self, messages: Sequence[StandardMessage]) -> FilterReport:
        report = FilterReport()
        for message in messages:
            decision = self._first_hit(message)
            if decision is None:
                report.kept.append(message)
                continue
            report.rule_hits[decision.rule_id or "unknown"] = report.rule_hits.get(decision.rule_id or "unknown", 0) + 1
            # 承接规则若为 review 阶段，则进待审核队列；否则直接剔除
            rule_stage = next((rule.stage for rule in self._rules if rule.rule_id == decision.rule_id), "pre")
            if rule_stage == "review":
                report.review_flagged.append((message, decision))
            else:
                report.rejected.append((message, decision))

        total = len(messages)
        logger.info(
            "data.filters.done",
            extra={
                "event": "data_filter",
                "status": "ok",
                "reason": (
                    f"kept={len(report.kept)},rejected={len(report.rejected)},"
                    f"review={len(report.review_flagged)},total={total}"
                ),
            },
        )
        return report

    def _first_hit(self, message: StandardMessage) -> FilterDecision | None:
        for rule in self._rules:
            if rule.predicate(message):
                return FilterDecision(kept=False, rule_id=rule.rule_id, reason=rule.reason)
        return None
