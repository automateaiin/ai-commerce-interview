"""脱敏：可配置规则集 + 占位符 + **导出门禁（双保险）**。

面试要点：
- 一组正则规则（手机、邮箱、证件、银行卡、API Key），命中替换成 `<phone>` 这类占位符；
- 日志只记录**命中类型**，绝不记录原值 —— 否则脱敏等于白做（原值进了日志系统）；
- 入库前脱一次、**导出训练集前再扫一次**：两次之间数据可能被其它流程改写，
  单点脱敏一定会漏；导出门禁是最后一道闸。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..infra.core.logging import get_logger

logger = get_logger("data.scrub")

# 规则集：可配置、可测试。顺序敏感 —— 卡号要先于手机号匹配（都是数字串）。
RULES: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("id_card", re.compile(r"\b\d{17}[\dXx]\b")),
    ("bank_card", re.compile(r"\b\d{16,19}\b")),
    ("phone_cn", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("api_key", re.compile(r"(?i)\b(?:bearer|api[-_]?key|sk-)[\w\-.]{6,}")),
]


@dataclass(slots=True)
class ScrubResult:
    text: str
    hits: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.hits.values())


class Scrubber:
    """脱敏器。规则可注入，便于针对不同业务加/减规则。"""

    def __init__(self, rules: list[tuple[str, re.Pattern[str]]] | None = None) -> None:
        self._rules = list(rules or RULES)

    def scrub(self, text: str) -> ScrubResult:
        hits: dict[str, int] = {}
        result = text
        for name, pattern in self._rules:
            result, count = pattern.subn(f"<{name}>", result)
            if count:
                hits[name] = count
        if hits:
            # 只记命中类型与次数，**不记原值**
            logger.info(
                "data.scrub.hit",
                extra={"event": "data_scrub", "status": "ok", "reason": ",".join(sorted(hits))},
            )
        return ScrubResult(text=result, hits=hits)

    def scan(self, text: str) -> dict[str, int]:
        """只扫描不替换 —— 导出门禁用它判断「是否还有敏感值漏出」。"""
        return {name: len(pattern.findall(text)) for name, pattern in self._rules if pattern.search(text)}

    def gate(self, payloads: list[str], *, context: str) -> list[str]:
        """导出门禁：任一 payload 命中敏感值就记警告并返回问题清单（调用方决定是否中断）。"""
        problems: list[str] = []
        for index, text in enumerate(payloads):
            hits = self.scan(text)
            if hits:
                problems.append(f"{context}#{index}:{','.join(sorted(hits))}")
                logger.warning(
                    "data.scrub.gate_blocked",
                    extra={"event": "data_scrub", "status": "error", "reason": ",".join(sorted(hits))},
                )
        return problems
