"""去重：**精确去重与近重复检测是两件事**，缺一不可。

为什么不能只按 content 去重：不同会话里出现同样的正常问答（如"多久发货"）会被误删，
所以精确去重用组合键 (session_id, ts, content_hash)。

为什么还要近重复：同一问题的多种改写如果全进训练集会造成分布偏斜；
更严重的是训练/测试集若包含近重复样本会造成**评估虚高（数据泄漏）**。

近重复用什么度量（这里有意讲清「为什么不直接用 SimHash」）：
在本项目的短中文文本上实测，SimHash 的区分度不足 —— 相似对 Hamming≈18、
无关对 Hamming≈33（64 位），阈值窗口太窄，调松就误杀、调紧就漏检。
改用**字符 bigram 的 Jaccard 相似度**后分离度很好：相似对 0.41~0.48，无关对 0.0，
阈值取 0.42 既稳又可解释。

诚实边界：这里是逐条两两比较（O(n²) 的最坏复杂度，配 max_compare 封顶）。
生产规模下要先用 SimHash 做 LSH 分桶把候选缩到常数级，再在桶内精确比较。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..infra.core.logging import get_logger
from .schemas import StandardMessage

logger = get_logger("data.dedupe")

SIMHASH_BITS = 64
DEFAULT_JACCARD_THRESHOLD = 0.42


def content_hash(text: str) -> str:
    """规范化后取 SHA1 —— 大小写/多余空白不该造成"不同"的假象。"""
    normalized = " ".join(text.split()).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def exact_key(message: StandardMessage) -> str:
    """组合唯一键：只按内容会让不同会话的相同问答被误删。"""
    return f"{message.session_id}|{int(message.ts.timestamp())}|{message.content_hash}"


def bigrams(text: str) -> set[str]:
    """字符 bigram 集合：对中文短文本比按空格分词更稳定。"""
    compact = "".join(text.split()).lower()
    return {compact[index : index + 2] for index in range(len(compact) - 1)} or {compact}


def jaccard(left: str, right: str) -> float:
    """字符 bigram 的 Jaccard 相似度。"""
    left_set, right_set = bigrams(left), bigrams(right)
    union = left_set | right_set
    return round(len(left_set & right_set) / max(len(union), 1), 4)


def near_duplicate(left: str, right: str, *, threshold: float = DEFAULT_JACCARD_THRESHOLD) -> bool:
    return jaccard(left, right) >= threshold


def simhash(text: str, *, bits: int = SIMHASH_BITS) -> int:
    """SimHash 指纹（保留给「大规模候选分桶」用，不作为最终判定）。"""
    weights = [0] * bits
    for token in bigrams(text):
        digest = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        for index in range(bits):
            weights[index] += 1 if (digest >> index) & 1 else -1
    fingerprint = 0
    for index, weight in enumerate(weights):
        if weight > 0:
            fingerprint |= 1 << index
    return fingerprint


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


@dataclass(slots=True)
class DedupeReport:
    kept: list[Any] = field(default_factory=list)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    near_duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def removed(self) -> int:
        return self.exact_duplicates + self.near_duplicates


class Deduper:
    """两步去重，且**两级粒度**：

    - 精确去重在**消息级**：防的是同一条消息被重复写入（导出、重放、同步抖动）；
    - 近重复在**语义单元级**（切片后的轮次）：防的是同一问答的改写版本进两份资产。

    为什么近重复不能放在消息级：把用户提问删掉、只留客服回答，语义单元就残缺了 ——
    去重应该删掉「整轮」，而不是删掉「半轮」。这是实测踩出来的：先按消息级近重复，
    结果问答对被拆成孤儿消息，训练样本直接因为「只有一问」被导出门禁拒掉。
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_JACCARD_THRESHOLD,
        enabled_near: bool = True,
        max_compare: int = 500,
    ) -> None:
        self._threshold = threshold
        self._enabled_near = enabled_near
        # 逐条比较的上限：超过就停止继续比较（避免脏数据把流水线拖垮），
        # 生产用 SimHash 分桶替代这个兜底。
        self._max_compare = max_compare

    def exact(self, messages: Sequence[StandardMessage]) -> DedupeReport:
        """消息级精确去重（组合键：session + 时间 + 内容哈希）。"""
        report = DedupeReport()
        seen: set[str] = set()
        for message in messages:
            key = exact_key(message)
            if key in seen:
                report.exact_duplicates += 1
                continue
            seen.add(key)
            report.kept.append(message)
        self._log("exact", report, len(messages))
        return report

    def near(self, units: Sequence[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]]]:
        """语义单元级近重复。units: [(unit_id, text)] → (保留的 unit_id, 重复对)。

        「保留最先出现者」是**刻意的**：流水线是有序的，先到的通常是更早/更权威的来源；
        若希望保留更完整的那一份，应在接入阶段按来源优先级排序，而不是让去重猜。
        """
        kept_ids: list[str] = []
        pairs: list[tuple[str, str]] = []
        if not self._enabled_near:
            return [unit_id for unit_id, _ in units], pairs
        kept_texts: list[tuple[str, str]] = []
        for unit_id, text in units:
            match = next(
                (
                    existing_id
                    for existing_id, existing_text in kept_texts[-self._max_compare :]
                    if jaccard(text, existing_text) >= self._threshold
                ),
                None,
            )
            if match is not None:
                pairs.append((match, unit_id))
                continue
            kept_texts.append((unit_id, text))
            kept_ids.append(unit_id)
        report = DedupeReport(near_duplicates=len(pairs), near_duplicate_pairs=pairs)
        self._log("near", report, len(units))
        return kept_ids, pairs

    @staticmethod
    def _log(stage: str, report: DedupeReport, total: int) -> None:
        logger.info(
            "data.dedupe.done",
            extra={
                "event": "data_dedupe",
                "status": "ok",
                "reason": (
                    f"stage={stage},exact={report.exact_duplicates},"
                    f"near={report.near_duplicates},total={total}"
                ),
            },
        )


def group_by_session(messages: Iterable[StandardMessage]) -> dict[str, list[StandardMessage]]:
    """按会话分组（切片与训练集分组切分都要用）。"""
    grouped: dict[str, list[StandardMessage]] = {}
    for message in messages:
        grouped.setdefault(message.session_id, []).append(message)
    for items in grouped.values():
        items.sort(key=lambda item: item.ts)
    return grouped
