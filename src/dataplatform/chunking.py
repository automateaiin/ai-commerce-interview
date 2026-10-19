"""切片：语义边界优先，时间窗只做初筛；token 上限 + overlap。

面试追问「为什么不用固定 5 分钟窗」：
固定窗会把**问题和答案切到两个切片**里 —— 检索时只召回半个语义单元，定位不准。
所以顺序是：先按会话语义边界（用户提问 + 客服回答=一个完整单元）切，
只有拿不到清晰边界时才 fallback 到时间窗；最后再按 token 上限硬切并加 overlap
（overlap 是为了避免语义在切口处断裂，代价是索引体积变大）。

**两种模式（可切换，默认保持语义边界模式不变）**：
1) `semantic`（默认）：会话轮次 + token 上限切分。适合对话/话术类文本。
2) `hierarchical`：L1/L2/L3 三级切片 + Auto-merging。适合长文档型知识。

三级切片的「为什么」（面试要能讲清的权衡）：
- 只切大块 → 检索粒度太粗，向量被平均掉，命中不准；
- 只切小块 → 召回准了但**上下文被切碎**，模型拿到的是断章，容易答偏；
- 所以分层：**用小块去检索（召回准确），命中后向上合并回父块（上下文完整）**。
  L1=句子滑动窗口（检索单元）、L2=段落级聚合、L3=章节/整篇（回答时的上下文）。
- Auto-merging 的判定：同一父块下命中的子块占比超过阈值 `merge_ratio`
  就整体替换为父块 —— 命中零散一两句时不合并（那属于「只需要这一句」的场景）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Sequence

from ..infra.core.logging import get_logger
from .schemas import StandardMessage, speaker_role

logger = get_logger("data.chunking")

MODE_SEMANTIC = "semantic"
MODE_HIERARCHICAL = "hierarchical"
CHUNK_MODES = (MODE_SEMANTIC, MODE_HIERARCHICAL)

SENTENCE_ENDINGS = "。！？!?；;\n"


@dataclass(slots=True)
class ChunkNode:
    """分级切片节点。L1 参与检索，L2/L3 是合并上下文时用的父块。"""

    node_id: str
    level: int
    text: str
    parent_id: str | None = None
    index: int = 0

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def split_sentences(text: str) -> list[str]:
    """按句读切句（保留分隔符）。空片段丢弃，避免产生无意义节点。"""
    sentences: list[str] = []
    start = 0
    for position, char in enumerate(text):
        if char in SENTENCE_ENDINGS:
            piece = text[start : position + 1]
            if piece.strip():
                sentences.append(piece)
            start = position + 1
    tail = text[start:]
    if tail.strip():
        sentences.append(tail)
    return sentences


def auto_merge(
    nodes: Sequence[ChunkNode],
    selected_ids: Sequence[str],
    *,
    ratio: float = 0.5,
) -> tuple[list[ChunkNode], list[str]]:
    """Auto-merging：同一父块下命中的子块占比达阈值 → 用父块替换这些子块。

    返回（合并后的节点列表, 合并原因）。**已命中的父块本身保留**，只吸收子块。
    """
    by_id = {node.node_id: node for node in nodes}
    selected = [by_id[node_id] for node_id in selected_ids if node_id in by_id]
    if not selected:
        return [], []

    selected_set = set(selected_ids)
    children: dict[str, list[ChunkNode]] = {}
    for node in nodes:
        if node.parent_id:
            children.setdefault(node.parent_id, []).append(node)

    replaced: set[str] = set()
    additions: list[ChunkNode] = []
    reasons: list[str] = []
    for parent_id, siblings in children.items():
        hit = [sibling for sibling in siblings if sibling.node_id in selected_set]
        if not hit or len(hit) / len(siblings) < ratio:
            continue
        parent = by_id.get(parent_id)
        if parent is None or parent.node_id in selected_set:
            continue
        additions.append(parent)
        replaced.update(sibling.node_id for sibling in hit)
        reasons.append(f"{parent_id}: {len(hit)}/{len(siblings)} 子块命中，合并为 L{parent.level} 上下文")

    merged = [node for node in selected if node.node_id not in replaced] + additions
    return merged, reasons

# 粗略 token 估算：中文约 1 字≈1 token，英文约 4 字符≈1 token，取折中 1.6 字符/token。
# 生产应换成真实 tokenizer；这里只要保证「截断行为可测、可复现」。
CHARS_PER_TOKEN = 1.6


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass(slots=True)
class Turn:
    """一个语义单元：边界清晰的一段对话。"""

    session_id: str
    messages: list[StandardMessage] = field(default_factory=list)
    boundary: str = "semantic"  # semantic | time_window
    start: object | None = None
    end: object | None = None

    @property
    def text(self) -> str:
        """轮的正文（不含说话人前缀，与 StagingConversation.text 口径一致）。"""
        return "\n".join(m.content for m in self.messages)

    @property
    def source_ids(self) -> list[str]:
        return [m.source_id for m in self.messages]


class Chunker:
    """消息 → 语义轮次 → 长度受控切片；或 长文本 → L1/L2/L3 三级切片。"""

    def __init__(
        self,
        *,
        time_window_s: int = 300,
        max_tokens: int = 220,
        overlap_tokens: int = 40,
        mode: str = MODE_SEMANTIC,
        l1_window: int = 2,
        l2_group: int = 3,
        l3_group: int = 9,
        merge_ratio: float = 0.5,
    ) -> None:
        if mode not in CHUNK_MODES:
            from ..infra.core.errors import ValidationError

            raise ValidationError(f"未知切片模式: {mode}", detail={"supported": list(CHUNK_MODES)})
        self._window = timedelta(seconds=time_window_s)
        self.mode = mode
        self._max_tokens = max_tokens
        self._overlap_tokens = min(overlap_tokens, max_tokens // 2)
        # 三级切片的窗口参数：L1 = 句子滑动窗口；L2/L3 = L1 的聚合组（可配）
        self._l1_window = max(1, l1_window)
        self._l2_group = max(1, l2_group)
        self._l3_group = max(self._l2_group, l3_group)
        self._merge_ratio = merge_ratio

    def split_turns(self, messages: list[StandardMessage]) -> list[Turn]:
        """按语义边界成轮；相隔超过时间窗则强制断开（初筛 + 兜底）。"""
        turns: list[Turn] = []
        current: Turn | None = None
        for message in messages:
            if current is None:
                current = Turn(session_id=message.session_id, start=message.ts, end=message.ts, messages=[message])
                continue
            within_window = (message.ts - current.end) <= self._window  # type: ignore[operator]
            # 语义边界：用户消息开启新一轮（客服回答跟随其后，不会被切开）
            is_new_turn = speaker_role(message.speaker) == "user" or not within_window
            if is_new_turn:
                turns.append(current)
                current = Turn(session_id=message.session_id, start=message.ts, end=message.ts, messages=[message])
            else:
                current.messages.append(message)
                current.end = message.ts
        if current is not None:
            turns.append(current)
        return turns

    def split_text(self, text: str) -> list[str]:
        """token 上限 + overlap。切点尽量落在换行/句读上，避免切碎句子。"""
        if estimate_tokens(text) <= self._max_tokens:
            return [text]
        chunks: list[str] = []
        start = 0
        step_chars = int(self._max_tokens * CHARS_PER_TOKEN)
        overlap_chars = int(self._overlap_tokens * CHARS_PER_TOKEN)
        while start < len(text):
            end = min(start + step_chars, len(text))
            if end < len(text):
                # 在候选窗口内回退到最近的换行或句号，保持语义完整
                window = text[start:end]
                cut = max(window.rfind("\n"), window.rfind("。"), window.rfind(". "))
                if cut > step_chars // 3:
                    end = start + cut + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
        return [chunk for chunk in chunks if chunk]

    def chunk_turn(self, turn: Turn, *, max_tokens: int | None = None) -> list[str]:
        """对单轮做长度受控切片，必要时用配置的上限覆盖默认值。"""
        if max_tokens is None:
            return self.split_text(turn.text)
        override = Chunker(
            time_window_s=int(self._window.total_seconds()),
            max_tokens=max_tokens,
            overlap_tokens=self._overlap_tokens,
            mode=self.mode,
            l1_window=self._l1_window,
            l2_group=self._l2_group,
            l3_group=self._l3_group,
            merge_ratio=self._merge_ratio,
        )
        return override.split_text(turn.text)

    def chunk_document(self, text: str) -> list[str]:
        """知识条目正文切片。`semantic` 模式按长度切；`hierarchical` 模式取 L1 作检索单元。"""
        if self.mode == MODE_HIERARCHICAL:
            return [node.text for node in self.hierarchical(text) if node.level == 1]
        return self.split_text(text)

    # -- 三级切片（L1/L2/L3）------------------------------------------------ #
    def hierarchical(self, text: str, *, prefix: str = "hn") -> list[ChunkNode]:
        """长文本 → L1（句子滑动窗口）/ L2（段落聚合）/ L3（章节聚合）。

        关键约束：**任何节点都不得越过 token 上限** —— 超限节点用 split_text 再切，
        切片后的每片保持同一 level 与 parent_id（否则血缘链会断）。
        """
        sentences = split_sentences(text)
        if not sentences:
            return []

        # 句子的字符区间：L2/L3 的正文直接按区间截取（不必拼接 L1，避免滑动窗口导致的重复）
        spans: list[tuple[int, int]] = []
        cursor = 0
        for sentence in sentences:
            start = text.find(sentence, cursor)
            cursor = start + len(sentence)
            spans.append((start, cursor))

        count = len(sentences)
        # 分组表：group g 覆盖 L1 序号区间 [g*size, (g+1)*size)，与 parent 归属（index // size）严格一致
        buckets = {
            level: [
                (group * size, min((group + 1) * size, count))
                for group in range((count + size - 1) // size)
            ]
            for level, size in ((2, self._l2_group), (3, self._l3_group))
        }

        nodes: list[ChunkNode] = []
        for index in range(count):
            last = min(index + self._l1_window - 1, count - 1)
            start, end = spans[index][0], spans[last][1]
            nodes.append(ChunkNode(node_id=f"{prefix}:L1:{index:03d}", level=1, text=text[start:end], index=index))
        for level in (2, 3):
            for group, (first, last) in enumerate(buckets[level]):
                nodes.append(
                    ChunkNode(
                        node_id=f"{prefix}:L{level}:{group:03d}",
                        level=level,
                        text=text[spans[first][0] : spans[last - 1][1]],
                        index=group,
                    )
                )

        # 父子关系：L1 → 所属 L2 组；L2 → 所属 L3 组（按 L1 序号整除分组）
        l2_id_of_group = {group: f"{prefix}:L2:{group:03d}" for group in range(len(buckets[2]))}
        l3_id_of_group = {group: f"{prefix}:L3:{group:03d}" for group in range(len(buckets[3]))}
        l2_group_of_l1 = {index: index // self._l2_group for index in range(count)}
        l3_group_of_l2 = {
            group: (group * self._l2_group) // self._l3_group for group in l2_id_of_group
        }
        for node in nodes:
            if node.level == 1:
                node.parent_id = l2_id_of_group.get(l2_group_of_l1[node.index])
            elif node.level == 2:
                node.parent_id = l3_id_of_group.get(l3_group_of_l2[node.index])
            else:
                node.parent_id = None

        return self._enforce_token_cap(nodes)

    def _enforce_token_cap(self, nodes: Sequence[ChunkNode]) -> list[ChunkNode]:
        """超限节点再切：`token 上限 + overlap` 对长文本仍然生效。

        **首片保留原 node_id**：父块被切开时如果整块改名，子块的 parent_id 就会指向
        一个不存在的节点，Auto-merging 永远合不上 —— 这是切片里最容易踩的坑。
        """
        capped: list[ChunkNode] = []
        for node in nodes:
            if node.tokens <= self._max_tokens:
                capped.append(node)
                continue
            pieces = self.split_text(node.text)
            for offset, piece in enumerate(pieces):
                node_id = node.node_id if offset == 0 else f"{node.node_id}.{offset:02d}"
                capped.append(replace(node, node_id=node_id, text=piece))
        return capped

    def merge_ratio(self) -> float:
        return self._merge_ratio

    def report(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "max_tokens": self._max_tokens,
            "overlap_tokens": self._overlap_tokens,
        }
