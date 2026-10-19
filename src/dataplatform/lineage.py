"""数据血缘：raw → standard → staging → article → chunk → vector → export 的可追溯链。

面试要点（「如何保证可追溯」的标准答案不是「记了日志」）：
为**每一层实体保留稳定 ID 与关联 ID**，并记录处理规则版本、审核人、更新时间、导出批次。
线上答案出错时，可以从回答引用的 chunk_id 反向追到 article、staging、raw 原始记录，
从而回答「这条知识是谁、在什么时候、依据什么规则放进去的」。

本实现是内存图（离线可跑）。生产落在元数据表里（每个实体一行 + parent 外键），
并加上处理规则版本与批次号。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..infra.core.logging import get_logger
from .schemas import Dataset, LineageNode

logger = get_logger("data.lineage")


@dataclass(slots=True)
class LineageGraph:
    """有向图：node → parents。支持正向追溯与反向影响面分析。"""

    nodes: dict[str, LineageNode] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, node: LineageNode) -> LineageNode:
        self.nodes[node.node_id] = node
        for parent in node.parents:
            self.children[parent].append(node.node_id)
        return node

    def get(self, node_id: str) -> LineageNode | None:
        return self.nodes.get(node_id)

    def ancestors(self, node_id: str) -> list[str]:
        """反向追溯：从 chunk 一路回到 raw。用于「这条知识从哪来」。"""
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            node = self.nodes.get(current)
            if node is None:
                continue
            for parent in node.parents:
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return sorted(seen)

    def descendants(self, node_id: str) -> list[str]:
        """正向影响面：改了这条 raw，会污染哪些 chunk / 导出批次。"""
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            for child in self.children.get(current, []):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return sorted(seen)

    def trace(self, node_id: str) -> dict[str, Any]:
        node = self.nodes.get(node_id)
        chain = [node_id, *self.ancestors(node_id)]
        return {
            "node": node.as_dict() if node else None,
            "ancestors": self.ancestors(node_id),
            "descendants": self.descendants(node_id),
            "chain": chain,
        }

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            kinds[node.kind] += 1
        return dict(kinds)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [node.as_dict() for node in self.nodes.values()]


def build_lineage(dataset: Dataset, *, export_batch: str | None = None) -> LineageGraph:
    """由一次流水线产物构建血缘图。每层都用稳定 ID 关联到上一层。"""
    graph = LineageGraph()

    for raw in dataset.raw:
        graph.add(LineageNode(node_id=f"raw::{raw.source_id}", kind="raw", attributes={"source": raw.source.value}))

    for message in dataset.messages:
        graph.add(
            LineageNode(
                node_id=f"std::{message.source_id}",
                kind="standard",
                parents=[f"raw::{message.source_id}"],
                attributes={"session_id": message.session_id, "parse_status": message.parse_status.value},
            )
        )

    for staging in dataset.staging:
        for message in staging.messages:
            graph.add(
                LineageNode(
                    node_id=f"staging::{staging.staging_id}",
                    kind="staging",
                    parents=[f"std::{message.source_id}"],
                    attributes={"status": staging.status.value, "reviewer": staging.reviewer},
                )
            )

    for article in dataset.articles:
        graph.add(
            LineageNode(
                node_id=f"article::{article.article_id}",
                kind="article",
                parents=[f"staging::{sid}" for sid in article.source_staging_ids],
                attributes={"version": article.version, "price_freshness": article.price_freshness},
            )
        )

    for chunk in dataset.chunks:
        graph.add(
            LineageNode(
                node_id=f"chunk::{chunk.chunk_id}",
                kind="chunk",
                parents=[f"article::{chunk.article_id}"],
                attributes={"chunk_index": chunk.chunk_index, "version": chunk.version},
            )
        )
        graph.add(
            LineageNode(
                node_id=f"vector::{chunk.chunk_id}",
                kind="vector",
                parents=[f"chunk::{chunk.chunk_id}"],
                attributes={"embedding_version": chunk.embedding_version},
            )
        )

    if export_batch:
        exported_ids = [f"sft::{sample.sample_id}" for sample in dataset.sft_samples]
        graph.add(
            LineageNode(
                node_id=f"export::{export_batch}",
                kind="export",
                parents=exported_ids,
                attributes={"samples": len(exported_ids)},
            )
        )

    logger.info(
        "data.lineage.built",
        extra={"event": "data_lineage", "status": "ok", "reason": str(graph.stats())},
    )
    return graph


def dump(graph: LineageGraph) -> Iterable[dict[str, Any]]:
    return graph.as_dicts()
