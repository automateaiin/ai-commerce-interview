"""节点级 Checkpointer：持久化 / 回滚 / 失败重试 / 续跑 / 回放。

面试要讲清的三个问题，都在这里有明确答案：

1) **为什么是「节点级」而不是「请求级」**：请求级快照只能整单重跑，一个 5 节点的内容工作流
   在第 4 个节点失败，整单重跑等于把前 3 次模型调用（和它们的钱）再付一遍。
   节点级快照把「已完成节点」变成可复用的资产，续跑只补没跑完的部分。

2) **回滚为什么要连带下游**：节点的输出是下游的输入，只把某个节点置为 pending，
   下游就会拿旧的上游输出继续用，产生「版本混用」的隐蔽错误。
   所以 rollback 必须沿 `descendants` 一起失效。

3) **回放为什么记录每一次尝试而不是最后一次**：排障时关心的是「第几次失败、失败原因是什么、
   有没有触发回退」，只留最终状态会把这段过程抹掉，指标里也就看不出重试率。

诚实边界：本实现是**进程内有界**的（生产应把 records 落到表/Redis，用 run_id + node 做幂等 UPSERT）。
这样写是因为它同时要支撑「跨请求续跑」的语义，而进程内实现能把这个语义完整跑通。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.infra.core.logging import get_logger

from .schemas import NodeAttempt, NodeRecord, NodeStatus

logger = get_logger("ops.workflow.checkpointer")


@dataclass(slots=True)
class _RunState:
    run_id: str
    workflow: str
    created_at: float
    status: str = "running"
    order: list[str] = field(default_factory=list)
    records: dict[str, NodeRecord] = field(default_factory=dict)


class Checkpointer:
    """节点级检查点存储。

    `max_runs` 给出上界：跨请求存活不等于无限增长，超出后按写入顺序淘汰最旧的 run
    （淘汰只影响回放能力，不影响正在跑的 run，因为当前 run 一定在 OrderedDict 末尾）。
    """

    def __init__(self, *, max_runs: int = 200) -> None:
        self._runs: OrderedDict[str, _RunState] = OrderedDict()
        self._max_runs = max(1, max_runs)

    # -- 生命周期 ---------------------------------------------------------- #
    def begin(self, run_id: str, workflow: str) -> None:
        """开局一个 run。对同一 run_id 重复调用是幂等的（续跑正依赖这个性质）。"""
        state = self._runs.get(run_id)
        if state is None:
            self._runs[run_id] = _RunState(run_id=run_id, workflow=workflow, created_at=time.time())
            self._evict()
            return
        self._runs.move_to_end(run_id)

    def exists(self, run_id: str) -> bool:
        return run_id in self._runs

    def runs(self) -> list[str]:
        return list(self._runs.keys())

    def status_of(self, run_id: str) -> str | None:
        state = self._runs.get(run_id)
        return state.status if state else None

    def set_status(self, run_id: str, status: str) -> None:
        state = self._runs.get(run_id)
        if state is not None:
            state.status = status

    # -- 记录读写 ---------------------------------------------------------- #
    def put(self, run_id: str, record: NodeRecord) -> NodeRecord:
        state = self._require(run_id)
        state.records[record.node] = record
        if record.node not in state.order:
            state.order.append(record.node)
        return record

    def get(self, run_id: str, node: str) -> NodeRecord | None:
        state = self._runs.get(run_id)
        return state.records.get(node) if state else None

    def records(self, run_id: str) -> list[NodeRecord]:
        state = self._runs.get(run_id)
        if state is None:
            return []
        return [state.records[name] for name in state.order if name in state.records]

    def completed(self, run_id: str) -> dict[str, Any]:
        """已成功节点的输出（续跑时直接回填，不再重跑）。"""
        return {
            record.node: (record.output or {})
            for record in self.records(run_id)
            if record.status == NodeStatus.SUCCEEDED.value
        }

    def failed_nodes(self, run_id: str) -> list[str]:
        return [r.node for r in self.records(run_id) if r.status == NodeStatus.FAILED.value]

    # -- 回滚 -------------------------------------------------------------- #
    def rollback(self, run_id: str, nodes: Iterable[str], *, reason: str) -> list[str]:
        """把指定节点（通常含其下游）置为待重跑，返回被失效的节点名。

        只清 `output` 与状态，**保留 history**：这样回放里仍能看到上一轮为什么失败，
        而不是把历史抹成空白。
        """
        state = self._require(run_id)
        invalidated: list[str] = []
        for name in nodes:
            record = state.records.get(name)
            if record is None:
                continue
            if record.status == NodeStatus.PENDING.value and not record.history:
                continue
            record.status = NodeStatus.PENDING.value
            record.output = None
            record.error = None
            record.fallback_reason = None
            record.history.append(
                NodeAttempt(attempt=record.attempt, status="rolled_back", error=reason, duration_ms=record.duration_ms)
            )
            invalidated.append(name)
        if invalidated:
            logger.info(
                "workflow.rollback",
                extra={"event": "ops.rollback", "msg": "节点回滚", "count": len(invalidated)},
            )
        return invalidated

    # -- 回放 -------------------------------------------------------------- #
    def replay(self, run_id: str) -> list[dict[str, Any]]:
        """按节点顺序展开每一次尝试的时间线，供接口回放与排障。"""
        timeline: list[dict[str, Any]] = []
        for record in self.records(run_id):
            if not record.history:
                timeline.append({"node": record.node, "attempt": 0, "status": record.status})
                continue
            for item in record.history:
                timeline.append({"node": record.node, **item.to_payload()})
        return timeline

    def snapshot(self, run_id: str) -> dict[str, Any]:
        state = self._runs.get(run_id)
        if state is None:
            return {"run_id": run_id, "found": False}
        return {
            "run_id": run_id,
            "found": True,
            "workflow": state.workflow,
            "status": state.status,
            "created_at": state.created_at,
            "records": [record.to_payload() for record in self.records(run_id)],
            "replay": self.replay(run_id),
        }

    def stats(self) -> dict[str, Any]:
        nodes = [record for run_id in self.runs() for record in self.records(run_id)]
        retried = sum(1 for record in nodes if len(record.history) > 1)
        fallback = sum(1 for record in nodes if record.fallback_reason)
        return {
            "runs": len(self._runs),
            "nodes": len(nodes),
            "retried_nodes": retried,
            "fallback_nodes": fallback,
            "failed_nodes": sum(1 for record in nodes if record.status == NodeStatus.FAILED.value),
        }

    # -- 内部 -------------------------------------------------------------- #
    def _require(self, run_id: str) -> _RunState:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"unknown run: {run_id}")
        self._runs.move_to_end(run_id)
        return state

    def _evict(self) -> None:
        while len(self._runs) > self._max_runs:
            evicted, _ = self._runs.popitem(last=False)
            logger.info(
                "workflow.checkpoint.evict",
                extra={"event": "ops.checkpoint.evict", "msg": "淘汰最旧检查点", "run_id": evicted},
            )


__all__ = ["Checkpointer"]
