"""运营内容工作流：DAG 图 + 节点 + 执行器。

对标简历里的「Microsoft Agent Framework WorkflowBuilder / LangGraph」，这里是**语义等价的原生实现**：
- **节点（Node）**：一次可重试的工作单元，声明自己的入参来自哪些上游节点（`depends_on`）；
- **边（Edge）**：`depends_on` 即边，显式声明，不做「隐式按顺序执行」；
- **图（Graph）**：定义期做**未知依赖 + 环**校验（把配置错误挡在跑之前）；
- **执行器（Runner）**：按拓扑层执行 —— 同一层内的节点**互相不依赖**，所以可以并行；
  层与层之间天然串行，这就是「只并行依赖关系正确的部分」的实现方式。

为什么坚持显式依赖而不是「按注册顺序跑」：
注册顺序是**实现细节**，依赖关系是**业务事实**。用前者驱动执行，一次无意的重排就能
把「先出卖点、再写文案」变成「先写文案、再出卖点」，而且没有任何地方会报错。

诚实边界：图与节点都是进程内对象（生产可把图定义落到表里），Checkpointer 亦为进程内有界实现。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Sequence

from src.infra.core.errors import InfraError, ValidationError
from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage

from .checkpointer import Checkpointer
from .schemas import NodeAttempt, NodeRecord, NodeStatus, WorkflowResult

logger = get_logger("ops.workflow.graph")


class WorkflowDefinitionError(ValidationError):
    """图定义本身非法（未知依赖 / 成环 / 重名）。"""

    code = "workflow_definition_invalid"


class NodeFailedError(InfraError):
    """节点重试耗尽后仍失败。run 在此停下，但**已完成节点不回滚**，供下一次续跑复用。"""

    code = "workflow_node_failed"
    http_status = 500
    safe_message = "工作流节点执行失败"


@dataclass(slots=True)
class NodeContext:
    """节点执行上下文：节点需要的公共依赖都从这儿拿，节点签名里就不必再传一堆参数。

    `child()` 产出的是**本次尝试的副本**（节点名 / 尝试次数 / 生效逻辑模型），
    这样并行执行的同层节点各自持有一份上下文，不会互相覆盖 `attempt` 之类的可变状态。
    """

    gateway: Any
    knowledge: Any
    prompts: Any
    brief: Any
    run_id: str
    parallel: Any = None
    settings: Any = None
    emitter: Any = None
    """事件发射器（SSE 模式注入）。为 None 时 `emit()` 是空操作 ——
    同一套节点代码既能在同步模式一次性返回，也能在流式模式逐 token 推送。"""
    node: str = ""
    attempt: int = 1
    logical_model: str | None = None

    def child(self, *, node: str, attempt: int, logical_model: str | None) -> "NodeContext":
        return replace(self, node=node, attempt=attempt, logical_model=logical_model)

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """发射一个事件。事件一定带上节点名，前端才能把 token 归到正确的节点上。"""
        if self.emitter is None:
            return
        payload = {"event": event, "node": self.node, "run_id": self.run_id, **(data or {})}
        await self.emitter(payload)


# 节点处理器：入参 = 上游节点输出（按 depends_on 名字取值），出参 = 结构化对象或 dict
NodeHandler = Callable[[dict[str, Any], NodeContext], Awaitable[Any]]


@dataclass(slots=True)
class Node:
    name: str
    handler: NodeHandler
    depends_on: tuple[str, ...] = ()
    kind: str = "transform"
    logical_model: str | None = None
    max_attempts: int | None = None
    optional: bool = False
    llm_report: bool = False
    """True 表示该节点用了模型，需要把 usage 归一进 run 的 artifacts 供成本归集。"""


class WorkflowGraph:
    """运营内容工作流的有向无环图。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._nodes: dict[str, Node] = {}

    # -- 定义期 -------------------------------------------------------------- #
    def add(self, node: Node) -> Node:
        if node.name in self._nodes:
            raise WorkflowDefinitionError(f"节点重名: {node.name}", detail={"node": node.name})
        self._nodes[node.name] = node
        return node

    def get(self, name: str) -> Node:
        try:
            return self._nodes[name]
        except KeyError as exc:
            raise WorkflowDefinitionError(f"未知节点: {name}", detail={"node": name}) from exc

    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def validate(self) -> None:
        """未知依赖与成环都必须在**跑之前**被挡住，而不是跑到一半才发现。"""
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise WorkflowDefinitionError(
                        f"节点 {node.name} 依赖不存在的节点 {dep}",
                        detail={"node": node.name, "missing_dependency": dep},
                    )
                if dep == node.name:
                    raise WorkflowDefinitionError(f"节点 {node.name} 依赖自身", detail={"node": node.name})
        self.layers()  # 成环时在下面抛错

    def layers(self) -> list[list[Node]]:
        """拓扑分层：同层节点互不依赖 → 可并行；层间必须串行。

        Kahn 算法变体：每轮取出所有前驱已就绪的节点作为一层；
        若某轮取不出任何节点但仍有剩余 → 说明存在环。
        """
        remaining = {node.name: set(node.depends_on) for node in self._nodes.values()}
        # 保留定义顺序，保证同一图的执行顺序稳定可复现
        order = list(self._nodes.keys())
        layers: list[list[Node]] = []
        resolved: set[str] = set()
        while remaining:
            ready = [name for name in order if name in remaining and not (remaining[name] - resolved)]
            if not ready:
                raise WorkflowDefinitionError(
                    "工作流存在环依赖", detail={"pending": sorted(remaining), "reason": "cycle"}
                )
            layers.append([self._nodes[name] for name in ready])
            for name in ready:
                resolved.add(name)
                remaining.pop(name)
        return layers

    def descendants(self, name: str) -> set[str]:
        """回滚要连带失效的下游集合（只失效单节点会造成版本混用）。"""
        self.get(name)
        result: set[str] = set()
        frontier = [name]
        while frontier:
            current = frontier.pop()
            for node in self._nodes.values():
                if current in node.depends_on and node.name not in result:
                    result.add(node.name)
                    frontier.append(node.name)
        return result

    def upstream(self, name: str) -> set[str]:
        self.get(name)
        result: set[str] = set()
        frontier = list(self._nodes[name].depends_on)
        while frontier:
            current = frontier.pop()
            if current in result:
                continue
            result.add(current)
            frontier.extend(self._nodes[current].depends_on)
        return result


@dataclass(slots=True)
class WorkflowRunner:
    """图执行器：拓扑层并行 + 节点级重试 + 检查点驱动的续跑。"""

    graph: WorkflowGraph
    checkpointer: Checkpointer
    ctx_template: NodeContext
    max_attempts: int = 2
    structured_retries: int = 1

    async def run(self, run_id: str, *, resume: bool = False) -> WorkflowResult:
        self.graph.validate()
        self.checkpointer.begin(run_id, self.graph.name)
        run_ctx = self.ctx_template.child(node="run", attempt=1, logical_model=None)
        await run_ctx.emit("run.start", {"workflow": self.graph.name, "resume": resume})

        started = time.perf_counter()
        completed = self.checkpointer.completed(run_id) if resume else {}
        state: dict[str, Any] = dict(completed)
        resumed_from = sorted(completed)
        artifacts: dict[str, Any] = {}
        failures: list[str] = []

        for layer in self.graph.layers():
            pending = [node for node in layer if node.name not in completed]
            if not pending:
                continue
            inputs_map = {
                node.name: {dep: state.get(dep) for dep in node.depends_on} for node in pending
            }
            outcomes = await asyncio.gather(
                *(self._invoke(node, inputs_map[node.name], run_id) for node in pending),
                return_exceptions=True,
            )
            for node, outcome in zip(pending, outcomes):
                if isinstance(outcome, BaseException):
                    if node.optional:
                        self._mark_skipped(run_id, node, outcome)
                        logger.warning(
                            "workflow.node.optional_failed",
                            extra={
                                "event": "ops_node",
                                "status": "skipped",
                                "node": node.name,
                                "run_id": run_id,
                                "reason": type(outcome).__name__,
                            },
                        )
                        continue
                    failures.append(node.name)
                else:
                    state[node.name] = outcome

        duration_ms = (time.perf_counter() - started) * 1000
        records = self.checkpointer.records(run_id)
        for record in records:
            if record.output is not None:
                artifacts[record.node] = record.output
        artifacts["models"] = sorted({r.logical_model for r in records if r.logical_model})
        artifacts["fallbacks"] = [
            {"node": r.node, "reason": r.fallback_reason} for r in records if r.fallback_reason
        ]
        artifacts["usage"] = self._collect_usage(run_id)
        status = "failed" if failures else "succeeded"
        self.checkpointer.set_status(run_id, status)
        await run_ctx.emit(
            "run.done",
            {
                "status": status,
                "workflow": self.graph.name,
                "resolved": [record.node for record in records if record.output is not None],
                "failed": failures,
                "duration_ms": round(duration_ms, 2),
            },
        )
        if failures:
            logger.warning(
                "workflow.run.failed",
                extra={
                    "event": "ops_workflow",
                    "status": "failed",
                    "run_id": run_id,
                    "node": failures[0],
                    "reason": "node_failed",
                    "changed_keys": len(records),
                },
            )
        else:
            logger.info(
                "workflow.run.done",
                extra={
                    "event": "ops_workflow",
                    "status": "succeeded",
                    "run_id": run_id,
                    "changed_keys": len(records),
                    "duration_ms": round(duration_ms, 2),
                },
            )
        return WorkflowResult(
            run_id=run_id,
            workflow=self.graph.name,
            status=status,
            records=records,
            artifacts=artifacts,
            resumed_from=resumed_from,
            duration_ms=duration_ms,
        )

    # -- 单节点执行 ---------------------------------------------------------- #
    async def _invoke(self, node: Node, inputs: dict[str, Any], run_id: str) -> Any:
        limit = node.max_attempts or self.max_attempts
        # 若该节点此前跑过（续跑/回滚重跑），把历史尝试**接续**下来 ——
        # 否则一次重跑就会把「上一轮为什么被回滚、第几次失败」整段抹掉，
        # 而回放的价值恰恰在于能看到这段过程。
        previous = self.checkpointer.get(run_id, node.name)
        record = NodeRecord(
            node=node.name,
            status=NodeStatus.RUNNING.value,
            logical_model=node.logical_model,
            history=list(previous.history) if previous else [],
        )
        record.started_at = time.time()
        self.checkpointer.put(run_id, record)
        node_ctx = self.ctx_template.child(node=node.name, attempt=1, logical_model=node.logical_model)
        await node_ctx.emit(
            "node.start",
            {
                "kind": node.kind,
                "logical_model": node.logical_model,
                "depends_on": list(node.depends_on),
                "optional": node.optional,
            },
        )

        last_error: BaseException | None = None
        for attempt in range(1, max(1, limit) + 1):
            ctx = self.ctx_template.child(node=node.name, attempt=attempt, logical_model=node.logical_model)
            attempt_started = time.perf_counter()
            try:
                async with stage("ops.node", node=node.name, attempt=attempt):
                    output = await node.handler(inputs, ctx)
            except BaseException as exc:  # noqa: BLE001 - 需要把任何失败都留痕后再重试
                elapsed = (time.perf_counter() - attempt_started) * 1000
                last_error = exc
                record.attempt = attempt
                record.error = self._describe(exc)
                record.history.append(
                    NodeAttempt(
                        attempt=attempt,
                        status="failed",
                        logical_model=node.logical_model,
                        error=self._describe(exc),
                        duration_ms=elapsed,
                    )
                )
                await node_ctx.emit(
                    "node.attempt_failed",
                    {"attempt": attempt, "reason": self._describe(exc), "duration_ms": round(elapsed, 2)},
                )
                logger.warning(
                    "workflow.node.attempt_failed",
                    extra={
                        "event": "ops_node_attempt",
                        "status": "failed",
                        "node": node.name,
                        "run_id": run_id,
                        "attempt": attempt,
                        "duration_ms": round(elapsed, 2),
                        "reason": type(exc).__name__,
                    },
                )
                if attempt >= max(1, limit) or not self._retryable(exc):
                    record.status = NodeStatus.FAILED.value
                    record.finished_at = time.time()
                    self.checkpointer.put(run_id, record)
                    await node_ctx.emit(
                        "node.failed",
                        {"attempt": attempt, "reason": self._describe(exc)},
                    )
                    raise NodeFailedError(
                        f"节点 {node.name} 执行失败",
                        detail={"node": node.name, "attempts": attempt, "cause": self._describe(exc)},
                    ) from exc
                continue

            elapsed = (time.perf_counter() - attempt_started) * 1000
            payload, fallback_reason, model = self._normalize_output(node, output)
            record.attempt = attempt
            record.output = payload
            record.logical_model = model
            record.fallback_reason = fallback_reason
            record.error = None
            record.status = NodeStatus.SUCCEEDED.value
            record.finished_at = time.time()
            record.history.append(
                NodeAttempt(
                    attempt=attempt,
                    status="succeeded",
                    logical_model=model,
                    fallback_reason=fallback_reason,
                    duration_ms=elapsed,
                )
            )
            self.checkpointer.put(run_id, record)
            logger.info(
                "workflow.node.done",
                extra={
                    "event": "ops_node",
                    "status": "ok",
                    "node": node.name,
                    "run_id": run_id,
                    "attempt": attempt,
                    "logical_model": model or "-",
                    "duration_ms": round(elapsed, 2),
                    "degraded": bool(fallback_reason),
                },
            )
            await node_ctx.emit(
                "node.done",
                {
                    "attempt": attempt,
                    "duration_ms": round(elapsed, 2),
                    "logical_model": model,
                    "fallback_reason": fallback_reason,
                    "output": payload,
                },
            )
            return payload

        raise NodeFailedError(f"节点 {node.name} 执行失败", detail={"node": node.name, "cause": str(last_error)})

    # -- 辅助 --------------------------------------------------------------- #
    def _mark_skipped(self, run_id: str, node: Node, exc: BaseException) -> None:
        record = self.checkpointer.get(run_id, node.name) or NodeRecord(node=node.name)
        record.status = NodeStatus.SKIPPED.value
        record.error = self._describe(exc)
        record.finished_at = record.finished_at or time.time()
        record.history.append(NodeAttempt(attempt=record.attempt or 1, status="skipped", error=self._describe(exc)))
        self.checkpointer.put(run_id, record)

    def _collect_usage(self, run_id: str) -> dict[str, Any]:
        """把模型用量/成本从节点输出里归一出来（节点自己记的 usage 字段）。"""
        prompt = completion = 0
        cost = 0.0
        per_node: dict[str, Any] = {}
        for record in self.checkpointer.records(run_id):
            usage = (record.output or {}).get("usage") if record.output else None
            if not isinstance(usage, dict):
                continue
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
            cost += float(usage.get("cost_usd") or 0.0)
            per_node[record.node] = usage
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cost_usd": round(cost, 6),
            "per_node": per_node,
        }

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        """错误归一化的价值就体现在这里：知道哪些能重试，而不是一律重试或一律不重试。"""
        if isinstance(exc, InfraError):
            return exc.retryable
        return True

    @staticmethod
    def _describe(exc: BaseException) -> str:
        if isinstance(exc, InfraError):
            return f"{exc.code}: {exc.message}"
        return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _normalize_output(node: Node, output: Any) -> tuple[dict[str, Any], str | None, str | None]:
        """节点输出统一成 dict 落库；结构化对象的 `_fallback_reason` 会提升为记录字段。"""
        fallback_reason: str | None = None
        model: str | None = node.logical_model
        if hasattr(output, "model_dump"):
            payload = output.model_dump()
        elif isinstance(output, dict):
            payload = dict(output)
        else:
            payload = {"value": output}
        fallback_reason = payload.pop("_fallback_reason", None) or fallback_reason
        model = payload.pop("_logical_model", None) or model
        return payload, fallback_reason, model


__all__ = ["Node", "NodeContext", "NodeFailedError", "WorkflowDefinitionError", "WorkflowGraph", "WorkflowRunner"]
