"""项目二门面：把工作流 / 流式 / 并行 / 多模态 / 直播 / 分析 / 自进化收在一个出口。

为什么要有门面（而不是让接口层直接拼各个模块）：
这些模块**互相咬合** —— 内容工作流产出的图片提示词要进资产库、直播切片也要进资产库、
运营分析的素材建议又要回流成下一轮内容生产的 brief。如果接口层自己组合，
「谁在什么时机调用谁」会散落在路由函数里，换一个入口就多一份不一致的逻辑。
门面把这条业务链固化成一个对象，接口层只负责鉴权、配额与响应包装。

状态归属（面试会追问「这些对象活多久」）：
- `checkpointer` / `asset store` / `selfevolve` 的回流收集器：**跨请求存活**，
  它们承载的正是「跨请求累积」的语义（续跑、素材库、回流攒够才出题）；
- 变长状态都有**上界**（见各模块的 max_* 参数），存活不等于无限增长。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Sequence

from src.infra.core.errors import ValidationError
from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage

from .analytics import OpsAnalytics
from .live import LiveProcessingResult, LiveProcessor
from .multimodal import AssetStore
from .parallel import ParallelReport, ParallelRunner, ParallelUnit
from .schemas import AssetModality, ContentBrief, TemplatePatch
from .selfevolve import EvolutionCycle, SelfEvolveLoop
from .streaming import OpsStreamer
from .workflow.checkpointer import Checkpointer
from .workflow.graph import NodeContext, WorkflowGraph, WorkflowRunner

logger = get_logger("ops.service")


def _sleep_blocking(seconds: float) -> dict[str, Any]:
    """并行基准用的**阻塞**工作量（真实等价物是出图 / 转码 / 写文件）。"""
    time.sleep(seconds)
    return {"slept_s": seconds}


@dataclass(slots=True)
class OpsService:
    """运营系统门面。"""

    graph: WorkflowGraph
    checkpointer: Checkpointer
    streamer: OpsStreamer
    parallel: ParallelRunner
    assets: AssetStore
    live: LiveProcessor
    analytics: OpsAnalytics
    selfevolve: SelfEvolveLoop
    ctx_template: NodeContext
    max_attempts: int = 2
    structured_retries: int = 1

    # -- 工作流 -------------------------------------------------------------- #
    @staticmethod
    def new_run_id(prefix: str = "content") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _runner_for(self, *, brief: ContentBrief, run_id: str, emitter: Any = None) -> WorkflowRunner:
        """按请求构造 Runner：run_id / brief / emitter 都是请求级参数。"""
        return WorkflowRunner(
            graph=self.graph,
            checkpointer=self.checkpointer,
            ctx_template=replace(self.ctx_template, brief=brief, run_id=run_id, emitter=emitter),
            max_attempts=self.max_attempts,
            structured_retries=self.structured_retries,
        )

    async def run_content(
        self, brief: ContentBrief, *, run_id: str | None = None, resume: bool = False
    ) -> Any:
        """跑一次内容工作流（或续跑）。"""
        rid = run_id or self.new_run_id()
        with_ = stage("ops.content.run", run_id=rid, resume=resume)
        async with with_:
            result = await self._runner_for(brief=brief, run_id=rid).run(rid, resume=resume)
        logger.info(
            "ops.content.run.done",
            extra={"event": "ops_workflow", "status": result.status, "run_id": rid, "changed_keys": len(result.records)},
        )
        return result

    def stream_content(
        self, brief: ContentBrief, *, run_id: str | None = None, resume: bool = False
    ) -> AsyncIterator[str]:
        """流式跑一次内容工作流：返回 SSE 帧的异步迭代器（供 StreamingResponse 消费）。"""
        rid = run_id or self.new_run_id()
        return self.streamer.stream(
            rid,
            resume=resume,
            make_runner=lambda emitter: self._runner_for(brief=brief, run_id=rid, emitter=emitter),
        )

    def workflow_snapshot(self, run_id: str) -> dict[str, Any]:
        return self.checkpointer.snapshot(run_id)

    def replay(self, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": self.checkpointer.status_of(run_id),
            "timeline": self.checkpointer.replay(run_id),
            "records": [record.to_payload() for record in self.checkpointer.records(run_id)],
        }

    def rollback(self, run_id: str, nodes: Sequence[str], *, reason: str = "manual") -> dict[str, Any]:
        """回滚指定节点（含其下游，避免版本混用），供下一轮 resume 重跑。"""
        if not self.checkpointer.exists(run_id):
            raise ValidationError(f"未知 run: {run_id}", detail={"run_id": run_id})
        invalidated: list[str] = []
        for node in nodes:
            if node not in {item.name for item in self.graph.nodes()}:
                raise ValidationError(f"未知节点: {node}", detail={"node": node})
            invalidated.extend(self.checkpointer.rollback(run_id, self.graph.descendants(node) | {node}, reason=reason))
        return {"run_id": run_id, "invalidated": sorted(set(invalidated)), "reason": reason}

    # -- 并行 ---------------------------------------------------------------- #
    async def benchmark_parallel(
        self, *, count: int = 4, sleep_s: float = 0.05, fail_every: int = 0
    ) -> ParallelReport:
        """并行基准：同批阻塞单元并发执行，报告里给出**实测**墙钟与串行下界。

        `fail_every>0` 时让第 N 个单元失败，用于验证「一个失败不取消其它单元」。
        """
        if count < 1:
            raise ValidationError("count 必须 >= 1")
        units = []
        for index in range(count):
            if fail_every and (index + 1) % fail_every == 0:
                units.append(ParallelUnit(name=f"unit{index}", fn=_boom, args=(), mode="thread"))
            else:
                units.append(ParallelUnit(name=f"unit{index}", fn=_sleep_blocking, args=(sleep_s,), mode="thread"))
        return await self.parallel.run(units)

    # -- 素材 ---------------------------------------------------------------- #
    async def import_asset(
        self,
        *,
        source_table: str,
        source_id: str,
        modality: AssetModality | str,
        title: str = "",
        product_id: str | None = None,
        market: str | None = None,
        tags: Sequence[str] = (),
        source: dict[str, Any] | None = None,
        retrieval_text: str | None = None,
    ) -> dict[str, Any]:
        asset, created = await self.assets.upsert(
            source_table=source_table,
            source_id=source_id,
            modality=modality,
            title=title,
            product_id=product_id,
            market=market,
            tags=tags,
            source=source,
            retrieval_text=retrieval_text,
        )
        return {"asset": asset.to_payload(), "created": created, "assets": self.assets.size}

    async def search_assets(
        self, query: str, *, top_k: int = 3, product_id: str | None = None, market: str | None = None
    ) -> dict[str, Any]:
        hits = await self.assets.search(query, top_k=top_k, product_id=product_id, market=market)
        return {"hits": [hit.to_payload() for hit in hits], "assets": self.assets.size}

    # -- 直播 ---------------------------------------------------------------- #
    async def process_live(
        self, source: dict[str, Any], *, duration_s: float, source_path: str = "live.mp4"
    ) -> LiveProcessingResult:
        if duration_s <= 0:
            raise ValidationError("直播时长必须为正数", detail={"duration_s": duration_s})
        return await self.live.process(source, duration_s=duration_s, source_path=source_path)

    # -- 分析 ---------------------------------------------------------------- #
    def analytics_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.analytics.report(payload)

    # -- 自进化 -------------------------------------------------------------- #
    def selfevolve_cycle(
        self,
        events: Sequence[dict[str, Any]],
        *,
        answers: dict[str, str] | None = None,
        evidence: dict[str, Sequence[str]] | None = None,
        default_source: str = "review",
    ) -> EvolutionCycle:
        if not events:
            raise ValidationError("自进化循环需要至少一条回流事件")
        return self.selfevolve.cycle(events, answers=answers, evidence=evidence, default_source=default_source)

    def selfevolve_report(self) -> dict[str, Any]:
        return self.selfevolve.report()

    def selfevolve_apply(self) -> dict[str, Any]:
        """人工审核后把最新一版模板激活（本方法不会被 cycle 自动触发）。"""
        report = self.selfevolve.report()
        cycles = report.get("cycles") or []
        if not cycles or not cycles[-1].get("template_patch"):
            raise ValidationError("没有可应用的模板迭代建议")
        patch = cycles[-1]["template_patch"]
        return self.selfevolve.apply_patch(TemplatePatch(**patch))

    def selfevolve_revert(self) -> dict[str, Any]:
        return self.selfevolve.revert()

    # -- 统计 ---------------------------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        return {
            "workflow": self.checkpointer.stats(),
            "streaming": self.streamer.stats(),
            "assets": self.assets.stats(),
            "parallel": {"max_concurrency": self.parallel.max_concurrency},
        }


def _boom() -> dict[str, Any]:
    raise RuntimeError("benchmark: intentional failure")


__all__ = ["OpsService"]
