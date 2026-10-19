"""数据平台门面：把「跑一条流水线」与「消费上次产物」收敛成一个入口。

为什么需要这一层（而不是把 DataPipeline 直接放进容器当单例）：
- `DataPipeline` 是**有状态**的（持有审核队列、向量索引、检索器）。
  若在多次请求间复用同一个实例，上一次请求的审核状态与索引内容会渗进下一次，
  出现「脏读」——这是数据链路里最难查的一类问题；
- 所以每次 `run` 都用工厂**新装一条**流水线，跑完把产物缓存下来，下次进来即全新；
- 下游只读动作（检索/评估/血缘/导出）需要「刚跑出来的那份产物」，
  因此缓存最近一次 `PipelineResult`，没有产物时**显式报错**而不是返回空结果。

**唯一例外（刻意的跨请求状态）**：`RefluxCollector` 是回流事件的沉淀池 ——
回流本身就是「多次业务事件累积成训练/评测数据」的过程，所以它必须跨请求存活，
否则每次请求只取到一条反馈，永远攒不出偏好对。这个例外的边界很清楚：
它是**只增不改**的收集器（有界队列 + 去重门禁），不参与流水线装配，因此不会带来脏读。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..infra.core.errors import NotFoundError
from ..infra.core.logging import get_logger
from ..infra.core.settings import Settings
from .evaluation import GoldenItem
from .factory import build_pipeline
from .pipeline import PipelineResult
from .quiz import AutoGrader, QuizBank, QuizItem
from .reflux import RefluxCollector, RefluxSource
from .training import DpoBuilder, LlamafactoryArtifact, build_llamafactory

logger = get_logger("data.service")


class DataPlatformService:
    """按请求装配流水线，并保留最近一次产物供只读接口消费。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._last: PipelineResult | None = None
        self._reflux = RefluxCollector()

    async def run(
        self,
        records: Sequence[dict[str, Any]],
        *,
        reviewer: str | None = "auto_policy",
        export_format: str | None = None,
        export_batch: str | None = None,
        training_export: bool = False,
        training_dataset: str = "beauty_cs",
    ) -> PipelineResult:
        pipeline = build_pipeline(self._settings)
        result = await pipeline.run(
            records,
            reviewer=reviewer,
            export_format=export_format,
            export_batch=export_batch,
            training_export=training_export,
            training_dataset=training_dataset,
        )
        self._last = result
        return result

    def last(self) -> PipelineResult:
        """取最近一次流水线产物；未跑过则报 404，而不是给一个误导性的空结果。"""
        if self._last is None:
            raise NotFoundError("尚未运行数据流水线，请先调用 POST /v1/data/pipeline/run")
        return self._last

    # -- 回流 --------------------------------------------------------------- #
    def collect_reflux(self, source: RefluxSource | str, payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """写入回流事件，并返回「这一批」落进哪些下游数据集。"""
        accepted = self._reflux.add_many(source, payloads)
        logger.info(
            "data.service.reflux",
            extra={"event": "data_reflux", "status": "ok", "reason": f"source={source},accepted={len(accepted)}"},
        )
        return self._reflux.report()

    def reflux_report(self) -> dict[str, Any]:
        return self._reflux.report()

    def reflux_training_samples(self) -> list[Any]:
        return self._reflux.to_training()

    def reflux_golden(self) -> list[GoldenItem]:
        return self._reflux.to_golden()

    # -- 训练产物 ----------------------------------------------------------- #
    def training_artifact(self, *, dataset_name: str = "beauty_cs", version: str = "v1") -> LlamafactoryArtifact:
        """把最近一次流水线的 SFT 样本 + 审核偏好对 + 回流偏好对合成训练产物。"""
        result = self.last()
        pairs = [
            *self._reflux.to_dpo_pairs(),
            *DpoBuilder().from_reviews(result.dataset.staging),
        ]
        samples = [*result.dataset.sft_samples, *self._reflux.to_training()]
        artifact = build_llamafactory(
            samples,
            pairs,
            dataset_name=dataset_name,
            version=version,
            base_model=self._settings.data_training_base_model,
        )
        return artifact

    # -- 出题与评分 ---------------------------------------------------------- #
    def quiz_items(self) -> list[QuizItem]:
        result = self.last()
        return QuizBank().generate(result.dataset.articles, chunks=result.dataset.chunks)

    async def grade_quiz(self, answers: dict[str, str], *, top_k: int = 3) -> dict[str, Any]:
        """按题目实测检索证据，再用启发式评分器打分（证据不一致/漏要点会被点名）。"""
        result = self.last()
        items = self.quiz_items()
        evidence: dict[str, list[str]] = {}
        for item in items:
            if item.quiz_id not in answers:
                continue
            assembled = await result.retriever.retrieve(item.question, top_k=top_k, product_id=item.product_id)
            evidence[item.quiz_id] = [block.text for block in assembled.blocks]
        return AutoGrader().grade_batch(items, answers, evidence=evidence)


__all__ = ["DataPlatformService"]
