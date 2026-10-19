"""流水线编排：把治理链路串起来，并用 stage 计时 + 指标让每一步都可观测。

顺序即语义（不能随意调换）：
    接入标准化 → 去重 → 脱敏 → 过滤 → 切片/成 staging → 审核 → 用途分流
        → 向量索引 → （评估）→ 训练导出 → 血缘

为什么必须有这一层：如果让各阶段各自「顺便」调用下一个阶段，
就会出现「脱敏还没跑就被索引了」这类顺序错误。编排集中在一处，
顺序可审计、可测试，也方便逐阶段替换实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..infra.core.logging import get_logger
from ..infra.core.metrics import metrics
from ..infra.core.tracing import stage
from .chunking import Chunker
from .dedupe import Deduper, group_by_session
from .export import ExportArtifact, Processor
from .filters import FilterChain, FilterReport
from .index import VectorIndex
from .ingest import Ingestor
from .lineage import LineageGraph, build_lineage
from .retrieval import Retriever
from .review import ReviewQueue
from .rerank import RerankPort
from .routing import Router
from .schemas import (
    Dataset,
    ReviewStatus,
    SamplePurpose,
    SourceKind,
    StagingConversation,
    StandardMessage,
)
from .scrub import Scrubber, ScrubResult
from .training import DpoBuilder, DpoPair, LlamafactoryArtifact, build_llamafactory

logger = get_logger("data.pipeline")


@dataclass(slots=True)
class PipelineResult:
    dataset: Dataset
    index: VectorIndex
    retriever: Retriever
    lineage: LineageGraph
    report: dict[str, Any] = field(default_factory=dict)
    export: ExportArtifact | None = None
    dpo_pairs: list[DpoPair] = field(default_factory=list)
    training: LlamafactoryArtifact | None = None


class DataPipeline:
    """数据治理与资产化主流程。各阶段组件可注入，便于单测与替换。"""

    def __init__(
        self,
        *,
        ingestor: Ingestor | None = None,
        deduper: Deduper | None = None,
        scrubber: Scrubber | None = None,
        filter_chain: FilterChain | None = None,
        chunker: Chunker | None = None,
        index: VectorIndex | None = None,
        verifier: Scrubber | None = None,
        reranker: RerankPort | None = None,
        merge_ratio: float = 0.5,
        dpo_builder: DpoBuilder | None = None,
    ) -> None:
        self.ingestor = ingestor or Ingestor()
        self.deduper = deduper or Deduper()
        self.scrubber = scrubber or Scrubber()
        # 导出门禁复用同一套规则，但可以是独立实例（两次扫描互不影响）
        self.verifier = verifier or scrubber or Scrubber()
        self.filter_chain = filter_chain or FilterChain()
        self.chunker = chunker or Chunker()
        self.review = ReviewQueue()
        self.router = Router(chunker=self.chunker)
        self.index = index or VectorIndex()
        self.reranker = reranker
        self.merge_ratio = merge_ratio
        self.dpo_builder = dpo_builder or DpoBuilder()

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
        dataset = Dataset(raw=[])

        # 1) 接入标准化
        with stage("data.pipeline.ingest"):
            raws, messages = self.ingestor.parse(records)
        dataset.raw = raws

        # 2) 精确去重（消息级）
        with stage("data.pipeline.dedupe"):
            dedupe_report = self.deduper.exact(messages)

        # 3) 脱敏（入库前第一次；导出前还有第二次）
        scrub_hits: dict[str, int] = {}
        scrubbed: list[StandardMessage] = []
        with stage("data.pipeline.scrub"):
            for message in dedupe_report.kept:
                result: ScrubResult = self.scrubber.scrub(message.content)
                for name, count in result.hits.items():
                    scrub_hits[name] = scrub_hits.get(name, 0) + count
                scrubbed.append(
                    StandardMessage(
                        source=message.source,
                        source_id=message.source_id,
                        session_id=message.session_id,
                        speaker=message.speaker,
                        raw_speaker=message.raw_speaker,
                        ts=message.ts,
                        content=result.text,
                        parse_status=message.parse_status,
                        meta=message.meta,
                    )
                )
        dataset.messages = scrubbed

        # 4) 过滤（规则链 + 原因）
        with stage("data.pipeline.filter"):
            filter_report: FilterReport = self.filter_chain.run(scrubbed)

        # 5) 成 staging（可信数据的唯一入口）
        staging_items = self._build_staging(filter_report)
        # 6) 近重复在**语义单元级**判定：删「整轮」，绝不删半轮（消息级近重复会拆散问答对）
        with stage("data.pipeline.near_dedupe", units=len(staging_items)):
            kept_ids, near_pairs = self.deduper.near([(item.staging_id, item.text) for item in staging_items])
        kept_set = set(kept_ids)
        staged = [item for item in staging_items if item.staging_id in kept_set]
        self.review.add_many(staged)
        if reviewer:
            # 自动放行「格式与内容都正常」的条目；被标记可疑的**留在待审核**
            for item in staged:
                if item.status is ReviewStatus.PENDING and not any(
                    message.parse_status.value == "partial" for message in item.messages
                ):
                    self.review.approve(item.staging_id, reviewer=reviewer, note="规则初筛通过")

        # 7) 用途分流（只吃已审核通过的数据）
        with stage("data.pipeline.routing"):
            articles, chunks, samples, routing_stats = self.router.route(self.review.approved())
        dataset.staging = self.review.all()
        dataset.articles = articles
        dataset.chunks = chunks
        dataset.sft_samples = samples

        # 7) 向量索引
        with stage("data.pipeline.index"):
            await self.index.upsert(chunks)

        # 8) 训练导出（可选）
        export_artifact: ExportArtifact | None = None
        if export_format:
            with stage("data.pipeline.export"):
                processor = Processor(verifier=self.verifier, version="v1")
                export_artifact = processor.export(samples, fmt=export_format, purpose=SamplePurpose.SFT_STYLE)

        # 9) 偏好对与训练数据集（LLaMA-Factory 可直接消费）
        with stage("data.pipeline.training"):
            dpo_pairs = self.dpo_builder.from_reviews(dataset.staging)
            training: LlamafactoryArtifact | None = None
            if training_export:
                training = build_llamafactory(
                    samples,
                    dpo_pairs,
                    dataset_name=training_dataset,
                    version="v1",
                    verifier=self.verifier,
                )

        # 10) 血缘
        with stage("data.pipeline.lineage"):
            lineage = build_lineage(dataset, export_batch=export_batch)

        report = {
            "ingest": {"raw": len(raws), "messages": len(messages)},
            "dedupe": {
                "kept": len(dedupe_report.kept),
                "exact_duplicates": dedupe_report.exact_duplicates,
                "near_duplicates": len(near_pairs),
                "near_duplicate_pairs": near_pairs,
            },
            "scrub": {"hits": scrub_hits, "total": sum(scrub_hits.values())},
            "filter": {
                "kept": len(filter_report.kept),
                "rejected": len(filter_report.rejected),
                "review_flagged": len(filter_report.review_flagged),
                "rule_hits": filter_report.rule_hits,
            },
            "review": self.review.stats(),
            "routing": routing_stats,
            "chunking": self.chunker.report(),
            "training": training.stats if training else {"dpo_pairs": len(dpo_pairs)},
            "index": {"size": self.index.size, **self.index.consistency()},
            "lineage": lineage.stats(),
        }
        dataset.stats = report
        metrics.incr("data_pipeline_run", "ok")
        logger.info(
            "data.pipeline.done",
            extra={"event": "data_pipeline", "status": "ok", "reason": f"chunks={len(chunks)},articles={len(articles)}"},
        )

        return PipelineResult(
            dataset=dataset,
            index=self.index,
            retriever=Retriever(self.index, reranker=self.reranker, merge_ratio=self.merge_ratio),
            lineage=lineage,
            report=report,
            export=export_artifact,
            dpo_pairs=dpo_pairs,
            training=training,
        )

    # ------------------------------------------------------------------ #
    def _build_staging(self, report: FilterReport) -> list[StagingConversation]:
        """把过滤结果组织成待审核单元：会话内按语义轮次成块，知识条目一条一块。"""
        staging: list[StagingConversation] = []
        counter = 0

        messages = list(report.kept)
        knowledge = [message for message in messages if message.session_id.startswith("kb::")]
        conversation = [message for message in messages if not message.session_id.startswith("kb::")]

        for message in knowledge:
            staging.append(
                StagingConversation(
                    staging_id=f"stg_{message.source_id}",
                    source=message.source,
                    session_id=message.session_id,
                    messages=[message],
                    product_id=message.meta.get("product_id"),
                    scene=message.meta.get("scene"),
                )
            )

        for session_id, group in group_by_session(conversation).items():
            for turn in self.chunker.split_turns(group):
                counter += 1
                staging.append(
                    StagingConversation(
                        staging_id=f"stg_{session_id}_{counter:02d}",
                        source=turn.messages[0].source,
                        session_id=session_id,
                        messages=list(turn.messages),
                        product_id=next((m.meta.get("product_ref") for m in turn.messages if m.meta.get("product_ref")), None),
                        scene=self._infer_scene(turn.messages[0]),
                    )
                )

        # 被标记可疑的条目也进队列，但保持 pending（不进资产）
        for message, decision in report.review_flagged:
            staging.append(
                StagingConversation(
                    staging_id=f"stg_{message.source_id}",
                    source=message.source,
                    session_id=message.session_id,
                    messages=[message],
                    product_id=message.meta.get("product_id"),
                    scene=message.meta.get("scene"),
                    reason=decision.reason,
                )
            )
        return staging

    @staticmethod
    def _infer_scene(message: StandardMessage) -> str | None:
        if message.source is SourceKind.LIVE_ASR:
            return "live_script"
        return "customer_chat"
