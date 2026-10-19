"""数据平台 REST 接口（项目三）。

与 bridge/router.py 共用鉴权与信封，但暴露的是**数据治理链路**：
跑流水线、检索证据、评估检索质量、导出训练集、追溯血缘。

两条设计约束（面试可直接讲）：
1) 这些是「数据面」操作，需要独立的 `data:rw` scope —— 只会话凭证（llm:chat）
   不应该能触发训练集导出这类高影响动作（最小权限）；
2) `pipeline/run` 是唯一写接口，其余都只读地消费「最近一次 run 的产物」；
   没有产物时直接 404，而不是静默返回空结果（空结果会被误读成「确实没有知识」）。
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends

from src.dataplatform.evaluation import Evaluator, GoldenItem, grouped_split, leak_check
from src.dataplatform.export import Processor
from src.dataplatform.fixtures import GOLDEN_ITEMS, RAW_RECORDS
from src.dataplatform.quiz import Expect, QuizType
from src.dataplatform.schemas import SamplePurpose

from ..container import Container
from ..core import context as ctx
from ..core.tracing import snapshot_timeline
from .router import current_auth, get_container
from .schemas import (
    DataEvaluateRequest,
    DataExportRequest,
    DataPipelineRunRequest,
    DataQuizRequest,
    DataRefluxRequest,
    DataSearchRequest,
    DataTrainingExportRequest,
)

router = APIRouter(prefix="/v1/data", tags=["data"])


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


@router.post("/pipeline/run")
async def run_pipeline(
    body: DataPipelineRunRequest,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """跑完整数据流水线，返回各阶段统计与产出的资产清单（不返回全量正文，响应保持可读）。"""
    auth.require_scope("data:rw")
    started = time.perf_counter()
    records = body.records if body.records is not None else RAW_RECORDS

    with ctx.context_scope(project_id=auth.project_id):
        result = await container.dataplatform.run(
            records,
            reviewer=body.reviewer,
            export_format=body.export_format,
            export_batch=body.export_batch,
            training_export=body.training_export,
            training_dataset=body.training_dataset,
        )

    dataset = result.dataset
    return {
        "success": True,
        "report": result.report,
        "articles": [
            {
                "article_id": article.article_id,
                "title": article.title,
                "scene": article.scene,
                "product_id": article.product_id,
                "price_freshness": article.price_freshness,
                "live_source": article.live_source,
                "version": article.version,
            }
            for article in dataset.articles
        ],
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "article_id": chunk.article_id,
                "text": chunk.text,
                "product_id": chunk.product_id,
                "scene": chunk.scene,
                "level": chunk.level,
                "parent_id": chunk.parent_id,
            }
            for chunk in dataset.chunks
        ],
        "sft_samples": [
            {"sample_id": sample.sample_id, "messages": sample.messages, "scene": sample.scene}
            for sample in dataset.sft_samples
        ],
        "export": (
            {
                "format": result.export.format,
                "stats": result.export.stats,
                "rejected": [
                    {"index": issue.index, "reason": issue.reason, "detail": issue.detail}
                    for issue in result.export.rejected
                ],
            }
            if result.export
            else None
        ),
        "trace_id": ctx.require().trace_id,
        "duration_ms": _elapsed(started),
        "stages": snapshot_timeline(),
    }


@router.post("/search")
async def search_knowledge(
    body: DataSearchRequest,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """检索并组装上下文：元数据预过滤 → 混合召回 → 重排 → 预算内组装。"""
    auth.require_scope("data:rw")
    started = time.perf_counter()
    result = container.dataplatform.last()

    with ctx.context_scope(project_id=auth.project_id):
        report = await result.retriever.search(
            body.query,
            top_k=body.top_k or container.settings.data_recall_top_k,
            product_id=body.product_id,
            scene=body.scene,
        )
        context = result.retriever.assemble(report.evidence)

    return {
        "success": True,
        "blocks": [
            {
                "chunk_id": block.chunk_id,
                "article_id": block.article_id,
                "text": block.text,
                "source": block.source,
                "score": block.score,
            }
            for block in context.blocks
        ],
        "render": context.render(),
        "total_chars": context.total_chars,
        "dropped_duplicates": context.dropped_duplicates,
        "truncated": context.truncated,
        "rerank": {
            "backend": report.rerank_backend,
            "candidates": report.candidates,
            "merged_children": report.merged_children,
            "reasons": report.reasons,
        },
        "trace_id": ctx.require().trace_id,
        "duration_ms": _elapsed(started),
    }


@router.post("/evaluate")
async def evaluate_retrieval(
    body: DataEvaluateRequest,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """在标准证据标注集上算 Recall@K / Hit Rate@K / MRR，并做分组切分与泄漏自检。"""
    auth.require_scope("data:rw")
    started = time.perf_counter()
    result = container.dataplatform.last()
    items = [GoldenItem(**item) for item in (body.items if body.items is not None else GOLDEN_ITEMS)]

    with ctx.context_scope(project_id=auth.project_id):
        metrics = await Evaluator(result.retriever).evaluate(items, k=body.k)
    splits = grouped_split(items)

    return {
        "success": True,
        "metrics": metrics.as_dict(),
        "per_query": metrics.per_query,
        "splits": {name: len(entries) for name, entries in splits.items()},
        "leaks": leak_check(splits),
        "trace_id": ctx.require().trace_id,
        "duration_ms": _elapsed(started),
    }


@router.post("/export")
async def export_training_set(
    body: DataExportRequest,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """把最近一次产出的 SFT 样本按目标框架导出（校验门禁 + 版本 + 统计）。"""
    auth.require_scope("data:rw")
    started = time.perf_counter()
    result = container.dataplatform.last()
    purpose = SamplePurpose(body.purpose) if body.purpose else None

    with ctx.context_scope(project_id=auth.project_id):
        artifact = Processor(version=container.settings.data_export_version).export(
            result.dataset.sft_samples, fmt=body.format, purpose=purpose
        )

    return {
        "success": True,
        "format": artifact.format,
        "version": artifact.version,
        "stats": artifact.stats,
        "records": artifact.records,
        "rejected": [
            {"index": issue.index, "reason": issue.reason, "detail": issue.detail} for issue in artifact.rejected
        ],
        "trace_id": ctx.require().trace_id,
        "duration_ms": _elapsed(started),
    }


@router.get("/lineage/{node_id:path}")
async def lineage_trace(
    node_id: str,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """追溯一个血缘节点：它从哪来（ancestors）、影响到谁（descendants）。

    node_id 形如 `chunk::KB-1001#c0`；URL 中的 `#` 需转义为 %23。
    """
    auth.require_scope("data:rw")
    result = container.dataplatform.last()
    return {
        "success": True,
        "trace": result.lineage.trace(node_id),
        "stats": result.lineage.stats(),
        "trace_id": ctx.require().trace_id,
    }


@router.post("/training/export")
async def export_training_artifact(
    body: DataTrainingExportRequest,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """生成 LLaMA-Factory 可直接消费的训练产物：SFT 数据集 + DPO 偏好数据集 + 训练 YAML。

    **诚实边界**：本仓库只做数据准备与验证链路（含空/过短/重复/敏感门禁），
    真正的训练在外部环境执行 —— 这里不安装 LLaMA-Factory、不跑 GPU。
    """
    auth.require_scope("data:rw")
    started = time.perf_counter()
    artifact = container.dataplatform.training_artifact(dataset_name=body.dataset_name, version=body.version)

    with ctx.context_scope(project_id=auth.project_id):
        files = artifact.files()

    return {
        "success": True,
        "dataset_name": artifact.dataset_name,
        "version": artifact.version,
        "dataset_info": artifact.dataset_info,
        "train_config_yaml": artifact.train_config_yaml,
        "stats": artifact.stats,
        "files": [{"name": name, "lines": text.count("\n") + 1 if text else 0} for name, text in files.items()],
        "preview": {
            "sft": artifact.sft_records[: body.preview],
            "dpo": artifact.dpo_records[: body.preview],
        },
        "rejected": artifact.rejected,
        "trace_id": ctx.require().trace_id,
        "duration_ms": _elapsed(started),
    }


@router.post("/reflux")
async def collect_reflux(
    body: DataRefluxRequest,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """写入回流事件（客服纠偏 / 人工审核 / Tool 调用 / HITL / 直播 ASR 五来源）。

    回流是**跨请求累积**的：单条反馈本身没有训练价值，攒成偏好对与金标才有 ——
    所以这里刻意写进长期存活的收集器，而不是随流水线一起重建。
    """
    auth.require_scope("data:rw")
    started = time.perf_counter()
    with ctx.context_scope(project_id=auth.project_id):
        report = container.dataplatform.collect_reflux(body.source, body.events)

    return {
        "success": True,
        "report": report,
        "dpo_pairs": report["dpo_pairs"],
        "trace_id": ctx.require().trace_id,
        "duration_ms": _elapsed(started),
    }


@router.get("/reflux")
async def reflux_report(
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """查看回流收集器的整体状况：各来源条数、可训练/可评测条数、被拒原因。"""
    auth.require_scope("data:rw")
    return {
        "success": True,
        "report": container.dataplatform.reflux_report(),
        "trace_id": ctx.require().trace_id,
    }


@router.post("/quiz")
async def quiz(
    body: DataQuizRequest,
    auth=Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """AI 美妆顾问考核：不传 answers 只出题；传了则按**实测检索证据**自动评分。

    评分为启发式（证据一致性 + 金标要点覆盖 + 拒答策略），用于粗筛与回归防退化，
    不是 LLM-as-judge —— 面试要主动说清这个边界。
    """
    auth.require_scope("data:rw")
    started = time.perf_counter()
    with ctx.context_scope(project_id=auth.project_id):
        if body.answers is None:
            items = container.dataplatform.quiz_items()
            payload: dict[str, Any] = {
                "items": [
                    {
                        "quiz_id": item.quiz_id,
                        "quiz_type": item.quiz_type.value,
                        "expect": item.expect.value,
                        "question": item.question,
                        "gold_answer": item.gold_answer,
                        "gold_chunk_ids": item.gold_chunk_ids,
                        "group": item.group,
                    }
                    for item in items
                ]
            }
        else:
            payload = await container.dataplatform.grade_quiz(body.answers, top_k=body.top_k)

    return {
        "success": True,
        **payload,
        "types": [quiz_type.value for quiz_type in QuizType],
        "expects": [expect.value for expect in Expect],
        "trace_id": ctx.require().trace_id,
        "duration_ms": _elapsed(started),
    }
