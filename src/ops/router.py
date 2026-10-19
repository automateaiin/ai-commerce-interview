"""项目二接口层：把运营系统挂到统一应用上。

与桥接层/项目一共用同一套鉴权与上下文规则（`current_auth` + `request_scope`），
权限口径统一为 `ops:rw` —— 运营内容生产、素材写入、直播处理、自进化迭代都是**数据面写操作**，
用独立 scope 而不是复用 `llm:chat`：内容运营线的凭证不该能触发直播切片与模板版本变更。

两个接口层细节（面试可讲）：
1) **SSE 里的上下文必须在生成器内部建立**：如果在路由函数里 `with request_scope(...)` 然后返回
   `StreamingResponse`，函数一返回上下文就退出了，而生成器是在**之后**才开始迭代的，
   于是流式期间的日志与成本归属全部丢失。所以这里把上下文与配额都放进生成器体内。
2) **配额要先预扣、跑完再补实际用量**：编排花掉的 token 必须记到发起项目头上，
   否则「哪条业务线在花钱」在成本报表里会失真。
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.infra.bridge.auth import AuthContext
from src.infra.bridge.router import current_auth, get_container, request_scope
from src.infra.container import Container
from src.infra.core import context as ctx
from src.infra.core.errors import NotFoundError
from src.infra.core.logging import get_logger
from src.infra.llm.gateway import estimate_messages_tokens

from .schemas import AssetModality, ContentBrief

logger = get_logger("ops.router")

router = APIRouter(prefix="/v1/ops", tags=["ops"])

_SCOPE = "ops:rw"


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class ContentRunRequest(BaseModel):
    brief: ContentBrief
    run_id: str | None = Field(default=None, max_length=64)
    resume: bool = False
    rollback_nodes: list[str] | None = None


class AssetImportRequest(BaseModel):
    source_table: str = Field(min_length=1, max_length=64)
    source_id: str = Field(min_length=1, max_length=128)
    modality: AssetModality
    title: str = ""
    product_id: str | None = None
    market: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)
    retrieval_text: str | None = None


class LiveProcessRequest(BaseModel):
    source: dict[str, Any] = Field(default_factory=dict)
    duration_s: float = Field(gt=0, le=60 * 60 * 8)
    source_path: str = "live.mp4"


class AnalyticsRequest(BaseModel):
    channels: list[dict[str, Any]] = Field(default_factory=list)
    materials: list[dict[str, Any]] = Field(default_factory=list)
    market: str | None = None


class SelfEvolveRequest(BaseModel):
    events: list[dict[str, Any]] = Field(min_length=1)
    answers: dict[str, str] | None = None
    evidence: dict[str, list[str]] | None = None
    default_source: str = "review"


class ParallelBenchmarkRequest(BaseModel):
    count: int = Field(default=4, ge=1, le=32)
    sleep_s: float = Field(default=0.05, gt=0, le=2.0)
    fail_every: int = Field(default=0, ge=0, le=8)


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def _quota_scope(container: Container, auth: AuthContext, text: str) -> Any:
    """预扣一次配额（请求 + 估算 token）。用完由调用方回填实际用量。"""
    estimated = estimate_messages_tokens([{"role": "user", "content": text}]) + 512
    limits = container.quota.limits_for(requests=auth.requests_per_window, tokens=auth.tokens_per_window)
    return container.quota.quota_scope(auth.project_id, requests=1, estimated_tokens=estimated, limits=limits)


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _ops(container: Container) -> Any:
    service = getattr(container, "ops", None)
    if service is None:
        raise NotFoundError("运营系统未装配", detail={"hint": "容器缺少 ops 组件"})
    return service


# --------------------------------------------------------------------------- #
# 内容工作流
# --------------------------------------------------------------------------- #
@router.post("/content/run")
async def content_run(
    body: ContentRunRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """跑一次运营内容工作流：选题 → 成分卖点 → (文案 ‖ 产品图)，节点级可续跑。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    started = time.perf_counter()
    run_id = body.run_id or ops.new_run_id()
    with request_scope(auth):
        if body.rollback_nodes:
            ops.rollback(run_id, body.rollback_nodes, reason="api_resume")
        async with _quota_scope(container, auth, body.brief.product_name or body.brief.product_id) as reservation:
            result = await ops.run_content(body.brief, run_id=run_id, resume=body.resume)
            reservation.meta["actual_tokens"] = (result.artifacts.get("usage") or {}).get("total_tokens", 0)
    return {"success": True, **result.to_payload(), "trace_id": ctx.require().trace_id, "duration_ms": _elapsed(started)}


@router.post("/content/resume")
async def content_resume(
    body: ContentRunRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """续跑：复用已完成节点，只补没跑完的部分；可先回滚指定节点（含下游）。"""
    auth.require_scope(_SCOPE)
    if not body.run_id:
        from src.infra.core.errors import ValidationError

        raise ValidationError("续跑必须提供 run_id")
    ops = _ops(container)
    started = time.perf_counter()
    with request_scope(auth):
        if body.rollback_nodes:
            ops.rollback(body.run_id, body.rollback_nodes, reason="api_resume")
        async with _quota_scope(container, auth, body.brief.product_name or body.brief.product_id) as reservation:
            result = await ops.run_content(body.brief, run_id=body.run_id, resume=True)
            reservation.meta["actual_tokens"] = (result.artifacts.get("usage") or {}).get("total_tokens", 0)
    return {"success": True, **result.to_payload(), "trace_id": ctx.require().trace_id, "duration_ms": _elapsed(started)}


@router.get("/workflow/{run_id}")
async def workflow_snapshot(
    run_id: str,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """回放一个 run：节点状态 + 每次尝试的时间线（排障时看「第几次失败、为什么」）。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    snapshot = ops.workflow_snapshot(run_id)
    if not snapshot.get("found"):
        raise NotFoundError(f"未知 run: {run_id}", detail={"run_id": run_id})
    return {
        "success": True,
        "run": snapshot,
        "replay": ops.replay(run_id)["timeline"],
        "trace_id": ctx.require().trace_id,
    }


@router.post("/content/stream")
async def content_stream(
    body: ContentRunRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> StreamingResponse:
    """事件级 SSE：逐条推 `node.start` / `token` / `node.done` / `run.done`。

    客户端断开时生成器被 `aclose()`，后台 run 会被取消并记账（不继续烧 token）。
    """
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    run_id = body.run_id or ops.new_run_id()
    brief = body.brief
    quota_key = brief.product_name or brief.product_id

    async def event_stream() -> AsyncIterator[str]:
        # 上下文与配额都必须在这里建立：路由函数返回后上下文就失效了，而生成器是之后才跑的
        with request_scope(auth):
            async with _quota_scope(container, auth, quota_key) as reservation:
                async for frame in ops.stream_content(brief, run_id=run_id, resume=body.resume):
                    yield frame
                result = ops.streamer.last_result
                if result is not None:
                    reservation.meta["actual_tokens"] = (result.artifacts.get("usage") or {}).get("total_tokens", 0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Run-Id": run_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭反向代理缓冲，否则事件会被攒着一次性下发
        },
    )


# --------------------------------------------------------------------------- #
# 多模态素材
# --------------------------------------------------------------------------- #
@router.post("/assets/import")
async def asset_import(
    body: AssetImportRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """导入一份多模态素材：端口抽取 → retrieval_text → 向量化 → 按 (table,id) 幂等 UPSERT。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    with request_scope(auth):
        result = await ops.import_asset(
            source_table=body.source_table,
            source_id=body.source_id,
            modality=body.modality,
            title=body.title,
            product_id=body.product_id,
            market=body.market,
            tags=body.tags,
            source=body.source,
            retrieval_text=body.retrieval_text,
        )
    return {"success": True, **result, "trace_id": ctx.require().trace_id}


@router.get("/assets/search")
async def asset_search(
    query: str = Query(min_length=1, max_length=500),
    top_k: int = Query(default=3, ge=1, le=20),
    product_id: str | None = Query(default=None),
    market: str | None = Query(default=None),
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """跨模态检索：文本查询能命中图片 / 视频 / 页面素材。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    with request_scope(auth):
        result = await ops.search_assets(query, top_k=top_k, product_id=product_id, market=market)
    return {"success": True, **result, "trace_id": ctx.require().trace_id}


# --------------------------------------------------------------------------- #
# 直播
# --------------------------------------------------------------------------- #
@router.post("/live/process")
async def live_process(
    body: LiveProcessRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """长直播处理：分段转录 → 重叠去重 → 分批分析 → 切片候选 → FFmpeg 命令（不执行）。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    started = time.perf_counter()
    with request_scope(auth):
        async with _quota_scope(container, auth, f"live:{body.source.get('title') or body.source_path}") as reservation:
            result = await ops.process_live(body.source, duration_s=body.duration_s, source_path=body.source_path)
            reservation.meta["actual_tokens"] = result.usage.get("prompt_tokens", 0) + result.usage.get(
                "completion_tokens", 0
            )
    return {"success": True, **result.to_payload(), "trace_id": ctx.require().trace_id, "duration_ms": _elapsed(started)}


# --------------------------------------------------------------------------- #
# 运营分析
# --------------------------------------------------------------------------- #
@router.post("/analytics/report")
async def analytics_report(
    body: AnalyticsRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """运营分析：可复算指标 + 渠道增量归因 + 素材爆款评分/标题/剪辑建议。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    with request_scope(auth):
        report = ops.analytics_report({"channels": body.channels, "materials": body.materials, "market": body.market})
    return {"success": True, **report, "trace_id": ctx.require().trace_id}


# --------------------------------------------------------------------------- #
# 自进化
# --------------------------------------------------------------------------- #
@router.post("/selfevolve/cycle")
async def selfevolve_cycle(
    body: SelfEvolveRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """跑一次自进化循环：回流 → 自动评测集 → 自动评分 → 模板/策略迭代**建议**（不自动发布）。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    with request_scope(auth):
        cycle = ops.selfevolve_cycle(
            body.events, answers=body.answers, evidence=body.evidence, default_source=body.default_source
        )
    return {"success": True, **cycle.to_payload(), "trace_id": ctx.require().trace_id}


@router.get("/selfevolve/report")
async def selfevolve_report(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """回流池状况 + 模板版本（含主动版本）+ 历次循环产物。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    return {"success": True, **ops.selfevolve_report(), "trace_id": ctx.require().trace_id}


@router.post("/selfevolve/apply")
async def selfevolve_apply(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """人工审核通过后**手动**激活最新一版模板（cycle 不会自动发布）。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    return {"success": True, **ops.selfevolve_apply(), "trace_id": ctx.require().trace_id}


@router.post("/selfevolve/revert")
async def selfevolve_revert(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """回滚到上一版模板。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    return {"success": True, **ops.selfevolve_revert(), "trace_id": ctx.require().trace_id}


# --------------------------------------------------------------------------- #
# 并行基准
# --------------------------------------------------------------------------- #
@router.post("/parallel/benchmark")
async def parallel_benchmark(
    body: ParallelBenchmarkRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """并行实测：同批阻塞单元并发执行，报告给出墙钟耗时与串行下界（不编造加速比）。"""
    auth.require_scope(_SCOPE)
    ops = _ops(container)
    with request_scope(auth):
        report = await ops.benchmark_parallel(count=body.count, sleep_s=body.sleep_s, fail_every=body.fail_every)
    return {"success": True, **report.to_payload(), "trace_id": ctx.require().trace_id}


__all__ = ["router"]
