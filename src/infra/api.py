"""应用组装：中间件、异常处理、健康检查、指标、管理端点。

三个必须放在应用层的横切能力（业务代码里不该出现）：
1) **Trace 中间件**：入口取/生成 trace_id 写 contextvar，出口写访问日志与阶段耗时。
   注意：这里用**纯 ASGI 中间件**而不是 BaseHTTPMiddleware ——
   后者会把下游放进不同的 task，导致 contextvar 传递断链（trace_id 在日志里丢失），
   这是排障时非常典型的坑。
2) **统一异常处理器**：把内部异常映射成统一响应体，堆栈只进日志不进响应。
3) **健康检查**：区分存活（liveness）与就绪（readiness）——
   存活只表示进程还在，就绪必须验证依赖（数据库、配置快照、知识索引）真的可用。
"""

from __future__ import annotations

import hmac
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .bridge import data_router as data_bridge_router
from .bridge import router as bridge_router
from .container import Container
from .core import context as ctx
from .core.errors import InfraError, normalize_exception
from .core.logging import get_logger, setup_logging
from .core.metrics import metrics
from .core.tracing import (
    TRACE_HEADER,
    new_trace_id,
    normalize_trace_id,
    slowest_stage,
    snapshot_timeline,
    start_timeline,
)
from .mcp.server import mount_mcp_servers
from .ratelimit.middleware import RateLimitMiddleware

logger = get_logger("infra.api")

# 演示前端：零构建静态页（原生 fetch + 手写 CSS），由 FastAPI 直接托管。
# 刻意不引入 npm/打包链：这是面试演示项目，多一条构建链就多一处「跑不起来」的理由。
STATIC_DIR = Path(__file__).resolve().parent / "static"


class TraceMiddleware:
    """纯 ASGI 请求追踪中间件：建立 trace 上下文 + 访问日志。"""

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        trace_id = normalize_trace_id(headers.get(TRACE_HEADER) or new_trace_id())
        token = ctx.set_context(ctx.RequestContext(trace_id=trace_id))
        # 开启本请求的阶段时间线：没有它，Stage 打点无处可记，慢在哪一环就无从定位
        start_timeline()
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def _send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers_list = list(message.get("headers", []))
                # 把 trace_id 回给调用方：客户端可用它去日志系统对账整条链路
                headers_list.append((TRACE_HEADER.encode(), trace_id.encode()))
                message = {**message, "headers": headers_list}
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            path = scope.get("path", "")
            if path not in ("/healthz", "/metrics"):  # 探针路径不写访问日志，避免噪声淹没真实请求
                metrics.incr("request", f"{scope.get('method', '')} {path}")
                metrics.observe_latency("request", path, duration_ms)
                if status_holder["status"] >= 400:
                    # 4xx 也要计入错误率：401/403/429 恰恰是攻击、越权与限流的第一手信号，
                    # 只统计 5xx 会让「谁在刷接口」这类问题在监控上完全隐形。
                    metrics.incr("error", f"{scope.get('method', '')} {path}")
                logger.info(
                    "http.request",
                    extra={
                        "event": "access",
                        "method": scope.get("method"),
                        "path": path,
                        "http_status": status_holder["status"],
                        "duration_ms": duration_ms,
                        "stages": snapshot_timeline(),
                        "slowest_stage": slowest_stage(),
                        "status": "ok" if status_holder["status"] < 400 else "error",
                    },
                )
            ctx.reset_context(token)


def _error_response(exc: InfraError, trace_id: str | None) -> JSONResponse:
    headers = {}
    # 配额超限与限流都是 429，都该告诉调用方「多久之后可以重试」；
    # 用统一的 retry_after 约定取字段，避免每加一种 429 就改一次这里。
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=exc.http_status,
        content={"success": False, "error": exc.to_payload(trace_id), "trace_id": trace_id},
        headers=headers,
    )


def create_app(container: Container | None = None) -> FastAPI:
    """创建应用。允许传入已构造的 Container（测试注入）。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or await Container.create()
        # 先挂载 MCP 应用（session_manager 是 streamable_http_app() 懒创建的），
        # 再进入它的生命周期，否则拿不到 session_manager。
        from contextlib import AsyncExitStack

        mount_mcp_servers(app, app.state.container.mcp_servers)
        async with AsyncExitStack() as stack:
            for server in app.state.container.mcp_servers.values():
                await stack.enter_async_context(server.session_manager.run())
            logger.info(
                "app.started",
                extra={
                    "event": "startup",
                    "status": "ok",
                    "route": ",".join(app.state.container.mcp_paths),
                },
            )
            try:
                yield
            finally:
                await app.state.container.aclose()
                logger.info("app.stopped", extra={"event": "shutdown", "status": "ok"})

    app = FastAPI(
        title="出海电商 AI Infra 与 MCP 服务集群",
        version="1.0.0",
        description="统一模型出口 + MCP 工具服务 + RAG/记忆 + 配额治理 + 可观测",
        lifespan=lifespan,
    )
    # 中间件顺序：最后添加的在最外层。TraceMiddleware 必须在最外层 ——
    # 这样限流拒绝（429）也处在 trace 上下文里，拒绝响应能带回 trace_id，
    # 而「谁在被限流」这件事才能和访问日志、指标对上号。
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TraceMiddleware)
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    # ---- 异常映射：业务代码只抛语义异常，响应形状在这里统一 ---- #
    @app.exception_handler(InfraError)
    async def _handle_infra_error(request: Request, exc: InfraError) -> JSONResponse:
        current = ctx.current()
        trace_id = current.trace_id if current else None
        logger.warning(
            "http.error",
            extra={
                "event": "error",
                "error_code": exc.code,
                "http_status": exc.http_status,
                "path": request.url.path,
                "status": "error",
            },
        )
        return _error_response(exc, trace_id)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        current = ctx.current()
        trace_id = current.trace_id if current else None
        issues = [{"field": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]} for e in exc.errors()[:5]]
        logger.warning(
            "http.validation_failed",
            extra={"event": "error", "error_code": "invalid_argument", "path": request.url.path, "status": "error"},
        )
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "invalid_argument",
                    "message": "请求参数不合法",
                    "retryable": False,
                    "issues": issues,
                },
                "trace_id": trace_id,
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # 未知异常：对外只给安全摘要，完整堆栈由日志层记录（这里显式打 exc_info）
        current = ctx.current()
        trace_id = current.trace_id if current else None
        error = normalize_exception(exc)
        logger.error(
            "http.unhandled_error",
            extra={
                "event": "error",
                "error_code": error.code,
                "http_status": 500,
                "path": request.url.path,
                "status": "error",
            },
            exc_info=exc,
        )
        return _error_response(error, trace_id)

    app.include_router(bridge_router.router)
    # 数据平台路由（项目三）：与桥接层同一套鉴权与异常信封
    app.include_router(data_bridge_router.router)
    # 应用层路由（项目一）：编排/双端 Agent/HITL/实时语音 + A2A 内部通道
    from src.agents.router import a2a_router, router as agents_router

    app.include_router(agents_router)
    app.include_router(a2a_router)
    # 宿主业务后端（commerce）：买家侧 /v1/shop/*（会员 token）与运营侧 /v1/admin/*（API Key + commerce:rw）
    from src.commerce.interface.routers import admin_router, shop_router

    app.include_router(shop_router)
    app.include_router(admin_router)
    # 应用层路由（项目二）：内容工作流 / SSE 流式 / 多模态素材 / 直播 / 分析 / 自进化（scope ops:rw）
    from src.ops.router import router as ops_router

    app.include_router(ops_router)

    # ---- 演示前端（官网页 / 运营台） ---------------------------------------- #
    # 全部 include_in_schema=False：这些是「给人看的页面」，不是 API 契约，
    # 混进 OpenAPI 只会让调用方以为它们需要按 API 的方式来调。
    # 用 FileResponse 显式返回具体文件而不是 StaticFiles(html=True)：
    # 后者依赖目录猜路径，页面一改名就会静默 404。
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    async def storefront() -> FileResponse:
        """公司官网页：浏览商品 + AI 导购客服。"""
        return FileResponse(STATIC_DIR / "shop.html", media_type="text/html")

    @app.get("/admin", include_in_schema=False)
    async def ops_console() -> FileResponse:
        """电商运营台：商品/订单/库存/券 + AI 运营与成本。"""
        return FileResponse(STATIC_DIR / "admin.html", media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """浏览器总会自动请求 favicon；回 204 而不是 404，免得访问日志里全是噪声。"""
        return Response(status_code=204)

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, Any]:
        """存活探针：只回答「进程还活着」，不检查依赖。"""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> JSONResponse:
        """就绪探针：验证依赖真的可用，不可用就不要接流量。"""
        active: Container = request.app.state.container
        checks: dict[str, Any] = {}
        healthy = True

        try:
            await active.memory.recent_messages(project_id="__probe__", session_id="__probe__", limit=1)
            checks["database"] = {"ok": True, "dialect": active.memory.dialect}
        except Exception as exc:  # noqa: BLE001
            healthy = False
            checks["database"] = {"ok": False, "reason": type(exc).__name__}

        catalog = active.catalog.current()
        checks["config"] = {"ok": bool(catalog.models), "revision": active.catalog.revision}
        checks["knowledge"] = {"ok": active.knowledge.size > 0, "chunks": active.knowledge.size}
        checks["quota"] = {"ok": True, "backend": active.quota.backend}
        checks["tools"] = {"ok": True, "count": len(active.registry.names())}
        healthy = healthy and checks["config"]["ok"] and checks["knowledge"]["ok"]

        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ready" if healthy else "degraded", "checks": checks},
        )

    @app.get("/metrics", tags=["ops"])
    async def metrics_view() -> dict[str, Any]:
        """轻量指标：请求量、错误率、各阶段 P95 延迟、Token 与成本。"""
        return metrics.snapshot()

    @app.post("/admin/config/reload", tags=["ops"])
    async def reload_config(request: Request) -> JSONResponse:
        """配置热重载（快照切换）。

        保护方式：需要 x-admin-token 匹配环境变量 ADMIN_TOKEN；
        **未配置 ADMIN_TOKEN 时端点直接关闭**，不做「本地就放行」的例外。
        """
        expected = os.environ.get("ADMIN_TOKEN")
        provided = request.headers.get("x-admin-token", "")
        from .core.errors import AuthenticationError, ForbiddenError

        if not expected:
            raise ForbiddenError("管理端点未启用：未配置 ADMIN_TOKEN")
        if not hmac.compare_digest(provided, expected):
            raise AuthenticationError("管理凭证无效")

        active: Container = request.app.state.container
        from .llm.catalog import parse_catalog
        from .llm.factory import apply_provider_mode

        revision = active.catalog.reload_from(
            active.settings.resolved_models_config_path(),
            loader=lambda raw: apply_provider_mode(parse_catalog(raw), active.settings.provider_mode),
        )
        # 目录变了，Provider 绑定也需要重建（与同一快照一起生效，避免新旧混用）
        from .llm.factory import build_providers

        active.gateway.replace_providers(build_providers(active.catalog.current(), active.settings))
        return JSONResponse(
            content={
                "success": True,
                "revision": revision,
                "logical_models": sorted(active.catalog.current().models),
                "trace_id": ctx.current().trace_id if ctx.current() else None,
            }
        )

    return app


app = create_app()
