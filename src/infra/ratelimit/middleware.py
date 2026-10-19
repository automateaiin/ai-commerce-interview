"""IP 维度限流中间件（纯 ASGI）。

为什么放在中间件而不是路由依赖：
鉴权失败（401）的请求也需要被限流 —— 否则拿无效 Key 狂刷鉴权逻辑照样能打满 CPU，
而依赖注入是在鉴权**之后**才执行的，拦不住这类流量。所以第一道闸必须在鉴权之前。

为什么用纯 ASGI 而不是 BaseHTTPMiddleware：后者把下游放进独立 task，
contextvar（trace_id）会断链，拒绝响应里就拿不到 trace_id 了。

Service 维度（业务线）在桥接层按凭证判定 —— 它需要先知道「你是谁」，
而 IP 维度不需要，两者分层正是这个原因。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..core import context as ctx
from ..core.errors import RateLimitedError
from ..core.logging import get_logger

logger = get_logger("infra.ratelimit.middleware")

# 探针路径不限流：限流把就绪探针打回 429，会被误判成实例不健康并触发摘流量
_EXEMPT_PATHS = ("/healthz", "/readyz", "/metrics", "/docs", "/openapi.json")


class RateLimitMiddleware:
    """按客户端 IP 的滑动窗口限流。容器未就绪或未配置时一律放行（fail-open）。"""

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        container = getattr(getattr(scope.get("app"), "state", None), "container", None)
        if container is None or getattr(container, "ratelimit", None) is None:
            # 启动早期（lifespan 尚未建好 Container）不能拒绝请求，否则会把启动过程打成 429
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        client = scope.get("client") or ("unknown", 0)
        ip = client[0] or "unknown"
        decision = await container.ratelimit.check(
            dimension="ip",
            key=ip,
            limit=container.settings.ratelimit_ip_limit,
            window_s=container.settings.ratelimit_ip_window_s,
        )
        if decision.allowed:
            await self.app(scope, receive, send)
            return

        await self._reject(scope, send, decision)

    async def _reject(self, scope: dict[str, Any], send: Callable[..., Any], decision: Any) -> None:
        current = ctx.current()
        trace_id = current.trace_id if current else None
        error = RateLimitedError(
            f"IP 维度请求过于频繁（窗口 {decision.limit} 次）",
            retry_after=decision.retry_after_s or 1,
            dimension=decision.dimension,
            detail={"limit": decision.limit, "key_hash": decision.key_hash, "backend": decision.backend},
        )
        body = json.dumps(
            {"success": False, "error": error.to_payload(trace_id), "trace_id": trace_id},
            ensure_ascii=False,
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": error.http_status,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"retry-after", str(error.retry_after).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


__all__ = ["RateLimitMiddleware"]
