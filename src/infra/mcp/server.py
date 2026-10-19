"""MCP 服务集群（Streamable HTTP）。

为什么拆多个 MCP Server，而不是全塞进一个 FastAPI 项目：
按**职责边界**拆 —— 商品/订单（交易事实）、知识（可更新证据）、记忆（会话与画像）、
Prompt（可变策略）各自独立演进、独立扩缩容。拆分依据是职责与复用需求，
不是为了数量好看；模块没有独立扩展需求时，同进程模块化实现即可。

身份传递（关键安全点）：MCP Server 只接受**内部调用**，
身份必须由调用方（桥接层）在完成鉴权后通过内部 header 注入，
并用 `INTERNAL_SERVICE_TOKEN` 作为内部凭证；
MCP Server 端绝不接受客户端直接传来的 project_id。

ASGI 中间件里设置的 contextvar 会随同一个 task 向下传递到工具函数，
因此工具实现无需感知 HTTP，直接 `context.require()` 就能拿到可信身份。
"""

from __future__ import annotations

import hmac
import json
import os
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from ..core import context as ctx
from ..core.errors import InfraError
from ..core.logging import get_logger
from ..core.settings import get_settings
from ..core.tracing import TRACE_HEADER, normalize_trace_id
from ..memory.store import MemoryStore
from ..prompts.registry import PromptRegistry
from ..rag.service import KnowledgeService
from ..tools.base import ToolContext, ToolExecutor, ToolRegistry

logger = get_logger("infra.mcp")

INTERNAL_TOKEN_HEADER = "x-internal-token"
PROJECT_HEADER = "x-project-id"
SCOPES_HEADER = "x-scopes"
SESSION_HEADER = "x-session-id"
USER_HEADER = "x-user-id"


class InternalIdentityMiddleware:
    """内部身份注入中间件（纯 ASGI，不依赖框架细节）。

    做的事：校验内部凭证 → 校验 header 形态 → 写入 contextvar。
    做错任何一步都返回 401/400，绝不放行「没有身份」的调用。
    """

    def __init__(self, app: Callable[..., Any], *, token: str, service: str) -> None:
        self.app = app
        self.token = token
        self.service = service

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        provided = headers.get(INTERNAL_TOKEN_HEADER, "")
        if not self.token or not hmac.compare_digest(provided, self.token):
            # 内部服务之间也做鉴权：不因为在内网就默认可信
            await _send_json(send, 401, {"code": "unauthenticated", "message": "缺少或错误的内部调用凭证"})
            return

        project_id = headers.get(PROJECT_HEADER)
        if not project_id:
            await _send_json(send, 400, {"code": "invalid_argument", "message": "缺少 x-project-id"})
            return

        scopes = frozenset(filter(None, headers.get(SCOPES_HEADER, "").split(",")))
        # trace_id 优先用显式 header；没有 header 时**继承外层中间件已建立的 trace_id**，
        # 不能另起一个 —— 否则同一次调用在接入层与 MCP 层会落在两条互不相干的链路上。
        outer = ctx.current()
        token = ctx.set_context(
            ctx.RequestContext(
                trace_id=normalize_trace_id(
                    headers.get(TRACE_HEADER) or (outer.trace_id if outer else None)
                ),
                project_id=project_id,
                session_id=headers.get(SESSION_HEADER) or None,
                user_id=headers.get(USER_HEADER) or None,
                scopes=scopes,
            )
        )
        try:
            await self.app(scope, receive, send)
        finally:
            ctx.reset_context(token)


async def _send_json(send: Callable[..., Any], status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json; charset=utf-8"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _tool_context() -> ToolContext:
    """从 contextvar 取身份。取不到直接抛错 —— 绝不用默认身份兜底。"""
    current = ctx.require()
    return ToolContext(
        trace_id=current.trace_id,
        project_id=current.project_id or "",
        session_id=current.session_id,
        user_id=current.user_id,
        scopes=frozenset(current.scopes),
    )


def _wrap(executor: ToolExecutor, tool_name: str) -> Callable[..., Any]:
    """把 ToolExecutor 调用包装成 MCP 工具函数。

    统一约定：MCP 层返回结构化 observation（ok / error_code / message），
    工具失败不抛协议异常，而是作为结果返回 —— 让 Agent 能基于失败继续决策。
    """

    async def _invoke(**kwargs: Any) -> dict[str, Any]:
        result = await executor.execute(tool_name, kwargs, _tool_context())
        return {
            "ok": result.ok,
            "tool": result.tool,
            "data": result.data,
            "error_code": result.error_code,
            "message": result.error_message,
            "retryable": result.retryable,
        }

    return _invoke


def _build_commerce_server(registry: ToolRegistry, executor: ToolExecutor) -> MCPServer:
    server = MCPServer(
        name="commerce",
        instructions="商品与订单事实工具。所有查询与写操作都在服务端按项目与用户权限校验。",
    )

    async def search_products(
        query: str = "",
        skin_type: str | None = None,
        max_price_usd: float | None = None,
        market: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """按关键词/肤质/价格/站点检索在售商品，返回商品 ID、价格、适用肤质与库存。"""
        return await _wrap(executor, "search_products")(
            query=query, skin_type=skin_type, max_price_usd=max_price_usd, market=market, limit=limit
        )

    async def get_product_detail(product_id: str) -> dict[str, Any]:
        """查询单个商品的成分、功效、使用方法、规格与库存。"""
        return await _wrap(executor, "get_product_detail")(product_id=product_id)

    async def get_order_status(order_id: str) -> dict[str, Any]:
        """查询当前用户订单的状态与明细，仅能查询属于当前用户的订单。"""
        return await _wrap(executor, "get_order_status")(order_id=order_id)

    async def create_order(
        product_id: str, quantity: int, idempotency_key: str, market: str | None = None
    ) -> dict[str, Any]:
        """创建待付款订单（写操作，必须携带幂等键）。"""
        return await _wrap(executor, "create_order")(
            product_id=product_id, quantity=quantity, idempotency_key=idempotency_key, market=market
        )

    server.add_tool(search_products)
    server.add_tool(get_product_detail)
    server.add_tool(get_order_status)
    server.add_tool(create_order)
    return server


def _build_knowledge_server(registry: ToolRegistry, executor: ToolExecutor, knowledge: KnowledgeService) -> MCPServer:
    server = MCPServer(name="knowledge", instructions="商品知识检索服务，返回带来源与有效期的证据片段。")

    async def search_knowledge(
        query: str, product_id: str | None = None, market: str | None = None, top_k: int = 3
    ) -> dict[str, Any]:
        """从知识库检索成分、功效、使用方法与政策类证据，返回来源与过期过滤诊断信息。"""
        return await _wrap(executor, "search_knowledge")(
            query=query, product_id=product_id, market=market, top_k=top_k
        )

    server.add_tool(search_knowledge)
    return server


def _build_memory_server(registry: ToolRegistry, executor: ToolExecutor, memory: MemoryStore) -> MCPServer:
    server = MCPServer(name="memory", instructions="会话记忆与用户画像服务，按项目与用户范围隔离。")

    async def get_conversation_history(limit: int = 10) -> dict[str, Any]:
        """读取当前会话最近的消息历史。"""
        return await _wrap(executor, "get_conversation_history")(limit=limit)

    async def update_user_profile(patch: dict[str, Any], source: str = "agent") -> dict[str, Any]:
        """字段级合并更新用户长期画像，不覆盖未提及的字段。"""
        return await _wrap(executor, "update_user_profile")(patch=patch, source=source)

    server.add_tool(get_conversation_history)
    server.add_tool(update_user_profile)
    return server


def _build_prompt_server(registry: PromptRegistry) -> MCPServer:
    server = MCPServer(name="prompt", instructions="Prompt 模板与版本管理服务。")

    async def get_prompt(name: str, version: str | None = None) -> dict[str, Any]:
        """获取指定版本（默认当前生效版本）的 Prompt 模板。"""
        template = registry.get(name, version)
        return {
            "name": template.name,
            "version": template.version,
            "template": template.template,
            "note": template.note,
            "active_version": registry.active_version(name),
        }

    async def render_prompt(name: str, variables: dict[str, Any]) -> dict[str, Any]:
        """渲染 Prompt，缺失变量会报错而不是静默留空。"""
        return {"name": name, "version": registry.active_version(name), "text": registry.render(name, **variables)}

    server.add_tool(get_prompt)
    server.add_tool(render_prompt)
    return server


def build_mcp_servers(
    *,
    registry: ToolRegistry,
    executor: ToolExecutor,
    knowledge: KnowledgeService,
    memory: MemoryStore,
    prompts: PromptRegistry,
) -> dict[str, MCPServer]:
    """构建全部 MCP Server。返回 {服务名: MCPServer}，由 api.py 挂载到 /mcp/{服务名}。"""
    return {
        "commerce": _build_commerce_server(registry, executor),
        "knowledge": _build_knowledge_server(registry, executor, knowledge),
        "memory": _build_memory_server(registry, executor, memory),
        "prompt": _build_prompt_server(prompts),
    }


def mount_mcp_servers(app: Any, servers: dict[str, MCPServer], *, base_path: str = "/mcp") -> list[str]:
    """把每个 MCP Server 挂到独立路径，并套上内部身份中间件。

    注意：内部凭证缺失时直接拒绝，不做「本地调用就放行」的例外 ——
    例外一旦存在，就会在某个环境被打开成为越权入口。
    """
    token = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
    if not token:
        logger.warning(
            "mcp.internal_token_missing",
            extra={
                "event": "config_warning",
                "reason": "INTERNAL_SERVICE_TOKEN 未配置，MCP 端点将拒绝所有调用",
            },
        )
    # Host 白名单（DNS rebinding 防护）：**只做加法，不做关闭**。
    # 默认只允许回环地址，真实部署时用 MCP_ALLOWED_HOSTS 显式声明内网域名/容器主机名；
    # 授权仍由内部凭证承担，Host 校验只是纵深防御的一层，因此不允许用开关绕过。
    extra_hosts = [item.strip() for item in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if item.strip()]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", *extra_hosts],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
    )
    mounted: list[str] = []
    for name, server in servers.items():
        path = f"{base_path}/{name}"
        # streamable_http_path="/" 让挂载后的对外地址就是 {base_path}/{name}，而不是再加一层 /mcp；
        # json_response=True 让响应是 JSON 而非 SSE 流 —— 便于内部服务与脚本消费，
        # 需要流式时可以按部署参数切回 SSE。
        inner = server.streamable_http_app(
            streamable_http_path="/", json_response=True, transport_security=security
        )
        app.mount(path, InternalIdentityMiddleware(inner, token=token, service=name))
        mounted.append(path)
    logger.info("mcp.mounted", extra={"event": "mcp_mounted", "reason": ",".join(mounted)})
    return mounted
