"""REST-to-MCP 桥接层。

四件事（面试口述顺序就是这个）：
1) **校验**：项目级 API Key → 鉴权得到 {project_id, scopes, 配额上限}；
2) **注入**：把授权结果写进 contextvar，下游工具从上下文取 project_id ——
   客户端传的 project_id 一律忽略（请求体里那个字段只是为了演示「不可信输入」）；
3) **调用**：通过 ToolInvoker 执行工具（默认进程内，可切 MCP Streamable HTTP）；
4) **包装**：统一响应信封 + trace_id + 阶段耗时，方便上游对账与排障。

配额与工具错误的分工：
- 传输/鉴权/配额问题 → 抛 InfraError，由全局处理器映射成 4xx/5xx（调用方需要改请求或重试）；
- 工具业务级失败 → HTTP 200 + success=false（这是给 Agent 消费的 observation，不是传输失败）。
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request

from ..container import Container
from ..core import context as ctx
from ..core.errors import ForbiddenError, RateLimitedError, ValidationError
from ..core.logging import get_logger, scrub
from ..core.metrics import metrics
from ..core.tracing import snapshot_timeline, stage
from ..eval.cases import default_suite
from ..llm.gateway import estimate_messages_tokens
from ..memory.store import build_context
from ..tools.base import ToolContext
from .auth import AuthContext
from .schemas import (
    AgentRunRequest,
    ChatCompletionRequest,
    ConfigView,
    EvalRunRequest,
    PromptPreviewRequest,
    QuotaView,
    ToolCallRequest,
)

logger = get_logger("infra.bridge")
router = APIRouter(prefix="/v1", tags=["bridge"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


async def current_auth(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
    x_agent_id: str | None = Header(default=None, alias="X-Agent-Id"),
) -> AuthContext:
    """鉴权依赖。支持 X-API-Key 与 Authorization: Bearer 两种携带方式。

    另外承担两件事，因为它们都必须在「已经知道你是谁」之后才能做：
    1) 记录调用方声明的 Agent 身份（用于审计与成本分摊，**不参与鉴权**）；
    2) Service 维度限流（IP 维度在中间件里，因为那时还不知道身份）。
    """
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    auth = await get_container(request).authenticator.authenticate(key)
    if x_agent_id:
        auth.agent = scrub(x_agent_id.strip())[:64]
    await _enforce_service_rate_limit(get_container(request), auth)
    return auth


async def _enforce_service_rate_limit(container: Container, auth: AuthContext) -> None:
    """按业务线做滑动窗口限流。限流器缺失（例如单测注入的精简容器）时直接放行。"""
    limiter = getattr(container, "ratelimit", None)
    if limiter is None:
        return
    decision = await limiter.check(
        dimension="service",
        key=auth.service_id,
        limit=container.settings.ratelimit_service_limit,
        window_s=container.settings.ratelimit_service_window_s,
    )
    if not decision.allowed:
        raise RateLimitedError(
            f"业务线 {auth.service_id} 请求过于频繁",
            retry_after=decision.retry_after_s or 1,
            dimension=decision.dimension,
            detail={"limit": decision.limit, "key_hash": decision.key_hash, "backend": decision.backend},
        )


def request_scope(
    auth: AuthContext, *, session_id: str | None = None, user_id: str | None = None
) -> Any:
    """派生请求上下文：把权限链前两级（Service / Agent）一并写进环境。

    这样下游（模型网关的成本聚合、工具的审计）无需再逐层传参就能拿到归属信息。
    """
    return ctx.context_scope(
        project_id=auth.project_id,
        session_id=session_id,
        user_id=user_id,
        api_key_id=auth.api_key_id,
        scopes=auth.scopes,
        service=auth.service_id,
        agent=auth.agent,
    )


def _tool_context(auth: AuthContext, session_id: str | None, user_id: str | None) -> ToolContext:
    current = ctx.require()
    return ToolContext(
        trace_id=current.trace_id,
        project_id=auth.project_id,
        session_id=session_id,
        user_id=user_id,
        scopes=auth.scopes,
        service=auth.service_id,
        agent=auth.agent,
    )


@router.post("/tools/{tool_name}")
async def call_tool(
    tool_name: str,
    body: ToolCallRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """通过桥接层调用单个工具。参数在服务端二次校验，权限在服务端强制判定。"""
    if body.client_project_id and body.client_project_id != auth.project_id:
        # 记录但不采纳：客户端试图指定别人的项目，属于明确的越权信号
        logger.warning(
            "bridge.ignored_client_project_id",
            extra={
                "event": "auth",
                "status": "error",
                "reason": "client_supplied_project_id_ignored",
                "project_id": auth.project_id,
            },
        )

    trace = ctx.require().trace_id
    started = time.perf_counter()

    with request_scope(auth, session_id=body.session_id, user_id=body.user_id):
        limits = container.quota.limits_for(
            requests=auth.requests_per_window, tokens=auth.tokens_per_window
        )
        async with container.quota.quota_scope(auth.project_id, requests=1, estimated_tokens=0, limits=limits):
            tool_ctx = _tool_context(auth, body.session_id, body.user_id)
            async with stage("bridge.tool", tool=tool_name):
                result = await container.invoker.invoke(tool_name, body.arguments, tool_ctx)

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "success": result.ok,
        "tool": result.tool,
        "data": result.data,
        "error": (
            None
            if result.ok
            else {
                "code": result.error_code,
                "message": result.error_message,
                "retryable": result.retryable,
                "terminal": result.terminal,
            }
        ),
        "trace_id": trace,
        "duration_ms": duration_ms,
    }


@router.get("/tools")
async def list_tools(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """列出当前凭证可见的工具（按 scopes 过滤）。"""
    available = [
        name
        for name in container.registry.names()
        if (
            container.registry.get(name).requires_scope is None
            or container.registry.get(name).requires_scope in auth.scopes
        )
    ]
    return {
        "success": True,
        "tools": available,
        "schemas": container.registry.schemas(only_scopes=auth.scopes),
        "trace_id": ctx.require().trace_id,
    }


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """统一模型出口：记忆组装 → 配额预留 → 模型调用 → 结算 → 落库。"""
    auth.require_scope("llm:chat")
    trace = ctx.require().trace_id
    started = time.perf_counter()

    with request_scope(auth, session_id=body.session_id, user_id=body.user_id):
        messages: list[dict[str, Any]] = [message.model_dump(exclude_none=True) for message in body.messages]
        diagnostics: dict[str, Any] = {}

        # 1) 会话记忆：只在显式传了 session_id 时读写，避免无谓的 DB 访问
        if body.session_id:
            history = await container.memory.recent_messages(
                project_id=auth.project_id, session_id=body.session_id, limit=container.settings.memory_recent_turns
            )
            if history:
                bundle = build_context(
                    system_prompt=messages[0]["content"] if messages[0]["role"] == "system" else "",
                    history=history,
                    current_user_message=next(
                        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
                    ),
                    token_budget=container.settings.memory_context_token_budget,
                    recent_turns=container.settings.memory_recent_turns,
                )
                # bundle.messages[0] 是 system 占位；调用方自带 system 时沿用调用方的，
                # 否则丢弃空占位 —— 关键是无论哪种分支，历史都必须真的进模型输入。
                head = [messages[0]] if messages and messages[0]["role"] == "system" else []
                messages = [*head, *bundle.messages[1:]]
                diagnostics["dropped_messages"] = bundle.dropped_messages
                diagnostics["history_messages"] = len(bundle.messages) - 1

        estimated = estimate_messages_tokens(messages) + (
            body.max_output_tokens or container.settings.quota_default_max_output_tokens
        )
        limits = container.quota.limits_for(requests=auth.requests_per_window, tokens=auth.tokens_per_window)

        async with container.quota.quota_scope(auth.project_id, estimated_tokens=estimated, limits=limits) as rsv:
            response = await container.gateway.complete(
                messages,
                logical_model=body.logical_model,
                task=body.task,
                max_output_tokens=body.max_output_tokens,
            )
            rsv.meta["actual_tokens"] = response.usage.total_tokens

            if body.session_id:
                await container.memory.append_message(
                    project_id=auth.project_id,
                    session_id=body.session_id,
                    user_id=body.user_id,
                    role="user",
                    content=next((m["content"] for m in reversed(messages) if m["role"] == "user"), ""),
                )
                if response.text:
                    await container.memory.append_message(
                        project_id=auth.project_id,
                        session_id=body.session_id,
                        user_id=body.user_id,
                        role="assistant",
                        content=response.text,
                    )

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "success": True,
        "text": response.text,
        "logical_model": response.logical_model,
        "model": response.model,
        "provider": response.provider,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost_usd": round(response.usage.cost_usd, 6),
        },
        "finish_reason": response.finish_reason,
        "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in response.tool_calls],
        "trace_id": trace,
        "duration_ms": duration_ms,
        "stages": snapshot_timeline(),
        "diagnostics": diagnostics,
    }


@router.post("/agent/run")
async def agent_run(
    body: AgentRunRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """子 Agent 编排入口：一次调用返回结构化结果（含 token 用量与执行轨迹）。"""
    auth.require_scope("llm:chat")
    if not container.settings.tool_max_concurrency:
        raise ValidationError("工具并发度配置异常")

    started = time.perf_counter()
    with request_scope(auth, session_id=body.session_id, user_id=body.user_id):
        tool_ctx = _tool_context(auth, body.session_id, body.user_id)
        estimated = estimate_messages_tokens([{"role": "user", "content": body.task}]) + 1024
        limits = container.quota.limits_for(requests=auth.requests_per_window, tokens=auth.tokens_per_window)

        async with container.quota.quota_scope(auth.project_id, estimated_tokens=estimated, limits=limits) as rsv:
            result = await container.harness.run(
                task=body.task,
                ctx=tool_ctx,
                logical_model=body.logical_model,
                max_steps=body.max_steps,
                tool_names=body.tool_names,
                confirmed=body.confirm_high_risk,
            )
            rsv.meta["actual_tokens"] = result.usage.total_tokens

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = result.to_payload(trace_id=ctx.require().trace_id, duration_ms=duration_ms)
    return payload


@router.get("/quota")
async def quota_view(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> QuotaView:
    """查看当前项目的配额用量（口径：请求数 + Token，窗口内累计）。"""
    usage = await container.quota.store.usage(auth.project_id, container.settings.quota_window_s)
    return QuotaView(
        project_id=auth.project_id,
        backend=container.quota.backend,
        requests=usage["requests"],
        tokens=usage["tokens"],
        limits={
            "requests_per_window": auth.requests_per_window or container.settings.quota_requests_per_window,
            "tokens_per_window": auth.tokens_per_window or container.settings.quota_tokens_per_window,
            "window_s": container.settings.quota_window_s,
        },
    )


@router.get("/config")
async def config_view(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> ConfigView:
    """暴露当前生效配置的**结构**（不含密钥），用于排查「线上到底加载了哪一版」。"""
    catalog = container.catalog.current()
    return ConfigView(
        revision=container.catalog.revision,
        logical_models=sorted(catalog.models),
        routing_default=catalog.routing.default,
        mcp_paths=container.mcp_paths,
        transport=container.invoker.transport,
    )


@router.post("/memory/profile")
async def update_profile(
    patch: dict[str, Any],
    user_id: str = Query(min_length=1),
    source: str = Query(default="api"),
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """直接更新用户画像（字段级合并）。用于演示 UPSERT 与并发覆盖问题。"""
    auth.require_scope("memory:rw")
    if not patch:
        raise ForbiddenError("patch 不能为空")
    profile = await container.memory.upsert_profile(
        project_id=auth.project_id, user_id=user_id, patch=patch, source=source
    )
    return {"success": True, "user_id": user_id, "profile": profile, "trace_id": ctx.require().trace_id}


# ---- 平台治理（项目三）---------------------------------------------------- #


@router.get("/audit")
async def audit_view(
    limit: int = Query(default=20, ge=1, le=200),
    decision: str | None = Query(default=None, description="allow | deny | error"),
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """查看本项目的审计链（Service → Agent → Tool → Resource）。

    只返回**本凭证所属项目**的记录：审计是跨项目的敏感数据，
    越权读取比不读更危险 —— 所以这里按 project_id 强制隔离，而不是「反正是内部接口」。
    """
    events = container.audit.query(project_id=auth.project_id, limit=limit, decision=decision)
    return {
        "success": True,
        "project_id": auth.project_id,
        "service": auth.service_id,
        "agent": auth.agent or "default",
        "entries": [event.to_payload() for event in events],
        "summary": container.audit.summary(),
        "trace_id": ctx.require().trace_id,
    }


@router.get("/cost")
async def cost_view(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """按 Service/Agent 查看用量与成本，以及当日预算水位。

    成本数据同样按业务线隔离：只返回本 Service 前缀下的聚合，
    避免把别的业务线的花费暴露给调用方。
    """
    snapshot = metrics.metering_snapshot()
    prefix = f"{auth.service_id}/"
    scoped = {key: value for key, value in snapshot["by_service_agent"].items() if key.startswith(prefix)}
    budget = await container.budget.usage(service=auth.service_id, agent=auth.agent)
    return {
        "success": True,
        "service": auth.service_id,
        "agent": auth.agent or "default",
        "metering": scoped,
        "totals": {
            "calls": sum(item["calls"] for item in scoped.values()),
            "total_tokens": sum(item["total_tokens"] for item in scoped.values()),
            "cost_usd": round(sum(item["cost_usd"] for item in scoped.values()), 6),
        },
        "budget": budget,
        "backends": {"budget": container.budget.backend, "ratelimit": container.ratelimit.backend},
        "trace_id": ctx.require().trace_id,
    }


@router.get("/prompts")
async def prompt_view(
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """列出 Prompt 模板中心的角色与片段版本（不含正文）。"""
    registry = container.prompts
    return {
        "success": True,
        "roles": [
            {
                "role": item["role"],
                "title": item["title"],
                "scenarios": item["scenarios"],
                "layers": [layer["name"] for layer in item["layers"]],
            }
            for item in (registry.role_blueprint(role) for role in registry.roles)
        ],
        "templates": {name: registry.versions(name) for name in registry.names()},
        "active": {name: registry.active_version(name) for name in registry.names()},
        "trace_id": ctx.require().trace_id,
    }


@router.post("/prompts/{role}/preview")
async def prompt_preview(
    role: str,
    body: PromptPreviewRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """按角色组装并渲染 Prompt 正文。渲染会暴露系统提示词，因此需要 `prompt:rw`。"""
    auth.require_scope("prompt:rw")
    blueprint = container.prompts.role_blueprint(role, scenario=body.scenario)
    text = container.prompts.compose(role, scenario=body.scenario, **body.variables)
    return {
        "success": True,
        "blueprint": blueprint,
        "rendered": text,
        "chars": len(text),
        "trace_id": ctx.require().trace_id,
    }


@router.post("/eval/run")
async def eval_run(
    body: EvalRunRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """跑一轮评测（离线 mock 用例），返回六维打分报告。需要 `eval:rw`。"""
    auth.require_scope("eval:rw")
    started = time.perf_counter()
    cases = list(default_suite())
    if body.case_ids:
        wanted = set(body.case_ids)
        cases = [case for case in cases if case.case_id in wanted]
    if body.max_cases:
        cases = cases[: body.max_cases]
    if not cases:
        raise ValidationError("没有匹配的评测用例", detail={"case_ids": body.case_ids})

    with request_scope(auth):
        report = await container.eval_pipeline.run(cases, suite_id=body.suite_id)

    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    report["trace_id"] = ctx.require().trace_id
    report["scope"] = {"project_id": auth.project_id, "service": auth.service_id, "agent": auth.agent or "default"}
    return {"success": True, "report": report}
