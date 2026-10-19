"""项目一接口层：把编排能力挂到统一应用上。

与桥接层共用同一套鉴权与上下文规则（`current_auth` + `request_scope`），
因为「谁在调用、这个项目能不能调这个能力」不该因为换了个路径就换一套判断。

权限与配额的口径在这里收口：
- 对话/编排类端点需要 `llm:chat`，检索类需要 `tools:knowledge`，
  高风险确认需要**独立权限** `hitl:approve`（客服的 key 不能自己批自己的单）；
- 所有真实调用都要在**配额作用域**内执行（与 /v1/agent/run 一致），
  编排花的 token 记到发起项目头上。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from src.infra.bridge.router import current_auth, get_container, request_scope
from src.infra.container import Container
from src.infra.core import context as ctx
from src.infra.core.errors import AuthenticationError, ForbiddenError, ValidationError
from src.infra.core.logging import get_logger
from src.infra.llm.gateway import estimate_messages_tokens
from src.infra.tools.base import ToolContext
from src.infra.bridge.auth import AuthContext

from .a2a import A2ARequest, build_scopes

logger = get_logger("agents.router")

router = APIRouter(prefix="/v1/agents", tags=["agents"])
a2a_router = APIRouter(prefix="/v1/a2a", tags=["a2a"])


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class RouteRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    scenario: str | None = None
    mode: str | None = None
    domain: str | None = None


class OrchestrateRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    user_id: str | None = None
    scenario: str | None = None
    mode: str | None = None
    domain: str | None = None
    confirm_high_risk: bool = False
    persist: bool = True


class OpsRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    scenario: str = Field(default="analysis", pattern="^(analysis|report|review)$")


class HitlStageRequest(BaseModel):
    action: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    session_id: str | None = None
    user_id: str | None = None
    project_id: str | None = Field(
        default=None, description="仅用于演示「不可信输入」：服务端一律忽略并采用凭证所属项目"
    )


class HitlApproveRequest(BaseModel):
    checkpoint_id: str = Field(min_length=1)
    resume_token: str = Field(min_length=1)
    note: str | None = None


class HitlApplyRequest(BaseModel):
    checkpoint_id: str = Field(min_length=1)


class RealtimeTurnRequest(BaseModel):
    text: str | None = None
    utterance: str | None = Field(default=None, description="模拟客户端上行语音；与 text 二选一")
    session_id: str | None = None
    user_id: str | None = None


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    strategy: str = Field(default="step_back", pattern="^(none|step_back|hyde)$")
    top_k: int = Field(default=3, ge=1, le=10)


class FeedbackRequest(BaseModel):
    kind: str = Field(pattern="^(correction|preferred|failure)$")
    question: str = Field(min_length=1)
    answer: str | None = None
    corrected: str | None = None
    agent: str = "customer"
    scenario: str | None = None
    tags: list[str] = Field(default_factory=list)


class A2ASendRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    agent: str | None = None
    scenario: str | None = None
    mode: str | None = None
    project_id: str = "a2a"
    user_id: str | None = None
    service: str | None = None
    trace_id: str | None = None
    scopes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def _tool_context(auth: AuthContext, *, session_id: str | None, user_id: str | None) -> ToolContext:
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


def _quota_scope(container: Container, auth: AuthContext, task: str):
    estimated = estimate_messages_tokens([{"role": "user", "content": task}]) + 512
    limits = container.quota.limits_for(requests=auth.requests_per_window, tokens=auth.tokens_per_window)
    return container.quota.quota_scope(auth.project_id, requests=1, estimated_tokens=estimated, limits=limits)


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
@router.post("/route")
async def route_only(
    body: RouteRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """只路由、不执行：用于观察「这个问题会被判成直连还是进 Agent」。"""
    auth.require_scope("llm:chat")
    with request_scope(auth):
        plan = await container.agents.orchestrator.route(
            body.task, scenario_hint=body.scenario, mode_hint=body.mode, domain_hint=body.domain
        )
    return {"success": True, "route": plan.to_payload(), "trace_id": ctx.require().trace_id}


@router.post("/orchestrate")
async def orchestrate(
    body: OrchestrateRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """项目一主入口：复杂度路由 → 直连 Tool 或按模式编排 → 事实绑定闸门。"""
    auth.require_scope("llm:chat")
    agents = container.agents
    with request_scope(auth, session_id=body.session_id, user_id=body.user_id):
        tool_ctx = _tool_context(auth, session_id=body.session_id, user_id=body.user_id)
        async with _quota_scope(container, auth, body.task) as reservation:
            result = await agents.orchestrator.handle(
                body.task,
                tool_ctx,
                scenario=body.scenario,
                mode=body.mode,
                domain=body.domain,
                confirmed=body.confirm_high_risk,
            )
            reservation.meta["actual_tokens"] = result.usage.total_tokens

        if body.persist and body.session_id:
            await agents.sessions.persist(
                project_id=auth.project_id,
                session_id=body.session_id,
                user_id=body.user_id,
                user_message=body.task,
                assistant_message=result.answer,
            )

    payload = result.to_payload()
    payload["prompt_examples"] = agents.feedback.prompt_examples(limit=2)
    return payload


@router.post("/ops/analyze")
async def ops_analyze(
    body: OpsRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """运营侧：经营分析 / 运营报告 / 客服质检（强制走 operations Agent）。"""
    auth.require_scope("llm:chat")
    agents = container.agents
    with request_scope(auth):
        tool_ctx = _tool_context(auth, session_id=None, user_id=None)
        async with _quota_scope(container, auth, body.task) as reservation:
            result = await agents.orchestrator.handle(
                body.task, tool_ctx, scenario=body.scenario, mode="sequential", domain="operations"
            )
            reservation.meta["actual_tokens"] = result.usage.total_tokens
    return result.to_payload()


# --------------------------------------------------------------------------- #
# HITL
# --------------------------------------------------------------------------- #
@router.post("/hitl/stage")
async def hitl_stage(
    body: HitlStageRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """登记一个高风险动作，返回检查点与 Resume Token（此时**尚未执行**）。"""
    auth.require_scope("llm:chat")
    with request_scope(auth, session_id=body.session_id, user_id=body.user_id):
        record = container.agents.hitl.stage(
            action=body.action,
            args=body.args,
            ctx=_tool_context(auth, session_id=body.session_id, user_id=body.user_id),
            reason=body.reason,
        )
    return {"success": True, "pending": record.to_payload(include_token=True), "trace_id": ctx.require().trace_id}


@router.post("/hitl/approve")
async def hitl_approve(
    body: HitlApproveRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """确认高风险动作：需要独立权限 `hitl:approve`，且 Resume Token 必须与检查点内容一致。"""
    with request_scope(auth):
        record = container.agents.hitl.approve(
            checkpoint_id=body.checkpoint_id,
            token=body.resume_token,
            ctx=_tool_context(auth, session_id=None, user_id=None),
            note=body.note,
        )
    return {"success": True, "pending": record.to_payload(), "trace_id": ctx.require().trace_id}


@router.post("/hitl/reject")
async def hitl_reject(
    body: HitlApplyRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    with request_scope(auth):
        record = container.agents.hitl.reject(
            checkpoint_id=body.checkpoint_id, ctx=_tool_context(auth, session_id=None, user_id=None)
        )
    return {"success": True, "pending": record.to_payload(), "trace_id": ctx.require().trace_id}


@router.post("/hitl/apply")
async def hitl_apply(
    body: HitlApplyRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """执行已确认的高风险动作：以**原始请求者身份**执行，下单归属不因审批而改变。"""
    auth.require_scope("hitl:approve")
    with request_scope(auth):
        payload = await container.agents.hitl.apply(
            checkpoint_id=body.checkpoint_id, ctx=_tool_context(auth, session_id=None, user_id=None)
        )
    return {"success": bool(payload.get("applied", {}).get("ok")), "pending": payload, "trace_id": ctx.require().trace_id}


@router.get("/hitl/{checkpoint_id}")
async def hitl_resume(
    checkpoint_id: str,
    resume_token: str = Query(min_length=1),
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """凭 Resume Token 找回同一件待办（会话断了也能续上）。"""
    with request_scope(auth):
        record = container.agents.hitl.resume(checkpoint_id=checkpoint_id, token=resume_token)
    return {"success": True, "pending": record.to_payload(), "trace_id": ctx.require().trace_id}


# --------------------------------------------------------------------------- #
# 实时语音（端口 + Mock）
# --------------------------------------------------------------------------- #
@router.post("/realtime/turn")
async def realtime_turn(
    body: RealtimeTurnRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """一轮实时语音：RTC→ASR→编排→LLM→TTS，返回带序号的事件流。"""
    auth.require_scope("llm:chat")
    if not body.text and not body.utterance:
        raise ValidationError("需要提供 text（文本轮）或 utterance（模拟语音轮）之一")
    agents = container.agents
    if body.utterance:
        agents.rtc.push(body.utterance)
    with request_scope(auth, session_id=body.session_id, user_id=body.user_id):
        tool_ctx = _tool_context(auth, session_id=body.session_id, user_id=body.user_id)
        async with _quota_scope(container, auth, body.text or body.utterance or ""):
            turn = await agents.realtime.handle_turn(tool_ctx, text=body.text)
    payload = turn.to_payload()
    payload["success"] = True
    payload["trace_id"] = ctx.require().trace_id
    return payload


# --------------------------------------------------------------------------- #
# 检索策略
# --------------------------------------------------------------------------- #
@router.post("/retrieval/rewrite")
async def retrieval_rewrite(
    body: RetrievalRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """Step-Back / HyDE 查询重写后的检索结果（与原始检索可对比）。"""
    auth.require_scope("tools:knowledge")
    with request_scope(auth):
        outcome = await container.agents.retrieval.retrieve(body.query, strategy=body.strategy, top_k=body.top_k)
    return {"success": True, "result": outcome.to_payload(), "trace_id": ctx.require().trace_id}


@router.post("/retrieval/compare")
async def retrieval_compare(
    body: RetrievalRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """同一问题跑 none / step_back / hyde 三种策略，给出召回对比。"""
    auth.require_scope("tools:knowledge")
    with request_scope(auth):
        report = await container.agents.retrieval.compare(body.query, top_k=body.top_k)
    return {"success": True, "report": report, "trace_id": ctx.require().trace_id}


# --------------------------------------------------------------------------- #
# 反馈闭环
# --------------------------------------------------------------------------- #
@router.post("/feedback")
async def feedback_record(
    body: FeedbackRequest,
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """记录一条反馈（纠偏 / 优质回复 / 失败案例），供 Prompt 与 RAG few-shot 消费。"""
    auth.require_scope("llm:chat")
    record = container.agents.feedback.record(
        kind=body.kind,
        question=body.question,
        answer=body.answer,
        corrected=body.corrected,
        agent=body.agent,
        scenario=body.scenario,
        tags=body.tags,
    )
    return {
        "success": True,
        "record": record.to_payload(),
        "stats": container.agents.feedback.stats(),
        "trace_id": ctx.require().trace_id,
    }


@router.get("/feedback/examples")
async def feedback_examples(
    agent: str | None = Query(default=None),
    auth: AuthContext = Depends(current_auth),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """查看反馈已沉淀出的两类 few-shot 产物（Prompt / RAG）。"""
    auth.require_scope("llm:chat")
    collector = container.agents.feedback
    return {
        "success": True,
        "prompt_examples": collector.prompt_examples(agent=agent),
        "rag_examples": collector.rag_examples(agent=agent),
        "stats": collector.stats(),
        "trace_id": ctx.require().trace_id,
    }


# --------------------------------------------------------------------------- #
# A2A（内部服务间调用）
# --------------------------------------------------------------------------- #
@a2a_router.post("/message:send")
async def a2a_send(
    body: A2ASendRequest,
    container: Container = Depends(get_container),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, Any]:
    """Agent 间消息通道：内部令牌鉴权 + session_id/trace_id 透传。

    注意：这里**没有业务 API Key** —— 调用方是内网服务，不是终端用户。
    未配置内部令牌时直接拒绝（不做「本地就放行」的例外）。
    """
    if not container.settings.resolved_internal_service_token():
        raise ForbiddenError("A2A 通道未启用：未配置内部服务令牌")
    if not container.agents.a2a.verify_internal(x_internal_token):
        raise AuthenticationError("内部服务令牌无效")
    payload = await container.agents.a2a.send(
        A2ARequest(
            task=body.task,
            session_id=body.session_id,
            agent=body.agent,
            scenario=body.scenario,
            mode=body.mode,
            project_id=body.project_id,
            user_id=body.user_id,
            service=body.service,
            trace_id=body.trace_id,
            scopes=build_scopes(body.scopes),
        )
    )
    return payload


__all__ = ["router", "a2a_router"]
