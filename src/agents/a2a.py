"""A2A（Agent-to-Agent）通信：``/message:send`` + session_id 透传。

什么时候真的需要 A2A？当「专家 Agent 不在本进程」时 —— 例如内容生产 Agent 部署在
另一个服务里。这时同进程的函数调用失效，必须有一条带鉴权、可透传上下文的服务间通道。

本实现给出的是**语义完整的离线等价物**：
- 同进程优先走直接调用（省一次网络往返，也少一个故障点）；
- 跨服务形态由 ``/v1/a2a/message:send`` 提供，用**内部共享令牌**做服务间鉴权，
  并透传 ``session_id`` 与 ``trace_id``（父链路）—— 否则接手 Agent 既记不住上下文，
  也无法在链路追踪里跟本请求串起来。

诚实边界：内部令牌只证明「调用方是受信内网服务」，未做逐服务的能力注册与最小权限下发；
调用方声明的 scopes 需由内网网关侧约束（见 PROJECT.md 第 8 章）。
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any, Iterable

from src.infra.core.logging import get_logger
from src.infra.core.tracing import new_trace_id
from src.infra.tools.base import ToolContext

from .orchestrator import Orchestrator

logger = get_logger("agents.a2a")


@dataclass(slots=True)
class A2ARequest:
    task: str
    session_id: str | None = None
    agent: str | None = None
    scenario: str | None = None
    mode: str | None = None
    project_id: str = "a2a"
    user_id: str | None = None
    service: str | None = None
    trace_id: str | None = None
    scopes: tuple[str, ...] = ()


class A2AService:
    def __init__(self, *, orchestrator: Orchestrator, internal_token: str) -> None:
        self._orchestrator = orchestrator
        self._token = internal_token

    def verify_internal(self, provided: str | None) -> bool:
        """常量时间比较，避免令牌比较本身成为计时侧信道。"""
        return bool(provided) and hmac.compare_digest(self._token, provided or "")

    async def send(self, request: A2ARequest) -> dict[str, Any]:
        """执行一次 agent 间调用：session_id 与 trace_id 原样透传。"""
        trace_id = request.trace_id or new_trace_id()
        ctx = ToolContext(
            trace_id=trace_id,
            project_id=request.project_id,
            session_id=request.session_id,
            user_id=request.user_id,
            scopes=frozenset(request.scopes),
            service=request.service or "a2a",
            agent=request.agent or "orchestrator",
        )
        result = await self._orchestrator.handle(
            request.task, ctx, scenario=request.scenario, mode=request.mode, domain=request.agent
        )
        payload = result.to_payload()
        payload["session_id"] = request.session_id  # 透传：接手方沿用同一会话
        payload["agent"] = request.agent
        logger.info(
            "agents.a2a.sent",
            extra={
                "event": "a2a",
                "reason": (request.agent or "auto"),
                "status": "ok" if result.success else "error",
            },
        )
        return payload


def build_scopes(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(sorted({value for value in (values or ()) if value}))


__all__ = ["A2AService", "A2ARequest", "build_scopes"]
