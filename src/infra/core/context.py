"""请求级上下文（contextvar）。

为什么用 contextvar 而不是层层传参：
1) trace_id / project_id / session_id 需要贯穿「桥接层 → 工具层 → Provider 层」，
   显式传参会污染每一个函数签名，而且漏传一个点链路就断了；
2) contextvar 天然随 asyncio 任务隔离，不同并发请求互不串号，比 threadlocal 更安全；
3) 日志/指标可以在任意深度自动取到当前请求的标识，不需要业务代码感知。

安全要点：project_id 只能由服务端凭证推导后写入（见 bridge/auth.py），
客户端传来的 project_id 一律丢弃，禁止写入本上下文。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True, slots=True)
class RequestContext:
    """一次请求的不可变身份上下文。"""

    trace_id: str
    project_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    api_key_id: str | None = None
    scopes: frozenset[str] = frozenset()
    service: str | None = None
    """业务线（由凭证推导，对应权限链第一级 Service）。"""
    agent: str | None = None
    """发起调用的 Agent 身份（对应权限链第二级 Agent），用于审计与按 Agent 聚合成本。"""

    def bind(self, **changes: object) -> "RequestContext":
        """派生一个新的上下文（原上下文不可变，避免并发请求互相污染）。"""
        return replace(self, **changes)  # type: ignore[arg-type]

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


_ctx: ContextVar[RequestContext | None] = ContextVar("infra_request_context", default=None)


def current() -> RequestContext | None:
    """取当前上下文；没有则返回 None（用于健康检查等无上下文场景）。"""
    return _ctx.get()


def require() -> RequestContext:
    """取当前上下文；不存在直接抛错，避免悄悄用错身份。"""
    ctx = _ctx.get()
    if ctx is None:
        raise RuntimeError("请求上下文未初始化，禁止在无 trace/project 上下文时执行下游调用")
    return ctx


def set_context(ctx: RequestContext) -> Token:
    return _ctx.set(ctx)


def reset_context(token: Token) -> None:
    _ctx.reset(token)


@contextmanager
def context_scope(**changes: object) -> Iterator[RequestContext]:
    """在局部范围内派生上下文（例如工具执行时补 session_id）。"""
    base = _ctx.get() or RequestContext(trace_id="unknown")
    token = _ctx.set(base.bind(**changes))
    try:
        yield _ctx.get()  # type: ignore[misc]
    finally:
        _ctx.reset(token)


def trace_id() -> str | None:
    ctx = _ctx.get()
    return ctx.trace_id if ctx else None


def project_id() -> str | None:
    ctx = _ctx.get()
    return ctx.project_id if ctx else None


def session_id() -> str | None:
    ctx = _ctx.get()
    return ctx.session_id if ctx else None


def user_id() -> str | None:
    ctx = _ctx.get()
    return ctx.user_id if ctx else None
