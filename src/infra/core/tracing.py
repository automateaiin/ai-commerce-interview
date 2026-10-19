"""链路追踪：trace_id 生成/透传 + 阶段耗时打点。

三条设计要点（面试追问「怎么定位慢在哪一环」的答案来源）：
1) trace_id 在入口生成（或沿用上游 header），写入 contextvar，并通过 HTTP header 透传下游；
2) 每一次阶段（鉴权/配额/检索/模型/工具）都记录 name+duration_ms+status，
   聚合成一条时间线，出问题时先看哪个 stage 占比最高，再往下钻；
3) trace 只做请求关联，绝不作为身份凭证或授权凭据。
"""

from __future__ import annotations

import secrets
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Iterator

from . import context as ctx
from .logging import get_logger

logger = get_logger("infra.trace")

TRACE_HEADER = "x-trace-id"

# 一次请求内所有 stage 的 (name, duration_ms, status)，用于定位瓶颈
_timeline: ContextVar[list[tuple[str, float, str]] | None] = ContextVar("infra_stage_timeline", default=None)


def new_trace_id() -> str:
    """生成 trace_id。用随机串而非自增 ID，避免暴露调用量并降低碰撞概率。"""
    return "tr_" + secrets.token_hex(8)


def normalize_trace_id(value: str | None) -> str:
    """沿用上游 trace_id 时做一次格式校验，避免把脏数据写进日志与下游 header。"""
    if value and value.isascii() and 0 < len(value) <= 64 and value.replace("_", "").replace("-", "").isalnum():
        return value
    return new_trace_id()


def start_timeline() -> None:
    _timeline.set([])


def snapshot_timeline() -> list[dict[str, Any]]:
    """返回本请求的阶段耗时快照，供访问日志 / 指标 / 排查使用。"""
    return [
        {"stage": name, "duration_ms": round(duration, 2), "status": status}
        for name, duration, status in (_timeline.get() or [])
    ]


def slowest_stage() -> dict[str, Any] | None:
    stages = snapshot_timeline()
    if not stages:
        return None
    return max(stages, key=lambda item: item["duration_ms"])


class Stage:
    """阶段计时器，同时支持 `async with` 与 `with`。

    用法：
        async with Stage("llm.complete", model=model):
            ...
    退出时自动写一条结构化日志（event=stage），失败时记录 error_code 并标记 status=error。
    """

    def __init__(self, name: str, **fields: Any) -> None:
        self.name = name
        self.fields = fields
        self._started = 0.0
        self._status = "ok"
        self._error: str | None = None

    # -- 内部 -------------------------------------------------------------- #
    def _enter(self) -> None:
        self._started = time.perf_counter()

    def _exit(self, exc: BaseException | None) -> None:
        duration_ms = (time.perf_counter() - self._started) * 1000
        if exc is not None:
            self._status = "error"
            self._error = getattr(exc, "code", None) or type(exc).__name__
        line = _timeline.get()
        if line is not None:
            line.append((self.name, duration_ms, self._status))
        logger.info(
            self.name,
            extra={
                "event": "stage",
                "stage": self.name,
                "duration_ms": round(duration_ms, 2),
                "status": self._status,
                **({"error_code": self._error} if self._error else {}),
                **self.fields,
            },
            # 只在失败时记录栈，正常路径不写栈以免日志膨胀
            exc_info=exc,
        )

    def __enter__(self) -> "Stage":
        self._enter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._exit(exc)
        return False

    async def __aenter__(self) -> "Stage":
        self._enter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._exit(exc)
        return False


def stage(name: str, **fields: Any) -> Stage:
    return Stage(name, **fields)


@contextmanager
def sync_stage(name: str, **fields: Any) -> Iterator[Stage]:
    with Stage(name, **fields) as current:
        yield current


@asynccontextmanager
async def traced_request(trace_id: str, **fields: Any) -> AsyncIterator[ctx.RequestContext]:
    """建立一次请求的 trace 上下文边界（入口/测试中使用）。"""
    start_timeline()
    token = ctx.set_context(ctx.RequestContext(trace_id=trace_id, **fields))
    try:
        yield ctx.require()
    finally:
        ctx.reset_context(token)
