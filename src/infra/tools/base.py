"""工具定义、注册表与执行器。

三条边界（面试高频）：
1) 工具 Schema 是「给模型看的契约」：name / description / input_schema 直接决定模型调用准确率；
2) 服务端必须**二次校验**参数，不能只信任模型生成的参数（模型输出不是可信命令）；
3) 执行器把任何失败归一成**结构化 ToolResult**，而不是抛异常穿透到 Agent ——
   因为工具失败本身就是下一步决策所需的 observation（模型可以补参、换工具、向用户澄清），
   但权限错误这类不可恢复的失败要标记 terminal，由上层终止循环。

只读工具与写操作在元数据上就分开：risk / idempotent / parallel_safe。
写操作默认不重试、不并行，并要求幂等键。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..core.audit import AuditLog, derive_resource
from ..core.errors import ForbiddenError, InfraError, NotFoundError, normalize_exception
from ..core.logging import get_logger
from ..core.metrics import metrics
from ..core.retry import DEFAULT_POLICY, RetryPolicy, retry_async
from ..core.tracing import stage

logger = get_logger("infra.tools")


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    HIGH_RISK = "high_risk"


@dataclass(frozen=True, slots=True)
class ToolContext:
    """工具执行上下文。

    身份信息来自服务端凭证推导（bridge/auth），**不接受客户端传入的 project_id**。

    `service` / `agent` 撑起权限链的前两级（Service → Agent → Tool → Resource）：
    `service` 是业务线（由凭证推导），`agent` 是发起调用的 Agent 身份（由调用方声明）。
    它们不参与鉴权（鉴权只看 scopes），只用于**审计与成本分摊** —— 混用两者会让
    「谁有权」和「谁在用」变成同一个问题，权限模型就没法独立演进了。
    """

    trace_id: str
    project_id: str
    session_id: str | None = None
    user_id: str | None = None
    scopes: frozenset[str] = frozenset()
    service: str | None = None
    agent: str | None = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


@dataclass(slots=True)
class ToolResult:
    """工具调用结果。ok=False 时也必须结构化，供模型/上层决策。"""

    tool: str
    ok: bool
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    terminal: bool = False
    """不可恢复失败（越权、业务规则拒绝），上层应终止而非继续重试。"""
    duration_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> dict[str, Any]:
        """转换成给模型看的 observation：只含安全摘要，不含堆栈与内部细节。"""
        if self.ok:
            return {"tool": self.tool, "status": "ok", "data": self.data}
        return {
            "tool": self.tool,
            "status": "error",
            "error_code": self.error_code or "tool_error",
            "message": self.error_message or "工具执行失败",
            "retryable": self.retryable,
        }


class Tool(Protocol):
    """工具契约。实现类需要提供下面这些元数据 + run 方法。"""

    name: str
    description: str
    args_model: type[BaseModel]
    requires_scope: str | None
    risk: ToolRisk
    timeout_s: float
    idempotent: bool
    parallel_safe: bool

    async def run(self, args: BaseModel, ctx: ToolContext) -> Any: ...


class FunctionTool:
    """函数式工具。用装饰器/直接构造的方式注册，避免为每个工具写一个类。"""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[[BaseModel, ToolContext], Awaitable[Any]],
        requires_scope: str | None = None,
        risk: ToolRisk = ToolRisk.READ_ONLY,
        timeout_s: float = 8.0,
        idempotent: bool = True,
        parallel_safe: bool = True,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.args_model = args_model
        self.handler = handler
        self.requires_scope = requires_scope
        self.risk = risk
        self.timeout_s = timeout_s
        self.idempotent = idempotent
        self.parallel_safe = parallel_safe
        self.retry_policy = retry_policy or DEFAULT_POLICY

    async def run(self, args: BaseModel, ctx: ToolContext) -> Any:
        return await self.handler(args, ctx)

    def openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI function-calling 的 tools 定义。

        docstring/描述写清楚，模型才知道什么时候该调这个工具 ——
        工具描述的模糊会直接变成「模型频繁调用不存在的工具」。
        """
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    """工具注册表。负责注册、查找、导出 Schema 与权限元数据。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool
        logger.info("tool.registered", extra={"event": "tool_registered", "tool": tool.name})
        return tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise NotFoundError(f"工具不存在: {name}", detail={"tool": name, "available": sorted(self._tools)})
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> Sequence[Tool]:
        return list(self._tools.values())

    def schemas(self, *, only_scopes: frozenset[str] | None = None) -> list[dict[str, Any]]:
        """导出可用工具的 Schema（可按调用方权限过滤，越权工具直接不暴露）。"""
        return [
            tool.openai_schema()  # type: ignore[attr-defined]
            for tool in self._tools.values()
            if only_scopes is None or (tool.requires_scope is None or tool.requires_scope in only_scopes)
        ]


class ToolExecutor:
    """统一执行入口：参数校验 → 权限校验 → 超时/重试 → 结果归一 → 审计留痕。"""

    def __init__(self, registry: ToolRegistry, *, max_concurrency: int = 8, audit: AuditLog | None = None) -> None:
        self._registry = registry
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._audit = audit

    def _audit_record(
        self,
        *,
        ctx: ToolContext,
        tool: str,
        raw_args: Mapping[str, Any] | None,
        decision: str,
        reason: str | None = None,
        risk: str | None = None,
        duration_ms: float = 0.0,
        error_code: str | None = None,
    ) -> None:
        """审计四级权限链。拒绝与失败同样留痕：只看成功调用，越权尝试会完全隐形。"""
        if self._audit is None:
            return
        self._audit.record(
            tool=tool,
            resource=derive_resource(tool, raw_args),
            decision=decision,
            service=ctx.service or ctx.project_id,
            agent=ctx.agent,
            caller=ctx.project_id,
            project_id=ctx.project_id,
            trace_id=ctx.trace_id,
            reason=reason,
            risk=risk,
            duration_ms=duration_ms,
            error_code=error_code,
        )

    async def execute(
        self,
        name: str,
        raw_args: Mapping[str, Any] | None,
        ctx: ToolContext,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            tool = self._registry.get(name)
        except NotFoundError as exc:
            # 未知工具是「模型可修正」的错误：返回可恢复 observation，而不是直接中断
            logger.warning(
                "tool.unknown", extra={"event": "tool_unknown", "tool": name, "error_code": exc.code}
            )
            self._audit_record(
                ctx=ctx, tool=name, raw_args=raw_args, decision="error",
                reason="unknown_tool", error_code=exc.code,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return ToolResult(tool=name, ok=False, error_code=exc.code, error_message=exc.message, retryable=False)

        # 1) 二次校验参数：模型给的 JSON 不是可信输入
        try:
            args = tool.args_model.model_validate(dict(raw_args or {}))
        except PydanticValidationError as exc:
            issues = [
                {"field": ".".join(str(p) for p in err["loc"]), "msg": err["msg"]} for err in exc.errors()[:5]
            ]
            self._audit_record(
                ctx=ctx, tool=name, raw_args=raw_args, decision="error", reason="invalid_arguments",
                risk=tool.risk.value, error_code="invalid_arguments",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return ToolResult(
                tool=name,
                ok=False,
                error_code="invalid_arguments",
                error_message="参数校验失败",
                retryable=False,
                meta={"issues": issues},
            )

        # 2) 权限校验：鉴权（你是谁）与授权（你能调什么）是两件事
        if tool.requires_scope and not ctx.has_scope(tool.requires_scope):
            logger.warning(
                "tool.forbidden",
                extra={"event": "tool_forbidden", "tool": name, "error_code": "permission_denied"},
            )
            self._audit_record(
                ctx=ctx, tool=name, raw_args=raw_args, decision="deny",
                reason=f"missing_scope:{tool.requires_scope}", risk=tool.risk.value,
                error_code="permission_denied", duration_ms=(time.perf_counter() - started) * 1000,
            )
            return ToolResult(
                tool=name,
                ok=False,
                error_code="permission_denied",
                error_message=f"缺少 scope: {tool.requires_scope}",
                retryable=False,
                terminal=True,
            )

        async def _invoke(attempt: int) -> Any:
            async with asyncio.timeout(tool.timeout_s):
                return await tool.run(args, ctx)

        async with stage("tool.execute", tool=name, risk=tool.risk.value):
            try:
                async with self._semaphore:
                    data = await retry_async(
                        _invoke,
                        name=f"tool.{name}",
                        policy=tool.retry_policy,
                        idempotent=tool.idempotent,
                    )
            except InfraError as exc:
                terminal = isinstance(exc, ForbiddenError)
                metrics.incr("tool_error", f"{name}:{exc.code}")
                logger.warning(
                    "tool.failed",
                    extra={"event": "tool_call", "tool": name, "error_code": exc.code, "status": "error"},
                )
                self._audit_record(
                    ctx=ctx, tool=name, raw_args=raw_args, decision="error", reason=exc.code,
                    risk=tool.risk.value, error_code=exc.code,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                return ToolResult(
                    tool=name,
                    ok=False,
                    error_code=exc.code,
                    error_message=exc.message,
                    retryable=exc.retryable,
                    terminal=terminal,
                    meta=exc.detail,
                )

        duration_ms = (time.perf_counter() - started) * 1000
        self._audit_record(
            ctx=ctx, tool=name, raw_args=raw_args, decision="allow", risk=tool.risk.value, duration_ms=duration_ms
        )
        metrics.incr("tool_call", name)
        return ToolResult(tool=name, ok=True, data=data, duration_ms=duration_ms)

    async def execute_many(
        self,
        calls: Sequence[tuple[str, str, Mapping[str, Any]]],
        ctx: ToolContext,
    ) -> list[ToolResult]:
        """批量执行。

        calls: [(call_id, tool_name, args), ...]

        并行前提：全部为只读工具（parallel_safe 且 risk=read_only）。
        只要出现写操作就退化为串行 —— 因为写操作之间有顺序依赖与幂等约束。
        """
        if not calls:
            return []
        tools = [self._registry._tools.get(name) for _, name, _ in calls]  # noqa: SLF001
        can_parallel = all(
            tool is not None and tool.parallel_safe and tool.risk is ToolRisk.READ_ONLY for tool in tools
        )
        if can_parallel:
            return list(await asyncio.gather(*(self.execute(name, args, ctx) for _, name, args in calls)))
        results: list[ToolResult] = []
        for _, name, args in calls:
            result = await self.execute(name, args, ctx)
            results.append(result)
            if result.terminal:
                break  # 越权/业务拒绝：立即停止，不再继续执行后续工具
        return results
