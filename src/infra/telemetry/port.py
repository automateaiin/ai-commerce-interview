"""TelemetryPort：可观测出口的统一抽象。

为什么要在 `core/tracing` 之上再包一层（面试追问「为什么不直接用 stage()」）：
- `core/tracing` 是**本进程的请求时间线**，解决「这次请求慢在哪一环」；
- TelemetryPort 解决的是「换掉后端」：今天写日志+进程内指标，明天接 OpenTelemetry / APM，
  业务代码不该跟着改。所以导出 span / event / counter 三种原语，由实现决定落到哪里。

默认实现 ConsoleTelemetry 把 span 委托给 `core/tracing.stage()`，
好处是阶段耗时既进请求时间线，又进 telemetry 出口，两者口径天然一致；
OTel 适配器（OpenTelemetryTelemetry）在依赖缺失时由 factory 显式降级，不静默。
"""

from __future__ import annotations

from typing import Any, ContextManager, Protocol

from ..core.logging import get_logger
from ..core.metrics import metrics
from ..core.tracing import Stage, stage

logger = get_logger("infra.telemetry")


class TelemetryPort(Protocol):
    """可观测出口。三种原语覆盖「一次调用 / 一个事件 / 一个计数」三类信号。"""

    backend: str

    def span(self, name: str, **attributes: Any) -> Any: ...

    def event(self, name: str, **fields: Any) -> None: ...

    def counter(self, kind: str, key: str, value: int = 1) -> None: ...

    def observe(self, kind: str, key: str, value_ms: float) -> None: ...


class ConsoleTelemetry:
    """默认实现：结构化日志 + 进程内指标 + 阶段时间线。零外部依赖，离线可用。"""

    backend = "console"

    def span(self, name: str, **attributes: Any) -> Stage:
        """返回与 `core/tracing.stage()` 同款的计时器（同时支持 with / async with）。

        直接复用 Stage 而不是另起一套：否则同一阶段会出现两条互相对不上的耗时数据。
        """
        return stage(name, **attributes)

    def event(self, name: str, **fields: Any) -> None:
        logger.info(name, extra={"event": "telemetry", "stage": name, **fields})

    def counter(self, kind: str, key: str, value: int = 1) -> None:
        metrics.incr(kind, key, value)

    def observe(self, kind: str, key: str, value_ms: float) -> None:
        metrics.observe_latency(kind, key, value_ms)


class _OtelSpan:
    """OTel span + 本地 Stage 的双写包装：既能进 APM，也不丢本进程时间线。"""

    def __init__(self, tracer: Any, name: str, attributes: dict[str, Any], local: Stage) -> None:
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._local = local
        self._span: Any = None
        self._scope: ContextManager[Any] | None = None

    def __enter__(self) -> "_OtelSpan":
        self._local.__enter__()
        self._span = self._tracer.start_span(self._name, attributes=self._attributes)
        self._scope = self._tracer.use_span(self._span, end_on_exit=True)
        self._scope.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._scope is not None:
            self._scope.__exit__(exc_type, exc, tb)
        self._local.__exit__(exc_type, exc, tb)
        return False

    async def __aenter__(self) -> "_OtelSpan":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)


class OpenTelemetryTelemetry:
    """OpenTelemetry 适配器（可选后端）。

    本仓库默认不安装 OTel 依赖（PROJECT.md 第 8 章：重依赖走端口 + 离线等价实现），
    因此本类只在依赖存在时可用；factory 会用 `probe()` 探测并在失败时降级到 Console。
    """

    backend = "otel"

    def __init__(self, service_name: str) -> None:
        from opentelemetry import trace  # 延迟导入：未安装时由 factory 提前拦截

        self._trace = trace
        self._tracer = trace.get_tracer(service_name)

    @staticmethod
    def probe() -> bool:
        try:
            import opentelemetry.trace  # noqa: F401
        except Exception:  # noqa: BLE001 - 依赖探测失败属于正常分支
            return False
        return True

    def span(self, name: str, **attributes: Any) -> _OtelSpan:
        return _OtelSpan(self._tracer, name, attributes, stage(name, **attributes))

    def event(self, name: str, **fields: Any) -> None:
        logger.info(name, extra={"event": "telemetry", "stage": name, **fields})

    def counter(self, kind: str, key: str, value: int = 1) -> None:
        metrics.incr(kind, key, value)

    def observe(self, kind: str, key: str, value_ms: float) -> None:
        metrics.observe_latency(kind, key, value_ms)


def build_telemetry(settings: Any) -> TelemetryPort:
    """按配置装配 telemetry 出口；后端不可用时**显式降级**并记录原因。"""
    backend = (getattr(settings, "telemetry_backend", "console") or "console").lower()
    if backend == "otel":
        if OpenTelemetryTelemetry.probe():
            logger.info(
                "telemetry.ready",
                extra={"event": "startup", "reason": "otel_adapter", "degraded": False},
            )
            return OpenTelemetryTelemetry(getattr(settings, "app_name", "ecom-ai-infra"))
        logger.warning(
            "telemetry.degraded",
            extra={
                "event": "startup",
                "status": "degraded",
                "degraded": True,
                "reason": "otel_dependency_missing",
            },
        )
    return ConsoleTelemetry()


__all__ = [
    "TelemetryPort",
    "ConsoleTelemetry",
    "OpenTelemetryTelemetry",
    "build_telemetry",
]
