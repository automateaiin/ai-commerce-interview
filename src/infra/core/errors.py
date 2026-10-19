"""统一异常体系。

设计目标（面试要能讲清楚的三点）：
1) 业务代码只抛语义化异常（如 QuotaExceededError），由统一处理器映射成 HTTP 状态与响应体，
   路由函数里不再出现 try/except 拼装 Response。
2) 每个异常自带 retryable 标记 —— 这是重试层能「只重试可重试错误」的前提。
   错误不归一，上层就没办法自适应重试或切模型。
3) 异常分两层信息：safe_message 可以暴露给调用方/模型；detail 只进服务端日志。
   绝不把堆栈、密钥、SQL 返回给外部。
"""

from __future__ import annotations

from typing import Any


class InfraError(Exception):
    """所有基础设施/业务异常的基类。"""

    code: str = "infra_error"
    http_status: int = 500
    retryable: bool = False
    safe_message: str = "服务内部错误"

    def __init__(self, message: str | None = None, *, detail: dict[str, Any] | None = None) -> None:
        self.message = message or self.safe_message
        self.detail: dict[str, Any] = detail or {}
        super().__init__(self.message)

    def to_payload(self, trace_id: str | None = None) -> dict[str, Any]:
        """对外统一错误体。只含安全信息。"""
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "trace_id": trace_id,
        }


# --------------------------------------------------------------------------- #
# 4xx：客户端/调用方可修正，不重试
# --------------------------------------------------------------------------- #
class ValidationError(InfraError):
    code = "invalid_argument"
    http_status = 400
    safe_message = "请求参数不合法"


class AuthenticationError(InfraError):
    code = "unauthenticated"
    http_status = 401
    safe_message = "凭证无效"


class ForbiddenError(InfraError):
    code = "permission_denied"
    http_status = 403
    safe_message = "无权访问该资源"


class NotFoundError(InfraError):
    code = "not_found"
    http_status = 404
    safe_message = "资源不存在"


class QuotaExceededError(InfraError):
    code = "quota_exceeded"
    http_status = 429
    safe_message = "调用配额不足"

    def __init__(self, message: str | None = None, *, detail=None, retry_after: int = 60) -> None:
        super().__init__(message, detail=detail)
        self.retry_after = retry_after


class RateLimitedError(InfraError):
    """限流拒绝（滑动窗口），与「配额超限」是两件事：

    - 配额（quota_exceeded）：窗口内**总量/计费**用尽，需要业务侧提额或等下一窗口；
    - 限流（rate_limited）：**瞬时速率**超阈值，通常退避几秒即可重试。
    两者的 error code 必须分开，否则监控上「谁在刷接口」会被误读成「额度用完了」。
    """

    code = "rate_limited"
    http_status = 429
    retryable = True
    safe_message = "请求过于频繁"

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: dict[str, Any] | None = None,
        retry_after: int = 1,
        dimension: str = "unknown",
    ) -> None:
        super().__init__(message, detail={**(detail or {}), "dimension": dimension})
        self.retry_after = retry_after
        self.dimension = dimension


# --------------------------------------------------------------------------- #
# 5xx：下游故障。区分「暂时性」（可重试）与「确定性」（不可原样重试）
# --------------------------------------------------------------------------- #
class UpstreamError(InfraError):
    code = "upstream_error"
    http_status = 502
    safe_message = "下游服务异常"
    # 5xx 默认按**暂态故障**处理：连接被拒、实例滚动、后端过载都会自愈，重试才有意义。
    # 是否真的重试还有第二道闸门 —— 重试层会再校验操作是否幂等（见 retry.py 的 idempotent），
    # 所以「写操作不重试」由重试层保证，不靠错误类型来保证。
    retryable = True


class UpstreamTimeoutError(UpstreamError):
    code = "upstream_timeout"
    http_status = 504
    retryable = True
    safe_message = "下游服务超时"


class UpstreamRateLimitError(UpstreamError):
    code = "upstream_rate_limited"
    http_status = 429
    retryable = True
    safe_message = "下游服务限流"

    def __init__(self, message: str | None = None, *, detail=None, retry_after_s: float | None = None) -> None:
        super().__init__(message, detail=detail)
        # 服务端返回的 Retry-After 优先于本地退避曲线
        self.retry_after_s = retry_after_s


class ProviderError(UpstreamError):
    """模型供应商返回的确定性错误（如模型不支持 tools）：可换模型，不可原样重试。"""

    code = "provider_error"
    retryable = False


class InternalError(InfraError):
    code = "internal_error"
    http_status = 500
    safe_message = "服务内部错误"


def normalize_exception(exc: BaseException) -> InfraError:
    """把任意异常归一成 InfraError，供统一处理器使用。

    未知异常一律折叠成 InternalError：对外只暴露安全摘要，
    原始异常由日志层用 exc_info 记录，避免细节泄漏。
    """
    if isinstance(exc, InfraError):
        return exc
    if isinstance(exc, TimeoutError):  # asyncio / socket 超时
        return UpstreamTimeoutError(detail={"origin": type(exc).__name__})
    return InternalError(detail={"origin": type(exc).__name__})
