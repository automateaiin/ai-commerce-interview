"""结构化日志。

为什么必须是 JSON 而不是文本行：
「一次请求经过网关、MCP、RAG、模型，出问题时要把它们串起来」，
前提是每条日志都带同一组字段。JSON 行可以直接被日志系统按字段检索，
文本日志只能靠正则捞，排障成本完全不同。

字段规范（PROJECT.md 4.1）：ts / level / logger / trace_id / project_id / session_id / stage / duration_ms / status / event / msg
其中 trace_id、project_id、session_id 由 contextvar 自动注入，业务代码不需要手动传。

隐私要求：默认不记录完整用户隐私与密钥；调用方需显式调用 scrub() 清洗后再写。
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any, Mapping

from . import context as ctx

# 结构化日志允许透出的额外字段白名单，防止有人把整个请求体塞进 extra 造成泄漏
_ALLOWED_EXTRA = {
    "event",
    "stage",
    "duration_ms",
    "status",
    "method",
    "path",
    "http_status",
    "error_code",
    "project_id",
    "session_id",
    "user_id",
    "api_key_id",
    "tool",
    "tool_name",
    "provider",
    "model",
    "logical_model",
    "route",
    "attempt",
    "sleep_ms",
    "max_attempts",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "cache_hit",
    "quota_reserved",
    "quota_settled",
    "quota_backend",
    "steps",
    "iterations",
    "finish_reason",
    "top_k",
    "hits",
    "degraded",
    "reason",
    "latency_ms",
    "stages",
    "slowest_stage",
    "client_ip",
    "revision",
    "changed_keys",
    # ---- 权限链与审计（Service → Agent → Tool → Resource）----
    "service",
    "agent",
    "caller",
    "resource",
    "decision",
    "audit_id",
    # ---- 限流 ----
    "dimension",
    "limit",
    "remaining",
    "retry_after_s",
    "key_hash",
    # ---- 成本预算 ----
    "budget_scope",
    "budget_used",
    "budget_limit",
    "budget_ratio",
    "downgrade",
    "downgraded_to",
    # ---- Prompt 模板中心 ----
    "role",
    "scenario",
    "layers",
    # ---- 评测 ----
    "case_id",
    "cases",
    "passed",
    "failed",
    "score",
    "dimension_scores",
}

_SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[email]"),
    (re.compile(r"\b\d{16,19}\b"), "[card]"),
    (re.compile(r"\b\d{11}\b"), "[phone]"),
    (re.compile(r"(?i)\b(bearer|api[-_]?key|sk-)[\w\-.]{6,}"), "[secret]"),
]


def scrub(value: Any) -> Any:
    """对字符串做敏感信息脱敏；容器类型递归处理。

    日志里永远不应该出现完整手机号/邮箱/银行卡/密钥，
    需要排查时保留长度与摘要即可。
    """
    if isinstance(value, str):
        masked = value
        for pattern, replacement in _SENSITIVE_PATTERNS:
            masked = pattern.sub(replacement, masked)
        if len(masked) > 400:
            masked = masked[:400] + "...[truncated]"
        return masked
    if isinstance(value, Mapping):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # contextvar 自动注入，业务代码零感知
        current = ctx.current()
        if current is not None:
            payload["trace_id"] = current.trace_id
            if current.project_id:
                payload["project_id"] = current.project_id
            if current.session_id:
                payload["session_id"] = current.session_id
            if current.user_id:
                payload["user_id"] = current.user_id
            if current.api_key_id:
                payload["api_key_id"] = current.api_key_id

        for key, value in record.__dict__.items():
            if key in _ALLOWED_EXTRA and value is not None:
                payload[key] = scrub(value)

        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["stack"] = self.formatException(record.exc_info)[-2000:]

        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def setup_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """进程级日志初始化（幂等，可被 uvicorn 与脚本重复调用）。"""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level.upper())

    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.handlers = [handler]

    # uvicorn 自带 handler，统一收编，避免出现两套格式
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers = [handler]
        uv.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
