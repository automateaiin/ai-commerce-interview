"""可观测性：结构化日志、脱敏、链路追踪、配置快照、指标。

工程化里最容易被问穿的一环。这些用例守住三条底线：
1) 日志字段可被机器检索（不是给人看的文本）；
2) 隐私与密钥不出现在日志里；
3) 坏配置永远不会生效。
"""

from __future__ import annotations

import logging

import pytest

from src.infra.core import context as ctx
from src.infra.core.config import ConfigSnapshot, expand_env
from src.infra.core.errors import ValidationError
from src.infra.core.logging import JsonFormatter, scrub
from src.infra.core.metrics import MetricsRegistry
from src.infra.core.tracing import (
    new_trace_id,
    normalize_trace_id,
    slowest_stage,
    snapshot_timeline,
    stage,
    start_timeline,
    traced_request,
)


# -- 脱敏 ------------------------------------------------------------------- #


def test_scrub_masks_隐私与密钥():
    text = "邮箱 a@b.com 手机 13800138000 卡号 4111111111111111 密钥 sk-abcdef123456"
    masked = scrub(text)

    for raw in ("a@b.com", "13800138000", "4111111111111111", "sk-abcdef123456"):
        assert raw not in masked
    for placeholder in ("[email]", "[phone]", "[card]", "[secret]"):
        assert placeholder in masked


def test_scrub_handles_containers_and_truncates_long_values():
    masked = scrub({"notes": ["联系 u1@example.com"], "blob": "x" * 500})
    # 递归清洗：敏感信息藏在嵌套结构里也要被处理
    assert "[email]" in masked["notes"][0]
    # 长文本截断：避免一次日志把整段请求体写进去
    assert masked["blob"].endswith("...[truncated]")
    assert len(masked["blob"]) < 500


def test_scrub_leaves_plain_values_untouched():
    assert scrub(123) == 123
    assert scrub(None) is None
    assert scrub("普通文本") == "普通文本"


# -- 结构化日志 -------------------------------------------------------------- #


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("infra.test", logging.INFO, __file__, 1, "evt", (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_injects_context_and_whitelists_extras():
    formatter = JsonFormatter()
    token = ctx.set_context(
        ctx.RequestContext(
            trace_id="tr_pytest_log",
            project_id="cs-bot",
            session_id="s1",
            user_id="u1",
            api_key_id="cs-bot",
        )
    )
    try:
        payload = formatter.format(_record(event="tool_call", tool="search_products", secret_field="不该出现"))
    finally:
        ctx.reset_context(token)

    assert payload.startswith("{")
    # 身份字段由 contextvar 自动注入，业务代码不需要手写
    for key, value in (
        ("trace_id", "tr_pytest_log"),
        ("project_id", "cs-bot"),
        ("session_id", "s1"),
        ("user_id", "u1"),
        ("api_key_id", "cs-bot"),
    ):
        assert f'"{key}": "{value}"' in payload
    # 白名单之外的字段一律丢弃：防止有人把整个请求体塞进 extra
    assert "secret_field" not in payload
    assert '"tool": "search_products"' in payload


def test_json_formatter_scrubs_values_before_writing():
    payload = JsonFormatter().format(_record(event="tool", tool="13800138000"))
    assert "13800138000" not in payload and "[phone]" in payload

    payload = JsonFormatter().format(_record(event="auth", reason="login by user_10086@example.com"))
    assert "@example.com" not in payload and "[email]" in payload


# -- 链路追踪 ---------------------------------------------------------------- #


def test_trace_id_is_validated_not_trusted():
    assert normalize_trace_id("tr_abc-123") == "tr_abc-123"
    assert normalize_trace_id("上游带了中文") .startswith("tr_")  # 脏数据不进日志与下游
    assert normalize_trace_id("x" * 200).startswith("tr_")
    assert new_trace_id().startswith("tr_")


def test_stage_timeline_locates_the_slowest_step():
    start_timeline()
    with stage("auth") as _:
        pass
    with stage("quota") as _:
        pass

    snapshot = snapshot_timeline()
    assert [item["stage"] for item in snapshot] == ["auth", "quota"]
    assert all(item["status"] == "ok" for item in snapshot)
    assert slowest_stage()["stage"] in {"auth", "quota"}


async def test_async_stage_is_recorded_and_marks_failure():
    start_timeline()
    async with stage("llm.complete", model="chat-lite"):
        pass

    with pytest.raises(ValidationError):
        with stage("tool.execute", tool="search_products"):
            raise ValidationError("参数不合法")

    snapshot = {item["stage"]: item for item in snapshot_timeline()}
    assert snapshot["llm.complete"]["status"] == "ok"
    # 失败阶段必须被标记：否则时间线会掩盖真正的错误环节
    assert snapshot["tool.execute"]["status"] == "error"


def test_stage_without_timeline_does_not_break():
    """没有显式开时间线时，打点不应该抛错（健康检查等无 trace 场景）。"""
    import contextvars

    def run() -> list[dict]:
        with stage("orphan"):
            pass
        return snapshot_timeline()

    # 空上下文里跑：既验证不抛错，也验证「没开时间线就不该记录」
    assert contextvars.Context().run(run) == []


async def test_traced_request_sets_context_and_timeline():
    async with traced_request("tr_pytest_traced", project_id="cs-bot") as current:
        assert current.trace_id == "tr_pytest_traced"
        assert ctx.require().project_id == "cs-bot"
        async with stage("rag.search"):
            pass
        assert [item["stage"] for item in snapshot_timeline()] == ["rag.search"]
    # 退出后上下文必须被还原，避免污染下一个请求
    assert ctx.current() is None


# -- 配置快照 ---------------------------------------------------------------- #


def test_expand_env_requires_explicit_default():
    assert expand_env("${PYTEST_MISSING_VAR:-fallback}") == "fallback"
    # 没写默认值就是配置错误：宁可启动失败，也不要在运行时炸出 None
    with pytest.raises(ValidationError):
        expand_env("${PYTEST_MISSING_VAR}")

    nested = expand_env({"a": ["${PYTEST_MISSING_VAR:-x}"], "b": {"c": "${PYTEST_MISSING_VAR:-y}"}})
    assert nested == {"a": ["x"], "b": {"c": "y"}}


def test_expand_env_prefers_real_environment(monkeypatch):
    monkeypatch.setenv("PYTEST_REAL_VAR", "from-env")
    assert expand_env("${PYTEST_REAL_VAR:-fallback}") == "from-env"


def test_bad_config_never_replaces_a_working_snapshot():
    def validator(value: dict) -> None:
        if "base_url" not in value:
            raise ValidationError("缺少 base_url")

    snapshot: ConfigSnapshot[dict] = ConfigSnapshot({"base_url": "https://ok"}, validator=validator, name="pytest")

    with pytest.raises(ValidationError):
        snapshot.swap({"base_url_missing": True})

    # 坏配置不生效：旧配置仍可用，版本号不变
    assert snapshot.current() == {"base_url": "https://ok"}
    assert snapshot.revision == 1

    assert snapshot.swap({"base_url": "https://new"}) == 2
    assert snapshot.current() == {"base_url": "https://new"}


# -- 指标 -------------------------------------------------------------------- #


def test_metrics_registry_aggregates_counts_latency_and_cost():
    registry = MetricsRegistry()
    registry.incr("request", "POST /v1/tools")
    registry.incr("request", "POST /v1/tools")
    registry.incr("error", "POST /v1/tools")
    for value in (10.0, 20.0, 30.0, 40.0):
        registry.observe_latency("request", "POST /v1/tools", value)
    registry.observe_tokens("mock", "chat-lite", 100, 20, cost_usd=0.0003)

    snapshot = registry.snapshot()
    assert snapshot["requests_total"] == 2
    assert snapshot["errors_total"] == 1
    stats = snapshot["latency"]["request:POST /v1/tools"]
    assert stats["count"] == 4 and stats["p50_ms"] == 20.0 and stats["max_ms"] == 40.0
    assert snapshot["tokens"]["prompt:mock/chat-lite"] == 100
    assert snapshot["tokens"]["completion:mock/chat-lite"] == 20
    assert snapshot["cost_usd"]["mock/chat-lite"] > 0
    assert snapshot["error_rate"] == 0.5
