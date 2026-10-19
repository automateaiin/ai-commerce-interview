"""LLM 适配层：统一出入参、统一错误类型、逻辑路由、能力声明。

这一层存在的意义只有一个：**让上层业务永远不需要知道供应商是谁**。
所以断言重点放在「归一化」上 —— 响应结构归一到 LLMResponse、错误归一到 InfraError 分类树、
逻辑名归一到物理端点。供应商私有字段永远不进上层。
"""

from __future__ import annotations

import pytest

from src.infra.core.errors import (
    AuthenticationError,
    ProviderError,
    UpstreamError,
    UpstreamRateLimitError,
    ValidationError,
)
from src.infra.llm.gateway import LLMGateway, estimate_messages_tokens, estimate_tokens
from src.infra.llm.mock import final
from src.infra.llm.openai_compat import OpenAICompatibleProvider
from src.infra.llm.provider import UnsupportedCapabilityError
from src.infra.llm.types import ToolCall


def test_tool_call_arguments_are_parsed_leniently_but_not_guessed():
    parsed = OpenAICompatibleProvider._parse_tool_calls(
        [
            {"id": "c1", "function": {"name": "search_products", "arguments": '{"query": "精华"}'}},
            # 非合法 JSON：保留原文交给上层报可恢复错误，绝不猜参数
            {"id": "c2", "function": {"name": "get_order_status", "arguments": '{"order_id": '}},
        ]
    )
    assert parsed[0].args == {"query": "精华"} and parsed[0].raw_arguments is None
    assert parsed[1].args == {}
    assert parsed[1].raw_arguments == '{"order_id": '


def test_http_status_is_mapped_into_a_typed_error_tree():
    rate_limited = OpenAICompatibleProvider._map_http_error(429, "slow down", model="m")
    assert isinstance(rate_limited, UpstreamRateLimitError)
    assert rate_limited.retryable is True and rate_limited.http_status == 429

    # 上游 401 对本服务是凭证配置问题：确定性错误，重试无意义
    auth_failed = OpenAICompatibleProvider._map_http_error(401, "unauthorized", model="m")
    assert isinstance(auth_failed, ProviderError) and auth_failed.retryable is False

    bad_request = OpenAICompatibleProvider._map_http_error(400, "bad", model="m")
    assert bad_request.retryable is False

    server_error = OpenAICompatibleProvider._map_http_error(503, "oops", model="m")
    assert isinstance(server_error, UpstreamError) and server_error.retryable is True

    # 错误详情必须可诊断，但不能把整段响应体塞进日志
    assert server_error.detail["http_status"] == 503
    assert len(server_error.detail["body"]) <= 300


async def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("PYTEST_ABSENT_KEY", raising=False)
    provider = OpenAICompatibleProvider(default_api_key_env="PYTEST_ABSENT_KEY")

    with pytest.raises(AuthenticationError):
        await provider.complete([{"role": "user", "content": "hi"}], "m", base_url="https://example.invalid")


async def test_missing_base_url_is_a_config_error():
    provider = OpenAICompatibleProvider(default_api_key_env="PATH")  # 用已存在的环境变量绕过密钥检查
    with pytest.raises(ValidationError):
        await provider.complete([{"role": "user", "content": "hi"}], "m")


@pytest.fixture
def gateway() -> LLMGateway:
    from src.infra.core.settings import Settings
    from src.infra.llm.factory import build_catalog_snapshot, build_providers

    settings = Settings()
    snapshot = build_catalog_snapshot(settings)
    return LLMGateway(snapshot, build_providers(snapshot.current(), settings))


def test_task_type_routes_to_the_explicit_override(gateway: LLMGateway):
    # 业务只说「这是复杂推理」，不该关心要接哪个物理模型
    assert gateway.resolve(task="complex_reasoning").logical_model == "chat-pro"
    assert gateway.resolve(task="product_consulting").logical_model == "chat-pro"
    # 图像理解必须落到具备图像能力的逻辑模型
    assert gateway.resolve(task="image_understanding").spec.capabilities
    # 未声明的任务类型回落到 routing.default
    assert gateway.resolve(task="something_unknown").logical_model == "chat-lite"
    assert gateway.resolve().logical_model == "chat-lite"


def test_explicit_logical_model_wins_over_task_type(gateway: LLMGateway):
    # 显式指定优先级最高：免得业务想锁死模型却被任务路由悄悄改掉
    resolved = gateway.resolve(logical_model="chat-lite", task="complex_reasoning")
    assert resolved.logical_model == "chat-lite"
    assert resolved.provider is gateway.provider_of("chat-lite")


async def test_unsupported_capability_is_rejected_before_the_call(gateway: LLMGateway):
    # vision 没有 tools 能力：带工具调用应当在发出请求前就失败，而不是让供应商报 400
    with pytest.raises(UnsupportedCapabilityError):
        await gateway.complete(
            [{"role": "user", "content": "看图"}], logical_model="vision", tools=[{"type": "function", "function": {"name": "t"}}]
        )


async def test_response_is_normalized_with_usage_and_cost(gateway: LLMGateway):
    mock = gateway.provider_of("chat-pro").mock
    mock.reset([final("咨询建议：按成分选。")])

    response = await gateway.complete([{"role": "user", "content": "敏感肌怎么选精华"}], task="product_consulting")

    # 归一化后的响应结构：业务只认这一套字段
    assert response.logical_model == "chat-pro"
    assert response.text.startswith("咨询建议")
    assert response.usage.total_tokens > 0
    assert response.latency_ms >= 0
    # 物理模型名可见（用于排障与成本归因），但业务代码只依赖逻辑名
    assert response.model == gateway.resolve(logical_model="chat-pro").spec.model
    # 成本由逻辑模型计价算出，不依赖供应商返回的任何私有字段
    assert response.usage.cost_usd > 0


def test_unknown_logical_model_is_a_validation_error(gateway: LLMGateway):
    with pytest.raises(ValidationError):
        gateway.resolve(logical_model="chat-ultra")


def test_providers_are_bound_per_logical_model(gateway: LLMGateway):
    """同一个供应商承载多个逻辑模型时不能互相覆盖 —— 这是最容易埋下来的坑。"""
    # 每个逻辑模型拿到独立的绑定实例：否则给 chat-lite 注入的 Mock 脚本会串到 chat-pro
    assert gateway.provider_of("chat-lite") is not gateway.provider_of("chat-pro")
    assert gateway.provider_of("chat-lite") is not gateway.provider_of("vision")
    # 同一个逻辑模型重复取必须是同一个实例（Provider 内可能持有连接池等状态）
    assert gateway.provider_of("chat-pro") is gateway.provider_of("chat-pro")


def test_token_estimation_is_an_upper_bound():
    # 预留必须偏保守：估低了会在结算时发现超额，估高只是暂时占用
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") > 0
    long_text = "中文内容" * 100
    assert estimate_messages_tokens([{"role": "user", "content": long_text}]) >= estimate_tokens(long_text)
