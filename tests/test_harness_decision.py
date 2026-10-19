"""项目二：决策解析与调用指纹（纯函数，无 IO）。

这两件事是 Harness 的地基：
- 决策解析把「供应商差异」挡在循环之外，循环只认 FINAL / TOOL_CALLS 两种；
- 调用指纹让「打转检测」不依赖工具名，换个参数顺序也骗不过去。
"""

from __future__ import annotations

import pytest

from src.harness.decision import DecisionKind, DecisionParseError, PlannedCall, parse_decision
from src.infra.llm.types import LLMResponse, ToolCall


def test_text_only_response_is_a_final_decision():
    decision = parse_decision(LLMResponse(text="  已完成回答。  ", finish_reason="stop"))

    assert decision.kind is DecisionKind.FINAL
    assert decision.text == "已完成回答。"
    assert decision.mixed is False


def test_tool_calls_win_and_text_is_recorded_as_preamble():
    """文本 + 工具调用混排：必须以工具调用为准，文本只当前言记录。"""
    decision = parse_decision(
        LLMResponse(
            text="我先查一下商品。",
            tool_calls=[ToolCall(id="call_1", name="search_products", args={"query": "精华"})],
            finish_reason="tool_calls",
        )
    )

    assert decision.kind is DecisionKind.TOOL_CALLS
    assert decision.mixed is True
    assert [call.name for call in decision.calls] == ["search_products"]


def test_empty_response_is_a_non_retryable_parse_error():
    with pytest.raises(DecisionParseError) as excinfo:
        parse_decision(LLMResponse(text="   ", finish_reason="stop"))

    assert excinfo.value.code == "decision_invalid"
    assert excinfo.value.retryable is False


def test_json_arguments_are_preserved_on_the_normalized_call():
    decision = parse_decision(
        LLMResponse(
            tool_calls=[
                ToolCall(id="call_9", name="create_order", args={"product_id": "SKU-1"}, raw_arguments="not-json")
            ]
        )
    )
    call = decision.calls[0]

    assert call.args == {"product_id": "SKU-1"}
    # 原始字符串保留给服务端排查，但绝不回填给模型（避免二次污染）
    assert call.raw_arguments == "not-json"


def test_fingerprint_ignores_key_order_but_not_values():
    """指纹必须对参数键顺序不敏感，否则模型换个字段顺序就能绕过打转检测。"""
    a = PlannedCall(id="1", name="search_products", args={"query": "精华", "limit": 5})
    b = PlannedCall(id="2", name="search_products", args={"limit": 5, "query": "精华"})
    c = PlannedCall(id="3", name="search_products", args={"query": "面霜", "limit": 5})

    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert a.fingerprint().startswith("search_products#")


def test_decision_describe_only_exposes_structure_not_argument_values():
    decision = parse_decision(
        LLMResponse(tool_calls=[ToolCall(id="c", name="get_order_status", args={"order_id": "ORD-90001"})])
    )
    described = decision.describe()

    assert described["kind"] == "tool_calls"
    assert described["calls"][0]["tool"] == "get_order_status"
    # args_digest 是指纹，不是参数原文：轨迹会被长期保留，不能泄露用户数据
    assert "ORD-90001" not in str(described)
