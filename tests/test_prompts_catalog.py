"""Prompt 模板中心：片段可版本化 + Role 决定组装顺序。

要守住的三点：
1) 同 name 多版本，只有一版 active —— 改措辞是一次可回滚的发布；
2) 组装顺序固定为 base → scenario，避免「改了基础口径、忘了改某个场景」的漂移；
3) 渲染缺变量**报错**而不是留空，否则模型会收到半截指令。
"""

from __future__ import annotations

import pytest

from src.infra.core.errors import ValidationError
from src.infra.prompts.registry import build_default_registry


def test_catalog_loads_versions_and_marks_only_one_active():
    registry = build_default_registry()

    assert registry.versions("cs.system") == ["v1", "v2"]
    assert registry.active_version("cs.system") == "v2"
    assert set(registry.roles) == {"content", "customer", "inspector", "operations"}


def test_compose_orders_base_layers_then_scenario_layers():
    registry = build_default_registry()
    blueprint = registry.role_blueprint("customer", scenario="after_sales")

    assert blueprint["role"] == "customer"
    assert [layer["name"] for layer in blueprint["layers"]] == [
        "cs.system",
        "customer.evidence",
        "customer.policy",
    ]

    rendered = registry.compose("customer", scenario="after_sales")
    assert "人工确认" in rendered  # 售后场景追加的政策约束确实进了正文


def test_compose_without_scenario_only_injects_base_layers():
    registry = build_default_registry()
    assert [layer["name"] for layer in registry.role_blueprint("customer")["layers"]] == ["cs.system"]


def test_unknown_scenario_is_rejected():
    registry = build_default_registry()
    with pytest.raises(ValidationError) as captured:
        registry.compose("customer", scenario="not_a_scenario")

    assert "not_a_scenario" in str(captured.value)


def test_missing_variable_fails_loudly_instead_of_rendering_blank():
    registry = build_default_registry()
    with pytest.raises(ValidationError) as captured:
        registry.render("agent.system")  # 模板里需要 {max_steps}

    assert captured.value.detail["missing"] == ["max_steps"]
    assert "4" in registry.render("agent.system", max_steps=4)


def test_role_must_be_enum_member():
    registry = build_default_registry()
    with pytest.raises(ValidationError) as captured:
        registry.compose("marketing")

    assert "marketing" in str(captured.value)
