"""项目二装配入口：把配置与 infra 组件装配成一个可运行的 Harness。

为什么单独一个 factory（而不是在 Harness.__init__ 里直接读全局 settings）：
Harness 是**纯执行逻辑**，它不该知道「配置从哪来、环境变量叫什么」；
把「配置 → 护栏参数」的映射集中在这里，测试就能只替换 factory 的入参，
而不必改动引擎内部 —— 这也是 harness 能脱离 infra 被单测的前提。

依赖方向：本模块位于 src.harness，只向下依赖 src.infra 的稳定接口。
"""

from __future__ import annotations

from src.infra.core.settings import Settings
from src.infra.prompts.registry import PromptRegistry
from src.infra.tools.base import ToolExecutor, ToolRegistry

from .engine import Harness
from .guardrails import Guardrails
from .observation import ObservationBuilder
from .policy import ToolPolicy
from .validate import DecisionValidator


def build_guardrails(settings: Settings) -> Guardrails:
    """把环境配置映射成护栏参数（唯一允许「字符串 → 结构化配置」的地方）。"""
    exempt = frozenset(
        name.strip() for name in settings.harness_repeat_exempt_tools.split(",") if name.strip()
    )
    return Guardrails(
        max_steps=settings.agent_max_steps,
        deadline_s=settings.agent_total_timeout_s,
        token_budget=settings.agent_token_budget,
        cost_budget_usd=settings.harness_cost_budget_usd,
        max_consecutive_errors=settings.harness_max_consecutive_errors,
        repeat_window_steps=settings.harness_repeat_window_steps,
        repeat_exempt_tools=exempt,
        max_calls_per_turn=settings.harness_max_calls_per_turn,
        max_observation_chars=settings.harness_max_observation_chars,
    )


def build_harness(
    *,
    settings: Settings,
    gateway: object,
    registry: ToolRegistry,
    executor: ToolExecutor,
    prompts: PromptRegistry,
    policy: ToolPolicy | None = None,
) -> Harness:
    """按配置装配 Harness。policy 允许调用方覆盖（不同项目不同白名单）。"""
    guardrails = build_guardrails(settings)
    active_policy = policy or ToolPolicy()
    return Harness(
        gateway=gateway,
        registry=registry,
        executor=executor,
        prompts=prompts,
        guardrails=guardrails,
        validator=DecisionValidator(
            policy=active_policy, max_calls_per_turn=guardrails.max_calls_per_turn
        ),
        observations=ObservationBuilder(
            max_chars=guardrails.max_observation_chars,
            page_size=guardrails.observation_page_size,
        ),
        policy=active_policy,
    )


__all__ = ["build_guardrails", "build_harness"]
