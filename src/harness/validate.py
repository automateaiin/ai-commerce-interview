"""派发前校验：四道闸门，任何一道不过就不进执行器。

为什么在 Harness 里还要再校验一次（执行器里已经有参数与权限校验）：
两者的**目的不同**。执行器校验是「保护系统」——防止非法调用打到下游；
Harness 校验是「保护循环」——在花掉一次执行、一次配额、一次重试之前就拦下来，
并且把失败变成**模型可读的 observation**（尤其是参数问题，模型必须能自己改对）。

所以这里是纵深防御的第一道，不是重复劳动：不同入口（REST 直调 / MCP / Agent）都要各自守住。

四道闸门：
1) 工具存在且在「当前身份可见集合」内 —— 不可见就当不存在（不回显内部工具清单）；
2) 项目级白名单 —— 业务策略不允许的动作直接终止，不浪费重试；
3) 高风险动作确认 —— 有业务副作用的动作不允许模型自己拍板；
4) 参数 Schema + 每轮调用数量上限 —— 拦明显错误与「一次调一堆」的失控行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from src.infra.tools.base import ToolContext, ToolRegistry

from .decision import LLMDecision, PlannedCall
from .policy import ToolPolicy


@dataclass(frozen=True, slots=True)
class Violation:
    """一条校验失败。terminal 决定它是「让模型改正」还是「终止并交接」。"""

    code: str
    message: str
    tool: str | None = None
    call_id: str | None = None
    terminal: bool = False
    issues: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_observation(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "status": "error",
            "error_code": self.code,
            "message": self.message,
        }
        if self.issues:
            # 参数问题必须把「哪个字段错、错在哪」回给模型，否则它只能瞎猜
            payload["issues"] = list(self.issues)
        if self.terminal:
            payload["terminal"] = True
        return payload


@dataclass(slots=True)
class ValidatedDecision:
    decision: LLMDecision
    executable: tuple[PlannedCall, ...]
    violations: tuple[Violation, ...]

    @property
    def terminal_violation(self) -> Violation | None:
        return next((item for item in self.violations if item.terminal), None)

    @property
    def recovered(self) -> bool:
        """有可恢复问题、但一条都没能执行：本轮应该只回填 observation 让模型改正。"""
        return bool(self.violations) and not self.executable


class DecisionValidator:
    def __init__(self, *, policy: ToolPolicy | None = None, max_calls_per_turn: int = 4) -> None:
        self._policy = policy or ToolPolicy()
        self._max_calls = max_calls_per_turn

    @property
    def policy(self) -> ToolPolicy:
        return self._policy

    def validate(
        self,
        decision: LLMDecision,
        *,
        registry: ToolRegistry,
        ctx: ToolContext,
        visible: frozenset[str] | None = None,
        confirmed: bool = False,
    ) -> ValidatedDecision:
        executable: list[PlannedCall] = []
        violations: list[Violation] = []

        for call in decision.calls:
            # 闸门 1：存在性。用 visible 而不是全量注册表 —— 越权工具根本不该被承认
            if not registry.has(call.name) or (visible is not None and call.name not in visible):
                violations.append(
                    Violation(
                        code="not_found",
                        message=f"工具不可用: {call.name}",
                        tool=call.name,
                        call_id=call.id,
                    )
                )
                continue

            tool = registry.get(call.name)

            # 闸门 2：项目级白名单（业务策略）
            if not self._policy.allows(tool):
                violations.append(
                    Violation(
                        code="tool_not_allowed",
                        message=f"当前项目不允许调用工具: {call.name}",
                        tool=call.name,
                        call_id=call.id,
                        terminal=True,
                    )
                )
                continue

            # 闸门 3：高风险动作必须有人确认
            if self._policy.requires_confirmation(tool) and not confirmed:
                violations.append(
                    Violation(
                        code="high_risk_confirmation_required",
                        message=f"{call.name} 属于高风险动作（{tool.risk.value}），需要人工确认后才能执行",
                        tool=call.name,
                        call_id=call.id,
                        terminal=True,
                    )
                )
                continue

            # 闸门 4a：参数不是合法 JSON —— 直接用原文报错，绝不尝试「猜」参数
            if call.raw_arguments is not None:
                violations.append(
                    Violation(
                        code="invalid_arguments",
                        message="工具参数不是合法 JSON",
                        tool=call.name,
                        call_id=call.id,
                        issues=({"field": "(root)", "msg": "arguments 不是合法 JSON"},),
                    )
                )
                continue

            # 闸门 4b：参数 Schema
            try:
                tool.args_model.model_validate(dict(call.args))
            except PydanticValidationError as exc:
                issues = tuple(
                    {"field": ".".join(str(part) for part in error["loc"]) or "(root)", "msg": error["msg"]}
                    for error in exc.errors()[:5]
                )
                violations.append(
                    Violation(
                        code="invalid_arguments",
                        message="参数校验失败",
                        tool=call.name,
                        call_id=call.id,
                        issues=issues,
                    )
                )
                continue

            # 闸门 4c：权限（scope）。越权是确定性错误，重试与改参都没用 → terminal
            if tool.requires_scope and not ctx.has_scope(tool.requires_scope):
                violations.append(
                    Violation(
                        code="permission_denied",
                        message=f"缺少 scope: {tool.requires_scope}",
                        tool=call.name,
                        call_id=call.id,
                        terminal=True,
                    )
                )
                continue

            # 闸门 4d：每轮调用数量上限（放在最后：只对真正可执行的调用计数）
            if len(executable) >= self._max_calls:
                violations.append(
                    Violation(
                        code="call_limit_exceeded",
                        message=f"单轮工具调用超过上限 {self._max_calls}",
                        tool=call.name,
                        call_id=call.id,
                    )
                )
                continue

            executable.append(call)

        return ValidatedDecision(
            decision=decision,
            executable=tuple(executable),
            violations=tuple(violations),
        )
