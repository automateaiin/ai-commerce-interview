"""三重护栏：静态上限、动态打转检测、连续失败熔断。

为什么是「三重」而不是只设一个 `max_steps`（这是本项目的核心答题点）：
- **静态上限**（轮数/总时长/token/成本）只能兜住「一直做但没做完」，
  兜不住「一直做同一件事」——模型打转时轮数往往远没到上限，预算已经烧完了；
- **动态检测**（重复调用指纹 + 时间窗 + 豁免）专门抓「原地打转」，
  比单纯数轮数准得多，也不会像硬编码黑名单那样误伤；
- **连续失败熔断**（错误率）抓的是「环境坏了」——
  工具连续失败时继续调用只是把预算和重试次数一起烧掉，应该尽早转人工。

三者缺一个，上线后都会以另一种形式翻车。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .decision import PlannedCall
from .handoff import HandoffReason
from .state import RunState, StopReason


@dataclass(frozen=True, slots=True)
class GuardrailTrip:
    stop_reason: StopReason
    handoff_reason: str | None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Guardrails:
    max_steps: int = 6
    deadline_s: float = 60.0
    token_budget: int = 20000
    cost_budget_usd: float = 0.0
    """0 表示不单独设成本上限（只按 token 计）。开了就是「按钱熔断」，比按 token 更贴业务。"""

    max_consecutive_errors: int = 3
    repeat_window_steps: int = 3
    """指纹时间窗：窗口内的重复算打转。窗口太小会误伤合法轮询，太大则浪费预算。"""

    repeat_exempt_tools: frozenset[str] = field(default_factory=frozenset)
    """合法轮询豁免：已知需要反复调用的工具（如轮询订单状态）不该被判定为打转。"""

    max_calls_per_turn: int = 4
    max_observation_chars: int = 1500
    observation_page_size: int = 20
    max_fingerprints: int = 256

    def check_limits(self, state: RunState, *, now: float) -> GuardrailTrip | None:
        """静态上限。放在每轮**开头**检查：先拦再用，不让已经超限的请求再花一次模型调用。"""
        # 用 state.max_steps（本次运行的有效上限）而不是 self.max_steps：
        # 调用方可以按任务收紧轮数（如 max_steps=2），护栏必须认这个「本次生效值」。
        if state.step >= state.max_steps:
            return GuardrailTrip(StopReason.MAX_STEPS, None, {"max_steps": state.max_steps, "rounds": state.step})
        if now > state.deadline:
            return GuardrailTrip(
                StopReason.DEADLINE_EXCEEDED,
                None,
                {"deadline_s": self.deadline_s, "overdue_ms": round((now - state.deadline) * 1000, 2)},
            )
        if state.token_budget and state.usage.total_tokens > state.token_budget:
            return GuardrailTrip(
                StopReason.TOKEN_BUDGET_EXCEEDED,
                HandoffReason.BUDGET.value,
                {"token_budget": state.token_budget, "used_tokens": state.usage.total_tokens},
            )
        if state.cost_budget_usd and state.usage.cost_usd >= state.cost_budget_usd:
            return GuardrailTrip(
                StopReason.COST_BUDGET_EXCEEDED,
                HandoffReason.BUDGET.value,
                {"cost_budget_usd": state.cost_budget_usd, "used_cost_usd": round(state.usage.cost_usd, 6)},
            )
        return None

    def detect_repeat(
        self, state: RunState, calls: Sequence[PlannedCall]
    ) -> tuple[GuardrailTrip | None, tuple[PlannedCall, ...]]:
        """打转检测。返回 (拦截原因, 允许执行的调用)。

        两个细节决定了它好不好用：
        - 用**参数指纹**而不是工具名：同名工具换个参数就是有效进展（违法的是重复调用）；
        - 同批内重复也算打转：一次返回两个一模一样的调用，是模型在「凑数量」。
        """
        fresh: list[PlannedCall] = []
        for call in calls:
            if call.name in self.repeat_exempt_tools:
                fresh.append(call)
                continue

            fingerprint = call.fingerprint()
            first_seen = state.fingerprints.get(fingerprint)
            if first_seen is not None and state.step - first_seen < self.repeat_window_steps:
                return (
                    GuardrailTrip(
                        StopReason.REPEATED_CALL_DETECTED,
                        HandoffReason.LOOP_DETECTED.value,
                        {
                            "tool": call.name,
                            "fingerprint": fingerprint,
                            "window_steps": self.repeat_window_steps,
                            "first_seen_round": first_seen + 1,
                            "current_round": state.step + 1,
                        },
                    ),
                    (),
                )

            state.fingerprints[fingerprint] = state.step
            fresh.append(call)

        if len(state.fingerprints) > self.max_fingerprints:
            # 指纹表必须设上限：长会话里它会长成内存泄漏
            for key in list(state.fingerprints)[: len(state.fingerprints) - self.max_fingerprints]:
                state.fingerprints.pop(key, None)

        return None, tuple(fresh)
