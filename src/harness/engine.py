"""主循环：一轮 = 一次模型决策 + 一批工具执行。

循环本身很简单，难的是**让它一定会停**并且**停下来时可交接**。
所以这里的结构刻意写成「每轮先查护栏，再决策，再校验，再执行」：

    ┌─ 每轮开头：静态护栏（轮数 / 总时长 / token / 成本）
    │
    ├─ 模型决策 ──► 终答？──► 完成，返回 answer
    │                 │
    │                 └─ 工具调用 ──► 派发前校验（存在/白名单/高风险/参数/数量/权限）
    │                                   │
    │                                   ├─ 终止性问题 ──► 停止 + 交接包
    │                                   └─ 可恢复问题 ──► 作为 observation 回填，下一轮让模型改
    │
    ├─ 打转检测（参数指纹 + 时间窗 + 豁免）──► 命中即停止 + 交接包
    │
    ├─ 批量执行（只读并行 / 写操作串行）
    │
    └─ 观察回填 + 连续失败熔断 + 历史压缩 ──► 进入下一轮

三个设计取舍值得在面试里讲：
- **护栏放在轮首**：已经超限的第 N+1 轮不会再花一次模型调用；
- **校验失败不抛异常**：除「终止性」问题外，全部转成 observation 回填，
  因为模型自己改参数的成功率远高于上层重试；
- **工具失败不是异常**：失败本身是下一步决策的输入，只有越权/业务拒绝才终止。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.infra.core.logging import get_logger
from src.infra.core.metrics import metrics
from src.infra.core.tracing import stage
from src.infra.llm.types import Usage
from src.infra.prompts.registry import PromptRegistry
from src.infra.tools.base import ToolContext, ToolExecutor, ToolRegistry

from .decision import DecisionKind, DecisionParseError, PlannedCall, parse_decision
from .guardrails import Guardrails
from .handoff import HandoffPacket, HandoffReason
from .observation import ObservationBuilder
from .policy import ToolPolicy
from .state import RunState, StopReason
from .trace import ExecutionTrace
from .validate import DecisionValidator

logger = get_logger("harness.engine")

_NEXT_ACTIONS: dict[str, list[str]] = {
    HandoffReason.BUDGET.value: ["确认是否需要继续该任务", "评估后可提高 token/成本预算再重跑", "已有中间结论可直接采用"],
    HandoffReason.ERROR_THRESHOLD.value: ["检查下游依赖与工具可用性", "修复后重新发起"], 
    HandoffReason.PERMISSION.value: ["为该身份补充所需 scope", "或改用具备权限的执行者重试"],
    HandoffReason.HIGH_RISK.value: ["向用户确认订单/退款等细节", "人工确认后在请求中带确认标记重试"],
    HandoffReason.LOOP_DETECTED.value: ["确认用户诉求是否表达清楚", "补充缺失的上下文后重试"],
    HandoffReason.INVALID_DECISION.value: ["检查 Prompt 与模型输出格式约束", "必要时切换逻辑模型重试"],
    HandoffReason.NO_TOOLS.value: ["检查该身份的工具配置与 scope 绑定"],
}


@dataclass(slots=True)
class HarnessResult:
    """一次运行的对外结果契约。

    它同时服务三种读者，这也是字段不能省的原因：
    - **业务前端**：answer（终答）+ transcript（可视化过程）；
    - **计费/配额**：usage（token 与成本，配额结算的依据）；
    - **排障与评估**：stop_reason + trace（为什么停、停在哪一步）。
    """

    answer: str | None
    steps: int
    stop_reason: str
    tool_calls: list[dict[str, Any]]
    usage: Usage
    transcript: list[dict[str, Any]]
    trace: ExecutionTrace
    handoff: HandoffPacket | None = None
    duration_ms: float = 0.0

    def to_payload(self, *, trace_id: str, duration_ms: float | None = None) -> dict[str, Any]:
        return {
            "success": self.answer is not None,
            "answer": self.answer,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "tool_calls": self.tool_calls,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "cost_usd": round(self.usage.cost_usd, 6),
            },
            "transcript": self.transcript,
            "handoff": self.handoff.to_payload() if self.handoff else None,
            "trace": self.trace.digest(),
            "trace_id": trace_id,
            "duration_ms": round(duration_ms if duration_ms is not None else self.duration_ms, 2),
        }


class Harness:
    def __init__(
        self,
        *,
        gateway: Any,
        registry: ToolRegistry,
        executor: ToolExecutor,
        prompts: PromptRegistry,
        guardrails: Guardrails | None = None,
        validator: DecisionValidator | None = None,
        observations: ObservationBuilder | None = None,
        policy: ToolPolicy | None = None,
        system_prompt_id: str = "agent.system",
    ) -> None:
        self._gateway = gateway
        self._registry = registry
        self._executor = executor
        self._prompts = prompts
        self._guardrails = guardrails or Guardrails()
        self._policy = policy or ToolPolicy()
        self._validator = validator or DecisionValidator(
            policy=self._policy, max_calls_per_turn=self._guardrails.max_calls_per_turn
        )
        self._observations = observations or ObservationBuilder(
            max_chars=self._guardrails.max_observation_chars,
            page_size=self._guardrails.observation_page_size,
        )
        self._system_prompt_id = system_prompt_id

    @property
    def guardrails(self) -> Guardrails:
        return self._guardrails

    async def run(
        self,
        *,
        task: str,
        ctx: ToolContext,
        logical_model: str | None = None,
        max_steps: int | None = None,
        tool_names: list[str] | None = None,
        confirmed: bool = False,
        system_prompt: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> HarnessResult:
        started = time.perf_counter()
        state = self._initial_state(
            task=task,
            ctx=ctx,
            logical_model=logical_model,
            max_steps=max_steps,
            started=started,
            system_prompt=system_prompt,
            history=history,
        )
        visible = self._visible_tools(ctx, tool_names)
        schemas = self._schemas(ctx, visible)

        logger.info(
            "harness.run_started",
            extra={"event": "agent_run", "reason": "started", "tool": ",".join(sorted(visible)) or None},
        )

        rounds = 0
        while True:
            trip = self._guardrails.check_limits(state, now=time.perf_counter())
            if trip is not None:
                self._stop(state, trip.stop_reason, trip.handoff_reason, detail=trip.detail)
                break

            if not visible:
                self._stop(state, StopReason.NO_TOOLS, HandoffReason.NO_TOOLS.value)
                break

            rounds += 1
            state.step = rounds

            # ---- 一次模型决策 -------------------------------------------- #
            async with stage("harness.decision", iterations=rounds):
                response = await self._gateway.complete(state.messages, logical_model=logical_model, tools=schemas)
            state.usage = state.usage.merged_with(response.usage)
            state.messages.append(response.to_message())
            state.trace.tokens = state.usage.total_tokens
            state.trace.cost_usd = state.usage.cost_usd
            state.trace.llm_ms += response.latency_ms
            state.trace.rounds = rounds

            try:
                decision = parse_decision(response)
            except DecisionParseError as exc:
                state.note(step=rounds, type="decision", status="error", error_code=exc.code)
                self._stop(state, StopReason.DECISION_INVALID, HandoffReason.INVALID_DECISION.value, failure=exc.message)
                break

            state.note(
                step=rounds,
                type="decision",
                kind=decision.kind.value,
                tools=[call.name for call in decision.calls] or None,
                mixed=decision.mixed or None,
                tokens=response.usage.total_tokens,
            )
            state.trace.record(
                step=rounds,
                kind="decision",
                kind_name=decision.kind.value,
                calls=[{"tool": call.name, "fingerprint": call.fingerprint()} for call in decision.calls] or None,
                mixed=decision.mixed or None,
                tokens=response.usage.total_tokens,
                duration_ms=response.latency_ms,
            )

            # ---- 终答：唯一的正常出口 ------------------------------------ #
            if decision.kind is DecisionKind.FINAL:
                state.answer = decision.text
                self._stop(state, StopReason.COMPLETED, None)
                break

            # ---- 派发前校验 ---------------------------------------------- #
            validated = self._validator.validate(
                decision, registry=self._registry, ctx=ctx, visible=visible, confirmed=confirmed
            )
            for violation in validated.violations:
                state.note(
                    step=rounds,
                    type="violation",
                    tool=violation.tool,
                    status="error",
                    error_code=violation.code,
                    terminal=violation.terminal or None,
                )
                state.trace.record(
                    step=rounds, kind="violation", tool=violation.tool, status="error",
                    error_code=violation.code, terminal=violation.terminal or None,
                )
                state.messages.append(self._observations.from_violation(violation))

            terminal = validated.terminal_violation
            if terminal is not None:
                # 终止性问题：越权、策略禁止、高风险待确认。模型改参数或重试都没用
                if terminal.code == "high_risk_confirmation_required":
                    self._stop(
                        state, StopReason.HANDOFF_REQUIRED, HandoffReason.HIGH_RISK.value,
                        detail={"tool": terminal.tool}, failure=terminal.message,
                    )
                elif terminal.code == "permission_denied":
                    self._stop(
                        state, StopReason.TERMINAL_TOOL_ERROR, HandoffReason.PERMISSION.value,
                        detail={"tool": terminal.tool}, failure=terminal.message,
                    )
                else:
                    self._stop(
                        state, StopReason.TERMINAL_TOOL_ERROR, None,
                        detail={"tool": terminal.tool, "error_code": terminal.code}, failure=terminal.message,
                    )
                break

            # ---- 打转检测 ------------------------------------------------- #
            repeat_trip, executable = self._guardrails.detect_repeat(state, validated.executable)
            if repeat_trip is not None:
                self._stop(state, repeat_trip.stop_reason, repeat_trip.handoff_reason, detail=repeat_trip.detail)
                break

            if not executable:
                # 全部被可恢复问题拦下：本轮不执行任何工具，只回填 observation，让模型自己改正
                state.messages = self._observations.trim(state.messages)
                continue

            # ---- 批量执行（只读并行 / 写操作串行由 executor 决定） -------- #
            async with stage("harness.tools", tools=len(executable)):
                results = await self._executor.execute_many(
                    [(call.id, call.name, call.args) for call in executable], ctx
                )

            failed = 0
            for call, result in zip(executable, results):
                state.messages.append(self._observations.from_result(call, result))
                status = "ok" if result.ok else "error"
                state.note(
                    step=rounds, type="tool", tool=call.name, status=status,
                    error_code=result.error_code, duration_ms=round(result.duration_ms, 2),
                )
                state.trace.record(
                    step=rounds, kind="tool", tool=call.name, status=status,
                    error_code=result.error_code, fingerprint=call.fingerprint(),
                    duration_ms=result.duration_ms,
                )
                state.trace.tool_ms += result.duration_ms
                state.tool_calls.append(
                    {
                        "id": call.id,
                        "name": call.name,
                        "ok": result.ok,
                        "error_code": result.error_code,
                    }
                )
                if not result.ok:
                    failed += 1

            terminal_result = next((result for result in results if result.terminal), None)
            if terminal_result is not None:
                self._stop(
                    state, StopReason.TERMINAL_TOOL_ERROR, HandoffReason.PERMISSION.value,
                    detail={"tool": terminal_result.tool}, failure=terminal_result.error_message,
                )
                break

            # ---- 连续失败熔断 -------------------------------------------- #
            if failed == len(results):
                state.error_streak += 1
            else:
                state.error_streak = 0
            if state.error_streak >= self._guardrails.max_consecutive_errors:
                self._stop(
                    state, StopReason.ERROR_THRESHOLD_EXCEEDED, HandoffReason.ERROR_THRESHOLD.value,
                    detail={"error_streak": state.error_streak, "threshold": self._guardrails.max_consecutive_errors},
                )
                break

            state.messages = self._observations.trim(state.messages)

        duration_ms = (time.perf_counter() - started) * 1000
        metrics.incr("harness_stop", state.stop_reason or "unknown")
        metrics.observe_latency("harness", "run", duration_ms)
        logger.info(
            "harness.run_finished",
            extra={
                "event": "agent_run",
                "status": "ok" if state.stop_reason == StopReason.COMPLETED.value else "error",
                "reason": state.stop_reason,
                "error_code": state.stop_reason if state.stop_reason != StopReason.COMPLETED.value else None,
            },
        )

        return HarnessResult(
            answer=state.answer,
            steps=state.step,
            stop_reason=state.stop_reason or StopReason.COMPLETED.value,
            tool_calls=state.tool_calls,
            usage=state.usage,
            transcript=state.transcript,
            trace=state.trace,
            handoff=state.handoff,
            duration_ms=duration_ms,
        )

    # -- 内部 ---------------------------------------------------------------- #

    def _initial_state(
        self,
        *,
        task: str,
        ctx: ToolContext,
        logical_model: str | None,
        max_steps: int | None,
        started: float,
        system_prompt: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> RunState:
        max_steps = max_steps or self._guardrails.max_steps
        # system_prompt 允许调用方（项目一各角色 Agent）注入按 Role 组装后的提示词；
        # 未提供时退回默认 agent.system 模板 —— 保持既有调用方行为不变。
        system = system_prompt or self._prompts.render(self._system_prompt_id, max_steps=max_steps)
        # history 允许调用方注入已回填的会话（只含 role/content，不含 system —— 由本方法负责 system）。
        # 把它做成显式入参而不是拼进 task，是为了保留消息的 role 结构：模型对「谁说的」很敏感。
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})
        state = RunState(
            task=task,
            ctx=ctx,
            trace=ExecutionTrace(trace_id=ctx.trace_id),
            messages=messages,
            logical_model=logical_model,
            max_steps=max_steps,
            deadline=started + self._guardrails.deadline_s,
            token_budget=self._guardrails.token_budget,
            cost_budget_usd=self._guardrails.cost_budget_usd,
        )
        return state

    def _visible_tools(self, ctx: ToolContext, tool_names: list[str] | None) -> frozenset[str]:
        """当前身份可见的工具集。

        越权工具**不暴露**给模型（而不是暴露后再拒绝）：
        既减少无效调用，也不向模型泄露它不该知道的内部工具清单。
        """
        names = set(self._registry.names())
        if tool_names:
            names &= set(tool_names)
        visible = {
            name
            for name in names
            if (tool := self._registry.get(name)).requires_scope is None
            or ctx.has_scope(tool.requires_scope)  # type: ignore[arg-type]
        }
        return frozenset(visible)

    def _schemas(self, ctx: ToolContext, visible: frozenset[str]) -> list[dict[str, Any]]:
        return [
            schema
            for schema in self._registry.schemas(only_scopes=ctx.scopes)
            if schema["function"]["name"] in visible
        ]

    def _stop(
        self,
        state: RunState,
        reason: StopReason,
        handoff_reason: str | None,
        *,
        detail: dict[str, Any] | None = None,
        failure: str | None = None,
    ) -> None:
        state.stop_reason = reason.value
        state.note(step=state.step, type="termination", status="ok" if reason is StopReason.COMPLETED else "error",
                   stop_reason=reason.value, handoff=handoff_reason)
        state.trace.termination(
            step=state.step,
            reason=reason.value,
            handoff=handoff_reason,
            detail=detail,
        )
        if handoff_reason is not None:
            state.handoff = HandoffPacket(
                reason=handoff_reason,
                summary=state.task[:200],
                completed_steps=state.step,
                trace_id=state.trace.trace_id,
                tool_results=[
                    {"tool": item.get("tool"), "status": item.get("status"), "error_code": item.get("error_code")}
                    for item in state.transcript
                    if item.get("type") == "tool"
                ],
                failure=failure,
                next_actions=_NEXT_ACTIONS.get(handoff_reason, []),
            )
        logger.info(
            "harness.terminated",
            extra={
                "event": "agent_run",
                "reason": reason.value,
                "status": "ok" if reason is StopReason.COMPLETED else "error",
                "iterations": state.step,
            },
        )


__all__ = ["Harness", "HarnessResult"]
