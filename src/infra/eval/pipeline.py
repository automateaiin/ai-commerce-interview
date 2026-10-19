"""评测流水线：用例 → 执行 → 六维打分 → 报告。

两条设计原则（面试会被追问）：

1) **评测跑的是真实执行链路**，不是另写一套简化流程。
   `HarnessEvalRunner` 直接把用例交给共享执行内核（harness）跑，
   评测结果才代表线上行为；如果评测自己实现一遍 Agent Loop，
   就会变成「测试通过、线上不通过」的两套逻辑（这类问题在 Agent 项目里极常见）。

2) **结果与执行解耦**：流水线只依赖 `EvalObservation`（回答/工具/证据/成本/延迟），
   因此后续可以把 runner 换成真实端点、换成项目一/项目二的 Agent，评测口径不变。
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..core import context as ctx
from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..core.metrics import metrics
from ..tools.base import ToolContext
from .cases import EvalCase
from .evaluators import DimensionScore, EvalLimits, EvalObservation, evaluate_case

logger = get_logger("infra.eval")

_DIMENSIONS = ("accuracy", "grounding", "tool_selection", "safety", "cost", "latency")


class EvalRunner(Protocol):
    async def run_case(self, case: EvalCase) -> EvalObservation: ...


class HarnessEvalRunner:
    """默认 runner：用共享执行内核 + 脚本化 Mock 模型跑用例（离线、可复现）。

    依赖 mock 模式：脚本化轮次由 MockProvider 提供，因此**必须**在 `PROVIDER_MODE=mock`
    下运行；否则直接报错而不是悄悄用真实模型 —— 评测烧真钱这件事要在启动时就拦住。
    """

    def __init__(
        self,
        *,
        harness: Any,
        gateway: Any,
        logical_model: str = "chat-lite",
        max_steps: int = 4,
        service: str = "eval",
        agent: str = "eval_runner",
    ) -> None:
        self._harness = harness
        self._gateway = gateway
        self._logical_model = logical_model
        self._max_steps = max_steps
        self._service = service
        self._agent = agent

    async def run_case(self, case: EvalCase) -> EvalObservation:
        provider = self._gateway.provider_of(self._logical_model)
        mock = getattr(provider, "mock", None)
        if mock is None:
            raise ValidationError(
                "评测用例需要脚本化 Mock 模型，当前端点不是 mock",
                detail={"logical_model": self._logical_model, "hint": "设置 PROVIDER_MODE=mock"},
            )
        mock.reset(list(case.scripts))

        # 身份沿用调用方的上下文（评测是平台能力，不该凭空造一个业务线身份），
        # 只把用例需要的 scope 并进来 —— 评测要能调通被测工具，但不该获得额外权限。
        current = ctx.current()
        scopes = frozenset({"llm:chat", *case.scopes})
        token = ctx.set_context(
            ctx.RequestContext(
                trace_id=f"eval_{case.case_id}",
                project_id=current.project_id if current else "eval-runner",
                scopes=(current.scopes | scopes) if current else scopes,
                service=(current.service if current else self._service) or self._service,
                agent=(current.agent if current else None) or self._agent,
            )
        )
        try:
            tool_ctx = ToolContext(
                trace_id=f"eval_{case.case_id}",
                project_id=current.project_id if current else "eval-runner",
                scopes=(current.scopes | scopes) if current else scopes,
                service=(current.service if current else self._service) or self._service,
                agent=(current.agent if current else None) or self._agent,
            )
            result = await self._harness.run(
                task=case.question,
                ctx=tool_ctx,
                logical_model=self._logical_model,
                max_steps=self._max_steps,
                tool_names=list(case.expected_tools) or None,
            )
        except Exception as exc:  # noqa: BLE001 - 用例异常要变成一条评分记录，而不是让整轮评测崩掉
            return EvalObservation(case_id=case.case_id, answer=None, error=f"{type(exc).__name__}: {exc}")
        finally:
            ctx.reset_context(token)

        history = mock.history[-1] if mock.history else []
        return EvalObservation(
            case_id=case.case_id,
            answer=result.answer,
            tool_calls=[item["name"] for item in result.tool_calls],
            evidence=_collect_evidence(history),
            cost_usd=result.usage.cost_usd,
            total_tokens=result.usage.total_tokens,
            latency_ms=result.duration_ms,
            stop_reason=result.stop_reason,
        )


def _collect_evidence(messages: Sequence[Any]) -> str:
    """汇总「模型实际看到过的证据」：工具结果 + 回填给模型的可恢复错误说明。

    跳过后两条之前的 user 消息（第 1 条是任务本身、第 2 条是 System 提示拼装后的输入），
    避免把用户问题当成证据，否则 Grounding 检查会形同虚设。
    """
    parts: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content") or ""
        if role == "tool" or (role == "user" and index > 1):
            parts.append(str(content))
    return "\n".join(parts)


class EvaluationPipeline:
    """评测编排：跑用例 → 分维度打分 → 汇总报告 + 指标打点。"""

    def __init__(self, runner: EvalRunner, *, limits: EvalLimits | None = None, telemetry: Any | None = None) -> None:
        self._runner = runner
        self._limits = limits or EvalLimits()
        self._telemetry = telemetry

    async def run(self, cases: Sequence[EvalCase], *, suite_id: str = "default") -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        dimension_stats = {dim: {"passed": 0, "total": 0} for dim in _DIMENSIONS}
        passed = 0

        for case in cases:
            observation = await self._runner.run_case(case)
            scores = evaluate_case(case, observation, self._limits)
            case_passed = all(score.passed for score in scores)
            passed += int(case_passed)
            for score in scores:
                stats = dimension_stats[score.dimension]
                stats["total"] += 1
                stats["passed"] += int(score.passed)
            results.append(self._case_payload(case, observation, scores, case_passed))

        total = len(cases)
        report = {
            "suite_id": suite_id,
            "cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "dimensions": {
                dim: {
                    "passed": stats["passed"],
                    "total": stats["total"],
                    "rate": round(stats["passed"] / stats["total"], 4) if stats["total"] else 0.0,
                }
                for dim, stats in dimension_stats.items()
            },
            "limits": {
                "accuracy_threshold": self._limits.accuracy_threshold,
                "max_cost_usd": self._limits.max_cost_usd,
                "max_latency_ms": self._limits.max_latency_ms,
            },
            "results": results,
        }
        metrics.incr("eval_case", f"{suite_id}:{'pass' if report['failed'] == 0 else 'fail'}")
        logger.info(
            "eval.suite_finished",
            extra={
                "event": "eval",
                "status": "ok" if report["failed"] == 0 else "error",
                "reason": suite_id,
                "cases": total,
                "passed": passed,
                "failed": report["failed"],
                "dimension_scores": {dim: stats["rate"] for dim, stats in report["dimensions"].items()},
            },
        )
        if self._telemetry is not None:
            self._telemetry.event("eval.suite_finished", suite_id=suite_id, passed=passed, cases=total)
        return report

    @staticmethod
    def _case_payload(
        case: EvalCase, observation: EvalObservation, scores: Sequence[DimensionScore], passed: bool
    ) -> dict[str, Any]:
        return {
            "case_id": case.case_id,
            "group": case.group,
            "passed": passed,
            "question": case.question,
            "answer": observation.answer,
            "stop_reason": observation.stop_reason,
            "tool_calls": observation.tool_calls,
            "cost_usd": round(observation.cost_usd, 6),
            "total_tokens": observation.total_tokens,
            "latency_ms": round(observation.latency_ms, 2),
            "dimension_scores": [score.to_payload() for score in scores],
            "error": observation.error,
        }


__all__ = ["EvalRunner", "HarnessEvalRunner", "EvaluationPipeline"]
