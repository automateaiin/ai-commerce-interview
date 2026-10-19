"""LLM Gateway：逻辑模型路由 + 能力校验 + 超时/重试 + 用量与成本归一。

一次调用的完整链路（面试白板题就画这条）：

    业务请求 → 逻辑模型名 → 配置快照(routing) → Provider 适配器 → 物理端点
                     ↑                                              ↓
              能力校验(缺能力直接拒绝)                        用量/成本归一 → 指标

设计要点：
- 每个请求只取**一次**配置快照，保证同一请求内路由决策一致（不会中途换模型）；
- 能力校验发生在调用前：必然失败的请求不发出，省时间也省 token；
- 重试只对暂时性错误生效，重试预算来自 settings.llm_max_retries；
- 成本按 spec 的单价换算并写进 Usage，这是后续配额结算与成本核算的依据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

from ..core import context as ctx
from ..core.config import ConfigSnapshot
from ..core.errors import QuotaExceededError
from ..core.logging import get_logger
from ..core.metrics import metrics
from ..core.retry import DEFAULT_POLICY, RetryPolicy, retry_async
from ..core.tracing import stage
from ..quota.budget import AgentBudgetGuard
from .catalog import ModelCatalog, ModelSpec
from .provider import CAP_TOOLS, LLMProvider, ensure_capabilities
from .types import Chunk, LLMResponse, Message

logger = get_logger("infra.llm.gateway")

# 预算预扣时的输出长度兜底（未显式指定 max_output_tokens 时用它估上界）
_DEFAULT_OUTPUT_TOKENS = 512


@dataclass(slots=True)
class _BudgetOutcome:
    """预算判定后**真正生效**的路由参数（未触发降级时原样透传入参）。"""

    service: str | None
    agent: str | None
    reserved_tokens: int
    logical_model: str | None
    max_output_tokens: int | None
    downgraded_to: str | None


def estimate_tokens(text: str) -> int:
    """估算 token 数（配额预留用）。

    刻意不做精确 tokenizer：预留只需要「上界估计」，
    真实用量以供应商返回的 usage 为准，结算时多退少补。
    经验口径：英文 ≈ 4 字符/token，中文 ≈ 1 字符/token，这里取折中系数 2.5 偏保守。
    """
    if not text:
        return 0
    return max(1, int(len(text) / 2.5))


def estimate_messages_tokens(messages: Sequence[Message]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):  # 多模态分段
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
        total += 4  # 角色/分隔符开销
    return total


@dataclass(slots=True)
class ResolvedModel:
    logical_model: str
    spec: ModelSpec
    provider: LLMProvider


class LLMGateway:
    """统一模型出口。上层业务只依赖本类，不感知任何供应商 SDK。"""

    def __init__(
        self,
        catalog: ConfigSnapshot[ModelCatalog],
        providers: dict[str, LLMProvider],
        *,
        retry_policy: RetryPolicy = DEFAULT_POLICY,
        budget: AgentBudgetGuard | None = None,
        lite_model: str = "",
        lite_max_output_tokens: int = 512,
    ) -> None:
        self._catalog = catalog
        self._providers = providers
        self._retry_policy = retry_policy
        # 预算守卫：为 None 时不做预算控制（单测与离线装配的默认形态）
        self._budget = budget
        self._lite_model = lite_model
        self._lite_max_output_tokens = lite_max_output_tokens

    # -- 路由 --------------------------------------------------------------- #
    def resolve(self, *, logical_model: str | None = None, task: str | None = None) -> ResolvedModel:
        """逻辑名/任务类型 → (spec, provider)。

        优先级：显式 logical_model > 任务类型覆盖 > routing.default。
        """
        catalog = self._catalog.current()
        name = logical_model
        if name is None:
            if task and task in catalog.routing.overrides:
                name = catalog.routing.overrides[task]
            else:
                name = catalog.routing.default
        spec = catalog.spec(name)
        # 优先按逻辑模型取绑定实例（同一 provider 的不同逻辑模型各有端点），
        # 退化到按 provider 名取，兼容只有一个实例的场景
        provider = self._providers.get(name) or self._providers.get(spec.provider)
        if provider is None:
            from ..core.errors import ProviderError

            raise ProviderError(
                f"未注册的 provider: {spec.provider}",
                detail={"logical_model": name, "available": sorted(self._providers)},
            )
        return ResolvedModel(logical_model=name, spec=spec, provider=provider)

    @property
    def revision(self) -> int:
        return self._catalog.revision

    def replace_providers(self, providers: dict[str, LLMProvider]) -> None:
        """配置重载后替换 Provider 绑定（与同一份快照一起生效，避免新旧混用）。"""
        self._providers = providers

    def provider_of(self, logical_model: str) -> LLMProvider:
        """取某逻辑模型的绑定 Provider（测试/演示脚本用它注入 Mock 脚本，避免直接碰内部字段）。"""
        return self._providers.get(logical_model) or self.resolve(logical_model=logical_model).provider

    # -- 调用 --------------------------------------------------------------- #
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        logical_model: str | None = None,
        task: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        idempotent: bool = True,
    ) -> LLMResponse:
        # 预算判定必须在路由之前：一旦需要降级，路由就得按降级后的模型名走，
        # 否则会出现「先按 chat-pro 解析、又按 chat-lite 调用」的自相矛盾。
        budget = await self._apply_budget(messages, logical_model=logical_model, max_output_tokens=max_output_tokens)
        resolved = self.resolve(logical_model=budget.logical_model, task=task)
        spec = resolved.spec
        effective_max_output = budget.max_output_tokens

        required = {CAP_TOOLS} if tools else set()
        ensure_capabilities(spec.model, required, set(spec.capabilities))

        async def _call(attempt: int) -> LLMResponse:
            response = await resolved.provider.complete(
                messages,
                spec.model,
                tools=tools,
                temperature=temperature if temperature is not None else spec.temperature,
                max_output_tokens=effective_max_output or spec.max_output_tokens,
                timeout_s=spec.timeout_s,
                extra=spec.extra or None,
            )
            response.logical_model = resolved.logical_model
            response.provider = spec.provider
            response.usage.cost_usd = spec.estimated_cost(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )
            return response

        async with stage(
            "llm.complete", logical_model=resolved.logical_model, model=spec.model, provider=spec.provider
        ):
            response = await retry_async(
                _call,
                name=f"llm.{resolved.logical_model}",
                policy=self._retry_policy,
                idempotent=idempotent,
            )

        if self._budget is not None:
            await self._budget.settle(
                service=budget.service, agent=budget.agent, reserved_tokens=budget.reserved_tokens,
                actual_tokens=response.usage.total_tokens,
            )
        self._record(resolved, response, service=budget.service, agent=budget.agent)
        if budget.downgraded_to:
            logger.info(
                "llm.downgraded",
                extra={
                    "event": "llm_call",
                    "status": "degraded",
                    "degraded": True,
                    "reason": "agent_daily_budget",
                    "agent": budget.agent,
                    "service": budget.service,
                    "downgrade": True,
                    "downgraded_to": budget.downgraded_to,
                },
            )
        return response

    async def _apply_budget(
        self,
        messages: Sequence[Message],
        *,
        logical_model: str | None,
        max_output_tokens: int | None,
    ) -> _BudgetOutcome:
        """预算预扣 + 降级决策，返回路由要用的**生效参数**。

        - 触硬上限：抛 429（quota_exceeded），止损线不可协商；
        - 触软阈值：改用 Lite 逻辑模型，并同时收紧本次输出上限 ——
          降级只换模型不控输出长度的话，成本一样会继续涨。
        """
        if self._budget is None:
            return _BudgetOutcome(None, None, 0, logical_model, max_output_tokens, None)

        current = ctx.current()
        service = (current.service or current.project_id) if current else None
        agent = current.agent if current else None
        estimated = estimate_messages_tokens(messages) + (max_output_tokens or _DEFAULT_OUTPUT_TOKENS)
        decision = await self._budget.decide(service=service, agent=agent, estimated_tokens=estimated)
        if not decision.allowed:
            raise QuotaExceededError(
                "Agent 日预算已用尽",
                detail={
                    "quota_type": "agent_daily_budget",
                    "budget_scope": decision.scope,
                    "used": decision.used_tokens,
                    "limit": decision.limit_tokens,
                },
                retry_after=600,
            )
        if not decision.downgrade:
            return _BudgetOutcome(service, agent, estimated, logical_model, max_output_tokens, None)

        lite = self._lite_model or self._catalog.current().routing.default
        if not lite or lite == logical_model:
            # 已经在用 Lite，或压根没有 Lite 目标：不重复降级，但预扣照常
            return _BudgetOutcome(service, agent, estimated, logical_model, max_output_tokens, None)
        return _BudgetOutcome(
            service=service,
            agent=agent,
            reserved_tokens=estimated,
            logical_model=lite,
            max_output_tokens=min(max_output_tokens or self._lite_max_output_tokens, self._lite_max_output_tokens),
            downgraded_to=lite,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        logical_model: str | None = None,
        task: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[Chunk]:
        """流式输出。注意：流式不适用于「重试整个请求」策略 ——
        一旦有 chunk 已经发给客户端，重试就会造成重复内容；
        因此这里只做单次调用，由调用方决定断流后的补偿语义。
        """
        resolved = self.resolve(logical_model=logical_model, task=task)
        spec = resolved.spec
        ensure_capabilities(spec.model, {CAP_TOOLS} if tools else set(), set(spec.capabilities))
        async for chunk in resolved.provider.stream(
            messages,
            spec.model,
            tools=tools,
            temperature=temperature if temperature is not None else spec.temperature,
            max_output_tokens=max_output_tokens or spec.max_output_tokens,
            timeout_s=spec.timeout_s,
        ):
            yield chunk

    def _record(
        self,
        resolved: ResolvedModel,
        response: LLMResponse,
        *,
        service: str | None = None,
        agent: str | None = None,
    ) -> None:
        metrics.incr("llm_call", f"{resolved.spec.provider}/{resolved.logical_model}")
        metrics.observe_latency(
            "llm", f"{resolved.spec.provider}/{resolved.logical_model}", response.latency_ms
        )
        metrics.observe_tokens(
            resolved.spec.provider,
            resolved.logical_model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.cost_usd,
        )
        # 同时按 Service/Agent 聚合：模型维度回答「哪个模型贵」，业务维度回答「谁花的钱」
        metrics.observe_metering(
            service=service or "unknown",
            agent=agent or "default",
            model=f"{resolved.spec.provider}/{resolved.logical_model}",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=response.usage.cost_usd,
            latency_ms=response.latency_ms,
        )
        logger.info(
            "llm.complete.done",
            extra={
                "event": "llm_call",
                "provider": resolved.spec.provider,
                "model": resolved.spec.model,
                "logical_model": resolved.logical_model,
                "service": service or "unknown",
                "agent": agent or "default",
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost_usd": round(response.usage.cost_usd, 6),
                "finish_reason": response.finish_reason,
                "latency_ms": response.latency_ms,
                "status": "ok",
            },
        )
