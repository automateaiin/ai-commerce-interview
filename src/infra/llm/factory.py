"""Provider 工厂与注册表。

按配置构造适配器实例：一个逻辑模型 = 一个绑定好端点与密钥来源的 Provider 包装。
这样 LLMProvider 协议里只有通用参数（messages/model/tools/gen），
供应商特有的 base_url / api_key_env 被关在包装类内部，不会泄漏到上层。
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Sequence

from ..core.config import ConfigSnapshot, load_yaml_file
from ..core.logging import get_logger
from ..core.settings import Settings, get_settings
from .catalog import ModelCatalog, ModelSpec, parse_catalog, validate_catalog
from .gateway import LLMGateway
from .mock import MockProvider
from .openai_compat import OpenAICompatibleProvider
from .provider import LLMProvider
from .types import Chunk, LLMResponse, Message

logger = get_logger("infra.llm.factory")


class BoundOpenAICompatibleProvider:
    """把一个逻辑模型的物理端点信息绑定到适配器上，对协议只暴露通用参数。"""

    def __init__(self, template: OpenAICompatibleProvider, spec: ModelSpec, logical_model: str) -> None:
        self.name = spec.provider
        self._template = template
        self._spec = spec
        self._logical_model = logical_model

    async def complete(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await self._template.complete(
            messages,
            model,
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            extra=extra,
            base_url=self._spec.base_url,
            api_key_env=self._spec.api_key_env,
            logical_model=self._logical_model,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[Chunk]:
        async for chunk in self._template.stream(
            messages,
            model,
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            base_url=self._spec.base_url,
            api_key_env=self._spec.api_key_env,
        ):
            yield chunk


class BoundMockProvider:
    """Mock 也需要最少量绑定：保留真实模型名以便日志与用量口径一致。"""

    def __init__(self, provider: MockProvider, spec: ModelSpec, logical_model: str) -> None:
        self.name = "mock"
        self._provider = provider
        self._spec = spec
        self._logical_model = logical_model

    @property
    def mock(self) -> MockProvider:
        """暴露底层 MockProvider，供测试/演示脚本注入脚本化轮次。"""
        return self._provider

    async def complete(self, messages, model, *, tools=None, temperature=None, max_output_tokens=None, timeout_s=None, extra=None) -> LLMResponse:  # noqa: ANN001
        return await self._provider.complete(messages, model, tools=tools, temperature=temperature)

    async def stream(self, messages, model, *, tools=None, temperature=None, max_output_tokens=None, timeout_s=None):  # noqa: ANN001
        async for chunk in self._provider.stream(messages, model, tools=tools):
            yield chunk


def apply_provider_mode(catalog: ModelCatalog, mode: str) -> ModelCatalog:
    """按运行模式改写 provider（不影响逻辑名、路由、计价与能力声明）。

    - mode=mock：全部逻辑模型走 MockProvider —— 离线可跑、可复现、不烧钱；
    - mode=auto：没有真实密钥时自动落到 mock，有密钥时走真实端点（避免「以为在调真模型」的空跑）；
    - mode=openai_compat 或其它：保持配置文件原样（用于强制走真实端点，缺密钥就报错）。
    """
    default_provider = "openai_compat"
    if mode == "mock":
        forced = "mock"
    elif mode == "auto":
        forced = "mock" if not os.environ.get("ARK_API_KEY") else default_provider
    else:
        forced = None
    if forced is None:
        return catalog
    models = {name: spec.model_copy(update={"provider": forced}) for name, spec in catalog.models.items()}
    logger.info(
        "catalog.provider_mode",
        extra={"event": "config_loaded", "provider": forced, "reason": f"mode={mode}"},
    )
    return catalog.model_copy(update={"models": models})


def build_catalog_snapshot(settings: Settings | None = None) -> ConfigSnapshot[ModelCatalog]:
    """加载并校验模型目录，构造初始快照。"""
    active = settings or get_settings()
    path = active.resolved_models_config_path()
    raw = load_yaml_file(path)
    catalog = apply_provider_mode(parse_catalog(raw), active.provider_mode)
    validate_catalog(catalog)
    snapshot: ConfigSnapshot[ModelCatalog] = ConfigSnapshot(catalog, validator=validate_catalog, name="model_catalog")
    logger.info(
        "catalog.loaded",
        extra={"event": "config_loaded", "route": str(path), "revision": snapshot.revision},
    )
    return snapshot


def build_providers(catalog: ModelCatalog, settings: Settings | None = None) -> dict[str, LLMProvider]:
    """按 provider 名称构造适配器注册表。

    每个逻辑模型拿到自己的绑定实例；同一个底层适配器模板可被复用，
    但 mock 与真实适配器互相独立，便于离线测试与联调切换。
    """
    active = settings or get_settings()
    http_template = OpenAICompatibleProvider(default_api_key_env="ARK_API_KEY")
    providers: dict[str, LLMProvider] = {}
    for logical_model, spec in catalog.models.items():
        if spec.provider == "mock":
            providers[logical_model] = BoundMockProvider(MockProvider(), spec, logical_model)
        elif spec.provider == "openai_compat":
            providers[logical_model] = BoundOpenAICompatibleProvider(http_template, spec, logical_model)
        else:
            logger.warning(
                "catalog.unknown_provider",
                extra={"event": "config_warning", "provider": spec.provider, "logical_model": logical_model},
            )
    return providers


def build_gateway(settings: Settings | None = None) -> LLMGateway:
    """一步构造网关（应用启动与测试共用）。"""
    active = settings or get_settings()
    snapshot = build_catalog_snapshot(active)
    providers = build_providers(snapshot.current(), active)
    from ..core.retry import RetryPolicy

    policy = RetryPolicy(max_attempts=active.llm_max_retries)
    return LLMGateway(snapshot, providers, retry_policy=policy)
