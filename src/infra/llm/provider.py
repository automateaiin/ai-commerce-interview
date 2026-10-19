"""LLMProvider 抽象接口。

抽象的是「能力契约」而不是某个函数：统一出入参 + 统一响应 + 统一错误。
适配器只做四件事：拼请求体、加鉴权、解析响应归一化、错误映射。
上层（网关、Agent Loop）只依赖这个 Protocol，不依赖任何供应商 SDK。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, Sequence, runtime_checkable

from ..core.errors import ProviderError
from .types import Chunk, LLMResponse, Message

# 模型能力常量：路由时用于校验「这个模型能不能干这件事」
CAP_TOOLS = "tools"
CAP_STREAM = "stream"
CAP_VISION = "vision"
CAP_JSON_MODE = "json_mode"


@runtime_checkable
class LLMProvider(Protocol):
    """所有模型适配器的统一契约。"""

    name: str

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
        """非流式补全。所有实现必须做超时与错误归一。"""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[Chunk]:
        """流式补全。不支持流式的适配器应抛 ProviderError，而不是静默降级。"""
        ...


class UnsupportedCapabilityError(ProviderError):
    """请求了模型不具备的能力（例如给纯文本模型传 tools）。

    这是**确定性错误**：不能靠重试解决，只能换模型或改请求，
    所以 retryable=False，并且要在路由层就拦下来，而不是等供应商返回 400。
    """

    code = "unsupported_capability"
    retryable = False
    safe_message = "模型不支持所请求的能力"


def ensure_capabilities(model: str, required: set[str], declared: set[str]) -> None:
    """在调用供应商之前做能力校验，避免把必然失败的请求发出去（省时省钱）。"""
    missing = required - declared
    if missing:
        raise UnsupportedCapabilityError(
            f"模型 {model} 缺少能力: {sorted(missing)}",
            detail={"model": model, "missing": sorted(missing), "declared": sorted(declared)},
        )
