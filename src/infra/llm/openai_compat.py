"""OpenAI 兼容适配器（火山方舟 Ark / OpenAI / 任何兼容端点）。

适配器只做四件事（面试要能背）：
1) 拼请求体：把内部统一消息转成 Chat Completions 格式；
2) 加鉴权：从 api_key_env 指定的环境变量取密钥，密钥不落日志；
3) 解析响应归一化：text / tool_calls / usage / finish_reason → LLMResponse；
4) 错误映射：HTTP 状态与网络异常 → 内部错误类型（含 retryable 判定）。

关键细节：OpenAI 的 function.arguments 是 **JSON 字符串**，
必须 json.loads 后放进 ToolCall.args；解析失败时保留原文并给出 raw_arguments，
让上层可以返回「参数不合法」的可恢复错误，而不是直接崩掉。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx

from ..core.errors import (
    AuthenticationError,
    InfraError,
    ProviderError,
    UpstreamError,
    UpstreamRateLimitError,
    UpstreamTimeoutError,
    ValidationError,
)
from .types import Chunk, LLMResponse, Message, ToolCall, Usage


class OpenAICompatibleProvider:
    """面向 `POST {base_url}/chat/completions` 的通用适配器。"""

    def __init__(
        self,
        name: str = "openai_compat",
        *,
        client: httpx.AsyncClient | None = None,
        default_api_key_env: str = "ARK_API_KEY",
    ) -> None:
        self.name = name
        self._client = client
        self._owns_client = client is None
        self._default_api_key_env = default_api_key_env

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- 内部工具方法 ------------------------------------------------------- #
    def _http(self, timeout_s: float | None) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=timeout_s or 30.0, limits=httpx.Limits(max_connections=50))
            self._owns_client = True
        return self._client

    def _resolve_api_key(self, api_key_env: str | None) -> str:
        """密钥只从环境变量取。缺失时直接失败，绝不静默用空 key 打请求。"""
        env_name = api_key_env or self._default_api_key_env
        value = os.environ.get(env_name)
        if not value:
            raise AuthenticationError(
                f"环境变量 {env_name} 未配置，无法调用该模型端点",
                detail={"api_key_env": env_name},
            )
        return value

    @staticmethod
    def _build_payload(
        messages: Sequence[Message],
        model: str,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_output_tokens: int | None,
        stream: bool,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if tools:
            payload["tools"] = tools
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def _map_http_error(status_code: int, body: str, *, model: str) -> InfraError:
        detail = {"http_status": status_code, "model": model, "body": body[:300]}
        if status_code == 429:
            return UpstreamRateLimitError(f"模型 {model} 触发限流", detail=detail)
        if status_code in (401, 403):
            # 对供应商来说是鉴权失败，对本服务来说是配置/凭证问题：确定性错误，不重试
            return ProviderError(f"模型 {model} 鉴权失败", detail=detail)
        if status_code == 400:
            return ProviderError(f"模型 {model} 拒绝该请求（参数或能力不匹配）", detail=detail)
        if status_code == 404:
            return ProviderError(f"模型端点不存在: {model}", detail=detail)
        if status_code >= 500:
            return UpstreamError(f"模型服务 {status_code} 错误", detail=detail)
        return UpstreamError(f"模型服务返回未知状态 {status_code}", detail=detail)

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for item in raw_calls:
            function = item.get("function") or {}
            arguments_text = function.get("arguments") or "{}"
            try:
                args = json.loads(arguments_text) if isinstance(arguments_text, str) else dict(arguments_text)
                if not isinstance(args, dict):
                    raise ValueError("arguments 不是对象")
                parsed.append(
                    ToolCall(
                        id=item.get("id") or f"call_{len(parsed)}",
                        name=function.get("name") or "",
                        args=args,
                    )
                )
            except (json.JSONDecodeError, ValueError):
                # 参数不是合法 JSON：不猜测、不修补，保留原文交给上层做可恢复错误
                parsed.append(
                    ToolCall(
                        id=item.get("id") or f"call_{len(parsed)}",
                        name=function.get("name") or "",
                        args={},
                        raw_arguments=arguments_text if isinstance(arguments_text, str) else None,
                    )
                )
        return parsed

    @classmethod
    def _parse_response(
        cls, payload: dict[str, Any], *, logical_model: str, model: str, latency_ms: float
    ) -> LLMResponse:
        try:
            choice = (payload.get("choices") or [{}])[0]
        except AttributeError as exc:
            raise ProviderError("模型响应结构异常", detail={"keys": sorted(payload)[:10]}) from exc
        message = choice.get("message") or {}
        usage_raw = payload.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
        )
        return LLMResponse(
            text=message.get("content"),
            tool_calls=cls._parse_tool_calls(message.get("tool_calls") or []),
            usage=usage,
            finish_reason=choice.get("finish_reason") or "stop",
            logical_model=logical_model,
            model=payload.get("model") or model,
            provider="openai_compat",
            latency_ms=latency_ms,
            raw=payload,
        )

    # -- 契约实现 ----------------------------------------------------------- #
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
        base_url: str | None = None,
        api_key_env: str | None = None,
        logical_model: str = "",
    ) -> LLMResponse:
        if not base_url:
            raise ValidationError("openai_compat 适配器必须提供 base_url", detail={"model": model})
        started = time.perf_counter()
        client = self._http(timeout_s)
        api_key = self._resolve_api_key(api_key_env)
        url = base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(messages, model, tools, temperature, max_output_tokens, False, extra)

        try:
            response = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"模型 {model} 调用超时", detail={"timeout_s": timeout_s}) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"模型 {model} 网络异常: {type(exc).__name__}") from exc

        if response.status_code >= 400:
            raise self._map_http_error(response.status_code, response.text, model=model)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return self._parse_response(
            response.json(), logical_model=logical_model, model=model, latency_ms=latency_ms
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
        extra: dict[str, Any] | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> AsyncIterator[Chunk]:
        if not base_url:
            raise ValidationError("openai_compat 适配器必须提供 base_url", detail={"model": model})
        client = self._http(timeout_s)
        api_key = self._resolve_api_key(api_key_env)
        url = base_url.rstrip("/") + "/chat/completions"
        payload = self._build_payload(messages, model, tools, temperature, max_output_tokens, True, extra)

        try:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=timeout_s,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "ignore")
                    raise self._map_http_error(response.status_code, body, model=model)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = self._parse_stream_chunk(data)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"模型 {model} 流式调用超时") from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"模型 {model} 流式网络异常: {type(exc).__name__}") from exc

    @staticmethod
    def _parse_stream_chunk(data: str) -> Chunk | None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return None
        choices = payload.get("choices") or [{}]
        choice = choices[0]
        delta = choice.get("delta") or {}
        usage_raw = payload.get("usage")
        usage = None
        if usage_raw:
            usage = Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
                total_tokens=int(usage_raw.get("total_tokens") or 0),
            )
        tool_delta = None
        if delta.get("tool_calls"):
            first = delta["tool_calls"][0]
            function = first.get("function") or {}
            tool_delta = {
                "index": first.get("index", 0),
                "id": first.get("id"),
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        if not delta.get("content") and tool_delta is None and choice.get("finish_reason") is None and usage is None:
            return None
        return Chunk(
            delta_text=delta.get("content") or "",
            tool_call_delta=tool_delta,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )
