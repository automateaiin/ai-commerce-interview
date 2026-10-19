"""模型目录（config/models.yaml 的强类型表示 + 校验）。

校验在这里做，而不是等到调用时才发现：配置是快照，坏配置必须在 reload 阶段被拒绝。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..core.errors import ValidationError


class ModelSpec(BaseModel):
    """一个逻辑模型的物理落地方式。"""

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 30.0
    max_output_tokens: int = 1024
    temperature: float = 0.3
    capabilities: list[str] = Field(default_factory=list)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    # 兼容供应商特有参数（例如 top_p、stop）；不允许放密钥
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def _provider_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider 不能为空")
        return value

    def estimated_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1000 * self.cost_per_1k_input
            + completion_tokens / 1000 * self.cost_per_1k_output
        )


class RoutingSpec(BaseModel):
    """任务类型 → 逻辑模型。避免所有请求都打到最贵的模型上。"""

    default: str
    overrides: dict[str, str] = Field(default_factory=dict)


class ApiKeySpec(BaseModel):
    """项目级 API Key（桥接层凭证）。

    生产环境应改为：库中存 key 的哈希 + 前缀（用于索引），支持轮换与吊销；
    这里保留明文只为了本地演示可跑。
    """

    key: str
    api_key_id: str
    project_id: str
    service: str | None = None
    """业务线标识（权限链第一级 Service）。留空时退化为 project_id，便于兼容旧配置。"""
    scopes: list[str] = Field(default_factory=list)
    requests_per_window: int | None = None
    tokens_per_window: int | None = None
    enabled: bool = True


class ModelCatalog(BaseModel):
    """整份模型目录。"""

    default_provider: str = "openai_compat"
    models: dict[str, ModelSpec]
    routing: RoutingSpec
    api_keys: list[ApiKeySpec] = Field(default_factory=list)

    def spec(self, logical_model: str) -> ModelSpec:
        try:
            return self.models[logical_model]
        except KeyError as exc:
            raise ValidationError(
                f"未定义的逻辑模型: {logical_model}",
                detail={"logical_model": logical_model, "available": sorted(self.models)},
            ) from exc

    def api_key_map(self) -> dict[str, ApiKeySpec]:
        return {item.key: item for item in self.api_keys if item.enabled}


def parse_catalog(raw: dict[str, Any]) -> ModelCatalog:
    """把 YAML 原始字典解析成强类型目录。解析失败统一抛 ValidationError。"""
    try:
        return ModelCatalog.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ValidationError("模型配置校验失败", detail={"reason": str(exc)[:400]}) from exc


def validate_catalog(catalog: ModelCatalog) -> None:
    """快照切换前的完整性校验：路由指向的模型必须存在。"""
    missing = [name for name in catalog.routing.overrides.values() if name not in catalog.models]
    if catalog.routing.default not in catalog.models:
        missing.append(catalog.routing.default)
    if missing:
        raise ValidationError("路由配置指向了不存在的逻辑模型", detail={"missing": sorted(set(missing))})
    if not catalog.models:
        raise ValidationError("配置中没有任何模型定义")
