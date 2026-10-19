"""Prompt 模板中心的 YAML 契约与加载。

契约（`config/prompts.yaml`）分两部分：
- `templates`：片段库，一个 name 可有多版，只有一版 active；
- `roles`：Agent Role → 片段组装顺序（base layers + scenario 追加）。

启动时做**交叉校验**（Role 引用的片段必须存在）：把配置错误挡在启动阶段，
而不是等某个用户在某个场景下触发渲染失败才发现 —— 后者在线上是「部分话术悄悄缺失」，
比直接报错危险得多。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import ValidationError


class AgentRole(str, Enum):
    """工程内受支持的 Agent Role。新增角色必须同时补进这里，避免拼写漂移。"""

    customer = "customer"
    operations = "operations"
    inspector = "inspector"
    content = "content"

    @classmethod
    def parse(cls, value: "AgentRole | str") -> "AgentRole":
        if isinstance(value, AgentRole):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValidationError(
                f"未知 Agent Role: {value}", detail={"valid_roles": [item.value for item in cls]}
            ) from exc


class TemplateVersion(BaseModel):
    """模板的一个版本。`active=true` 表示当前生效版本。"""

    model_config = ConfigDict(extra="forbid")

    version: str = "v1"
    template: str
    note: str = ""
    active: bool = False


class RoleSpec(BaseModel):
    """角色组装规则：base 片段恒定注入，scenario 按任务追加。"""

    model_config = ConfigDict(extra="forbid")

    title: str
    variables: list[str] = Field(default_factory=list)
    layers: list[str] = Field(default_factory=list)
    scenarios: dict[str, list[str]] = Field(default_factory=dict)


class PromptCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    templates: dict[str, list[TemplateVersion]]
    roles: dict[str, RoleSpec] = Field(default_factory=dict)


def load_prompt_catalog(path: str | Path) -> PromptCatalog:
    """读取并校验 YAML 模板中心。任何不合法都在这里失败，不留给运行期。"""
    target = Path(path)
    if not target.exists():
        raise ValidationError(f"Prompt 配置文件不存在: {target}", detail={"path": str(target)})
    try:
        payload: Any = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        catalog = PromptCatalog.model_validate(payload)
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - pydantic/yaml 的错误统一转成项目内语义异常
        raise ValidationError(f"Prompt 配置解析失败: {target}: {exc}", detail={"path": str(target)}) from exc

    _validate_references(catalog, target)
    return catalog


def _validate_references(catalog: PromptCatalog, path: Path) -> None:
    for role_name, spec in catalog.roles.items():
        # 角色名必须在枚举内：防止 YAML 里写了个新角色却没人实现
        AgentRole.parse(role_name)
        referenced = [*spec.layers, *[name for names in spec.scenarios.values() for name in names]]
        missing = sorted({name for name in referenced if name not in catalog.templates})
        if missing:
            raise ValidationError(
                f"Prompt 角色 {role_name} 引用了不存在的模板: {missing}",
                detail={"path": str(path), "role": role_name, "missing": missing},
            )


def compose_layers(spec: RoleSpec, *, role: str, scenario: str | None) -> list[str]:
    """按顺序展开片段名：base layers + scenario layers。"""
    layers = list(spec.layers)
    if scenario:
        if scenario not in spec.scenarios:
            raise ValidationError(
                f"角色 {role} 不支持场景 {scenario}",
                detail={"role": role, "valid_scenarios": sorted(spec.scenarios)},
            )
        layers.extend(spec.scenarios[scenario])
    return layers


__all__ = [
    "AgentRole",
    "TemplateVersion",
    "RoleSpec",
    "PromptCatalog",
    "load_prompt_catalog",
    "compose_layers",
]
