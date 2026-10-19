"""Prompt 版本管理 + YAML 模板中心。

为什么 Prompt 要版本化（面试会被问「为什么不用字符串常量」）：
Prompt 与模型、工具 Schema 一样属于「线上可变的策略」。一旦改了措辞，
就必须能回答三个问题：现在生效的是哪一版？上一版是什么？怎么回滚回去？
把 Prompt 当代码常量写死，等于把线上策略变更变成了不可追溯的发布。

为什么还要 Role 组合层（第二层抽象）：
同一个 Agent 在不同场景下只该差几句约束（例如「售后场景必须走人工确认」），
如果每个场景各写一份完整 Prompt，就会出现「改了基础口径、忘了改其中 5 个场景」的漂移。
所以：片段（template）可复用、可版本化，Role 只负责声明组装顺序。
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from ..core.errors import NotFoundError, ValidationError
from ..core.logging import get_logger
from .templates import AgentRole, PromptCatalog, RoleSpec, compose_layers, load_prompt_catalog

logger = get_logger("infra.prompts")


@dataclass(slots=True)
class PromptTemplate:
    name: str
    version: str
    template: str
    note: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = False


class PromptRegistry:
    """版本化模板注册表。同一 name 可有多个 version，只有一个 active。"""

    def __init__(self, *, roles: Mapping[str, RoleSpec] | None = None) -> None:
        self._templates: dict[str, dict[str, PromptTemplate]] = {}
        self._active: dict[str, str] = {}
        self._roles: dict[str, RoleSpec] = dict(roles or {})

    def register(self, name: str, template: str, *, version: str = "v1", note: str = "", activate: bool = False) -> PromptTemplate:
        if not template.strip():
            raise ValidationError(f"Prompt {name} 不能为空")
        item = PromptTemplate(name=name, version=version, template=template, note=note)
        versions = self._templates.setdefault(name, {})
        is_first = not versions
        versions[version] = item
        if activate or is_first:
            self.activate(name, version)
        logger.info(
            "prompt.registered",
            extra={"event": "prompt_registered", "reason": f"{name}@{version}", "changed_keys": note or version},
        )
        return item

    def activate(self, name: str, version: str) -> None:
        versions = self._templates.get(name)
        if not versions or version not in versions:
            raise NotFoundError(f"Prompt 版本不存在: {name}@{version}", detail={"name": name, "version": version})
        for item in versions.values():
            item.active = item.version == version
        self._active[name] = version

    def rollback(self, name: str) -> PromptTemplate:
        """回滚到上一版：版本顺序即回滚顺序。"""
        versions = self._templates.get(name)
        if not versions:
            raise NotFoundError(f"Prompt 不存在: {name}")
        ordered = sorted(versions)
        current = self._active.get(name)
        if len(ordered) < 2 or current is None:
            raise ValidationError(f"Prompt {name} 没有可回滚的历史版本")
        index = ordered.index(current)
        target = ordered[max(index - 1, 0)]
        self.activate(name, target)
        return versions[target]

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        versions = self._templates.get(name)
        if not versions:
            raise NotFoundError(f"Prompt 不存在: {name}")
        key = version or self._active.get(name)
        if key is None or key not in versions:
            raise NotFoundError(f"Prompt 版本不存在: {name}@{version}")
        return versions[key]

    def render(self, name: str, /, **variables: object) -> str:
        """渲染模板。缺失变量直接报错，不静默留空 —— 静默留空会让模型收到残缺指令。"""
        template = self.get(name)
        required = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template.template)
            if field_name and not field_name.startswith("_")
        }
        missing = {key for key in required if key not in variables}
        if missing:
            raise ValidationError(
                f"渲染 Prompt {name} 缺少变量: {sorted(missing)}", detail={"missing": sorted(missing)}
            )
        return template.template.format(**variables)

    def names(self) -> list[str]:
        """所有已注册的模板名（供运营视角查看模板中心全貌）。"""
        return sorted(self._templates)

    def versions(self, name: str) -> list[str]:
        return sorted(self._templates.get(name, {}))

    def active_version(self, name: str) -> str | None:
        return self._active.get(name)

    # -- Role 组合（YAML 模板中心） ------------------------------------------ #
    @property
    def roles(self) -> list[str]:
        return sorted(self._roles)

    def role_spec(self, role: AgentRole | str) -> RoleSpec:
        key = AgentRole.parse(role).value
        spec = self._roles.get(key)
        if spec is None:
            raise NotFoundError(f"未注册的 Agent Role: {key}", detail={"role": key, "valid_roles": self.roles})
        return spec

    def compose(self, role: AgentRole | str, *, scenario: str | None = None, **variables: object) -> str:
        """按角色组装并渲染最终 System Prompt。

        组装顺序：base layers → scenario layers，片段之间用空行分隔，
        让模型能看出「基础口径」与「本场景追加约束」的边界。
        """
        spec = self.role_spec(role)
        layers = compose_layers(spec, role=AgentRole.parse(role).value, scenario=scenario)
        parts = [self.render(name, **variables) for name in layers]
        logger.info(
            "prompt.composed",
            extra={
                "event": "prompt_composed",
                "role": AgentRole.parse(role).value,
                "scenario": scenario or "-",
                "layers": len(layers),
                "status": "ok",
            },
        )
        return "\n\n".join(parts)

    def role_blueprint(self, role: AgentRole | str, *, scenario: str | None = None) -> dict[str, object]:
        """角色的组装视图（供接口/文档展示「这一角色由哪些片段拼成」）。"""
        key = AgentRole.parse(role).value
        spec = self.role_spec(key)
        layers = compose_layers(spec, role=key, scenario=scenario)
        return {
            "role": key,
            "title": spec.title,
            "scenario": scenario,
            "variables": list(spec.variables),
            "scenarios": sorted(spec.scenarios),
            "layers": [{"name": name, "version": self.active_version(name)} for name in layers],
        }


def _default_catalog_path() -> Path:
    # 文件位置：src/infra/prompts/registry.py → 仓库根目录
    return Path(__file__).resolve().parents[3] / "config" / "prompts.yaml"


def build_default_registry(path: str | Path | None = None) -> PromptRegistry:
    """装配默认 Prompt 集。

    优先加载 YAML 模板中心（单一事实来源）；文件缺失或不可解析时**显式降级**到内置模板，
    保证离线与单测环境依然可用 —— 降级原因写进日志，不做静默吞异常。
    """
    target = Path(path) if path is not None else _default_catalog_path()
    if target.exists():
        try:
            catalog: PromptCatalog = load_prompt_catalog(target)
            registry = PromptRegistry(roles=catalog.roles)
            for name, versions in catalog.templates.items():
                for item in versions:
                    registry.register(
                        name, item.template, version=item.version, note=item.note, activate=item.active
                    )
            logger.info(
                "prompt.catalog_loaded",
                extra={
                    "event": "startup",
                    "status": "ok",
                    "reason": target.name,
                    "changed_keys": f"roles={len(catalog.roles)}",
                },
            )
            return registry
        except ValidationError as exc:
            logger.warning(
                "prompt.catalog_invalid",
                extra={
                    "event": "startup",
                    "status": "degraded",
                    "degraded": True,
                    "reason": exc.code,
                    "changed_keys": str(target.name),
                },
            )
    else:
        logger.warning(
            "prompt.catalog_missing",
            extra={
                "event": "startup",
                "status": "degraded",
                "degraded": True,
                "reason": "prompts_yaml_not_found",
                "changed_keys": str(target),
            },
        )
    return _build_builtin_registry()


def _build_builtin_registry() -> PromptRegistry:
    """内置兜底 Prompt 集（客服 / 推荐 / Agent Loop），仅在 YAML 不可用时启用。"""
    registry = PromptRegistry()
    registry.register(
        "cs.system",
        "你是高端美妆出海电商的智能客服。请依据提供的证据回答，"
        "涉及成分、功效、价格与库存时不得凭记忆编造；证据不足时明确说明无法确认，并建议转人工。"
        "回答语言与用户保持一致。",
        version="v1",
        note="客服基础口径：强约束依据证据回答",
    )
    registry.register(
        "cs.system",
        "你是高端美妆出海电商的智能客服。请依据提供的证据回答，"
        "涉及成分、功效、价格与库存时不得凭记忆编造；证据不足时明确说明无法确认，并建议转人工。"
        "回答语言与用户保持一致。先给出结论，再给依据，控制在 120 字以内。",
        version="v2",
        note="增加输出长度与结构约束（对 harness 更友好）",
    )
    registry.register(
        "agent.system",
        "你是电商业务子 Agent。你可以调用工具获取事实，然后给出结论。\n"
        "规则：1) 事实必须来自工具返回结果，不得编造；2) 信息不足时继续调用工具或向用户澄清；"
        "3) 工具报错时可根据错误信息调整参数；4) 最多 {max_steps} 步内结束。",
        version="v1",
        note="子 Agent 基础规则，步数上限由运行时注入",
    )
    registry.register(
        "rag.answer",
        "以下是检索到的知识片段，请仅依据它们回答：\n{evidence}\n\n用户问题：{question}",
        version="v1",
        note="RAG 证据注入模板",
    )
    return registry
