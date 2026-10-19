"""不可变配置快照 + 原子切换。

为什么不是「改了 YAML 就自动生效」：
进程内持有一个**不可变**配置对象，重载时先解析成新对象、校验必填与映射完整性，
通过后才原子替换引用；每个请求在入口取一次快照，全程用同一份，
因此不会在请求中途读到「写了一半」的配置。
校验失败就保留旧配置并记录错误 —— 坏配置永远不生效。

面试追问「并发行为」时的答案：CPython 中属性重新绑定是原子操作，
读者要么看到旧对象、要么看到新对象，不存在中间态；且 `current()` 只在请求入口调用一次。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import yaml

from .errors import ValidationError
from .logging import get_logger

logger = get_logger("infra.config")

T = TypeVar("T")

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """展开 `${VAR}` / `${VAR:-default}`。支持字符串、list、dict 递归展开。

    配置里可以写 `api_key_env: ARK_API_KEY`，但绝不把密钥本体写进 YAML ——
    密钥只存在于环境变量或密钥管理服务。
    """
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            env_value = os.environ.get(name)
            if env_value is None:
                if default is None:
                    raise ValidationError(f"配置引用了未定义的环境变量 {name}")
                return default
            return env_value

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """读取 YAML 并展开环境变量。文件不存在/格式错误一律抛 ValidationError。"""
    file_path = Path(path)
    if not file_path.exists():
        raise ValidationError(f"配置文件不存在: {file_path}", detail={"path": str(file_path)})
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # noqa: PERF203
        raise ValidationError(f"YAML 解析失败: {file_path}", detail={"path": str(file_path)}) from exc
    if not isinstance(raw, dict):
        raise ValidationError("配置文件顶层必须是映射（dict）", detail={"path": str(file_path)})
    return expand_env(raw)


class ConfigSnapshot(Generic[T]):
    """带版本号的配置快照容器。

    只做三件事：读当前快照、校验后用新对象替换、暴露版本号。
    不做文件监听 —— 监听属于部署形态问题，与配置语义分开。
    """

    def __init__(self, value: T, *, validator: Callable[[T], None] | None = None, name: str = "config") -> None:
        self._value = value
        self._validator = validator
        self._name = name
        self._revision = 1

    @property
    def revision(self) -> int:
        return self._revision

    def current(self) -> T:
        """返回当前快照。请求入口调用一次即可，不要在每个函数里重复取。"""
        return self._value

    def swap(self, new_value: T) -> int:
        """校验通过后原子替换，返回新版本号；校验失败抛错且**保持旧配置可用**。"""
        if self._validator is not None:
            self._validator(new_value)
        self._value = new_value
        self._revision += 1
        logger.info(
            f"{self._name}.reload",
            extra={"event": "config_reload", "revision": self._revision},
        )
        return self._revision

    def reload_from(self, path: str | Path, loader: Callable[[dict[str, Any]], T]) -> int:
        """从文件重载：解析 → 构造 → 校验 → 切换。

        任一步失败都抛错，旧快照不受影响（这是「热更新读到半成品」的正解）。
        """
        raw = load_yaml_file(path)
        return self.swap(loader(raw))
