"""编排模式注册表：按名分发。

有了 Registry，路由/规划层只需要产出「模式名」，不必 import 具体类 ——
换实现、加新模式都是往注册表里塞一个对象，调用方零改动。
"""

from __future__ import annotations

from src.infra.core.errors import NotFoundError

from .base import OrchestrationMode
from .fanout import FanOutFanInMode
from .handoff import HandoffMode
from .sequential import SequentialMode


class ModeRegistry:
    def __init__(self, modes: list[OrchestrationMode] | None = None) -> None:
        self._modes: dict[str, OrchestrationMode] = {}
        for item in modes or []:
            self.register(item)

    def register(self, mode: OrchestrationMode) -> OrchestrationMode:
        if mode.name in self._modes:
            raise ValueError(f"编排模式重名: {mode.name}")
        self._modes[mode.name] = mode
        return mode

    def get(self, name: str) -> OrchestrationMode:
        mode = self._modes.get(name)
        if mode is None:
            raise NotFoundError(f"未知编排模式: {name}", detail={"valid_modes": self.names()})
        return mode

    def has(self, name: str) -> bool:
        return name in self._modes

    def names(self) -> list[str]:
        return sorted(self._modes)


def build_mode_registry(*, max_handoffs: int) -> ModeRegistry:
    """装配三种模式。Handoff 的上限来自配置（超限即停在交接态）。"""
    return ModeRegistry(
        [
            SequentialMode(),
            FanOutFanInMode(),
            HandoffMode(max_handoffs=max_handoffs),
        ]
    )


__all__ = ["ModeRegistry", "build_mode_registry"]
