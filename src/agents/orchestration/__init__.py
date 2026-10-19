"""多 Agent 编排（项目一）。

三种模式统一到 `OrchestrationMode` Protocol + `ModeRegistry`：
- ``sequential``：按依赖串行，前序输出显式注入后序 prompt；
- ``fanout``：无依赖分支并行，最后收敛成统一结论；
- ``handoff``：主 Agent 跑不动（高风险/越权/缺信息）时带交接包转派专家，且有转派次数上限。

模式层不认识任何业务概念，只认「依赖、并行、交接」，因此新增 Agent 不必改模式代码。
"""

from .base import AgentRunner, ModeOutcome, OrchestrationMode
from .fanout import FanOutFanInMode
from .handoff import HandoffMode
from .registry import ModeRegistry, build_mode_registry
from .sequential import SequentialMode

__all__ = [
    "AgentRunner",
    "ModeOutcome",
    "OrchestrationMode",
    "SequentialMode",
    "FanOutFanInMode",
    "HandoffMode",
    "ModeRegistry",
    "build_mode_registry",
]
