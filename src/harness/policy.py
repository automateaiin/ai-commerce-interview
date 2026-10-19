"""执行策略：工具白名单与高风险动作确认。

为什么策略要独立成一层（而不是塞进工具实现里）：
「这个工具能不能做」是**项目/业务策略**，「这个工具怎么做」才是工具实现的职责。
策略会随业务方变化（客服不允许下单、销售允许下单但必须确认），
把它写进工具内部，等于每次业务调整都要改平台代码。

两层控制：
1) 白名单：项目级允许集合（不配 = 不限制，但默认按最小可用集配置）；
2) 确认门：产生业务副作用的动作（高风险）不允许模型自己拍板，必须带显式人工确认才执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.infra.tools.base import Tool, ToolRisk


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    allowed_tools: frozenset[str] | None = None
    """项目级白名单。None = 不额外限制（仍受 scopes 与 tool_names 约束）。"""

    confirm_risks: frozenset[ToolRisk] = field(default_factory=lambda: frozenset({ToolRisk.HIGH_RISK}))
    """需要人工确认的风险等级集合。默认只有 high_risk。"""

    def allows(self, tool: Tool) -> bool:
        return self.allowed_tools is None or tool.name in self.allowed_tools

    def requires_confirmation(self, tool: Tool) -> bool:
        return tool.risk in self.confirm_risks
