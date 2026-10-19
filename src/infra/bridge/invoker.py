"""工具调用通道：桥接层到工具服务之间的抽象。

两种实现（面试要讲清取舍）：
- `InProcessInvoker`：同进程直接调 ToolExecutor。零网络开销，适合模块化单体阶段；
- `McpHttpInvoker`（见 mcp/client.py）：走 MCP Streamable HTTP 调用远端服务。
  适合服务独立部署/独立扩缩容，但引入网络超时、连接管理与鉴权。

桥接层只依赖 ToolInvoker 协议，所以「先单体、后拆分」时业务代码不用改。
默认走进程内通道，通过 MCP_TRANSPORT=http 切换到协议调用。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..tools.base import ToolContext, ToolExecutor, ToolResult


class ToolInvoker(Protocol):
    transport: str

    async def invoke(self, tool_name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult: ...

    async def list_tools(self, ctx: ToolContext) -> list[str]: ...


class InProcessInvoker:
    """同进程调用：复用同一份 ToolRegistry 与执行器，保证校验逻辑只有一份。"""

    transport = "inprocess"

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    async def invoke(self, tool_name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        return await self._executor.execute(tool_name, arguments, ctx)

    async def list_tools(self, ctx: ToolContext) -> list[str]:
        return self._executor._registry.names()  # noqa: SLF001
