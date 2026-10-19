"""工具注册工厂：把各业务域工具装配进同一个注册表。

新增一个业务域（例如「营销素材检索」）时，只需要在这里注册，
MCP Server / 桥接层 / Agent Loop 会自动看到它 —— 这是「统一工具协议」的实际收益。
"""

from __future__ import annotations

from ..memory.store import MemoryStore
from ..rag.service import KnowledgeService
from .base import ToolRegistry
from .commerce import CommercePort, build_commerce_tools
from .knowledge import build_knowledge_tools, build_memory_tools


def build_registry(
    memory: MemoryStore, knowledge: KnowledgeService, commerce: CommercePort
) -> ToolRegistry:
    """装配工具注册表。业务端口（commerce）由组合根注入 —— 
    注册表只要求「满足 Protocol」，不认识具体是 SQLite 还是别的实现。"""
    registry = ToolRegistry()
    for tool in build_commerce_tools(commerce):
        registry.register(tool)
    for tool in build_knowledge_tools(knowledge):
        registry.register(tool)
    for tool in build_memory_tools(memory):
        registry.register(tool)
    return registry
