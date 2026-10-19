"""应用层：用例编排（事务边界、幂等、事件发布、AI 钩子）。

它只依赖 domain 定义的 Protocol 与自己的 ports，
因此可以在没有数据库、没有 Redis、没有真实 LLM 的情况下被单测。
"""

from __future__ import annotations

__all__ = ["__doc__"]
