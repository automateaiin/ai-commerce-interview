"""业务 ID 生成。放在 domain 的原因：它是**纯 stdlib 的标识规则**，
用例层与仓储层都要用；放在任一侧都会让另一侧跨层依赖。

为什么不用数据库自增 ID：自增会暴露业务量（竞品能通过订单号推算日单量），
分库后还需要额外的号段服务；随机 ID 天然无碰撞前提下的全局唯一（这里配合唯一键兜底）。
"""

from __future__ import annotations

import secrets


def new_id(prefix: str, *, length: int = 8) -> str:
    return f"{prefix}_{secrets.token_hex(length // 2)}"


def new_prefixed_id(prefix: str, *, length: int = 4) -> str:
    """带连字符的可读 ID，例如 `SKU-3F2A`。仅用于对外可见的商品号。"""
    return f"{prefix}-{secrets.token_hex(length // 2).upper()}"


__all__ = ["new_id", "new_prefixed_id"]
