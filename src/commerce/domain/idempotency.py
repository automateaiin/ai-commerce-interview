"""幂等请求指纹：把一次写请求的语义压成一个稳定哈希。

放在 domain（纯 stdlib）而不是 application 或 infrastructure：
它被**两侧同时使用** —— 用例层写幂等表时算指纹，基础设施层落库时存指纹。
放在任一侧都会造成另一侧的跨层依赖；放在领域层则两边都能干净引用。

为什么必须「键排序 + 紧凑分隔符」：
`{"a":1,"b":2}` 与 `{"b":2,"a":1}` 是同一个下单意图，
如果直接 `json.dumps` 默认（带空格、按插入顺序），两个语义相同的请求会得到不同哈希，
幂等判定就失效了 —— 用户重放会真的重复下单。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def hash_request(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["hash_request"]
