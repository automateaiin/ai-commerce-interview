"""桥接层鉴权：API Key → 项目身份。

鉴权与授权是两件事，必须分开（面试高频追问）：
- **鉴权（Authentication）**：你是谁 → 用 API Key 查到 project_id / api_key_id；
- **授权（Authorization）**：你能做什么 → 校验 scopes 工具白名单，
  工具内部还要再按业务规则校验（例如订单是否属于该用户）。

安全要点：
- project_id **由服务端凭证推导**，客户端在请求体里传的 project_id 一律忽略；
- Key 比较用 `hmac.compare_digest` 做常量时间比较，避免时序侧信道；
- 生产环境 Key 应以哈希存储并用前缀索引查找，同时支持轮换与吊销（本实现为演示明文表）。
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field

from ..core.config import ConfigSnapshot
from ..core.errors import AuthenticationError, ForbiddenError
from ..core.logging import get_logger
from ..llm.catalog import ApiKeySpec, ModelCatalog

logger = get_logger("infra.bridge.auth")


@dataclass(slots=True)
class AuthContext:
    """一次调用的可信身份上下文。

    `service` 由**凭证**推导（不可伪造）；`agent` 由调用方声明（用于审计归属，
    因此**不参与鉴权**）—— 把两者混在一起，「谁有权」和「谁在用」就会变成同一个问题。
    """

    api_key_id: str
    project_id: str
    scopes: frozenset[str] = frozenset()
    requests_per_window: int | None = None
    tokens_per_window: int | None = None
    service: str | None = None
    agent: str | None = None
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def service_id(self) -> str:
        """Service 层的稳定标识。凭证未声明业务线时退化为 project_id，保证链路不断。"""
        return self.service or self.project_id

    def require_scope(self, scope: str, *, tool: str | None = None) -> None:
        if scope not in self.scopes:
            raise ForbiddenError(
                f"当前凭证缺少 scope: {scope}",
                detail={"required_scope": scope, "tool": tool, "api_key_id": self.api_key_id},
            )


class ApiKeyAuthenticator:
    """API Key 校验器。凭证表来自配置快照（可热更新、可轮换）。"""

    def __init__(self, catalog: ConfigSnapshot[ModelCatalog]) -> None:
        self._catalog = catalog

    async def authenticate(self, raw_key: str | None) -> AuthContext:
        if not raw_key:
            raise AuthenticationError("缺少 API Key", detail={"header": "x-api-key"})

        keys: dict[str, ApiKeySpec] = self._catalog.current().api_key_map()
        record = keys.get(raw_key)
        if record is None:
            # 用一次假比较抹平时序差异：存在与不存在的耗时要一致
            hmac.compare_digest(raw_key, raw_key)
            logger.warning(
                "auth.failed", extra={"event": "auth", "error_code": "unauthenticated", "status": "error"}
            )
            raise AuthenticationError("API Key 无效或已停用")

        if not hmac.compare_digest(record.key, raw_key):
            raise AuthenticationError("API Key 校验失败")

        auth = AuthContext(
            api_key_id=record.api_key_id,
            project_id=record.project_id,
            scopes=frozenset(record.scopes),
            requests_per_window=record.requests_per_window,
            tokens_per_window=record.tokens_per_window,
            service=record.service,
        )
        logger.info(
            "auth.ok",
            extra={
                "event": "auth",
                "api_key_id": auth.api_key_id,
                "project_id": auth.project_id,
                "service": auth.service_id,
                "status": "ok",
            },
        )
        return auth
