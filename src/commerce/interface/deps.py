"""商城 REST 接口层。

分层意图：`interface/` 只做「协议转换」——把 HTTP 请求翻译成用例入参、
把用例结果翻译成响应体。**不写业务规则**（那是 domain/application 的事），
也不直接碰 SQL（那是 infrastructure 的事）。

鉴权分两套，且互不通用（面试可讲）：
- `/v1/shop/*` 的浏览与下单走**会员 token**（`Authorization: Bearer <member token>`），
  商品浏览开放（货架本来就该公开），购物车/订单必须带 token；
- `/v1/admin/*` 走**项目级 API Key + `commerce:rw` scope**。
  两套凭证来自不同签发方，所以「会员 token 冒充后台」和「API Key 冒充会员」都不成立。
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, Request

from src.infra.bridge.auth import AuthContext
from src.infra.bridge.router import current_auth
from src.infra.container import Container
from src.infra.core.errors import AuthenticationError

from ..service import CommerceService


def get_commerce(request: Request) -> CommerceService:
    """从应用状态取业务门面。Container 是唯一组合根，门面挂在它上面。"""
    container: Container = request.app.state.container
    return container.commerce


@dataclass(slots=True)
class MemberPrincipal:
    """会员身份。**只带主体与过期时间**：等级/折扣一律查库，不由 token 承载。"""

    member_id: str
    expires_at: int


async def current_member(
    request: Request,
    authorization: str | None = Header(default=None),
) -> MemberPrincipal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError("缺少会员令牌（Authorization: Bearer <token>）", detail={"header": "authorization"})
    token = authorization[7:].strip()
    verified = get_commerce(request).signer.verify(token)  # 过期/签名错由领域层抛 401
    return MemberPrincipal(member_id=verified.member_id, expires_at=verified.expires_at)


def require_commerce_admin(auth: AuthContext = Depends(current_auth)) -> AuthContext:
    """后台权限闸门：`commerce:rw`。scope 校验统一走 AuthContext，不在这里手写 if。"""
    auth.require_scope("commerce:rw")
    return auth


__all__ = ["MemberPrincipal", "current_member", "get_commerce", "require_commerce_admin"]
