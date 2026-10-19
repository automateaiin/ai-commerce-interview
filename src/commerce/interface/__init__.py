"""接口层：HTTP 协议适配（deps / schemas / routers）。

对外只暴露两个路由对象，具体端点见 `routers/shop.py`（买家侧）与 `routers/admin.py`（运营侧）。
"""

from .deps import MemberPrincipal, current_member, get_commerce, require_commerce_admin
from .routers import admin_router, shop_router

__all__ = ["MemberPrincipal", "admin_router", "current_member", "get_commerce", "require_commerce_admin", "shop_router"]
