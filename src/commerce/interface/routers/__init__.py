"""商城 REST 路由汇总。两个前缀各自导出独立 router：

- `shop.router`  → `/v1/shop/*`（买家侧，会员 token）
- `admin.router` → `/v1/admin/*`（运营侧，API Key + `commerce:rw`）

`tests` 与 `scripts/smoke.py` 直接 include 这两个 router，无需经过 infra 的组装逻辑。
"""

from .admin import router as admin_router
from .shop import router as shop_router

__all__ = ["admin_router", "shop_router"]
