"""应用层用例：一个用例 = 一个 UoW = 一次「读聚合 → 判规则 → 落库 → commit」。

`application/` 里没有 SQL、没有 HTTP、没有模型调用，只有编排；
这也是「换个数据库不用改用例、换个协议不用改用例」的前提。

- `catalog.py`：商品目录（读为主 + 后台上下架 / 改库存）
- `cart.py`：购物车（价格快照 + 库存前置提示）
- `order.py`：下单（原子扣库存 + 幂等 + Outbox + 状态迁移）
- `member.py`：注册 / 登录 / 资料 / 地址
- `coupon.py`：领券 / 我的券 / 券模板
"""

from .cart import CartUseCase
from .catalog import CatalogUseCase
from .common import cart_payload, order_payload, product_brief, product_detail, ucm
from .coupon import CouponUseCase
from .member import MemberUseCase
from .order import OrderLine, OrderUseCase

__all__ = [
    "CartUseCase",
    "CatalogUseCase",
    "CouponUseCase",
    "MemberUseCase",
    "OrderLine",
    "OrderUseCase",
    "cart_payload",
    "order_payload",
    "product_brief",
    "product_detail",
    "ucm",
]
