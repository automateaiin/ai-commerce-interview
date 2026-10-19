"""演示种子数据：把原先只存在于内存工具层的美妆商品/订单搬进真实业务库。

动机（也是本项目的一条主线）：项目一/二里的电商工具原本读的是内存 mock
（`src/infra/data/demo_data.py`），那是「两套真相源」——工具说有货，业务库里查不到。
把同一批数据灌进业务库后，AI 工具链与真实交易链路终于指向同一个事实。

三条实现约束：
1. **幂等**：只在库为空时灌入，重复启动不会重复插（`force=True` 供测试重置）；
2. **可追溯**：订单状态用与线上一致的英文码（SHIPPED/PENDING_PAYMENT/COMPLETED），
   中文标签只在对外的适配层（`tools_port.py`）翻译，领域层不认识中文；
3. **不制造意外**：演示订单不设 `expires_at`，以免被超时 sweeper 顺手取消，
   让「打开就能演示」这件事保持稳定。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from src.infra.core.logging import get_logger
from src.infra.data.demo_data import ORDERS, PRODUCTS

from ..domain.models import (
    Member,
    MemberLevel,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    ProductStatus,
    Sku,
    Stock,
)
from ..domain.security import hash_password
from ..domain.values import InciIngredient, Money
from .db.models import MemberRecord, ProductRecord
from .db.session import CommerceDatabase
from .repositories import (
    SqlMemberRepository,
    SqlOrderRepository,
    SqlProductRepository,
    SqlStockRepository,
)

logger = get_logger("commerce.seed")

# 演示账号（店面页可直接用这组凭据登录，省掉「先注册再演示」的绕路）
DEMO_MEMBER_ID = "u_10086"
DEMO_MEMBER_EMAIL = "demo@example.com"
DEMO_MEMBER_PASSWORD = "Demo12345!"

_LEGACY_STATUS = {"已发货": OrderStatus.SHIPPED, "待付款": OrderStatus.PENDING_PAYMENT, "已完成": OrderStatus.COMPLETED}


def _build_product(entry: dict) -> tuple[Product, int]:
    """把演示商品条目转成领域聚合。单规格商品不建矩阵（避免表达式与 SKU 维度错位）。"""
    price = Money.of(entry["price_usd"])
    sku = Sku(sku_id=f"{entry['product_id']}-01", product_id=entry["product_id"], shade="", spec="", price=price)
    product = Product(
        product_id=entry["product_id"],
        name=entry["name"],
        brand=entry["brand"],
        category=entry["category"],
        market=entry["market"],
        price=price,
        skin_types=list(entry["skin_types"]),
        highlights=list(entry["highlights"]),
        usage=entry["usage"],
        status=ProductStatus.ON_SALE,
        skus=[sku],
        inci=[InciIngredient(name=name, function="配方成分") for name in entry["ingredients"]],
    )
    return product, int(entry["stock"])


async def seed_demo(db: CommerceDatabase, *, force: bool = False) -> dict[str, int]:
    """灌入演示数据。返回各表写入条数（便于 smoke 断言「真的进来了」）。"""
    async with db.session() as session:
        existing = int((await session.execute(select(func.count()).select_from(ProductRecord))).scalar_one())
        if existing and not force:
            return {"products": 0, "members": 0, "orders": 0, "skipped": 1}

        products = SqlProductRepository(session)
        stocks = SqlStockRepository(session)
        members = SqlMemberRepository(session)
        orders = SqlOrderRepository(session)

        product_names: dict[str, str] = {}
        for entry in PRODUCTS:
            product, stock_units = _build_product(entry)
            await products.upsert(product)
            await stocks.upsert(Stock(sku_id=product.skus[0].sku_id, available=stock_units))
            product_names[product.product_id] = product.name

        salt_hex, hash_hex = hash_password(DEMO_MEMBER_PASSWORD)
        demo = Member(
            member_id=DEMO_MEMBER_ID,
            email=DEMO_MEMBER_EMAIL,
            nickname="演示会员",
            level=MemberLevel.GOLD,
            salt_hex=salt_hex,
            hash_hex=hash_hex,
        )
        await members.save(demo)
        guest = Member(member_id="u_20001", email="guest@example.com", nickname="演示访客")
        session.add(MemberRecord(member_id=guest.member_id, email=guest.email, nickname=guest.nickname))
        await session.flush()

        for order_id, entry in ORDERS.items():
            status = _LEGACY_STATUS.get(entry["status"], OrderStatus.PENDING_PAYMENT)
            created = datetime.fromisoformat(str(entry["updated_at"]).replace("Z", "+00:00"))
            order = Order(
                order_id=order_id,
                member_id=entry["user_id"],
                items=[
                    OrderItem(
                        sku_id=f"{item['product_id']}-01",
                        product_id=item["product_id"],
                        name=product_names.get(item["product_id"], item["product_id"]),
                        quantity=int(item["quantity"]),
                        unit_price=Money.of(item["price_usd"]),
                    )
                    for item in entry["items"]
                ],
                status=status,
                carrier=entry["carrier"],
                created_at=created,
                expires_at=None,
                paid_at=created if status is not OrderStatus.PENDING_PAYMENT else None,
            )
            await orders.save(order, project_id="shop")

        await session.commit()

    counts = {"products": len(PRODUCTS), "members": 2, "orders": len(ORDERS), "skipped": 0}
    logger.info("commerce.seed.done", extra={"event": "startup", **counts, "status": "ok"})
    return counts


__all__ = [
    "DEMO_MEMBER_EMAIL",
    "DEMO_MEMBER_ID",
    "DEMO_MEMBER_PASSWORD",
    "seed_demo",
]
