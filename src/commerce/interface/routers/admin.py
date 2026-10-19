"""运营侧接口（`/v1/admin/*`）。

为什么和消费者侧分成两个文件、两个前缀，而不是用 scope 在同一个 router 里做分支：
- 前缀不同 → 网关/日志/监控可以按路径**一眼区分买家流量与运营流量**，
  限流与告警策略也能分别配置，不需要解析请求体才知道是哪一类；
- 文件不同 → 「谁能上架商品」这种权限问题在 code review 时是显式的，
  而不是藏在某个 `if scope in ...` 后面。

全部端点要求 `commerce:rw` scope；`require_commerce_admin` 统一闸门。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query

from src.infra.bridge.auth import AuthContext
from src.infra.core.errors import ValidationError

from ...application.usecases.common import order_payload, product_brief, ucm
from ...domain.models import ProductStatus, utcnow
from ...service import CommerceService
from ..deps import get_commerce, require_commerce_admin
from ..schemas import CouponTemplateCreateRequest, StockAdjustRequest

router = APIRouter(prefix="/v1/admin", tags=["commerce-admin"], dependencies=[Depends(require_commerce_admin)])


@router.get("/orders/stats")
async def order_stats(
    days: int = Query(default=7, ge=1, le=90, description="统计最近 N 天（按创建时间）"),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    since = utcnow() - timedelta(days=days)
    return {"stats": await commerce.orders.stats(since=since), "since": since.isoformat()}


@router.get("/products")
async def list_products(
    keyword: str | None = Query(default=None, max_length=60),
    status: str | None = Query(default=None, description="按状态过滤：DRAFT / ON_SALE / OFF_SHELF；留空为全部"),
    limit: int = Query(default=50, ge=1, le=100),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    """后台商品列表（**全状态**）。

    买家侧 `/v1/shop/products` 只看得到 `ON_SALE`，若不提供这条，下架商品就再也回不到货架上 ——
    「内容不外泄」与「运营可复原」是两个都要满足的约束。
    """
    products, total = await commerce.catalog.search(keyword=keyword, status=status, limit=limit)
    stocks = await commerce.catalog.stocks_for(
        [sku.sku_id for product in products for sku in product.skus]
    )
    items: list[dict[str, Any]] = []
    for product in products:
        default_sku = product.skus[0].sku_id if product.skus else None
        item = product_brief(product, stocks.get(default_sku))
        # 列表带上 SKU：库存/改价的操作对象是 SKU，缺了它后台只能再点一次详情
        item["skus"] = [
            {"sku_id": sku.sku_id, "shade": sku.shade, "spec": sku.spec, "price_usd": ucm(sku.price)}
            for sku in product.skus
        ]
        items.append(item)
    return {"items": items, "count": len(items), "total": total}


@router.get("/orders")
async def list_orders(
    status: str | None = Query(
        default=None, max_length=32, description="按状态字面值过滤：PENDING_PAYMENT / PAID / SHIPPED / CANCELLED"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    """跨会员订单列表（运营视角）。买家侧 `/v1/shop/orders` 只能看自己的，这条补上运营的全量视图。"""
    orders = await commerce.orders.list_all(status=status, limit=limit)
    return {"items": [order_payload(order) for order in orders], "count": len(orders)}


@router.post("/orders/sweep")
async def sweep_expired_orders(commerce: CommerceService = Depends(get_commerce)) -> dict[str, Any]:
    """手动触发「超时未付款」清理。定时器与手动入口复用同一段用例，行为一致。"""
    return await commerce.orders.sweep_expired()


@router.post("/orders/{order_id}/ship")
async def ship_order(
    order_id: str,
    carrier: str = Query(min_length=1, max_length=40, description="承运商，如 DHL / FedEx"),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    order = await commerce.orders.ship(order_id, carrier=carrier)
    return {"order": order.to_dict()}


@router.get("/stock/low")
async def low_stock(
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    stocks = await commerce.catalog.low_stock()
    return {"items": [stock.to_dict() for stock in stocks], "count": len(stocks)}


@router.post("/stock/adjust")
async def adjust_stock(
    body: StockAdjustRequest, commerce: CommerceService = Depends(get_commerce)
) -> dict[str, Any]:
    stock = await commerce.catalog.adjust_stock(body.sku_id, body.available)
    return {"stock": stock.to_dict()}


@router.post("/products/{product_id}/publish")
async def publish_product(
    product_id: str, commerce: CommerceService = Depends(get_commerce)
) -> dict[str, Any]:
    product = await commerce.catalog.transition(product_id, ProductStatus.ON_SALE)
    return {"product": product.to_dict()}


@router.post("/products/{product_id}/unpublish")
async def unpublish_product(
    product_id: str, commerce: CommerceService = Depends(get_commerce)
) -> dict[str, Any]:
    product = await commerce.catalog.transition(product_id, ProductStatus.OFF_SHELF)
    return {"product": product.to_dict()}


@router.post("/coupons/templates")
async def create_coupon_template(
    body: CouponTemplateCreateRequest,
    auth: AuthContext = Depends(require_commerce_admin),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    if body.kind not in {"FULL_REDUCTION", "PERCENT"}:
        raise ValidationError("未知券类型", detail={"kind": body.kind})
    template = await commerce.coupons.create_template(
        name=body.name,
        kind=body.kind,
        threshold_usd=body.threshold_usd,
        value_cents=body.value_cents,
        total=body.total,
        per_member_limit=body.per_member_limit,
        stackable=body.stackable,
    )
    return {"template": template.to_dict(), "created_by": auth.api_key_id}
