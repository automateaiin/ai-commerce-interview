"""消费者侧接口（`/v1/shop/*`）。

两个刻意的取舍（面试可讲）：
1) **浏览开放、交易鉴权**：商品列表与详情不需要凭证（货架是公开信息，硬要 API Key
   会让「分享一个商品链接」变成不可能）；而购物车、订单、我的券一律要会员 token ——
   「看得见」和「动得了」是两种权限面。
2) **下单接口与白名单 Tool 走同一段用例**：`/v1/shop/orders` 与 Agent 侧的
   `create_order` 都调 `OrderUseCase.place_order`，所以幂等键、原子扣库存、
   Outbox 事件在两条入口上完全一致 —— 不存在「网页有幂等、Agent 没有」这种裂缝。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from src.infra.core.errors import NotFoundError

from ...application.usecases.common import cart_payload, order_payload, product_brief, product_detail
from ...application.usecases.order import OrderLine
from ...domain.models import CouponStatus, ProductStatus
from ...service import CommerceService
from ..deps import MemberPrincipal, current_member, get_commerce
from ..schemas import (
    AddressCreateRequest,
    CartAddRequest,
    MemberLoginRequest,
    MemberRegisterRequest,
    PlaceOrderRequest,
)

router = APIRouter(prefix="/v1/shop", tags=["commerce-shop"])


def _default_sku(product) -> str | None:
    return product.skus[0].sku_id if product.skus else None


# --------------------------------------------------------------------------- #
# 商品（公开）
# --------------------------------------------------------------------------- #
@router.get("/products")
async def list_products(
    keyword: str | None = Query(default=None, max_length=60),
    category: str | None = Query(default=None, max_length=40),
    skin_type: str | None = Query(default=None, max_length=20),
    market: str | None = Query(default=None, max_length=16),
    max_price_usd: float | None = Query(default=None, ge=0),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    products, total = await commerce.catalog.search(
        keyword=keyword,
        category=category,
        skin_type=skin_type,
        market=market,
        max_price_usd=max_price_usd,
        offset=offset,
        limit=limit,
    )
    stocks = await commerce.catalog.stocks_for([sku for sku in (_default_sku(p) for p in products) if sku])
    items = [product_brief(product, stocks.get(_default_sku(product))) for product in products]
    return {"items": items, "count": len(items), "total": total, "offset": offset}


@router.get("/products/{product_id}")
async def get_product(product_id: str, commerce: CommerceService = Depends(get_commerce)) -> dict[str, Any]:
    product = await commerce.catalog.detail(product_id)
    if product.status is not ProductStatus.ON_SALE:
        # 未上架的商品在买家侧必须按「不存在」处理：返回「已下架」等于泄露了运营动作
        raise NotFoundError(f"商品不存在: {product_id}", detail={"product_id": product_id})
    stocks = await commerce.catalog.stocks_for([sku for sku in (_default_sku(product) or "",) if sku])
    return product_detail(product, stocks.get(_default_sku(product)))


# --------------------------------------------------------------------------- #
# 会员
# --------------------------------------------------------------------------- #
@router.post("/members/register")
async def register(body: MemberRegisterRequest, commerce: CommerceService = Depends(get_commerce)) -> dict[str, Any]:
    member = await commerce.members.register(email=body.email, password=body.password, nickname=body.nickname)
    return {"member": member.to_dict()}


@router.post("/members/login")
async def login(body: MemberLoginRequest, commerce: CommerceService = Depends(get_commerce)) -> dict[str, Any]:
    member, token, expires_at = await commerce.members.login(email=body.email, password=body.password)
    return {"member": member.to_dict(), "token": token, "expires_at": expires_at}


@router.get("/members/me")
async def me(
    principal: MemberPrincipal = Depends(current_member), commerce: CommerceService = Depends(get_commerce)
) -> dict[str, Any]:
    return {"member": (await commerce.members.profile(principal.member_id)).to_dict()}


@router.post("/members/addresses")
async def add_address(
    body: AddressCreateRequest,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    address = await commerce.members.add_address(
        principal.member_id,
        receiver=body.receiver,
        phone=body.phone,
        country=body.country,
        city=body.city,
        line1=body.line1,
        is_default=body.is_default,
    )
    return {"address": address.masked()}


# --------------------------------------------------------------------------- #
# 购物车
# --------------------------------------------------------------------------- #
@router.get("/cart")
async def view_cart(
    principal: MemberPrincipal = Depends(current_member), commerce: CommerceService = Depends(get_commerce)
) -> dict[str, Any]:
    return {"cart": cart_payload(await commerce.cart.get(principal.member_id))}


@router.post("/cart/items")
async def add_cart_item(
    body: CartAddRequest,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    await commerce.cart.add(
        principal.member_id, body.product_id, body.quantity, sku_id=body.sku_id, choice=body.choice
    )
    # 回整车而不是回刚加的那一行：客户端需要拿到最新总价才能刷新结算栏
    return {"cart": cart_payload(await commerce.cart.get(principal.member_id))}


@router.delete("/cart/items/{sku_id}")
async def remove_cart_item(
    sku_id: str,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    await commerce.cart.remove(principal.member_id, sku_id)
    return {"cart": cart_payload(await commerce.cart.get(principal.member_id))}


@router.delete("/cart")
async def clear_cart(
    principal: MemberPrincipal = Depends(current_member), commerce: CommerceService = Depends(get_commerce)
) -> dict[str, Any]:
    await commerce.cart.clear(principal.member_id)
    return {"cleared": True}


# --------------------------------------------------------------------------- #
# 订单
# --------------------------------------------------------------------------- #
@router.post("/orders")
async def place_order(
    body: PlaceOrderRequest,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    payload = await commerce.orders.place_order(
        principal.member_id,
        [OrderLine(**line.model_dump()) for line in body.items],
        idempotency_key=body.idempotency_key,
        project_id="shop",
        coupon_id=body.coupon_id,
        market=body.market,
    )
    return {"order": payload}


@router.get("/orders")
async def list_my_orders(
    limit: int = Query(default=20, ge=1, le=100),
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    orders = await commerce.orders.list_by_member(principal.member_id, limit=limit)
    return {"items": [order_payload(order) for order in orders], "count": len(orders)}


@router.get("/orders/{order_id}")
async def get_order(
    order_id: str,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    return {"order": await commerce.orders.get_order_payload(order_id, user_id=principal.member_id)}


@router.post("/orders/{order_id}/pay")
async def pay_order(
    order_id: str,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    order = await commerce.orders.pay(order_id, actor_member_id=principal.member_id)
    return {"order": order_payload(order)}


@router.post("/orders/{order_id}/cancel")
async def cancel_order(
    order_id: str,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    order = await commerce.orders.cancel(order_id, actor_member_id=principal.member_id)
    return {"order": order_payload(order, replay=False)}


# --------------------------------------------------------------------------- #
# 优惠券
# --------------------------------------------------------------------------- #
@router.get("/coupons/templates")
async def list_coupon_templates(commerce: CommerceService = Depends(get_commerce)) -> dict[str, Any]:
    templates = await commerce.coupons.list_templates()
    return {"items": [template.to_dict() for template in templates], "count": len(templates)}


@router.post("/coupons/{template_id}/receive")
async def receive_coupon(
    template_id: str,
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    coupon = await commerce.coupons.receive(principal.member_id, template_id)
    return {"coupon": coupon.to_dict()}


@router.get("/coupons/mine")
async def my_coupons(
    status: str | None = Query(default=None, description="按状态过滤：RECEIVED / USED / EXPIRED"),
    principal: MemberPrincipal = Depends(current_member),
    commerce: CommerceService = Depends(get_commerce),
) -> dict[str, Any]:
    coupons = await commerce.coupons.mine(principal.member_id)
    if status:
        wanted = CouponStatus(status)
        coupons = [coupon for coupon in coupons if coupon.status is wanted]
    return {"items": [coupon.to_dict() for coupon in coupons], "count": len(coupons)}
