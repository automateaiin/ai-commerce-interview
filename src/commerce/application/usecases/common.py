"""用例层的公共小工具：金额换算与对外 DTO 片段。

为什么不散落在各用例里：
金额从 `Money`（整数分）到「对外 float 美元」的换算**必须口径一致**，
否则订单接口与工具接口会出现 128.0 与 127.99999 这种对不上的数字；
把它收敛成一个函数，是「对外契约只有一个出口」的最小做法。
"""

from __future__ import annotations

from typing import Any

from ...domain.models import Cart, Order, Product, Stock
from ...domain.values import Money


def ucm(money: Money) -> float:
    """金额 → 对外美元浮点。只用于展示层，内部计算一律用 Money。"""
    return round(float(money.amount), 2)


def product_brief(product: Product, stock: Stock | None) -> dict[str, Any]:
    """商品列表项：只给「消费者决策需要」的字段，不铺全量成分与批次。"""
    return {
        "product_id": product.product_id,
        "name": product.name,
        "brand": product.brand,
        "category": product.category,
        "market": product.market,
        "price_usd": ucm(product.price),
        "skin_types": list(product.skin_types),
        "highlights": list(product.highlights),
        "status": product.status.value,
        "stock": stock.available if stock else 0,
    }


def product_detail(product: Product, stock: Stock | None) -> dict[str, Any]:
    """商品详情：在列表项基础上补齐成分、用法、规格与批次到期。"""
    return {
        **product_brief(product, stock),
        "usage": product.usage,
        "ingredients": [item.name for item in product.inci_normalized()],
        "skus": [
            {"sku_id": sku.sku_id, "shade": sku.shade, "spec": sku.spec, "price_usd": ucm(sku.price)}
            for sku in product.skus
        ],
        "batches": [batch.to_dict() for batch in product.batches],
        "version": product.version,
    }


def order_payload(order: Order, *, replay: bool = False) -> dict[str, Any]:
    """订单对外契约。工具与 REST 共用**同一份**结构，避免两处各写一套。"""
    items = [
        {
            "product_id": item.product_id,
            "sku_id": item.sku_id,
            "name": item.name,
            "quantity": item.quantity,
            "unit_price_usd": ucm(item.unit_price),
            "subtotal_usd": ucm(item.subtotal()),
        }
        for item in order.items
    ]
    payload: dict[str, Any] = {
        "order_id": order.order_id,
        "member_id": order.member_id,
        "user_id": order.member_id,
        "status": order.status.value,
        "status_label": order.status.label,
        "items": items,
        "goods_total_usd": ucm(order.goods_total()),
        "discount_usd": ucm(order.discount),
        "payable_usd": ucm(order.payable()),
        "coupon_id": order.coupon_id,
        "carrier": order.carrier,
        "created_at": order.created_at.isoformat(),
        "expires_at": order.expires_at.isoformat() if order.expires_at else None,
        "idempotent_replay": replay,
    }
    if len(items) == 1:
        # 单行订单额外回填工具契约里的快捷字段：Agent 侧不需要再拆 items 才能拿到总量
        payload["product_id"] = items[0]["product_id"]
        payload["quantity"] = items[0]["quantity"]
        payload["total_usd"] = items[0]["subtotal_usd"]
    return payload


def cart_payload(cart: Cart) -> dict[str, Any]:
    data = cart.to_dict()
    for item in data["items"]:
        item["subtotal_usd"] = ucm(Money(item["subtotal"]["cents"], item["subtotal"]["currency"]))
    data["total_usd"] = ucm(cart.line_total())
    return data


__all__ = ["cart_payload", "order_payload", "product_brief", "product_detail", "ucm"]
