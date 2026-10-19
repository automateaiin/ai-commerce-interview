"""给 infra 白名单 Tool 用的业务端口实现（`CommercePort` 的适配器）。

这个文件是**依赖倒置的落点**，也是唯一允许「infra 视角」存在的文件：

- infra 侧 `src/infra/tools/commerce.py` 只声明 `CommercePort` Protocol（它需要什么），
  对本文件与 `CommerceService` 一无所知 —— 所以换掉整个业务库，infra 一行都不用改；
- 本文件知道两边：把 infra 需要的四个动作翻译成用例调用，并把领域结果
  翻译回**工具契约**（字段名、中文状态标签）。

三个必须守住的契约（破了它们，Agent 侧的既有断言就会连带失败）：
1) `get_order_status` 的 `status` 必须是**中文标签**（历史口径）；
2) 查别人的订单必须是 `permission_denied` 的 403，而不是 404 —— 
   「不存在」和「不是你的」要能区分，否则 Agent 无法给出正确话术；
3) 扣库存失败/商品不存在必须是结构化错误（`not_found` / 库存不足），
   由 Harness 当作 observation 回填，**不能**是 500。
"""

from __future__ import annotations

from typing import Any

from src.infra.core.errors import ForbiddenError, NotFoundError

from .application.usecases.common import product_brief, product_detail
from .application.usecases.order import OrderLine
from .service import CommerceService


class CommerceToolsAdapter:
    """实现 infra 的 `CommercePort`。结构上满足 Protocol 即可，不继承（鸭子类型）。"""

    def __init__(self, service: CommerceService) -> None:
        self._service = service
        self.backend = f"commerce:{service.backend}"

    @staticmethod
    def _first_sku(product) -> str | None:
        return product.skus[0].sku_id if product.skus else None

    async def search_products(
        self,
        *,
        query: str | None = None,
        skin_type: str | None = None,
        max_price_usd: float | None = None,
        market: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        products, _total = await self._service.catalog.search(
            keyword=query,
            skin_type=skin_type,
            market=market,
            max_price_usd=max_price_usd,
            offset=0,
            limit=limit,
        )
        skus = [sku for sku in (self._first_sku(product) for product in products) if sku]
        stocks = await self._service.catalog.stocks_for(skus)
        items = [product_brief(product, stocks.get(self._first_sku(product))) for product in products]
        return {"items": items, "count": len(items), "market": market}

    async def product_detail(self, product_id: str) -> dict[str, Any]:
        product = await self._service.catalog.detail(product_id)  # 不存在 → NotFoundError(not_found)
        sku = self._first_sku(product)
        stocks = await self._service.catalog.stocks_for([sku] if sku else [])
        return product_detail(product, stocks.get(sku))

    async def order_status(self, order_id: str, *, user_id: str | None = None) -> dict[str, Any]:
        payload = await self._service.orders.get_order_payload(order_id, user_id=user_id)
        return {
            "order_id": payload["order_id"],
            "status": payload["status"],  # 中文标签，保持既有工具契约
            "carrier": payload["carrier"],
            "items": [
                {"product_id": item["product_id"], "quantity": item["quantity"], "price_usd": item["price_usd"]}
                for item in payload["items"]
            ],
            "total_usd": payload["total_usd"],
            "updated_at": payload["created_at"],
        }

    async def create_order(
        self,
        *,
        product_id: str,
        quantity: int,
        idempotency_key: str,
        market: str | None = None,
        user_id: str | None = None,
        project_id: str = "shop",
    ) -> dict[str, Any]:
        if not user_id:
            # 下单是**有主体**的动作：拿到 ToolContext 却没主体，只能是上游没注入身份
            raise ForbiddenError("下单需要已认证的用户身份（user_id 缺失）", detail={"tool": "create_order"})
        if not idempotency_key:
            raise NotFoundError("下单缺少幂等键", detail={"tool": "create_order"})
        payload = await self._service.orders.place_order(
            user_id,
            [OrderLine(product_id=product_id, quantity=int(quantity))],
            idempotency_key=idempotency_key,
            project_id=project_id,
            market=market,
        )
        # 工具侧沿用「status 为中文标签」的口径；REST 侧仍用 status=枚举值 + status_label
        payload["status"] = payload["status_label"]
        return payload


__all__ = ["CommerceToolsAdapter"]
