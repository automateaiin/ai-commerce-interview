"""商品目录用例：读为主，写只有「上架/下架」与「人工改库存」两处后台动作。

一个刻意的分界（面试可讲）：
- **消费者侧**（REST /v1/shop/products、白名单 Tool search_products）只能看到
  `ON_SALE` 的商品：未上架商品出现在货架上，是最典型的内容泄露；
- **后台侧**（/v1/admin/*）才允许按任意状态检索、上下架、改库存。
  同一个仓储，靠**用例参数**区分权限面，而不是在仓储里写 if。

`publish` 走领域状态机（`Product.transition_to`）：未定价或无组合的商品不能上架，
这条规则在领域层，用例不做二次判断 —— 否则规则就会有两个版本。
"""

from __future__ import annotations

from typing import Sequence

from src.infra.core.errors import NotFoundError

from ...domain.models import Product, ProductStatus, Stock
from ...domain.services import StockAllocator
from ..ports import UowFactory


class CatalogUseCase:
    def __init__(self, uow_factory: UowFactory) -> None:
        self._uow = uow_factory
        self._allocator = StockAllocator()

    async def search(
        self,
        *,
        keyword: str | None = None,
        category: str | None = None,
        skin_type: str | None = None,
        market: str | None = None,
        status: str | None = ProductStatus.ON_SALE.value,
        max_price_usd: float | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Product], int]:
        max_price_cents = int(round(max_price_usd * 100)) if max_price_usd is not None else None
        async with self._uow() as uow:
            products, total = await uow.products.search(
                keyword=keyword,
                category=category,
                skin_type=skin_type,
                market=market,
                status=status,
                max_price_cents=max_price_cents,
                offset=offset,
                limit=limit,
            )
            await uow.commit()
        return products, total

    async def detail(self, product_id: str) -> Product:
        async with self._uow() as uow:
            product = await uow.products.get(product_id)
            await uow.commit()
        if product is None:
            raise NotFoundError(f"商品不存在: {product_id}", detail={"product_id": product_id})
        return product

    async def transition(self, product_id: str, target: ProductStatus) -> Product:
        """上架 / 下架。非法迁移由领域状态机抛 409（`invalid_state_transition`）。"""
        async with self._uow() as uow:
            product = await uow.products.get(product_id)
            if product is None:
                raise NotFoundError(f"商品不存在: {product_id}", detail={"product_id": product_id})
            product.transition_to(target)
            await uow.products.upsert(product)
            await uow.commit()
        return product

    async def stocks_for(self, sku_ids: Sequence[str]) -> dict[str, Stock]:
        """批量取库存。用于商品列表/详情回填「有没有货」。"""
        result: dict[str, Stock] = {}
        async with self._uow() as uow:
            for sku_id in sku_ids:
                stock = await uow.stocks.get(sku_id)
                if stock is not None:
                    result[sku_id] = stock
            await uow.commit()
        return result

    async def low_stock(self) -> list[Stock]:
        async with self._uow() as uow:
            stocks = await uow.stocks.list_low()
            await uow.commit()
        return stocks

    async def adjust_stock(self, sku_id: str, available: int) -> Stock:
        """后台人工改库存。这里设的是**绝对值**而不是增量：
        增量接口在「补货同时有人下单」时会算错基准，绝对值能让操作可复核。"""
        async with self._uow() as uow:
            stock = await uow.stocks.get(sku_id)
            if stock is None:
                raise NotFoundError(f"库存记录不存在: {sku_id}", detail={"sku_id": sku_id})
            stock.available = max(0, int(available))
            await uow.stocks.upsert(stock)
            await uow.commit()
        return stock

    def resolve_sku(self, product: Product, choice: dict[str, str] | None = None) -> str:
        """把「商品 + 维度选择」解析成可售 SKU。规则来自领域服务，不由用例重复实现。"""
        return self._allocator.resolve(product, choice)


__all__ = ["CatalogUseCase"]
