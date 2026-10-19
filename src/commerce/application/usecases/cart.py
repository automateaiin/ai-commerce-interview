"""购物车用例。

三条边界（面试可讲）：
1) **加购要快照价格**：购物车存的是「加入时的单价」。不存快照而每次实时取价，
   会让用户在结算页看到「价格和我刚加的不一样」；仓储层在合并行时**刷新快照**是为了
   「以最新价结算」，两者不冲突 —— 快照的语义是「结算时价的来源」，不是「锁价」。
2) **加购校验库存但不占用库存**：占用必须有超时释放，那是下单（预留）的职责。
   购物车只做「能不能加」的前置提示，真正的原子扣减在下单事务里。
3) **不校验会员存在性**：会员维度由 token 推导，购物车表以 member_id 为键。
   为不存在的会员建车是安全的（车没有资金含义），而查会员会给读路径加一次无谓查询。
"""

from __future__ import annotations

from src.infra.core.errors import ValidationError

from ...domain.errors import StockNotEnoughError
from ...domain.models import Cart, CartItem
from ...domain.values import Money
from ..ports import CachePort, UowFactory
from .catalog import CatalogUseCase

_MAX_CART_QUANTITY = 99


class CartUseCase:
    def __init__(self, uow_factory: UowFactory, catalog: CatalogUseCase, cache: CachePort) -> None:
        self._uow = uow_factory
        self._catalog = catalog
        self._cache = cache

    async def get(self, member_id: str) -> Cart:
        async with self._uow() as uow:
            cart = await uow.cart.get_by_member(member_id)
            await uow.commit()
        return cart

    async def add(
        self,
        member_id: str,
        product_id: str,
        quantity: int,
        *,
        sku_id: str | None = None,
        choice: dict[str, str] | None = None,
    ) -> CartItem:
        if not 1 <= int(quantity) <= _MAX_CART_QUANTITY:
            raise ValidationError(f"加购数量必须在 1~{_MAX_CART_QUANTITY}", detail={"quantity": quantity})

        product = await self._catalog.detail(product_id)
        product.assert_sellable()
        resolved = sku_id or self._catalog.resolve_sku(product, choice)
        price = next((sku.price for sku in product.skus if sku.sku_id == resolved), product.price)

        async with self._uow() as uow:
            stock = await uow.stocks.get(resolved)
            if stock is None:
                raise StockNotEnoughError(f"该规格暂无库存记录: {resolved}", detail={"sku_id": resolved})
            existing = (await uow.cart.get_by_member(member_id)).find(resolved)
            requested = (existing.quantity if existing else 0) + int(quantity)
            if stock.sellable < requested:
                raise StockNotEnoughError(
                    f"库存不足，当前可售 {stock.sellable}",
                    detail={"sku_id": resolved, "available": stock.sellable, "requested": requested},
                )
            await uow.cart.save_item(member_id, resolved, product.product_id, int(quantity), price.cents, price.currency)
            await uow.commit()

        # 购物车缓存以会员为键：加购后必须失效，否则用户看到的还是旧车
        await self._cache.delete(f"cart:{member_id}")
        return CartItem(
            sku_id=resolved,
            product_id=product.product_id,
            quantity=requested,
            price_snapshot=Money(price.cents, price.currency),
        )

    async def remove(self, member_id: str, sku_id: str) -> None:
        async with self._uow() as uow:
            await uow.cart.remove_item(member_id, sku_id)
            await uow.commit()
        await self._cache.delete(f"cart:{member_id}")

    async def clear(self, member_id: str) -> None:
        async with self._uow() as uow:
            await uow.cart.clear(member_id)
            await uow.commit()
        await self._cache.delete(f"cart:{member_id}")


__all__ = ["CartUseCase"]
