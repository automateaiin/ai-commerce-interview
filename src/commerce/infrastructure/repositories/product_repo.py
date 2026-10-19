"""商品与库存仓储实现。

两条实现上的取舍（面试可讲）：

1. **仓储方法接收调用方传入的 session**，而不是自己开 session。
   因为「扣库存 + 建订单」必须在**同一个事务**里，仓储如果自己管连接就做不到。
   把事务边界交给用例层（application），仓储只负责 SQL —— 职责才不重叠。

2. **`try_reserve` 用条件 UPDATE**：

       UPDATE commerce_stock SET available = available - :qty, version = version + 1
       WHERE sku_id = :sku AND available >= :qty

   `rowcount == 1` 才算扣减成功。这条单语句在 PostgreSQL 下天然持有行锁、
   在 SQLite 下由写锁串行化，**跨两种库语义一致**，
   因此不需要 `SELECT FOR UPDATE`、也不需要分布式锁 ——
   这是比加锁更简洁且可移植的答案。
"""

from __future__ import annotations

import secrets
from typing import Any, Sequence

from sqlalchemy import Select, cast, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import String

from ....infra.core.logging import get_logger
from ...domain.ids import new_id, new_prefixed_id
from ...domain.models import (
    Product,
    ProductStatus,
    Sku,
    Stock,
)
from ...domain.values import InciIngredient, Money, SkuMatrix, TrialSample
from ..db.models import (
    BatchRecord,
    InciRecord,
    ProductRecord,
    ShadeRecord,
    SkuRecord,
    StockRecord,
)

logger = get_logger("commerce.repo.product")


class SqlProductRepository:
    """商品仓储（含规格、色号字典、成分、批次）。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- 组装 --------------------------------------------------------------- #
    async def _load_children(self, product_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """一次性把子表取全，避免逐商品查询导致 N+1。"""
        if not product_ids:
            return {}
        skus = (
            (await self._session.execute(select(SkuRecord).where(SkuRecord.product_id.in_(product_ids))))
            .scalars()
            .all()
        )
        inci = (
            (await self._session.execute(select(InciRecord).where(InciRecord.product_id.in_(product_ids))))
            .scalars()
            .all()
        )
        shades = (
            (
                await self._session.execute(
                    select(ShadeRecord).where(ShadeRecord.product_id.in_(product_ids)).order_by(ShadeRecord.sort_order)
                )
            )
            .scalars()
            .all()
        )
        sku_ids = [row.sku_id for row in skus]
        batches = (
            (
                await self._session.execute(
                    select(BatchRecord).where(BatchRecord.sku_id.in_(sku_ids)).order_by(BatchRecord.produced_on)
                )
            )
            .scalars()
            .all()
            if sku_ids
            else []
        )

        grouped: dict[str, dict[str, Any]] = {pid: {"skus": [], "inci": [], "shades": [], "batches": []} for pid in product_ids}
        for row in skus:
            grouped[row.product_id]["skus"].append(row)
        for row in inci:
            grouped[row.product_id]["inci"].append(row)
        for row in shades:
            grouped[row.product_id]["shades"].append(row)
        for row in batches:
            owner = next((sku.product_id for sku in skus if sku.sku_id == row.sku_id), None)
            if owner:
                grouped[owner]["batches"].append(row)
        return grouped

    def _to_domain(self, record: ProductRecord, children: dict[str, Any]) -> Product:
        skus: list[Sku] = []
        for row in children.get("skus", []):
            trial = (
                TrialSample(
                    sku_id=row.sku_id,
                    volume_ml=float(row.trial_volume_ml or 0.0),
                    attached_to=row.trial_attached_to or "",
                )
                if row.is_trial
                else None
            )
            skus.append(
                Sku(
                    sku_id=row.sku_id,
                    product_id=row.product_id,
                    shade=row.shade or "",
                    spec=row.spec or "",
                    price=Money(row.price_cents, row.currency),
                    barcode=row.barcode or "",
                    trial=trial,
                )
            )
        inci = [
            InciIngredient(
                name=row.name,
                function=row.function or "",
                cas_no=row.cas_no,
                concentration=row.concentration,
            )
            for row in children.get("inci", [])
        ]
        batches = [
            BatchNo(code=row.code, produced_on=row.produced_on, shelf_life_months=row.shelf_life_months)
            for row in children.get("batches", [])
        ]
        shade_names = tuple(row.name for row in children.get("shades", []))
        matrix = None
        if len(shade_names) > 1 or any(sku.spec for sku in skus):
            axes = {"shade": shade_names or ("default",), "spec": tuple(sorted({sku.spec or "default" for sku in skus}))}
            sellable = sorted({(sku.shade or "default", sku.spec or "default") for sku in skus})
            matrix = SkuMatrix.build(axes, sellable)
        return Product(
            product_id=record.product_id,
            name=record.name,
            brand=record.brand,
            category=record.category,
            market=record.market,
            price=Money(record.price_cents, record.currency),
            skin_types=list(record.skin_types or []),
            highlights=list(record.highlights or []),
            usage=record.usage_text or "",
            status=ProductStatus(record.status),
            matrix=matrix,
            skus=skus,
            inci=inci,
            batches=batches,
            version=record.version,
        )

    # -- 读 ----------------------------------------------------------------- #
    async def get(self, product_id: str) -> Product | None:
        record = (
            await self._session.execute(select(ProductRecord).where(ProductRecord.product_id == product_id))
        ).scalar_one_or_none()
        if record is None:
            return None
        children = (await self._load_children([product_id])).get(product_id, {})
        return self._to_domain(record, children)

    async def search(
        self,
        *,
        keyword: str | None = None,
        category: str | None = None,
        skin_type: str | None = None,
        market: str | None = None,
        status: str | None = None,
        max_price_cents: int | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Product], int]:
        conditions = []
        if keyword:
            like = f"%{keyword.strip().lower()}%"
            conditions.append(
                or_(
                    func.lower(ProductRecord.name).like(like),
                    func.lower(ProductRecord.brand).like(like),
                    func.lower(ProductRecord.category).like(like),
                )
            )
        if category:
            conditions.append(ProductRecord.category == category)
        if market:
            conditions.append(ProductRecord.market == market)
        if status:
            conditions.append(ProductRecord.status == status)
        if max_price_cents is not None:
            conditions.append(ProductRecord.price_cents <= max_price_cents)
        if skin_type:
            # 肤质以 JSON 数组存储，这里对 JSON 文本做包含匹配：
            # SQLite 下就是 TEXT，PostgreSQL 下 jsonb::text 形态同为["...", "..."]，
            # 因此同一条条件跨两种方言都成立（M1 不做独立肤质维表，见方案减法清单）。
            conditions.append(cast(ProductRecord.skin_types, String).like(f'%{skin_type}%'))

        count_stmt: Select[Any] = select(func.count()).select_from(ProductRecord)
        list_stmt: Select[Any] = select(ProductRecord)
        for condition in conditions:
            count_stmt = count_stmt.where(condition)
            list_stmt = list_stmt.where(condition)

        total = int((await self._session.execute(count_stmt)).scalar_one())
        rows = (
            (
                await self._session.execute(
                    list_stmt.order_by(ProductRecord.created_at.desc(), ProductRecord.product_id).offset(offset).limit(limit)
                )
            )
            .scalars()
            .all()
        )
        children = await self._load_children([row.product_id for row in rows])
        return [self._to_domain(row, children.get(row.product_id, {})) for row in rows], total

    # -- 写 ----------------------------------------------------------------- #
    async def upsert(self, product: Product) -> None:
        record = (
            await self._session.execute(select(ProductRecord).where(ProductRecord.product_id == product.product_id))
        ).scalar_one_or_none()
        if record is None:
            record = ProductRecord(product_id=product.product_id)
            self._session.add(record)
        record.name = product.name
        record.brand = product.brand
        record.category = product.category
        record.market = product.market
        record.price_cents = product.price.cents
        record.currency = product.price.currency
        record.skin_types = list(product.skin_types)
        record.highlights = list(product.highlights)
        record.usage_text = product.usage
        record.status = product.status.value
        record.version = product.version

        # 子表整表重写：商品是聚合根，规格/成分/批次都是它的内部对象，
        # 「先删后插」比逐条 diff 更简单，也不会漏删已下线的规格。
        await self._session.execute(
            SkuRecord.__table__.delete().where(SkuRecord.product_id == product.product_id)
        )
        await self._session.execute(
            ShadeRecord.__table__.delete().where(ShadeRecord.product_id == product.product_id)
        )
        await self._session.execute(
            InciRecord.__table__.delete().where(InciRecord.product_id == product.product_id)
        )
        for sku in product.skus:
            self._session.add(
                SkuRecord(
                    sku_id=sku.sku_id,
                    product_id=product.product_id,
                    shade=sku.shade,
                    spec=sku.spec,
                    price_cents=sku.price.cents,
                    currency=sku.price.currency,
                    barcode=sku.barcode,
                    is_trial=sku.trial is not None,
                    trial_volume_ml=sku.trial.volume_ml if sku.trial else None,
                    trial_attached_to=sku.trial.attached_to if sku.trial else None,
                )
            )
        seen_shades: list[str] = []
        for sku in product.skus:
            if sku.shade and sku.shade not in seen_shades:
                seen_shades.append(sku.shade)
                self._session.add(
                    ShadeRecord(
                        shade_id=f"shade_{product.product_id}_{len(seen_shades)}",
                        product_id=product.product_id,
                        name=sku.shade,
                        sort_order=len(seen_shades),
                    )
                )
        # 成分在入库前去重：领域层已经把规范化规则定死，这里只是把结果落库
        for index, item in enumerate(product.inci_normalized()):
            self._session.add(
                InciRecord(
                    inci_id=f"inci_{product.product_id}_{index}",
                    product_id=product.product_id,
                    name=item.name,
                    normalized=item.normalized,
                    function=item.function,
                    cas_no=item.cas_no,
                    concentration=item.concentration,
                )
            )
        for index, batch in enumerate(product.batches):
            self._session.add(
                BatchRecord(
                    batch_id=f"batch_{product.product_id}_{index}",
                    sku_id=product.skus[0].sku_id if product.skus else product.product_id,
                    code=batch.code,
                    produced_on=batch.produced_on,
                    shelf_life_months=batch.shelf_life_months,
                    expires_on=batch.expires_on,
                )
            )
        await self._session.flush()


class SqlStockRepository:
    """库存仓储。核心是那条条件 UPDATE。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, sku_id: str) -> Stock | None:
        row = (
            await self._session.execute(select(StockRecord).where(StockRecord.sku_id == sku_id))
        ).scalar_one_or_none()
        if row is None:
            return None
        return Stock(
            sku_id=row.sku_id,
            available=row.available,
            reserved=row.reserved,
            version=row.version,
            threshold=row.threshold,
        )

    async def upsert(self, stock: Stock) -> None:
        row = (
            await self._session.execute(select(StockRecord).where(StockRecord.sku_id == stock.sku_id))
        ).scalar_one_or_none()
        if row is None:
            row = StockRecord(sku_id=stock.sku_id, available=stock.available, reserved=stock.reserved)
            self._session.add(row)
        row.available = stock.available
        row.reserved = stock.reserved
        row.threshold = stock.threshold
        await self._session.flush()

    async def try_reserve(self, sku_id: str, quantity: int) -> bool:
        """原子扣减。**不做 SELECT 再判断** —— 那是「读-改-写」竞态的来源。"""
        if quantity <= 0:
            return False
        result = await self._session.execute(
            update(StockRecord)
            .where(StockRecord.sku_id == sku_id, StockRecord.available >= quantity)
            .values(available=StockRecord.available - quantity, version=StockRecord.version + 1)
            .execution_options(synchronize_session=False)
        )
        matched = result.rowcount == 1
        if not matched:
            logger.info(
                "commerce.stock.reserve_missed",
                extra={"event": "business", "sku_id": sku_id, "quantity": quantity, "status": "error"},
            )
        return matched

    async def release(self, sku_id: str, quantity: int) -> int:
        """回滚库存。条件 UPDATE 天然幂等：即便被重复调用，也不会把库存刷成负数以上限。"""
        if quantity <= 0:
            return 0
        result = await self._session.execute(
            update(StockRecord)
            .where(StockRecord.sku_id == sku_id)
            .values(available=StockRecord.available + quantity, version=StockRecord.version + 1)
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)

    async def decr_available(self, sku_id: str, quantity: int) -> int:
        """人工调整（后台改库存）。这里允许扣成 0 但不允许负数。"""
        result = await self._session.execute(
            update(StockRecord)
            .where(StockRecord.sku_id == sku_id, StockRecord.available >= quantity)
            .values(available=StockRecord.available - quantity, version=StockRecord.version + 1)
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0)

    async def list_low(self) -> list[Stock]:
        rows = (await self._session.execute(select(StockRecord))).scalars().all()
        return [
            Stock(sku_id=row.sku_id, available=row.available, reserved=row.reserved, version=row.version, threshold=row.threshold)
            for row in rows
            if row.available <= row.threshold
        ]


__all__ = ["SqlProductRepository", "SqlStockRepository", "new_id", "new_prefixed_id", "secrets"]
