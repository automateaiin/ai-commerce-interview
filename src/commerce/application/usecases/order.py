"""订单用例：本项目最重的一个用例，也是「原子扣库存 + 幂等下单 + Outbox」三件套的落点。

按发生顺序讲清楚每一步的**理由**（面试基本会按这个顺序追问）：

1) **业务限流在看库存之前**：会员 × 下单频率先过 `CachePort.incr_window`。
   放在最前面是因为它是**最便宜的拒绝** —— 被限流的请求不该消耗任何数据库事务。
2) **幂等键先查后写，且靠唯一约束兜底**：顺序是「先 `get` 命中就返回」，
   只有在并发穿透时才会走到 `save` 并撞唯一键 —— 那时捕获 `IntegrityError`，
   回读别人存下的响应。**顺序查-判断在并发下必然漏**，所以兜底必须是数据库约束。
3) **请求指纹（request_hash）防「同键不同体」**：同一个 key 携带不同购物意图时，
   返回上一次的结果是最典型的资金级事故，因此宁可 409 也不返回旧单。
4) **扣库存是条件 UPDATE**：`try_reserve(...) == False` 直接抛 `StockNotEnoughError`，
   事务整体回滚。这里是唯一会发生「超卖」的位置，所以它必须是单语句原子判定，
   **不能**「先 SELECT 判断再 UPDATE」。
5) **事件只进 Outbox，钩子在 commit 之后**：见 `application/events.py`。
   订单事务里绝不 await 推荐/知识抽取这类副作用。

事务边界一律用「一个用例 = 一个 UoW」，且 `commit()` 写在用例里而不是 `__aexit__`：
读到 `commit()` 就知道「副作用从这里开始」，这是能被 code review 抓住的写法。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from sqlalchemy.exc import IntegrityError

from src.infra.core.errors import ForbiddenError, NotFoundError, RateLimitedError, ValidationError

from ...domain.errors import (
    CouponNotApplicableError,
    DuplicateRequestConflictError,
    InvalidStateTransitionError,
    StockNotEnoughError,
)
from ...domain.events import OrderCancelled, OrderPlaced, StockLow
from ...domain.ids import new_id
from ...domain.idempotency import hash_request
from ...domain.models import Member, MemberLevel, Order, OrderItem, OrderStatus, utcnow
from ...domain.services import PricingPolicy, StockAllocator
from ...domain.values import Money
from ..events import EventRouter, OutboxPublisher
from ..ports import CachePort, UowFactory
from .common import order_payload

_MAX_LINES = 5
_MAX_LINE_QUANTITY = 10


@dataclass(slots=True)
class OrderLine:
    """下单行。`sku_id` 与 `choice` 二选一：直接指定规格，或给出维度让分配器解析。"""

    product_id: str
    quantity: int = 1
    sku_id: str | None = None
    choice: dict[str, str] | None = None


class OrderUseCase:
    def __init__(
        self,
        uow_factory: UowFactory,
        *,
        cache: CachePort,
        dispatcher: EventRouter,
        order_ttl_minutes: int = 30,
        per_member_hour_limit: int = 20,
    ) -> None:
        self._uow = uow_factory
        self._cache = cache
        self._dispatcher = dispatcher
        self._ttl = order_ttl_minutes
        self._hour_limit = per_member_hour_limit

    # -- 下单 ------------------------------------------------------------------ #
    async def place_order(
        self,
        member_id: str,
        lines: Sequence[OrderLine],
        *,
        idempotency_key: str,
        project_id: str = "shop",
        coupon_id: str | None = None,
        market: str | None = None,
    ) -> dict:
        self._validate_lines(lines)
        request_hash = hash_request(
            {
                "project_id": project_id,
                "member_id": member_id,
                "market": market,
                "coupon_id": coupon_id,
                "lines": [
                    {"product_id": line.product_id, "sku_id": line.sku_id, "quantity": line.quantity}
                    for line in lines
                ],
            }
        )

        # 1) 最便宜的拒绝放在最前面：被限流的请求不产生任何数据库事务
        counted = await self._cache.incr_window(f"order-rate:{member_id}", window_s=3600)
        if counted > self._hour_limit:
            raise RateLimitedError(
                f"下单过于频繁（每小时上限 {self._hour_limit} 笔）",
                retry_after=60,
                dimension="member_order",
                detail={"limit": self._hour_limit, "count": counted},
            )

        try:
            async with self._uow() as uow:
                # 2) 幂等先查：命中且指纹一致 -> 直接回放，一行都不写
                stored = await uow.idempotency.get(project_id, idempotency_key)
                if stored is not None:
                    if stored["request_hash"] != request_hash:
                        raise DuplicateRequestConflictError(
                            "幂等键已被另一次不同内容的请求使用",
                            detail={"idempotency_key": idempotency_key, "project_id": project_id},
                        )
                    return {**dict(stored["response"]), "idempotent_replay": True}

                order, payable = await self._build_order(
                    uow,
                    member_id,
                    lines,
                    coupon_id=coupon_id,
                    market=market,
                    idempotency_key=idempotency_key,
                )
                await uow.orders.save(order, project_id=project_id)

                payload = order_payload(order, replay=False)
                await uow.idempotency.save(
                    project_id=project_id, key=idempotency_key, request_hash=request_hash, response=payload
                )

                # 3) 事件与业务写在同一事务：commit 成则事件必在
                publisher = OutboxPublisher(uow.outbox)
                publisher.add(
                    OrderPlaced(
                        event_id=new_id("evt"),
                        order_id=order.order_id,
                        member_id=member_id,
                        amount_cents=payable.cents,
                        currency=payable.currency,
                        item_summary="; ".join(f"{item.name}x{item.quantity}" for item in order.items),
                    )
                )
                await self._flag_low_stock(uow, publisher, order)
                events = list(publisher.pending)
                await publisher.flush()  # **必须在 commit 前**
                await uow.commit()
        except IntegrityError as exc:
            # 并发穿透唯一键：另一个同键请求已经落库，回读它的响应（绝不重复下单）
            return await self._replay_after_conflict(project_id, idempotency_key, request_hash, exc)

        # 4) 提交之后才触发 AI 钩子：钩子失败不影响已成立的订单
        await self._dispatcher.dispatch(events)
        return payload

    async def _build_order(
        self,
        uow,
        member_id: str,
        lines: Sequence[OrderLine],
        *,
        coupon_id: str | None,
        market: str | None,
        idempotency_key: str,
    ) -> tuple[Order, Money]:
        allocator = StockAllocator()
        member = await uow.members.get(member_id) or Member(
            member_id=member_id, email=f"{member_id}@unknown.local", level=MemberLevel.NORMAL
        )

        items: list[OrderItem] = []
        reserves: dict[str, int] = {}
        for line in lines:
            product = await uow.products.get(line.product_id)
            if product is None:
                raise NotFoundError(f"商品不存在: {line.product_id}", detail={"product_id": line.product_id})
            if market and product.market != market:
                raise ValidationError(
                    f"该商品不在 {market} 站点售卖", detail={"product_id": product.product_id, "market": market}
                )
            product.assert_sellable()
            sku_id = line.sku_id or allocator.resolve(product, line.choice)
            sku = next((item for item in product.skus if item.sku_id == sku_id), None)
            if sku is None:
                raise NotFoundError(
                    f"规格不存在: {sku_id}", detail={"sku_id": sku_id, "product_id": product.product_id}
                )
            items.append(
                OrderItem(
                    sku_id=sku.sku_id,
                    product_id=product.product_id,
                    name=product.name,
                    quantity=int(line.quantity),
                    unit_price=sku.price,
                )
            )
            reserves[sku_id] = reserves.get(sku_id, 0) + int(line.quantity)

        # 累加以「零钱」起手而不是内置 sum()：Money 的加法要求币种一致，
        # 显式循环能让「以第一行的币种为基准」这件事一眼可见
        goods_total = items[0].unit_price * 0
        for item in items:
            goods_total = goods_total + item.subtotal()

        coupon = template = None
        if coupon_id:
            coupon = await uow.coupons.get(coupon_id)
            if coupon is None:
                raise NotFoundError(f"优惠券不存在: {coupon_id}", detail={"coupon_id": coupon_id})
            if coupon.member_id != member_id:
                raise ForbiddenError("该券不属于当前会员", detail={"coupon_id": coupon_id})
            template = await uow.coupons.get_template(coupon.template_id)

        breakdown = PricingPolicy().quote(goods_total=goods_total, member=member, coupon=coupon, template=template)
        # 订单只存「一处折扣合计」：同时存会员折扣与券折扣会让 payable 有两个算法
        discount = (breakdown.member_discount + breakdown.coupon_discount).clamp_at_zero()

        # 扣库存：条件 UPDATE，未命中即「库存不足」，由事务回滚兜住
        for sku_id, quantity in reserves.items():
            if not await uow.stocks.try_reserve(sku_id, quantity):
                raise StockNotEnoughError(
                    f"库存不足或已被抢空（{sku_id}）",
                    detail={"sku_id": sku_id, "requested": quantity},
                )

        order = Order(
            order_id=new_id("ORD", length=10),
            member_id=member_id,
            items=items,
            discount=discount,
            coupon_id=coupon_id,
            expires_at=utcnow() + timedelta(minutes=self._ttl),
            idempotency_key=idempotency_key,
        )
        if coupon is not None:
            # 核销必须在同一事务：券被并发用掉时 rowcount=0，直接放弃本次下单
            if await uow.coupons.mark_used(coupon.coupon_id, order.order_id) != 1:
                raise CouponNotApplicableError("优惠券已被使用", detail={"coupon_id": coupon.coupon_id})
        return order, breakdown.payable

    async def _flag_low_stock(self, uow, publisher: OutboxPublisher, order: Order) -> None:
        """扣减后触达预警线的 SKU 上次级事件。独立读取是为了拿**扣减后**的真实余量。"""
        seen: set[str] = set()
        for item in order.items:
            if item.sku_id in seen:
                continue
            seen.add(item.sku_id)
            stock = await uow.stocks.get(item.sku_id)
            if stock is not None and stock.is_low():
                publisher.add(
                    StockLow(
                        event_id=new_id("evt"),
                        sku_id=stock.sku_id,
                        available=stock.available,
                        threshold=stock.threshold,
                    )
                )

    async def _replay_after_conflict(
        self, project_id: str, key: str, request_hash: str, exc: IntegrityError
    ) -> dict:
        async with self._uow() as uow:
            stored = await uow.idempotency.get(project_id, key)
            await uow.rollback()
        if stored is None:
            raise exc
        if stored["request_hash"] != request_hash:
            raise DuplicateRequestConflictError(
                "幂等键已被另一次不同内容的请求使用", detail={"idempotency_key": key, "project_id": project_id}
            )
        return {**dict(stored["response"]), "idempotent_replay": True}

    @staticmethod
    def _validate_lines(lines: Sequence[OrderLine]) -> None:
        if not lines:
            raise ValidationError("订单至少需要一行商品")
        if len(lines) > _MAX_LINES:
            raise ValidationError(f"单笔订单最多 {_MAX_LINES} 行", detail={"lines": len(lines)})
        for line in lines:
            if not 1 <= int(line.quantity) <= _MAX_LINE_QUANTITY:
                raise ValidationError(
                    f"单行数量必须在 1~{_MAX_LINE_QUANTITY}",
                    detail={"product_id": line.product_id, "quantity": line.quantity},
                )

    # -- 查询 ------------------------------------------------------------------ #
    async def get_order_payload(self, order_id: str, *, user_id: str | None = None) -> dict:
        """订单查询（工具与 REST 共用）。

        归属校验放在这里而不是路由里：`user_id` 是**业务上下文**，不是 HTTP 概念。
        工具侧由 ToolContext 注入、REST 侧由会员 token 推导 —— 同一个判定点，
        避免「某条路径忘了校验」这种经典漏洞。
        注意 `status` 回的是**中文标签**：这是工具契约的历史口径（演示数据即中文），
        同时保留 `status_code` 供程序判断，避免调用方去解析中文字符串。
        """
        order = await self.get(order_id)
        if user_id and order.member_id != user_id:
            raise ForbiddenError(
                "该订单不属于当前用户，拒绝查询",
                detail={"order_id": order_id, "member_id": order.member_id},
            )
        return {
            "order_id": order.order_id,
            "status": order.status.label,
            "status_code": order.status.value,
            "carrier": order.carrier,
            "items": [
                {
                    "product_id": item.product_id,
                    "sku_id": item.sku_id,
                    "name": item.name,
                    "quantity": item.quantity,
                    "price_usd": round(float(item.unit_price.amount), 2),
                }
                for item in order.items
            ],
            "total_usd": round(float(order.goods_total().amount), 2),
            "payable_usd": round(float(order.payable().amount), 2),
            "created_at": order.created_at.isoformat(),
        }

    async def get(self, order_id: str) -> Order:
        async with self._uow() as uow:
            order = await uow.orders.get(order_id)
            await uow.commit()
        if order is None:
            raise NotFoundError(f"订单不存在: {order_id}", detail={"order_id": order_id})
        return order

    async def list_by_member(self, member_id: str, *, limit: int = 20) -> list[Order]:
        async with self._uow() as uow:
            orders = await uow.orders.list_by_member(member_id, limit=limit)
            await uow.commit()
        return orders

    async def list_all(self, *, status: str | None = None, limit: int = 50) -> list[Order]:
        """运营侧跨会员订单列表。与买家侧 `list_by_member` 分开，避免「同一方法被两条权限面复用」。"""
        async with self._uow() as uow:
            orders = await uow.orders.list_all(status=status, limit=limit)
            await uow.commit()
        return orders

    async def stats(self, *, since=None) -> dict:
        async with self._uow() as uow:
            data = await uow.orders.stats(since=since)
            await uow.commit()
        return data

    # -- 状态迁移 -------------------------------------------------------------- #
    async def pay(self, order_id: str, *, actor_member_id: str | None = None) -> Order:
        """付款。CAS 期望旧状态是 PENDING_PAYMENT：与超时 sweeper 并发时只有一个能赢。"""
        async with self._uow() as uow:
            order = await uow.orders.get(order_id)
            if order is None:
                raise NotFoundError(f"订单不存在: {order_id}", detail={"order_id": order_id})
            if actor_member_id and order.member_id != actor_member_id:
                raise ForbiddenError("该订单不属于当前会员", detail={"order_id": order_id})
            order.transition_to(OrderStatus.PAID)
            matched = await uow.orders.update_status(
                order_id, OrderStatus.PAID.value, expected=OrderStatus.PENDING_PAYMENT.value, paid_at=utcnow()
            )
            if matched != 1:
                raise InvalidStateTransitionError("订单已被取消或状态已变更", detail={"order_id": order_id})
            await uow.commit()
        order.status = OrderStatus.PAID
        return order

    async def ship(self, order_id: str, *, carrier: str) -> Order:
        async with self._uow() as uow:
            order = await uow.orders.get(order_id)
            if order is None:
                raise NotFoundError(f"订单不存在: {order_id}", detail={"order_id": order_id})
            order.transition_to(OrderStatus.SHIPPED)
            matched = await uow.orders.update_status(
                order_id, OrderStatus.SHIPPED.value, expected=OrderStatus.PAID.value, carrier=carrier
            )
            if matched != 1:
                raise InvalidStateTransitionError("只有已付款订单可以发货", detail={"order_id": order_id})
            await uow.commit()
        order.carrier = carrier
        return order

    async def cancel(
        self, order_id: str, *, reason: str = "user_cancelled", actor_member_id: str | None = None
    ) -> Order:
        """取消 = CAS 改状态 + 回滚库存 + 退券 + 落事件，四件事同一事务。"""
        async with self._uow() as uow:
            order = await uow.orders.get(order_id)
            if order is None:
                raise NotFoundError(f"订单不存在: {order_id}", detail={"order_id": order_id})
            if actor_member_id and order.member_id != actor_member_id:
                raise ForbiddenError("该订单不属于当前会员", detail={"order_id": order_id})
            expected = order.status.value
            order.transition_to(OrderStatus.CANCELLED)  # 非法迁移（已发货/已完成）在这里被领域层拒绝
            matched = await uow.orders.update_status(
                order_id,
                OrderStatus.CANCELLED.value,
                expected=expected,
                cancelled_at=utcnow(),
                cancel_reason=reason,
            )
            if matched != 1:
                raise InvalidStateTransitionError("订单状态已变更，请重新读取", detail={"order_id": order_id})
            released = await self._release(uow, order)
            if order.coupon_id:
                await uow.coupons.mark_released(order.coupon_id)
            publisher = OutboxPublisher(uow.outbox)
            publisher.add(
                OrderCancelled(event_id=new_id("evt"), order_id=order_id, reason=reason, released_items=released)
            )
            events = list(publisher.pending)
            await publisher.flush()
            await uow.commit()

        await self._dispatcher.dispatch(events)
        order.status = OrderStatus.CANCELLED
        return order

    async def sweep_expired(self) -> dict:
        """超时未付款清理。返回被取消的订单数。

        多实例并发跑也不会重复回滚库存 —— 因为 `cancel_expired` 是逐单 CAS 抢占，
        只有一个实例能拿到 `rowcount == 1`。
        """
        async with self._uow() as uow:
            expired = await uow.orders.cancel_expired(utcnow())
            publisher = OutboxPublisher(uow.outbox)
            for order in expired:
                await self._release(uow, order)
                if order.coupon_id:
                    await uow.coupons.mark_released(order.coupon_id)
                publisher.add(
                    OrderCancelled(
                        event_id=new_id("evt"),
                        order_id=order.order_id,
                        reason="timeout",
                        released_items=len(order.items),
                    )
                )
            events = list(publisher.pending)
            await publisher.flush()
            await uow.commit()

        await self._dispatcher.dispatch(events)
        return {"cancelled": len(expired), "order_ids": [order.order_id for order in expired]}

    @staticmethod
    async def _release(uow, order: Order) -> int:
        """按 SKU 汇总回滚库存。汇总而不是逐行回滚，避免同一 SKU 多行时重复加回。"""
        aggregated: dict[str, int] = {}
        for item in order.items:
            aggregated[item.sku_id] = aggregated.get(item.sku_id, 0) + item.quantity
        for sku_id, quantity in aggregated.items():
            await uow.stocks.release(sku_id, quantity)
        return len(aggregated)


__all__ = ["OrderLine", "OrderUseCase"]
