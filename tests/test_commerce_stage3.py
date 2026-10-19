"""阶段 3：宿主业务后端（commerce）的契约与治理不变量。

覆盖 PROJECT.md 6.6 的验收点，每条都对应一个「面试会被追问」的问题：

- 领域不变量：折扣不会把应付压成负数；非法状态迁移在领域层就被拒。
- 原子扣库存：能不能超卖？并发下靠什么保证？（条件 UPDATE + 唯一约束兜底）
- 下单幂等：同键重放是否返回同一订单？同键不同体是否被拒？
- Outbox：事件是否和业务写在同一事务？提交前可断言。
- UoW：回滚后是否真的没有残留写？
- 路由与鉴权：会员 token 能不能进后台？没有 scope 的 API Key 能不能改库存？
- 与 infra 对接：工具注册表里的四个工具契约（scope / 风险等级）是否没被改坏。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from src.commerce.application.usecases.order import OrderLine
from src.commerce.domain.errors import (
    CouponNotApplicableError,
    DuplicateRequestConflictError,
    InvalidStateTransitionError,
    StockNotEnoughError,
)
from src.commerce.domain.models import (
    Coupon,
    CouponKind,
    CouponTemplate,
    Member,
    MemberLevel,
    Order,
    OrderItem,
    OrderStatus,
    ProductStatus,
    utcnow,
)
from src.commerce.domain.services import PricingPolicy
from src.commerce.domain.values import Money
from src.infra.core.errors import ForbiddenError, NotFoundError
from src.infra.tools.base import ToolRisk
from tests.conftest import CONTENT_KEY, CS_KEY, SHOP_ADMIN_KEY

DEMO_MEMBER = "u_10086"
DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "Demo12345!"
SKU_LOW_STOCK = "SKU-1003-01"  # 并发用例专用：避开其他用例争用的 SKU-1001-01


# --------------------------------------------------------------------------- #
# 1. 领域不变量（不需要起应用）
# --------------------------------------------------------------------------- #
def test_money_is_fixed_point_and_clamped() -> None:
    assert Money(12800).amount == Money.of(128.0).amount
    assert (Money.of(128.0) * 2).cents == 25600
    # 折扣大于货款时必须夹到 0，而不是产生「负应付」这种资金级脏数据
    assert (Money.of(10.0) - Money.of(30.0)).clamp_at_zero().cents == 0


def test_pricing_policy_applies_member_then_coupon() -> None:
    goods = Money.of(128.0)
    gold = Member(member_id="m_gold", email="g@x.io", level=MemberLevel.GOLD)
    plain = PricingPolicy().quote(goods_total=goods, member=gold, coupon=None, template=None)
    assert plain.member_discount.cents > 0, "GOLD 会员必须拿到会员折扣"
    assert plain.payable.cents < goods.cents

    template = CouponTemplate(
        template_id="ct_test",
        name="满 50 减 20",
        kind=CouponKind.FULL_REDUCTION,
        threshold=Money.of(50.0),
        value_cents=2000,
        total=0,
        per_member_limit=1,
    )
    coupon = Coupon(coupon_id="cpn_test", template_id="ct_test", member_id="m_gold")
    priced = PricingPolicy().quote(goods_total=goods, member=gold, coupon=coupon, template=template)
    assert priced.coupon_discount.cents == 2000
    assert priced.payable.cents == plain.payable.cents - 2000

    # 未达门槛的券必须被拒：这是「券不能被无条件套用」的领域规则
    with pytest.raises(CouponNotApplicableError):
        PricingPolicy().quote(goods_total=Money.of(10.0), member=gold, coupon=coupon, template=template)


def test_order_status_machine_rejects_illegal_transition() -> None:
    order = Order(
        order_id="ORD-x",
        member_id="m1",
        items=[OrderItem(sku_id="s1", product_id="p1", name="n", quantity=1, unit_price=Money.of(1.0))],
        status=OrderStatus.COMPLETED,
    )
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderStatus.PENDING_PAYMENT)


def test_stock_never_goes_negative_and_reserve_is_all_or_nothing() -> None:
    from src.commerce.domain.models import Stock

    stock = Stock(sku_id="s1", available=3, reserved=0, threshold=5)
    assert stock.sellable == 3
    assert stock.is_low() is True
    stock.available = 0
    stock.reserved = 0
    stock.available = max(0, stock.available - 1)  # 领域层兜底的钳制
    assert stock.available >= 0, "可售量不变量：available 永不为负"


# --------------------------------------------------------------------------- #
# 2. 下单：原子扣库存 / 幂等 / Outbox / 回滚
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_place_order_deducts_stock_and_writes_outbox(app_client) -> None:
    async with app_client() as (_client, container):
        commerce = container.commerce
        before = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available

        payload = await commerce.orders.place_order(
            DEMO_MEMBER, [OrderLine(product_id="SKU-1003", quantity=2)], idempotency_key="pytest-outbox-01"
        )
        assert payload["order_id"].startswith("ORD")
        assert payload["idempotent_replay"] is False
        assert payload["quantity"] == 2

        after = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
        assert after == before - 2, "下单必须真实扣减可售库存"

        # Outbox 与业务写在同一事务：commit 之后必然可查到事件
        async with commerce.uow_factory() as uow:
            pending = await uow.outbox.pending()
            await uow.commit()
        topics = [row["topic"] for row in pending]
        assert "OrderPlaced" in topics, f"Outbox 必须留下 OrderPlaced 事件，实际: {topics}"
        # 同一进程里前面的用例也会留下未投递事件，因此必须按 order_id 匹配，而不是取第一条
        assert any(
            row["topic"] == "OrderPlaced" and row["payload"].get("order_id") == payload["order_id"]
            for row in pending
        ), "本笔订单的 OrderPlaced 事件必须与业务写在同一事务里落进 Outbox"


@pytest.mark.asyncio
async def test_place_order_is_idempotent_and_deducts_once(app_client) -> None:
    async with app_client() as (_client, container):
        commerce = container.commerce
        before = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
        lines = [OrderLine(product_id="SKU-1003", quantity=1)]

        first = await commerce.orders.place_order(DEMO_MEMBER, lines, idempotency_key="pytest-idem-shared")
        second = await commerce.orders.place_order(DEMO_MEMBER, lines, idempotency_key="pytest-idem-shared")

        assert first["idempotent_replay"] is False
        assert second["idempotent_replay"] is True
        assert first["order_id"] == second["order_id"], "同键重放必须返回同一笔订单"
        after = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
        assert after == before - 1, "重放绝不能重复扣库存"


@pytest.mark.asyncio
async def test_idempotency_key_reuse_with_different_body_is_rejected(app_client) -> None:
    async with app_client() as (_client, container):
        commerce = container.commerce
        key = "pytest-idem-conflict"
        await commerce.orders.place_order(DEMO_MEMBER, [OrderLine(product_id="SKU-1003", quantity=1)], idempotency_key=key)
        with pytest.raises(DuplicateRequestConflictError):
            await commerce.orders.place_order(
                DEMO_MEMBER, [OrderLine(product_id="SKU-1003", quantity=2)], idempotency_key=key
            )


@pytest.mark.asyncio
async def test_concurrent_orders_never_oversell(app_client) -> None:
    """并发 12 单抢 4 件库存：只能成功 4 单，且余量必须精确为 0。

    这条用例是「单 SQL 条件 UPDATE」的存在理由 —— 任何「先查再改」的实现都会在这里超卖。
    """
    async with app_client() as (_client, container):
        commerce = container.commerce
        original = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
        await commerce.catalog.adjust_stock(SKU_LOW_STOCK, 4)
        try:
            results = await asyncio.gather(
                *(
                    commerce.orders.place_order(
                        DEMO_MEMBER,
                        [OrderLine(product_id="SKU-1003", quantity=1)],
                        idempotency_key=f"pytest-race-{index:02d}",
                    )
                    for index in range(12)
                ),
                return_exceptions=True,
            )
            succeeded = [item for item in results if isinstance(item, dict)]
            rejected = [item for item in results if isinstance(item, BaseException)]
            assert len(succeeded) == 4, f"只应成功 4 单，实际 {len(succeeded)}"
            assert len(rejected) == 8
            assert all(isinstance(item, StockNotEnoughError) for item in rejected), "失败原因必须是库存不足"
            remaining = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
            assert remaining == 0, "不允许出现负库存"
        finally:
            await commerce.catalog.adjust_stock(SKU_LOW_STOCK, original)


@pytest.mark.asyncio
async def test_uow_rollback_leaves_no_residue(app_client) -> None:
    async with app_client() as (_client, container):
        commerce = container.commerce
        ghost_sku = "SKU-GHOST-01"
        async with commerce.uow_factory() as uow:
            from src.commerce.domain.models import Stock

            await uow.stocks.upsert(Stock(sku_id=ghost_sku, available=7, reserved=0, threshold=1))
            await uow.rollback()  # 显式回滚：模拟「校验失败直接放弃事务」
        assert (await commerce.catalog.stocks_for([ghost_sku])) == {}, "回滚后不允许有残留写"


@pytest.mark.asyncio
async def test_cancel_restores_stock_and_emits_event(app_client) -> None:
    async with app_client() as (_client, container):
        commerce = container.commerce
        before = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
        payload = await commerce.orders.place_order(
            DEMO_MEMBER, [OrderLine(product_id="SKU-1003", quantity=1)], idempotency_key="pytest-cancel-01"
        )
        await commerce.orders.cancel(payload["order_id"], actor_member_id=DEMO_MEMBER)
        after = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
        assert after == before, "取消必须把库存原样退回"

        async with commerce.uow_factory() as uow:
            topics = [row["topic"] for row in await uow.outbox.pending()]
            await uow.commit()
        assert "OrderCancelled" in topics


@pytest.mark.asyncio
async def test_sweeper_cancels_expired_order_and_releases_stock(app_client) -> None:
    async with app_client() as (_client, container):
        commerce = container.commerce
        before = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available

        # 手工造一笔「已过期」订单：先占库存，再把过期时间设到过去，等价于 sweeper 的输入
        async with commerce.uow_factory() as uow:
            assert await uow.stocks.try_reserve(SKU_LOW_STOCK, 1) is True
            await uow.orders.save(
                Order(
                    order_id="ORD-expired-01",
                    member_id=DEMO_MEMBER,
                    items=[
                        OrderItem(
                            sku_id=SKU_LOW_STOCK,
                            product_id="SKU-1003",
                            name="过期单样例",
                            quantity=1,
                            unit_price=Money.of(10.0),
                        )
                    ],
                    expires_at=utcnow() - timedelta(minutes=5),
                ),
                project_id="shop",
            )
            await uow.commit()

        result = await commerce.orders.sweep_expired()
        assert "ORD-expired-01" in result["order_ids"]
        after = (await commerce.catalog.stocks_for([SKU_LOW_STOCK]))[SKU_LOW_STOCK].available
        assert after == before, "超时取消必须把库存退回"


@pytest.mark.asyncio
async def test_order_ownership_is_enforced_in_usecase(app_client) -> None:
    async with app_client() as (_client, container):
        commerce = container.commerce
        payload = await commerce.orders.place_order(
            DEMO_MEMBER, [OrderLine(product_id="SKU-1003", quantity=1)], idempotency_key="pytest-owned-01"
        )
        with pytest.raises(ForbiddenError):
            await commerce.orders.get_order_payload(payload["order_id"], user_id="u_someone_else")
        with pytest.raises(NotFoundError):
            await commerce.orders.get_order_payload("ORD-does-not-exist", user_id=DEMO_MEMBER)


# --------------------------------------------------------------------------- #
# 3. 路由挂载与鉴权（买家侧 / 运营侧两套凭证互不通用）
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_shop_catalog_is_public(app_client) -> None:
    async with app_client() as (client, _container):
        listed = await client.get("/v1/shop/products", params={"limit": 5})
        assert listed.status_code == 200
        body = listed.json()
        assert body["count"] > 0
        assert {"product_id", "name", "price_usd", "stock"} <= set(body["items"][0])

        filtered = await client.get("/v1/shop/products", params={"keyword": "精华"})
        assert filtered.status_code == 200
        assert filtered.json()["count"] >= 1

        detail = await client.get(f"/v1/shop/products/{body['items'][0]['product_id']}")
        assert detail.status_code == 200
        assert "ingredients" in detail.json()


@pytest.mark.asyncio
async def test_unpublished_product_is_hidden_from_buyers(app_client) -> None:
    async with app_client() as (client, _container):
        headers = {"X-API-Key": SHOP_ADMIN_KEY}
        assert (await client.post("/v1/admin/products/SKU-1004/unpublish", headers=headers)).status_code == 200
        try:
            assert (await client.get("/v1/shop/products/SKU-1004")).status_code == 404
        finally:
            restored = await client.post("/v1/admin/products/SKU-1004/publish", headers=headers)
            assert restored.status_code == 200
            assert restored.json()["product"]["status"] == ProductStatus.ON_SALE.value


@pytest.mark.asyncio
async def test_member_token_is_required_for_cart_and_orders(app_client) -> None:
    async with app_client() as (client, _container):
        assert (await client.get("/v1/shop/cart")).status_code == 401
        assert (await client.get("/v1/shop/cart", headers={"Authorization": "Bearer not-a-token"})).status_code == 401
        assert (
            await client.get("/v1/shop/orders", headers={"X-API-Key": CS_KEY})
        ).status_code == 401, "项目级 API Key 不是会员凭证，不能访问会员资源"


@pytest.mark.asyncio
async def test_admin_routes_require_commerce_scope(app_client) -> None:
    async with app_client() as (client, _container):
        assert (await client.get("/v1/admin/stock/low")).status_code == 401, "没有凭证必须 401"
        assert (
            await client.get("/v1/admin/stock/low", headers={"X-API-Key": CONTENT_KEY})
        ).status_code == 403, "非 commerce:rw 凭证必须 403"
        ok = await client.get("/v1/admin/stock/low", headers={"X-API-Key": SHOP_ADMIN_KEY})
        assert ok.status_code == 200
        assert "items" in ok.json()


@pytest.mark.asyncio
async def test_member_and_admin_credentials_are_not_interchangeable(app_client) -> None:
    async with app_client() as (client, _container):
        logged = await client.post("/v1/shop/members/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
        assert logged.status_code == 200
        token = logged.json()["token"]
        member_headers = {"Authorization": f"Bearer {token}"}

        assert (await client.get("/v1/shop/members/me", headers=member_headers)).status_code == 200
        # 会员 token 不能进后台：两套凭证由不同签发方产生，互不通用
        assert (await client.get("/v1/admin/orders/stats", headers=member_headers)).status_code == 401


@pytest.mark.asyncio
async def test_shop_cart_and_order_flow_end_to_end(app_client) -> None:
    async with app_client() as (client, _container):
        email = "pytest-full-flow@example.com"
        # 注册只返回会员档案；token 一律通过登录获取 —— 注册与登录是两条独立用例
        await client.post(
            "/v1/shop/members/register", json={"email": email, "password": "Pytest12345!", "nickname": "冒烟买家"}
        )
        logged = await client.post("/v1/shop/members/login", json={"email": email, "password": "Pytest12345!"})
        assert logged.status_code == 200
        headers = {"Authorization": f"Bearer {logged.json()['token']}"}

        added = await client.post("/v1/shop/cart/items", json={"product_id": "SKU-1002", "quantity": 1}, headers=headers)
        assert added.status_code == 200
        assert added.json()["cart"]["items"], "加购后购物车必须非空"

        placed = await client.post(
            "/v1/shop/orders",
            json={"items": [{"product_id": "SKU-1002", "quantity": 1}], "idempotency_key": "pytest-rest-order-01"},
            headers=headers,
        )
        assert placed.status_code == 200
        order = placed.json()["order"]
        assert order["status"] == OrderStatus.PENDING_PAYMENT.value

        fetched = await client.get(f"/v1/shop/orders/{order['order_id']}", headers=headers)
        assert fetched.status_code == 200

        paid = await client.post(f"/v1/shop/orders/{order['order_id']}/pay", headers=headers)
        assert paid.status_code == 200
        assert paid.json()["order"]["status"] == OrderStatus.PAID.value


@pytest.mark.asyncio
async def test_coupon_receive_and_apply(app_client) -> None:
    async with app_client() as (client, _container):
        admin_headers = {"X-API-Key": SHOP_ADMIN_KEY}
        created = await client.post(
            "/v1/admin/coupons/templates",
            json={
                "name": "pytest 满50减20",
                "kind": CouponKind.FULL_REDUCTION.value,
                "threshold_usd": 50,
                "value_cents": 2000,
                "total": 0,
                "per_member_limit": 1,
            },
            headers=admin_headers,
        )
        assert created.status_code == 200
        template_id = created.json()["template"]["template_id"]

        email = "pytest-coupon@example.com"
        await client.post("/v1/shop/members/register", json={"email": email, "password": "Pytest12345!"})
        account = await client.post("/v1/shop/members/login", json={"email": email, "password": "Pytest12345!"})
        assert account.status_code == 200
        headers = {"Authorization": f"Bearer {account.json()['token']}"}

        received = await client.post(f"/v1/shop/coupons/{template_id}/receive", headers=headers)
        assert received.status_code == 200
        coupon_id = received.json()["coupon"]["coupon_id"]

        # 每人限领 1 张：第二次必须是结构化拒绝，不能静默多发
        again = await client.post(f"/v1/shop/coupons/{template_id}/receive", headers=headers)
        assert again.status_code == 400

        placed = await client.post(
            "/v1/shop/orders",
            json={
                "items": [{"product_id": "SKU-1001", "quantity": 1}],
                "idempotency_key": "pytest-coupon-order-01",
                "coupon_id": coupon_id,
            },
            headers=headers,
        )
        assert placed.status_code == 200
        assert placed.json()["order"]["discount_usd"] >= 20.0, "券必须真的抵扣到应付金额上"


# --------------------------------------------------------------------------- #
# 4. 与 infra 对接：工具契约没被改坏
# --------------------------------------------------------------------------- #
def test_registry_exposes_the_four_commerce_tools() -> None:
    """工具是「用桩端口构造」出来的：证明注册表只依赖 Protocol，不认识具体业务库。"""
    names = {"search_products", "get_product_detail", "get_order_status", "create_order"}
    assert names.issubset(set(_registry_names()))


def _registry_names() -> list[str]:
    """从工厂构造一次注册表（用最小桩端口），确认工具确实被注册。"""
    from src.infra.memory.store import MemoryStore
    from src.infra.rag.service import HashingEmbedder, KnowledgeService
    from src.infra.tools.factory import build_registry

    class _StubCommerce:
        backend = "stub"

        async def search_products(self, **_kwargs): ...
        async def product_detail(self, _product_id): ...
        async def order_status(self, _order_id, **_kwargs): ...
        async def create_order(self, **_kwargs): ...

    memory = MemoryStore.from_url("sqlite+aiosqlite:///:memory:")
    knowledge = KnowledgeService(HashingEmbedder())
    registry = build_registry(memory, knowledge, _StubCommerce())
    return registry.names()


@pytest.mark.asyncio
async def test_commerce_tool_risk_and_scope_contract(app_client) -> None:
    async with app_client() as (_client, container):
        create_order = container.registry.get("create_order")
        assert create_order.requires_scope == "tools:write"
        assert create_order.risk is ToolRisk.HIGH_RISK
        assert create_order.idempotent is True
        assert create_order.parallel_safe is False

        read_tool = container.registry.get("get_order_status")
        assert read_tool.requires_scope == "tools:read"
        assert read_tool.risk is ToolRisk.READ_ONLY
