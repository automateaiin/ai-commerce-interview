"""业务后端装配：由 Settings 造出一个可注入的 `CommerceService`。

装配顺序即是依赖顺序，每一步都有理由：

1) **建库 + 建表**：业务库独立于 infra 库（见 `infrastructure/db/session.py` 的说明）；
2) **灌演示数据**：只在库为空时执行，重复启动幂等 —— 这样「首次启动有数据可点、
   第二次启动不会串数据」两件事同时成立；
3) **缓存**：Redis 可用则用 Redis（多副本一致），否则进程内 TTL 缓存。
   这里复用 infra 已建好的 Redis 连接，**不另开一条**；
4) **凭证签发器**：密钥未配置时用进程内随机密钥并**打告警日志**，
   绝不硬编码默认密钥（硬编码等于把 token 伪造能力写进了仓库）；
5) **事件链路**：`NoopHook` 是当前默认实现 —— 真正的消费端（项目一/二的
   RAG 索引、训练回流）接入时替换这一个对象即可，用例与事务边界都不用动。
6) **周期任务**：订单超时清理挂在应用 lifespan 启动、关闭时反向停止。
"""

from __future__ import annotations

import secrets

from src.infra.core.logging import get_logger

from .application.ai_hooks import NoopHook
from .application.events import EventRouter
from .application.ports import AIHookPort
from .application.usecases import CartUseCase, CatalogUseCase, CouponUseCase, MemberUseCase, OrderUseCase
from .domain.security import MemberTokenSigner
from .infrastructure.cache import build_cache
from .infrastructure.db.session import CommerceDatabase
from .infrastructure.scheduler import PeriodicScheduler
from .infrastructure.seed import seed_demo
from .infrastructure.uow import build_uow_factory
from .service import CommerceService

logger = get_logger("commerce.factory")


async def build_commerce_service(
    settings,
    *,
    redis_client=None,
    hook: AIHookPort | None = None,
) -> CommerceService:
    db = CommerceDatabase.from_url(settings.resolved_commerce_database_url())
    await db.init_schema()

    if settings.commerce_seed_enabled:
        counts = await seed_demo(db, force=settings.commerce_seed_force)
        logger.info(
            "commerce.seed",
            extra={"event": "startup", "status": "ok", "skipped": bool(counts.get("skipped")), **counts},
        )

    cache = build_cache(redis_client)
    uow_factory = build_uow_factory(db)

    secret = settings.commerce_member_token_secret
    if not secret:
        secret = secrets.token_hex(32)
        logger.warning(
            "commerce.member_token.ephemeral_secret",
            extra={
                "event": "startup",
                "status": "degraded",
                "degraded": True,
                "reason": "commerce_member_token_secret_unset",
                "effect": "tokens_invalidate_on_restart",
            },
        )
    signer = MemberTokenSigner(secret, ttl_s=settings.commerce_member_token_ttl_s)

    active_hook: AIHookPort = hook or NoopHook()
    dispatcher = EventRouter(active_hook)

    catalog = CatalogUseCase(uow_factory)
    cart = CartUseCase(uow_factory, catalog, cache)
    orders = OrderUseCase(
        uow_factory,
        cache=cache,
        dispatcher=dispatcher,
        order_ttl_minutes=settings.commerce_order_ttl_minutes,
        per_member_hour_limit=settings.commerce_order_rate_per_hour,
    )
    members = MemberUseCase(uow_factory, signer)
    coupons = CouponUseCase(uow_factory)

    async def _sweep() -> None:
        result = await orders.sweep_expired()
        if result["cancelled"]:
            logger.info(
                "commerce.order_sweeper.cancelled",
                extra={"event": "scheduler", "status": "ok", "cancelled": result["cancelled"]},
            )

    scheduler = PeriodicScheduler(
        "commerce.order_sweeper", interval_s=settings.commerce_sweeper_interval_s, job=_sweep
    )
    service = CommerceService(
        db=db,
        cache=cache,
        uow_factory=uow_factory,
        signer=signer,
        hook=active_hook,
        dispatcher=dispatcher,
        catalog=catalog,
        cart=cart,
        orders=orders,
        members=members,
        coupons=coupons,
        scheduler=scheduler,
    )
    await scheduler.start()
    return service


__all__ = ["build_commerce_service"]
