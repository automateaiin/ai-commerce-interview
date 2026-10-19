"""业务门面：把「一批用例 + 一个工作单元工厂 + 一个凭证签发器」组装成一个可注入的对象。

为什么门面用「属性暴露用例」而不是为每个方法写一层转发：
用例本身就是业务能力的**最小命名单元**（`orders.place_order`、`coupons.receive`），
再套一层 `CommerceService.place_order → OrderUseCase.place_order` 只会让调用链变长、
让 IDE 的跳转变绕，且每加一个用例就要改两处。
门面真正的职责只有三件：**装配、生命周期、跨用例的公共动作**（超时清理与关闭）。

`backend` 字段用于日志与排障：一眼看出当前跑在 sqlite 还是 PostgreSQL，
而不是等出问题再去猜方言。
"""

from __future__ import annotations

from typing import Any

from .application.events import EventRouter
from .application.ports import AIHookPort, CachePort, UowFactory
from .application.usecases import CartUseCase, CatalogUseCase, CouponUseCase, MemberUseCase, OrderUseCase
from .domain.security import MemberTokenSigner
from .infrastructure.db.session import CommerceDatabase
from .infrastructure.scheduler import PeriodicScheduler


class CommerceService:
    """宿主业务后端门面。`container.commerce` 指向本对象。"""

    def __init__(
        self,
        *,
        db: CommerceDatabase,
        cache: CachePort,
        uow_factory: UowFactory,
        signer: MemberTokenSigner,
        hook: AIHookPort,
        dispatcher: EventRouter,
        catalog: CatalogUseCase,
        cart: CartUseCase,
        orders: OrderUseCase,
        members: MemberUseCase,
        coupons: CouponUseCase,
        scheduler: PeriodicScheduler | None = None,
    ) -> None:
        self.db = db
        self.cache = cache
        self.uow_factory = uow_factory
        self.signer = signer
        self.hook = hook
        self.dispatcher = dispatcher
        self.catalog = catalog
        self.cart = cart
        self.orders = orders
        self.members = members
        self.coupons = coupons
        self.scheduler = scheduler
        self.backend = db.dialect

    @property
    def scheduler_stats(self) -> dict[str, Any]:
        return self.scheduler.stats if self.scheduler is not None else {"enabled": False, "running": False}

    async def aclose(self) -> None:
        """优雅关闭：先停后台任务再关连接池。

        顺序不能反：先 dispose 引擎会让正在跑的那一轮 sweeper 拿到已关闭的连接，
        留下一个必然会炸的异常 —— 这正是「关闭顺序」值得单独写一行注释的原因。
        """
        if self.scheduler is not None:
            await self.scheduler.stop()
        await self.db.dispose()


__all__ = ["CommerceService"]
