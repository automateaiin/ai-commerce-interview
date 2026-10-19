"""电商业务工具（商品 / 订单）。

这一层要讲清楚三件事：

1) **参数校验 ≠ 业务授权**。参数能通过 Pydantic 校验，不代表这个调用方有权限动这个资源。
   例如 order_id 格式合法，但订单不属于当前 user_id 时必须拒绝（由业务端口判定）。

2) **写操作与只读操作策略不同**。写工具标记 risk=WRITE、parallel_safe=False，
   并要求调用方提供幂等键（idempotency_key），否则重试会造成重复下单。

3) **本文件不依赖业务库实现**。它只声明 `CommercePort`（「我需要哪四个动作、
   各自返回什么形状」），真正的实现在业务后端（`src/commerce/tools_port.py`），
   由唯一组合根注入。因此把 SQLite 换成 PostgreSQL、把商品表改成别的存储，
   **这里的代码一行都不用改** —— 这是依赖倒置在本仓库最直接的一次应用。
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..core.logging import get_logger
from .base import FunctionTool, ToolContext, ToolRisk

logger = get_logger("infra.tools.commerce")


# --------------------------------------------------------------------------- #
# 业务端口：infra 对业务侧的**唯一**依赖面（只声明，不实现）
# --------------------------------------------------------------------------- #
class CommercePort(Protocol):
    """业务后端必须提供的能力。实现方保证：

    - 查不到商品/订单 → 抛 `not_found`（可恢复的结构化错误，不是 500）；
    - 越权访问他人订单 → 抛 `permission_denied`；
    - 库存不足 → 抛结构化错误，**绝不**返回超卖成功的订单；
    - 下单幂等：同 `(project_id, idempotency_key)` 重复调用返回**同一笔**订单，
      并以 `idempotent_replay=true` 标记这是回放。
    """

    backend: str

    async def search_products(
        self,
        *,
        query: str | None = None,
        skin_type: str | None = None,
        max_price_usd: float | None = None,
        market: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]: ...

    async def product_detail(self, product_id: str) -> dict[str, Any]: ...

    async def order_status(self, order_id: str, *, user_id: str | None = None) -> dict[str, Any]: ...

    async def create_order(
        self,
        *,
        product_id: str,
        quantity: int,
        idempotency_key: str,
        market: str | None = None,
        user_id: str | None = None,
        project_id: str = "shop",
    ) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# 参数模型：字段描述就是模型看到的工具契约，写清楚才能提高调用准确率
# --------------------------------------------------------------------------- #
class SearchProductsArgs(BaseModel):
    query: str = Field(description="商品关键词，例如「敏感肌精华」「保湿面霜」，可为空表示不限定关键词")
    skin_type: str | None = Field(default=None, description="肤质筛选：敏感肌 / 干性 / 油性 / 混合性 / 极干性")
    max_price_usd: float | None = Field(default=None, description="价格上限（美元）")
    market: str | None = Field(default=None, description="站点市场，例如 US")
    limit: int = Field(default=5, ge=1, le=20, description="返回条数上限")


class ProductIdArgs(BaseModel):
    product_id: str = Field(description="商品 ID，形如 SKU-1001")


class OrderStatusArgs(BaseModel):
    order_id: str = Field(description="订单号，形如 ORD-90001")


class CreateOrderArgs(BaseModel):
    product_id: str = Field(description="商品 ID")
    quantity: int = Field(ge=1, le=10, description="购买数量")
    idempotency_key: str = Field(
        min_length=8,
        description="幂等键。同一次下单意图必须复用同一个 key，重复调用不会重复下单",
    )
    market: str | None = Field(default=None, description="站点市场")


# --------------------------------------------------------------------------- #
# 工具构造：handler 通过闭包捕获业务端口
# --------------------------------------------------------------------------- #
def build_commerce_tools(commerce: CommercePort) -> list[FunctionTool]:
    """用业务端口构造四个电商工具。

    handler 用闭包而不是模块级函数，是为了**把端口作为显式依赖注入**：
    模块级函数只能靠全局单例拿到业务库（不可替换、不可测试），
    闭包则让「这个工具依赖哪个端口」在构造处一眼可见。
    """

    async def _search_products(args: SearchProductsArgs, ctx: ToolContext) -> dict[str, Any]:
        return await commerce.search_products(
            query=args.query,
            skin_type=args.skin_type,
            max_price_usd=args.max_price_usd,
            market=args.market,
            limit=args.limit,
        )

    async def _get_product_detail(args: ProductIdArgs, ctx: ToolContext) -> dict[str, Any]:
        return await commerce.product_detail(args.product_id)

    async def _get_order_status(args: OrderStatusArgs, ctx: ToolContext) -> dict[str, Any]:
        # 业务授权：订单归属校验。参数合法但在业务侧仍必须拒绝越权查询。
        return await commerce.order_status(args.order_id, user_id=ctx.user_id)

    async def _create_order(args: CreateOrderArgs, ctx: ToolContext) -> dict[str, Any]:
        """创建订单。幂等键命中直接返回首次结果，避免重试造成重复下单。"""
        return await commerce.create_order(
            product_id=args.product_id,
            quantity=args.quantity,
            idempotency_key=args.idempotency_key,
            market=args.market,
            user_id=ctx.user_id,
            project_id=ctx.project_id,
        )

    return [
        FunctionTool(
            name="search_products",
            description=(
                "按关键词/肤质/价格/站点检索在售商品，返回商品 ID、名称、价格、适用肤质、卖点与库存。"
                "用于回答「有什么商品」「适合什么肤质」「预算多少」这类问题。"
            ),
            args_model=SearchProductsArgs,
            handler=_search_products,
            requires_scope="tools:read",
            risk=ToolRisk.READ_ONLY,
            timeout_s=3.0,
        ),
        FunctionTool(
            name="get_product_detail",
            description=(
                "查询单个商品的完整信息，包含成分、功效、使用方法、规格与库存。"
                "已知商品 ID 且需要精确事实（成分、用法、注意事项）时使用。"
            ),
            args_model=ProductIdArgs,
            handler=_get_product_detail,
            requires_scope="tools:read",
            risk=ToolRisk.READ_ONLY,
            timeout_s=3.0,
        ),
        FunctionTool(
            name="get_order_status",
            description=(
                "查询当前用户某个订单的状态、承运商、商品明细与金额。"
                "只能查询属于当前用户的订单；缺少 user_id 的业务上下文时不要调用。"
            ),
            args_model=OrderStatusArgs,
            handler=_get_order_status,
            requires_scope="tools:read",
            risk=ToolRisk.READ_ONLY,
            timeout_s=3.0,
        ),
        FunctionTool(
            name="create_order",
            description=(
                "为用户创建一笔待付款订单。属于写操作，必须携带稳定的幂等键；"
                "同一下单意图重复调用会返回首次结果而不是重复下单。"
            ),
            args_model=CreateOrderArgs,
            handler=_create_order,
            requires_scope="tools:write",
            # 高风险：下单有真实业务副作用，不允许模型自己拍板。
            # Harness 的确认门要求调用方显式带 confirm 标记才会执行（见 harness/policy.py）。
            risk=ToolRisk.HIGH_RISK,
            timeout_s=8.0,
            idempotent=True,
            parallel_safe=False,
        ),
    ]


__all__ = ["CommercePort", "build_commerce_tools"]
