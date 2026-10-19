"""商城接口的请求模型。

字段约束刻意写成「和用例里的业务上限一致」：
`quantity <= 10`、`items <= 5` 这类限制如果在 Pydantic 与用例里各写一遍，
迟早会漂移成「接口放行、用例报错」的体验事故。这里的做法是**接口先挡一层**，
用例仍保留同样的校验（纵深防御），两者数值取自同一份常量。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..application.usecases.order import _MAX_LINE_QUANTITY, _MAX_LINES


class MemberRegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=120, description="登录邮箱")
    password: str = Field(min_length=8, max_length=128, description="口令，至少 8 位")
    nickname: str = Field(default="", max_length=40, description="昵称")


class MemberLoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=120)
    password: str = Field(min_length=8, max_length=128)


class AddressCreateRequest(BaseModel):
    receiver: str = Field(min_length=1, max_length=40)
    phone: str = Field(min_length=4, max_length=32, description="联系电话（对外响应会打码）")
    country: str = Field(min_length=2, max_length=40)
    city: str = Field(min_length=1, max_length=40)
    line1: str = Field(min_length=1, max_length=120)
    is_default: bool = False


class CartAddRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=64, description="商品 ID")
    quantity: int = Field(default=1, ge=1, le=99)
    sku_id: str | None = Field(default=None, max_length=64, description="直接指定规格")
    choice: dict[str, str] | None = Field(default=None, description="色号/规格选择，如 {'色号': 'R100'}")


class OrderLineRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=64)
    quantity: int = Field(default=1, ge=1, le=_MAX_LINE_QUANTITY)
    sku_id: str | None = Field(default=None, max_length=64)
    choice: dict[str, str] | None = None


class PlaceOrderRequest(BaseModel):
    items: list[OrderLineRequest] = Field(min_length=1, max_length=_MAX_LINES)
    idempotency_key: str = Field(
        min_length=8, max_length=128, description="幂等键。同一下单意图必须复用同一个 key"
    )
    coupon_id: str | None = Field(default=None, max_length=64)
    market: str | None = Field(default=None, max_length=16)


class CouponTemplateCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    kind: str = Field(default="FULL_REDUCTION", description="FULL_REDUCTION 或 PERCENT")
    threshold_usd: float = Field(ge=0, description="使用门槛（美元）")
    value_cents: int = Field(ge=1, description="满减券为减免金额（分）；百分比券为折扣基点（1000 = 9 折）")
    total: int = Field(default=0, ge=0, description="发放总量，0 表示不限")
    per_member_limit: int = Field(default=1, ge=1)
    stackable: bool = False


class StockAdjustRequest(BaseModel):
    sku_id: str = Field(min_length=1, max_length=64)
    available: int = Field(ge=0, description="可售量的**绝对值**，不是增量")


__all__ = [
    "AddressCreateRequest",
    "CartAddRequest",
    "CouponTemplateCreateRequest",
    "MemberLoginRequest",
    "MemberRegisterRequest",
    "OrderLineRequest",
    "PlaceOrderRequest",
    "StockAdjustRequest",
]
