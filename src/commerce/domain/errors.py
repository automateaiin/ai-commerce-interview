"""领域异常：继承 infra 的 `InfraError`，因此天然带 `code / http_status / retryable`。

为什么领域异常也要进统一错误树：
领域规则拒绝（库存不足、券不适用）与基础设施故障在**响应形状上必须一致**，
否则调用方要写两套解析；同时 `api.py` 的全局处理器一处即可完成映射，
业务代码里不会再出现 `try/except` 拼 Response 的写法。

注意这里的分类不是按「谁的代码」而是按**调用方能否修正**来分的：
- 4xx 类（参数/规则/权限）不重试；
- `InvalidStateTransitionError` 用 409：这是**并发冲突**，调用方需要重新读取状态再决定，
  重试同样的请求只会再次失败。
"""

from __future__ import annotations

from src.infra.core.errors import (
    AuthenticationError,
    ForbiddenError,
    InfraError,
    ValidationError,
)


class DomainValidationError(ValidationError):
    """领域规则校验失败（金额非法、批次号格式、维度缺失等）。"""

    code = "domain_rule_violated"


class UnsellableSampleError(DomainValidationError):
    """试用装等非卖品被单独下单。"""

    code = "sample_not_sellable"


class ProductNotSellableError(DomainValidationError):
    """商品/规格当前不可售（未上架或已下架）。"""

    code = "product_not_sellable"


class StockNotEnoughError(DomainValidationError):
    """可售库存不足。这是**条件 UPDATE 未命中的业务表达**，不是并发故障。"""

    code = "stock_not_enough"
    safe_message = "库存不足"


class CouponNotApplicableError(DomainValidationError):
    """优惠券门槛/叠加/时间窗不满足。"""

    code = "coupon_not_applicable"


class WeakPasswordError(DomainValidationError):
    code = "weak_password"


class MemberExistsError(ValidationError):
    """邮箱已注册。409 语义（资源冲突）比 400 更准确。"""

    code = "member_exists"
    http_status = 409


class MemberAuthenticationError(AuthenticationError):
    code = "member_unauthenticated"
    safe_message = "会员凭证无效"


class CartOwnershipError(ForbiddenError):
    """购物车不属于当前会员（横向越权）。"""

    code = "cart_forbidden"


class InvalidStateTransitionError(InfraError):
    """订单/商品状态机的非法迁移。

    用 409 而不是 500：这是「当前状态不允许该动作」的**确定性冲突**，
    重试无意义，调用方需要先读取最新状态。
    """

    code = "invalid_state_transition"
    http_status = 409
    safe_message = "当前状态不允许该操作"


class DuplicateRequestConflictError(InfraError):
    """同一幂等键携带了不同的请求体：宁可报错，也不返回上一次的结果。"""

    code = "idempotency_key_reused"
    http_status = 409
    safe_message = "幂等键已被不同请求使用"


__all__ = [
    "CartOwnershipError",
    "CouponNotApplicableError",
    "DomainValidationError",
    "DuplicateRequestConflictError",
    "InvalidStateTransitionError",
    "MemberAuthenticationError",
    "MemberExistsError",
    "ProductNotSellableError",
    "StockNotEnoughError",
    "UnsellableSampleError",
    "WeakPasswordError",
]
