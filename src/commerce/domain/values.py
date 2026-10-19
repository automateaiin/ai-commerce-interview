"""领域值对象：不可变、可比较、无身份——描述「是什么」而不是「哪一个」。

三个值对象的选择理由（面试可讲）：

1. **Money 用「分」的整数**：浮点无法精确表示 0.1，做累计与折扣会在小数点后漂移，
   而电商金额必须是精确十进制；同时金额必须自带币种，否则跨市场结算时
   `128` 到底是美元还是欧元都说不清。
2. **SkuMatrix（色号 × 规格）**：美妆商品天然是多维的（一支口红多个色号、
   一支精华多个规格）。把维度与可售组合显式建模，才能回答
   「这个色号有没有货」「这个组合存不存在」；平铺 SKU 列表则每次都要遍历匹配，
   而且无法表达「组合不合法」。
3. **BatchNo / ShelfLife + InciIngredient**：美妆受法规约束——成分表必须能规范化比对
   （否则重复成分会让浓度合计算错），批次必须能算出到期日（过期批次不可售）。
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .errors import DomainValidationError, UnsellableSampleError

_CENT = Decimal("0.01")
_BATCH_RE = re.compile(r"^[A-Z0-9]{4,20}$")
_WS_RE = re.compile(r"[\s\-_/]+")


# --------------------------------------------------------------------------- #
# 金额
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Money:
    """精确金额。内部只存「分」，对外用 Decimal 呈现。"""

    cents: int
    currency: str = "USD"

    @classmethod
    def of(cls, amount: object, currency: str = "USD") -> "Money":
        """从 float / str / Decimal 构造。先转 str 再进 Decimal，避免二进制浮点误差。"""
        try:
            value = Decimal(str(amount)).quantize(_CENT, rounding=ROUND_HALF_UP)
        except Exception as exc:  # noqa: BLE001 - 任何非法金额都归一为领域校验失败
            raise DomainValidationError(f"金额格式非法: {amount!r}") from exc
        return cls(cents=int(value * 100), currency=currency.upper())

    def _check(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise DomainValidationError(f"币种不一致: {self.currency} vs {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents + other.cents, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents - other.cents, self.currency)

    def __mul__(self, factor: int) -> "Money":
        if factor < 0:
            raise DomainValidationError("金额不能乘负数")
        return Money(self.cents * factor, self.currency)

    __rmul__ = __mul__

    def __lt__(self, other: "Money") -> bool:
        self._check(other)
        return self.cents < other.cents

    def __ge__(self, other: "Money") -> bool:
        self._check(other)
        return self.cents >= other.cents

    @property
    def is_negative(self) -> bool:
        return self.cents < 0

    @property
    def amount(self) -> Decimal:
        return (Decimal(self.cents) / 100).quantize(_CENT)

    def clamp_at_zero(self) -> "Money":
        """折扣金额不得超过原价：负的应付金额必须归零，绝不能变成「平台倒贴」。"""
        return Money(max(self.cents, 0), self.currency)

    def to_dict(self) -> dict[str, Any]:
        return {"amount": str(self.amount), "currency": self.currency, "cents": self.cents}


# --------------------------------------------------------------------------- #
# 色号 × 规格 矩阵
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SkuMatrix:
    """可售组合矩阵。`axes` 是各维度取值，`sellable` 是真实存在的组合。"""

    axes: dict[str, tuple[str, ...]]
    sellable: frozenset[tuple[str, ...]]

    @classmethod
    def build(cls, axes: dict[str, tuple[str, ...]], sellable: list[tuple[str, ...]]) -> "SkuMatrix":
        ordering = tuple(axes)
        for combo in sellable:
            if len(combo) != len(ordering):
                raise DomainValidationError(f"组合维度数不匹配: {combo}")
            for axis, value in zip(ordering, combo):
                if value not in axes[axis]:
                    raise DomainValidationError(f"组合引用了未声明的取值: {axis}={value}")
        return cls(axes=dict(axes), sellable=frozenset(sellable))

    def axis_names(self) -> list[str]:
        return list(self.axes)

    def options_of(self, axis: str) -> tuple[str, ...]:
        if axis not in self.axes:
            raise DomainValidationError(f"未知维度: {axis}")
        return self.axes[axis]

    def resolve(self, choice: dict[str, str]) -> tuple[str, ...]:
        ordering = tuple(self.axes)
        missing = [axis for axis in ordering if axis not in choice]
        if missing:
            raise DomainValidationError(f"缺少维度取值: {missing}", detail={"missing": missing})
        return tuple(choice[axis] for axis in ordering)

    def is_sellable(self, choice: dict[str, str]) -> bool:
        return self.resolve(choice) in self.sellable

    def combinations(self) -> list[dict[str, str]]:
        ordering = tuple(self.axes)
        return [dict(zip(ordering, combo)) for combo in sorted(self.sellable)]


# --------------------------------------------------------------------------- #
# 成分（INCI）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class InciIngredient:
    """INCI 成分条目。`normalized` 是比对的唯一依据。"""

    name: str
    function: str
    cas_no: str | None = None
    concentration: float | None = None

    @property
    def normalized(self) -> str:
        return _WS_RE.sub("", self.name or "").lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function": self.function,
            "cas_no": self.cas_no,
            "concentration": self.concentration,
        }


def normalize_inci(items: list[InciIngredient]) -> list[InciIngredient]:
    """按规范化名称去重，保留首次出现顺序。

    重复成分会让浓度合计虚高，而浓度在配方里有法规上限——所以去重发生在领域层，
    不能寄希望于录入方「不会录重」。
    """
    seen: set[str] = set()
    result: list[InciIngredient] = []
    for item in items:
        key = item.normalized
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


# --------------------------------------------------------------------------- #
# 批次 + 保质期
# --------------------------------------------------------------------------- #
def _days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


@dataclass(frozen=True, slots=True)
class BatchNo:
    """生产批次 + 保质期。到期日 = 生产日 + 保质月数，按自然月推进。"""

    code: str
    produced_on: date
    shelf_life_months: int

    def __post_init__(self) -> None:
        if not _BATCH_RE.match(self.code or ""):
            raise DomainValidationError(f"批次号格式非法: {self.code!r}", detail={"code": self.code})
        if not 1 <= self.shelf_life_months <= 60:
            raise DomainValidationError("保质期月数必须在 1~60 之间")

    @property
    def expires_on(self) -> date:
        month_index = self.produced_on.month - 1 + self.shelf_life_months
        year = self.produced_on.year + month_index // 12
        month = month_index % 12 + 1
        day = min(self.produced_on.day, _days_in_month(year, month))
        return date(year, month, day)

    def days_left(self, at: date | None = None) -> int:
        return (self.expires_on - (at or date.today())).days

    def is_expired(self, at: date | None = None) -> bool:
        return self.days_left(at) < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "produced_on": self.produced_on.isoformat(),
            "shelf_life_months": self.shelf_life_months,
            "expires_on": self.expires_on.isoformat(),
        }


# --------------------------------------------------------------------------- #
# 试用装
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TrialSample:
    """试用装。与正装共用配方，但**不可单独售卖**。"""

    sku_id: str
    volume_ml: float
    attached_to: str

    def assert_sellable(self) -> None:
        # 规则写在值对象里而不是散落在业务代码：只要有人试图下单试用装，
        # 就一定会走到这里，没有「忘记校验」的路径。
        raise UnsellableSampleError(
            f"试用装不可单独售卖: {self.sku_id}，请购买正装 {self.attached_to}",
            detail={"sku_id": self.sku_id, "attached_to": self.attached_to},
        )

    def to_dict(self) -> dict[str, Any]:
        return {"sku_id": self.sku_id, "volume_ml": self.volume_ml, "attached_to": self.attached_to}


__all__ = [
    "BatchNo",
    "InciIngredient",
    "Money",
    "SkuMatrix",
    "TrialSample",
    "normalize_inci",
]
