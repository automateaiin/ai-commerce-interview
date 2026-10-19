"""纯领域层：只依赖 stdlib（异常体系复用 infra 的 InfraError），可脱离数据库单测。

设计动机：把「业务规则」与「存储/框架」彻底分开 —— 金额精度、色号矩阵、批次保质期、
优惠券门槛这些规则**不因为换了数据库而改变**，所以它们不应该知道 SQLAlchemy 的存在。
"""

from __future__ import annotations

__all__ = ["__doc__"]
