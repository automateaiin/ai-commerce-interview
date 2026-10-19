"""项目四：美妆电商业务层。

以「美妆公司自营商城」为背景，移植 macrozheng/mall 的**领域拆分思想**
（PMS 商品 / UMS 会员 / OMS 订单 / SMS 营销），打通
「浏览商品 → 加购物车 → 下单 → 模拟支付 → 履约/取消」主干，并预埋 AI 钩子。

分层（单向依赖，禁止倒置）：

    interface → application → domain
                     ↑
              infrastructure（实现 domain 的仓储 Protocol）

对外只通过 `service.py`（门面）与 `factory.py`（装配）暴露。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
