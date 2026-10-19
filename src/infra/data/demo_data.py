"""演示业务数据（离线可跑）。

刻意与代码分离：这些数据代表「真实业务系统」，工具层只通过仓储接口读它。
生产环境这里换成商品中心 / 订单中心 / 知识库的远程调用，
工具代码与上层 Agent 不需要任何改动。
"""

from __future__ import annotations

from typing import Any

# ---- 商品主数据（高端美妆，出海场景） -------------------------------------- #
PRODUCTS: list[dict[str, Any]] = [
    {
        "product_id": "SKU-1001",
        "name": "雪绒花修护精华 30ml",
        "brand": "ALPINE LAB",
        "category": "精华",
        "price_usd": 128.0,
        "market": "US",
        "skin_types": ["敏感肌", "干性", "混合性"],
        "ingredients": ["雪绒花提取物", "神经酰胺", "泛醇"],
        "highlights": ["无酒精", "无香精", "临床测试低刺激"],
        "usage": "洁面后取 2-3 滴，按压于面部，早晚各一次。",
        "stock": 320,
    },
    {
        "product_id": "SKU-1002",
        "name": "玫瑰玻尿酸保湿面霜 50ml",
        "brand": "ALPINE LAB",
        "category": "面霜",
        "price_usd": 96.0,
        "market": "US",
        "skin_types": ["干性", "极干性"],
        "ingredients": ["大马士革玫瑰水", "三重玻尿酸", "角鲨烷"],
        "highlights": ["24 小时保湿", "可配合精华叠加"],
        "usage": "精华后取黄豆大小，由内向外均匀涂抹。",
        "stock": 180,
    },
    {
        "product_id": "SKU-1003",
        "name": "夜间视黄醇焕肤精华 30ml",
        "brand": "DERMA NOVA",
        "category": "精华",
        "price_usd": 168.0,
        "market": "US",
        "skin_types": ["油性", "混合性"],
        "ingredients": ["视黄醇 0.3%", "烟酰胺", "生育酚"],
        "highlights": ["仅夜间使用", "需建立耐受", "孕期慎用"],
        "usage": "夜间使用，初次每周 2 次，逐步增加频率。",
        "stock": 95,
    },
    {
        "product_id": "SKU-1004",
        "name": "氨基酸温和洁面乳 120ml",
        "brand": "ALPINE LAB",
        "category": "洁面",
        "price_usd": 42.0,
        "market": "US",
        "skin_types": ["敏感肌", "干性", "混合性", "油性"],
        "ingredients": ["氨基酸表活", "甘油", "甘草酸二钾"],
        "highlights": ["弱酸性", "不紧绷"],
        "usage": "早晚各一次，取适量加水揉搓起泡后按摩面部。",
        "stock": 540,
    },
]

# ---- 订单数据（演示用；真实场景来自订单中心） ------------------------------ #
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-90001": {
        "order_id": "ORD-90001",
        "user_id": "u_10086",
        "status": "已发货",
        "carrier": "DHL",
        "items": [{"product_id": "SKU-1001", "quantity": 1, "price_usd": 128.0}],
        "total_usd": 128.0,
        "updated_at": "2026-10-12T09:20:00Z",
    },
    "ORD-90002": {
        "order_id": "ORD-90002",
        "user_id": "u_10086",
        "status": "待付款",
        "carrier": None,
        "items": [{"product_id": "SKU-1004", "quantity": 2, "price_usd": 42.0}],
        "total_usd": 84.0,
        "updated_at": "2026-10-15T21:03:00Z",
    },
    "ORD-90003": {
        "order_id": "ORD-90003",
        "user_id": "u_20001",
        "status": "已完成",
        "carrier": "FedEx",
        "items": [{"product_id": "SKU-1002", "quantity": 1, "price_usd": 96.0}],
        "total_usd": 96.0,
        "updated_at": "2026-09-28T11:40:00Z",
    },
}

# ---- 知识库（含一条已过期促销，用于演示有效期过滤） ------------------------ #
KNOWLEDGE: list[dict[str, Any]] = [
    {
        "chunk_id": "kb_001",
        "article_id": "art_ingredient_1001",
        "title": "雪绒花修护精华 · 成分与适用肤质",
        "text": "雪绒花修护精华含雪绒花提取物、神经酰胺与泛醇，主打屏障修护与舒缓。配方无酒精、无香精，"
        "适合敏感肌、干性与混合性肌肤。首次使用建议先在耳后小面积试用。",
        "product_id": "SKU-1001",
        "market": "US",
        "source": "商品资料库 v3",
        "tags": ["成分", "适用肤质", "敏感肌"],
        "valid_to": None,
    },
    {
        "chunk_id": "kb_002",
        "article_id": "art_usage_1001",
        "title": "雪绒花修护精华 · 使用方法",
        "text": "洁面后取 2-3 滴，按压于面部吸收，早晚各一次。可与同系列玫瑰玻尿酸保湿面霜叠加使用，"
        "先精华后面霜，避免与高浓度酸类产品同时使用。",
        "product_id": "SKU-1001",
        "market": "US",
        "source": "商品资料库 v3",
        "tags": ["使用方法", "搭配"],
        "valid_to": None,
    },
    {
        "chunk_id": "kb_003",
        "article_id": "art_promo_1002",
        "title": "玫瑰面霜 · 限时促销（已结束）",
        "text": "玫瑰玻尿酸保湿面霜限时买一送一，活动价 68 美元。活动截止 2026-08-31。",
        "product_id": "SKU-1002",
        "market": "US",
        "source": "活动系统快照",
        "tags": ["促销", "价格"],
        "valid_to": "2026-08-31T23:59:59Z",
    },
    {
        "chunk_id": "kb_004",
        "article_id": "art_ingredient_1003",
        "title": "夜间视黄醇精华 · 使用注意",
        "text": "夜间视黄醇焕肤精华含 0.3% 视黄醇，仅建议夜间使用。初次使用需建立耐受，每周 2 次起步。"
        "孕期、哺乳期不建议使用，使用期间需加强防晒。不可与果酸、水杨酸叠加。",
        "product_id": "SKU-1003",
        "market": "US",
        "source": "商品资料库 v3",
        "tags": ["视黄醇", "注意事项", "孕期"],
        "valid_to": None,
    },
    {
        "chunk_id": "kb_005",
        "article_id": "art_logistics_us",
        "title": "美国站物流与退换政策",
        "text": "美国站标准配送 5-8 个工作日，满 79 美元免运费。支持 30 天无理由退换，"
        "已开封的护肤类商品因卫生原因不支持退换。",
        "product_id": None,
        "market": "US",
        "source": "客服政策库 v2",
        "tags": ["物流", "退换", "政策"],
        "valid_to": None,
    },
]
