"""离线示例原始数据（三类来源的真实形态）。

为什么要有一份脏数据：数据平台的价值**只有在脏数据上才能体现**。
这份 fixture 里刻意混入重复、近重复、系统通知、广告噪声、乱码、
手机号/邮箱、含价格的活动话术、发送人映射失败、以及直播 ASR 片段 ——
每一个都能对应到治理链路里的某一条规则，便于演示与回归。
"""

from __future__ import annotations

from typing import Any

RAW_RECORDS: list[dict[str, Any]] = [
    # ---- 可入训练集的风格样本（无易变事实、无敏感信息，一问一答） ----
    {
        "source": "chat_export",
        "id": "chat_g1",
        "session_id": "sess_G",
        "sender": "wxid_bob",
        "timestamp": "2026-03-09T10:00:00Z",
        "text": "你们一般多久发货呀？",
        "is_group": False,
    },
    {
        "source": "chat_export",
        "id": "chat_g2",
        "session_id": "sess_G",
        "sender": "wxid_alice",
        "timestamp": "2026-03-09T10:01:00Z",
        "text": "亲，付款后 48 小时内发出哦，节假日会顺延，发出后可在订单页查看物流。",
        "is_group": False,
    },
    # ---- 正常问答（会话 A） ----
    {
        "source": "chat_export",
        "id": "chat_a1",
        "session_id": "sess_A",
        "sender": "wxid_bob",
        "timestamp": "2026-03-02T09:12:00Z",
        "text": "我是敏感肌，这款雪绒花修护精华能用吗？",
        "is_group": False,
    },
    {
        "source": "chat_export",
        "id": "chat_a2",
        "session_id": "sess_A",
        "sender": "wxid_alice",
        "timestamp": "2026-03-02T09:12:40Z",
        "text": "雪绒花修护精华适合敏感肌使用，成分含积雪草与神经酰胺，不含酒精与香精；建议先在耳后测试。",
        "is_group": False,
    },
    # ---- 精确重复（同 session/同时间/同内容） ----
    {
        "source": "chat_export",
        "id": "chat_a2_dup",
        "session_id": "sess_A",
        "sender": "wxid_alice",
        "timestamp": "2026-03-02T09:12:40Z",
        "text": "雪绒花修护精华适合敏感肌使用，成分含积雪草与神经酰胺，不含酒精与香精；建议先在耳后测试。",
        "is_group": False,
    },
    # ---- 近重复：会话 A 那一轮问答的改写版（跨会话，应在语义单元级被判重复） ----
    {
        "source": "chat_export",
        "id": "chat_h1",
        "session_id": "sess_H",
        "sender": "wxid_bob",
        "timestamp": "2026-03-08T09:00:00Z",
        "text": "敏感肌可以用这款雪绒花修护精华吗？",
        "is_group": False,
    },
    {
        "source": "chat_export",
        "id": "chat_h2",
        "session_id": "sess_H",
        "sender": "wxid_alice",
        "timestamp": "2026-03-08T09:01:00Z",
        "text": "雪绒花修护精华适合敏感肌，成分含积雪草与神经酰胺；建议先在耳后测试。",
        "is_group": False,
    },
    # ---- 同场景换写法的另一轮问答（相似度未达近重复阈值 → 作为独立 SFT 风格样本） ----
    {
        "source": "chat_export",
        "id": "chat_b1",
        "session_id": "sess_B",
        "sender": "wxid_bob",
        "timestamp": "2026-03-03T14:02:00Z",
        "text": "请问敏感肌可以用这款雪绒花修护精华吗？",
        "is_group": False,
    },
    {
        "source": "chat_export",
        "id": "chat_b2",
        "session_id": "sess_B",
        "sender": "wxid_alice",
        "timestamp": "2026-03-03T14:03:00Z",
        "text": "可以的，它不含酒精与香精，敏感肌可放心使用，建议先做耳后测试。",
        "is_group": False,
    },
    # ---- 需要过滤：系统通知 / 广告噪声 / 乱码 / 过短 ----
    {"source": "chat_export", "id": "chat_sys", "session_id": "sess_C", "sender": "wxid_bob", "timestamp": "2026-03-04T10:00:00Z", "text": "已加入群聊", "is_group": True},
    {"source": "chat_export", "id": "chat_ad", "session_id": "sess_C", "sender": "wxid_carol", "timestamp": "2026-03-04T10:01:00Z", "text": "本店最低价 加微信 私聊下单 扫码进群", "is_group": True},
    {"source": "chat_export", "id": "chat_moji", "session_id": "sess_C", "sender": "wxid_bob", "timestamp": "2026-03-04T10:02:00Z", "text": "è¿™æ˜¯ä¹±ç \ufffd", "is_group": True},
    {"source": "chat_export", "id": "chat_short", "session_id": "sess_C", "sender": "wxid_alice", "timestamp": "2026-03-04T10:03:00Z", "text": "嗯", "is_group": True},
    # ---- 含敏感信息（入库时脱敏） ----
    {
        "source": "chat_export",
        "id": "chat_pii",
        "session_id": "sess_D",
        "sender": "wxid_bob",
        "timestamp": "2026-03-05T11:30:00Z",
        "text": "订单有问题请联系我 13800138000 或邮箱 bob@example.com",
        "is_group": False,
    },
    # ---- 含价格的活动话术（时效性 → 不进长期知识） ----
    {
        "source": "chat_export",
        "id": "chat_price",
        "session_id": "sess_E",
        "sender": "wxid_carol",
        "timestamp": "2026-03-06T16:20:00Z",
        "text": "活动价 立减 优惠券后只要 299 元，限时折扣到本周五",
        "is_group": False,
    },
    # ---- 发送人映射失败（partial，转人工判断） ----
    {
        "source": "chat_export",
        "id": "chat_unknown",
        "session_id": "sess_F",
        "sender": "wxid_zzz",
        "timestamp": "2026-03-07T08:00:00Z",
        "text": "** 这个成分会不会致痘？",
        "is_group": True,
    },
    # ---- 商品知识：可长期使用的规格与用法 ----
    {
        "source": "product_knowledge",
        "article_id": "KB-1001",
        "title": "雪绒花修护精华 · 规格与使用方法",
        "scene": "product_consulting",
        "question": "雪绒花修护精华怎么用、规格是多少？",
        "answer": "雪绒花修护精华含积雪草与神经酰胺，规格 30ml，早晚洁面后取 2-3 滴按压吸收；开封后建议 6 个月内用完。",
        "conditions": ["适用于敏感肌", "避光常温保存"],
        "product_id": "SKU-1001",
        "market": "sea",
        "tags": ["精华", "敏感肌", "成分"],
        "updated_at": "2026-03-01T00:00:00Z",
        "valid_from": "2026-03-01T00:00:00Z",
    },
    # ---- 商品知识：含价格与有效期（结构化保存但不进语义索引） ----
    {
        "source": "product_knowledge",
        "article_id": "KB-1002",
        "title": "雪绒花修护精华 · 春季促销价",
        "scene": "promotion",
        "question": "雪绒花修护精华现在多少钱？",
        "answer": "春季活动价 299 元，原价 399 元，优惠券限时折扣至 2026-03-31。",
        "product_id": "SKU-1001",
        "market": "sea",
        "price": "299",
        "price_live_source": "catalog-api:/v1/products/SKU-1001/price",
        "updated_at": "2026-03-01T00:00:00Z",
        "valid_from": "2026-03-01T00:00:00Z",
        "valid_to": "2026-03-31T00:00:00Z",
    },
    # ---- 直播 ASR 转写 ----
    {
        "source": "live_asr",
        "segment_id": "asr_001",
        "live_id": "live_20260325",
        "speaker": "wxid_carol",
        "started_at": "2026-03-25T12:00:00Z",
        "start_s": 95,
        "text": "这款精华的主推搭配是先用化妆水再用它，敏感肌也能用，成分里没有酒精。",
        "product_ref": "SKU-1001",
        "asset_id": "asset_live_01",
    },
]


# 评估集：带标准证据标注。group 用于分组切分（同一会话/客户必须同组）
GOLDEN_ITEMS: list[dict[str, Any]] = [
    {
        "question": "敏感肌可以用雪绒花修护精华吗？",
        "gold_chunk_ids": ["KB-1001#c0"],
        "group": "sess_A",
        "product_id": "SKU-1001",
        "scene": "product_consulting",
    },
    {
        "question": "雪绒花修护精华的成分和规格是什么？",
        "gold_chunk_ids": ["KB-1001#c0"],
        "group": "sess_A",
        "product_id": "SKU-1001",
        "scene": "product_consulting",
    },
    {
        "question": "这款精华怎么使用？",
        "gold_chunk_ids": ["KB-1001#c0"],
        "group": "sess_B",
        "product_id": "SKU-1001",
        "scene": "product_consulting",
    },
]
