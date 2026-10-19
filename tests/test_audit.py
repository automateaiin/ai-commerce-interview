"""审计：四级权限链（Service → Agent → Tool → Resource）落成可查记录。

两个设计点要能被测出来：
1) 资源由**参数推导**，不由调用方声明（否则等于让被审计者自己写审计）；
2) 拒绝与失败同样留痕 —— 只记成功调用，越权尝试就完全隐形。
"""

from __future__ import annotations

from src.infra.core.audit import AuditLog, derive_resource


def test_resource_derived_from_id_argument():
    assert derive_resource("get_order_status", {"order_id": "ORD-90001"}) == "order:ORD-90001"
    assert derive_resource("get_product_detail", {"product_id": "SKU-1001"}) == "product:SKU-1001"
    # sku / chunk / article 是同一实体的不同叫法，带 _id 后缀时需归一到统一命名
    assert derive_resource("search_products", {"sku_id": "SKU-1001"}) == "product:SKU-1001"
    assert derive_resource("search_knowledge", {"chunk_id": "KB-1001#c0"}) == "knowledge:KB-1001#c0"


def test_resource_falls_back_to_business_keyword_then_wildcard():
    # 无 *_id 时退化为「工具名 + 业务关键词」，仍能看出这次调用针对什么
    assert derive_resource("search_products", {"sku": "SKU-1001"}) == "search_products:SKU-1001"
    assert derive_resource("search_knowledge", {"query": "敏感肌 精华"}).startswith("search_knowledge:")
    assert derive_resource("list_tools", {}) == "list_tools:*"


def test_resource_is_scrubbed():
    text = derive_resource("search_knowledge", {"query": "联系我 13800138000"})
    assert "13800138000" not in text


def test_records_are_queryable_by_project_and_decision():
    log = AuditLog()
    log.record(
        tool="get_order_status", resource="order:ORD-90001", decision="allow",
        service="cs-service", agent="customer_agent", caller="cs-bot", project_id="cs-bot",
        trace_id="tr_1", risk="read_only",
    )
    log.record(
        tool="create_order", resource="product:SKU-1001", decision="deny",
        service="cs-service", agent="customer_agent", caller="cs-bot", project_id="cs-bot",
        reason="missing_scope:tools:write", error_code="permission_denied",
    )
    log.record(
        tool="create_order", resource="product:SKU-1001", decision="allow",
        service="sales-ops", agent="unknown", caller="sales-agent", project_id="sales-agent",
    )

    cs_all = log.query(project_id="cs-bot")
    assert {event.project_id for event in cs_all} == {"cs-bot"}
    assert len(cs_all) == 2

    denies = log.query(project_id="cs-bot", decision="deny")
    assert len(denies) == 1
    assert denies[0].reason == "missing_scope:tools:write"
    assert denies[0].error_code == "permission_denied"

    assert log.query(project_id="sales-agent")[0].tool == "create_order"


def test_summary_counts_every_decision_including_denies():
    log = AuditLog()
    log.record(tool="t", resource="t:*", decision="allow", project_id="p")
    log.record(tool="t", resource="t:*", decision="deny", project_id="p")
    log.record(tool="t", resource="t:*", decision="error", project_id="p")

    summary = log.summary()
    assert summary["total"] == 3
    assert summary["by_decision"] == {"allow": 1, "deny": 1, "error": 1}


def test_buffer_is_bounded_to_avoid_memory_leak():
    log = AuditLog(max_entries=5)
    for index in range(20):
        log.record(tool="t", resource=f"t:{index}", decision="allow", project_id="p")

    assert log.summary()["total"] == 5
    assert log.query(project_id="p", limit=10)[0].resource == "t:19"
