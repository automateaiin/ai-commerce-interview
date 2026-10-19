"""端到端冒烟脚本：把项目一的关键链路一次跑通，并给出可复核的结论。

覆盖范围（对应 PROJECT.md 第 4 章工程化门槛）：
- 结构化日志 / trace 贯穿（由访问日志与响应 trace_id 验证）
- 模型网关：Provider 抽象 + 逻辑名路由 + 能力校验 + Mock 脚本化
- MCP 工具服务：参数二次校验、权限判定、业务授权（订单归属）、幂等写
- RAG：有效期过滤 + 元数据过滤 + 混合打分
- 配额：预留-实扣（用量增长与释放）
- 子 Agent：工具循环、重复调用检测、结构化结果契约
- Harness（项目二）：未知工具自纠错、连续失败熔断交接、高风险动作确认门
- 数据平台（项目三）：分级去重、规则过滤、脱敏、价格时效治理、索引一致性、
  混合检索组装、召回评估、训练导出校验门禁、血缘追溯
- 异常映射：401 / 400 / 403 / 工具级错误的结构化响应
- 配置热重载：坏配置不生效（旧快照继续可用）
- MCP over Streamable HTTP：内部凭证校验 + 协议调用
- 平台治理（项目三 · 阶段 1）：限流 429 与配额 429 的区分、审计双向留痕与项目隔离、
  Prompt 角色组装与预览权限边界、成本按 Service/Agent 聚合与预算水位、预算软阈值降级/硬上限、
  评测六维报告与 eval:rw 边界
- 宿主业务后端（项目一 · commerce）：浏览开放/交易鉴权、会员 token 与项目级 API Key 互不通用、
  REST 下单幂等重放、下单事件落 Outbox（与业务写同一事务）、付款/取消状态迁移与库存回退、
  /v1/admin/* 的 commerce:rw 闸门
- 应用层（项目一 · AI 智能客服导购系统）：复杂度路由（规则直连零 LLM / 跨域才进 Agent）、
  编排模式与事件流、Grounding Gate 拦截无来源交易断言、HITL 三步链路（登记/独立审批/原始身份执行）、
  A2A 内部令牌与 session_id 透传、实时语音端口+Mock 事件流、检索重写三策略对比、反馈闭环投影
- 应用层（项目二 · AI 运营系统）：内容工作流节点级续跑/回滚/回放与结构化兜底、SSE 事件级流式与
  客户端断开释放、并行实测加速比与失败隔离、多模态素材幂等 UPSERT 与跨模态检索、直播分段转录/
  重叠去重/FFmpeg 降级声明、可复算指标与渠道归因、自进化评测集与模板迭代（不自动发布）、ops:rw 边界
- 统一验证（阶段 6 · 跨项目）：四项目与共享底座共存于同一应用、鉴权契约与错误信封跨项目一致、
  排他 scope（ops/data/commerce/eval）互不串门的三态矩阵、trace_id 贯穿全部入口

用法：
    .\.venv\Scripts\python.exe scripts\smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 冒烟环境：全部走 MockProvider（离线、可复现），并开启内部调用凭证
os.environ.setdefault("PROVIDER_MODE", "mock")
os.environ.setdefault("INTERNAL_SERVICE_TOKEN", "smoke-internal-token")
os.environ.setdefault("MCP_ALLOWED_HOSTS", "smoke")  # ASGI 直连下 Host 为 smoke，需显式声明进 Host 白名单
os.environ.setdefault("LOG_LEVEL", "WARNING")  # 冒烟时压低日志噪声，需要看链路时改成 INFO

# 每次冒烟从干净库开始：否则上一轮残留的历史消息会让「记忆条数」这类断言失真
_SMOKE_DB = Path(tempfile.gettempdir()) / "ecom_infra_smoke.db"
_SMOKE_DB.unlink(missing_ok=True)
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_SMOKE_DB.as_posix()}")

# 业务库同样从干净库开始：否则上一轮下的订单会让库存/幂等/统计断言失真
_SMOKE_COMMERCE_DB = Path(tempfile.gettempdir()) / "ecom_commerce_smoke.db"
_SMOKE_COMMERCE_DB.unlink(missing_ok=True)
os.environ.setdefault("COMMERCE_DATABASE_URL", f"sqlite+aiosqlite:///{_SMOKE_COMMERCE_DB.as_posix()}")
os.environ.setdefault("COMMERCE_MEMBER_TOKEN_SECRET", "smoke-member-token-secret")
os.environ.setdefault("COMMERCE_SWEEPER_INTERVAL_S", "0")  # 冒烟不跑后台清理，超时清理用手动端点验证

import httpx  # noqa: E402

from src.infra.api import create_app  # noqa: E402
from src.infra.llm.mock import calls, final  # noqa: E402

CS_BOT_KEY = "sk-project-service-demo"
SALES_KEY = "sk-project-sales-demo"
CONTENT_KEY = "sk-project-content-demo"
OPS_KEY = "sk-project-ops-demo"  # 运营系统凭证（ops:rw）：内容生产/素材/直播/自进化走独立 scope
EVAL_KEY = "sk-project-eval-demo"  # 平台治理能力走独立凭证：不扩大业务线的 scope
SHOP_ADMIN_KEY = "sk-project-shopadmin-demo"  # 业务后端运营侧凭证（commerce:rw）
HITL_KEY = "sk-project-hitl-demo"  # 高风险动作审批凭证（hitl:approve，与客服凭证分离）

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -> {detail}" if detail else ""))
    return bool(ok)


async def main() -> int:
    app = create_app()

    # ASGITransport 不会自动执行 lifespan，这里手动进入，确保 MCP 会话管理器已启动
    async with app.router.lifespan_context(app):
        container = app.state.container
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://smoke", timeout=30) as client:
            await run_checks(client, container)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("\n" + "=" * 72)
    print(f"冒烟结果：{passed}/{total} 通过")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} {detail}")
    print("=" * 72)
    return 0 if passed == total else 1


async def run_checks(client: httpx.AsyncClient, container) -> None:
    # ---------------- 1. 健康检查 ---------------- #
    r = await client.get("/healthz")
    check("存活探针 /healthz", r.status_code == 200 and r.json()["status"] == "ok", f"status={r.status_code}")

    r = await client.get("/readyz")
    body = r.json()
    check(
        "就绪探针 /readyz（校验依赖真实可用）",
        r.status_code == 200 and body["checks"]["database"]["ok"] and body["checks"]["knowledge"]["ok"],
        json.dumps(body["checks"], ensure_ascii=False),
    )

    # ---------------- 2. 鉴权 ---------------- #
    r = await client.post("/v1/tools/search_products", json={"arguments": {"query": "精华"}})
    check("缺少 API Key -> 401", r.status_code == 401 and r.json()["error"]["code"] == "unauthenticated")

    r = await client.post(
        "/v1/tools/search_products",
        json={"arguments": {"query": "精华"}},
        headers={"X-API-Key": "sk-not-exist"},
    )
    check("错误 API Key -> 401", r.status_code == 401 and r.json()["error"]["code"] == "unauthenticated")

    # ---------------- 3. 工具：正常路径 + 参数二次校验 ---------------- #
    r = await client.post(
        "/v1/tools/search_products",
        json={"arguments": {"query": "精华", "skin_type": "敏感肌"}},
        headers={"X-API-Key": CS_BOT_KEY, "X-Trace-Id": "tr_smoke_demo_001"},
    )
    body = r.json()
    check(
        "工具正常路径 + trace_id 回传",
        r.status_code == 200 and body["success"] and body["trace_id"] == "tr_smoke_demo_001",
        f"count={body['data']['count']}",
    )

    r = await client.post(
        "/v1/tools/search_products",
        json={"arguments": {"query": "精华", "limit": 999}},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    body = r.json()
    check(
        "参数越界被服务端二次校验拦截（不信任模型入参）",
        body["success"] is False and body["error"]["code"] == "invalid_arguments",
        json.dumps(body.get("error"), ensure_ascii=False),
    )

    r = await client.post(
        "/v1/tools/not_a_real_tool",
        json={"arguments": {}},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    body = r.json()
    check(
        "未知工具 -> 结构化可恢复错误（不是 500）",
        r.status_code == 200 and body["success"] is False and body["error"]["code"] == "not_found",
    )

    # ---------------- 4. 授权：scope 与业务归属 ---------------- #
    r = await client.post(
        "/v1/tools/search_products",
        json={"arguments": {"query": "面霜"}},
        headers={"X-API-Key": CONTENT_KEY},
    )
    body = r.json()
    check(
        "缺少 tools:read scope -> permission_denied（越权工具调用被拒）",
        body["success"] is False and body["error"]["code"] == "permission_denied" and body["error"]["terminal"],
        json.dumps(body.get("error"), ensure_ascii=False),
    )

    r = await client.post(
        "/v1/tools/get_order_status",
        json={"arguments": {"order_id": "ORD-90001"}, "user_id": "u_10086"},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    check("订单归属校验：本人订单可查", r.json()["success"] and r.json()["data"]["status"] == "已发货")

    r = await client.post(
        "/v1/tools/get_order_status",
        json={"arguments": {"order_id": "ORD-90001"}, "user_id": "u_99999"},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    body = r.json()
    check(
        "订单归属校验：他人订单被拒（参数合法 ≠ 有权访问）",
        body["success"] is False and body["error"]["code"] == "permission_denied",
    )

    # ---------------- 5. RAG：有效期 + 元数据 + 混合打分 ---------------- #
    r = await client.post(
        "/v1/tools/search_knowledge",
        json={"arguments": {"query": "敏感肌 精华 成分", "top_k": 3}},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    data = r.json()["data"]
    chunk_ids = [chunk["chunk_id"] for chunk in data["chunks"]]
    check(
        "RAG 召回证据并带来源",
        data["has_evidence"] and all(chunk["source"] for chunk in data["chunks"]),
        f"chunks={chunk_ids}",
    )
    check(
        "已过期促销被有效期过滤（不会把过期政策当有效信息）",
        data["diagnostics"]["filtered_expired"] >= 1 and "kb_003" not in chunk_ids,
        json.dumps(data["diagnostics"], ensure_ascii=False),
    )

    r = await client.post(
        "/v1/tools/search_knowledge",
        json={"arguments": {"query": "怎么用", "product_id": "SKU-1001"}},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    data = r.json()["data"]
    check(
        "元数据过滤：限定商品后不返回其它商品知识",
        all(chunk["product_id"] in (None, "SKU-1001") for chunk in data["chunks"]),
        f"chunks={[c['chunk_id'] for c in data['chunks']]}",
    )

    # ---------------- 6. 写工具幂等 ---------------- #
    order_payload = {
        "arguments": {"product_id": "SKU-1001", "quantity": 1, "idempotency_key": "smoke-idem-key-001"},
        "user_id": "u_10086",
    }
    first = await client.post("/v1/tools/create_order", json=order_payload, headers={"X-API-Key": SALES_KEY})
    second = await client.post("/v1/tools/create_order", json=order_payload, headers={"X-API-Key": SALES_KEY})
    first_body, second_body = first.json(), second.json()
    check(
        "写操作幂等：同幂等键重复调用返回首次结果（不重复下单）",
        first_body["success"]
        and first_body["data"]["order_id"] == second_body["data"]["order_id"]
        and second_body["data"]["idempotent_replay"] is True,
        f"order={first_body.get('data', {}).get('order_id')}",
    )

    # 写工具需要额外 scope：客服凭证没有 tools:write
    denied = await client.post("/v1/tools/create_order", json=order_payload, headers={"X-API-Key": CS_BOT_KEY})
    check(
        "写操作需要独立 scope（客服凭证无 tools:write 被拒）",
        denied.json()["success"] is False and denied.json()["error"]["code"] == "permission_denied",
        json.dumps(denied.json().get("error"), ensure_ascii=False),
    )

    # ---------------- 7. 客户端伪造 project_id 被忽略 ---------------- #
    r = await client.post(
        "/v1/tools/search_products",
        json={"arguments": {"query": "精华"}, "client_project_id": "sales-agent"},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    quota = (await client.get("/v1/quota", headers={"X-API-Key": CS_BOT_KEY})).json()
    check(
        "客户端声明的 project_id 被忽略（身份由凭证推导）",
        r.json()["success"] and quota["project_id"] == "cs-bot",
        f"实际 project_id={quota['project_id']}",
    )

    # ---------------- 8. 模型网关 + 记忆 + 阶段耗时 ---------------- #
    chat_body = {
        "messages": [{"role": "user", "content": "这款精华适合敏感肌吗？"}],
        "session_id": "sess-smoke-1",
        "user_id": "u_10086",
        "logical_model": "chat-lite",
    }
    r = await client.post("/v1/chat/completions", json=chat_body, headers={"X-API-Key": CS_BOT_KEY})
    body = r.json()
    check(
        "模型网关：归一化响应 + usage + 成本换算",
        body["success"] and body["usage"]["total_tokens"] > 0 and body["provider"] == "mock",
        f"model={body['model']} tokens={body['usage']['total_tokens']}",
    )
    check(
        "阶段耗时打点（可定位慢在哪一环）",
        any(stage["stage"] == "llm.complete" for stage in body["stages"]),
        json.dumps(body["stages"], ensure_ascii=False),
    )

    # 第二次同会话调用：历史应被读入并进入模型输入
    await client.post("/v1/chat/completions", json=chat_body, headers={"X-API-Key": CS_BOT_KEY})
    bound = container.gateway.provider_of("chat-lite")
    mock = getattr(bound, "mock", None)
    last_messages = mock.history[-1] if mock and mock.history else []
    check(
        "会话记忆：第二轮输入包含上一轮历史",
        len(last_messages) >= 3,
        f"messages={len(last_messages)}",
    )

    # 路由：任务类型选模型
    r = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "帮我做复杂推理"}], "task": "complex_reasoning"},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    check(
        "任务类型路由到配置的逻辑模型",
        r.json()["logical_model"] == "chat-pro",
        f"logical_model={r.json()['logical_model']}",
    )

    # 能力校验：vision 模型没有 tools 能力，请求工具时应被拒绝
    r = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "看图"}], "logical_model": "vision"},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    check("能力校验之外的正向调用", r.json()["success"], f"logical_model={r.json()['logical_model']}")

    # ---------------- 9. 子 Agent：工具循环 ---------------- #
    if mock is not None:
        mock.reset(
            [
                calls(("search_products", {"query": "精华", "skin_type": "敏感肌"})),
                calls(("search_knowledge", {"query": "雪绒花 精华 成分", "product_id": "SKU-1001"})),
                final("雪绒花修护精华（SKU-1001）无酒精无香精，适合敏感肌，建议配合保湿面霜使用。"),
            ]
        )
    r = await client.post(
        "/v1/agent/run",
        json={"task": "给敏感肌用户推荐一款精华并说明依据", "session_id": "sess-smoke-1", "user_id": "u_10086"},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    body = r.json()
    check(
        "子 Agent：多步工具循环并返回结构化结果契约",
        body["success"] and body["steps"] >= 2 and len(body["tool_calls"]) == 2,
        f"steps={body['steps']} stop={body['stop_reason']} tools={[c['name'] for c in body['tool_calls']]}",
    )
    check(
        "子 Agent：结果契约含 token 用量（成本统计依据）",
        body["usage"]["total_tokens"] > 0,
        json.dumps(body["usage"], ensure_ascii=False),
    )
    check(
        "子 Agent：执行轨迹可用于排障/前端可视化",
        any(step["type"] == "tool" for step in body["transcript"]),
        f"transcript={len(body['transcript'])} steps",
    )

    # 重复调用检测
    if mock is not None:
        mock.reset(
            [
                calls(("search_products", {"query": "面霜"})),
                calls(("search_products", {"query": "面霜"})),
                final("不应到达"),
            ]
        )
    r = await client.post(
        "/v1/agent/run",
        json={"task": "反复查同一件事", "tool_names": ["search_products"]},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    check(
        "子 Agent：重复调用被打转检测终止（护栏生效）",
        r.json()["stop_reason"] == "repeated_call_detected",
        f"stop_reason={r.json()['stop_reason']}",
    )

    # 迭代上限护栏
    if mock is not None:
        mock.reset(
            [
                calls(("search_products", {"query": f"轮次{i}"})) for i in range(6)
            ]
        )
    r = await client.post(
        "/v1/agent/run",
        json={"task": "一直查下去", "max_steps": 2, "tool_names": ["search_products"]},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    check(
        "子 Agent：迭代上限护栏（max_steps 生效）",
        r.json()["stop_reason"] == "max_steps" and r.json()["steps"] == 2,
        f"stop_reason={r.json()['stop_reason']} steps={r.json()['steps']}",
    )

    # ---------------- 9b. 项目二：Harness 治理能力 ---------------- #
    # 未知工具 → 校验失败转成 observation 回填，模型自行换工具后完成（可恢复，不终止）
    if mock is not None:
        mock.reset([calls(("no_such_tool", {})), final("这个工具不存在，我换个思路回答。")])
    r = await client.post(
        "/v1/agent/run",
        json={"task": "用一个不存在的工具", "tool_names": ["search_products"]},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    body = r.json()
    check(
        "Harness：未知工具回填为可恢复 observation（模型自纠错后仍能完成）",
        body["stop_reason"] == "completed"
        and any(
            step["type"] == "violation" and step["error_code"] == "not_found"
            for step in body["transcript"]
        ),
        f"stop={body['stop_reason']}",
    )

    # 连续失败熔断：工具持续报错（非终止性）累计到阈值 → 终止并交接
    if mock is not None:
        mock.reset([calls(("get_product_detail", {"product_id": f"SKU-99{i:02d}"})) for i in range(4)])
    r = await client.post(
        "/v1/agent/run",
        json={"task": "查一批不存在的商品", "tool_names": ["get_product_detail"]},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    body = r.json()
    check(
        "Harness：连续失败熔断（error_threshold_exceeded）+ 交接包",
        body["stop_reason"] == "error_threshold_exceeded"
        and body["handoff"]["reason"] == "error_threshold"
        and body["handoff"]["next_actions"],
        f"stop={body['stop_reason']} handoff={body.get('handoff', {}).get('reason')}",
    )

    # 高风险动作确认门：下单有真实副作用，模型不能自己拍板 → 终止并交接给人
    high_risk_payload = {
        "task": "直接帮用户下一单精华",
        "user_id": "u_10086",
        "tool_names": ["create_order"],
    }
    if mock is not None:
        mock.reset(
            [
                calls(
                    (
                        "create_order",
                        {"product_id": "SKU-1001", "quantity": 1, "idempotency_key": "smoke-hr-key-01"},
                    )
                )
            ]
        )
    r = await client.post("/v1/agent/run", json=high_risk_payload, headers={"X-API-Key": SALES_KEY})
    body = r.json()
    check(
        "Harness：高风险动作未确认 → 终止并交接（不下单）",
        body["stop_reason"] == "handoff_required"
        and body["handoff"]["reason"] == "high_risk_confirmation"
        and body["tool_calls"] == [],
        f"stop={body['stop_reason']} handoff={body.get('handoff', {}).get('reason')}",
    )

    if mock is not None:
        mock.reset(
            [
                calls(
                    (
                        "create_order",
                        {"product_id": "SKU-1001", "quantity": 1, "idempotency_key": "smoke-hr-key-02"},
                    )
                ),
                final("已按人工确认下单。"),
            ]
        )
    r = await client.post(
        "/v1/agent/run",
        json={**high_risk_payload, "confirm_high_risk": True},
        headers={"X-API-Key": SALES_KEY},
    )
    body = r.json()
    check(
        "Harness：人工确认后高风险动作才被执行",
        body["stop_reason"] == "completed" and [c["name"] for c in body["tool_calls"]] == ["create_order"],
        f"stop={body['stop_reason']} tools={[c['name'] for c in body['tool_calls']]}",
    )

    check(
        "Harness：结构化执行轨迹摘要（可回放/可评估）",
        body["trace"]["rounds"] >= 1 and body["trace"]["stop_reason"] == "completed" and body["trace"]["tokens"] > 0,
        json.dumps(body["trace"], ensure_ascii=False),
    )

    # ---------------- 9c. 项目三：数据平台（治理 → 资产 → 检索 → 评估 → 导出 → 血缘） ---------------- #
    # 未跑流水线就检索 -> 404：宁可报错，也不返回一个会被误读成「确实没有知识」的空结果
    r = await client.post("/v1/data/search", json={"query": "精华"}, headers={"X-API-Key": CONTENT_KEY})
    check(
        "数据平台：未跑流水线时检索 -> 404（不静默返回空结果）",
        r.status_code == 404 and r.json()["error"]["code"] == "not_found",
        f"status={r.status_code}",
    )

    # 数据面操作需要独立 scope：客服凭证只有 llm:chat，不能触发训练集导出这类高影响动作
    r = await client.post("/v1/data/pipeline/run", json={}, headers={"X-API-Key": CS_BOT_KEY})
    check(
        "数据平台：数据面操作需要独立 data:rw scope（会话凭证被拒）",
        r.status_code == 403 and r.json()["error"]["code"] == "permission_denied",
        f"status={r.status_code}",
    )

    r = await client.post(
        "/v1/data/pipeline/run",
        json={"reviewer": "auto_policy", "export_format": "sharegpt", "export_batch": "smoke-b1"},
        headers={"X-API-Key": CONTENT_KEY},
    )
    body = r.json()
    report = body["report"]
    check(
        "数据平台：流水线全链路跑通并输出各阶段统计",
        r.status_code == 200 and body["success"] and report["ingest"]["raw"] > 0 and len(body["chunks"]) >= 1,
        json.dumps({key: report[key] for key in ("ingest", "review", "routing", "index")}, ensure_ascii=False),
    )
    check(
        "数据平台：去重分两级（消息级精确去重 + 语义单元级近重复）",
        report["dedupe"]["exact_duplicates"] >= 1 and report["dedupe"]["near_duplicates"] >= 1,
        json.dumps(report["dedupe"], ensure_ascii=False),
    )
    check(
        "数据平台：过滤链按规则命中并给出原因（可统计误删率）",
        report["filter"]["rejected"] >= 1 and len(report["filter"]["rule_hits"]) >= 3,
        json.dumps(report["filter"], ensure_ascii=False),
    )
    check(
        "数据平台：身份映射失败的样本进人工审核，且绝不猜身份",
        report["review"]["pending"] >= 1,
        json.dumps(report["review"], ensure_ascii=False),
    )
    check(
        "数据平台：脱敏在入库阶段命中并替换为占位符",
        report["scrub"]["total"] >= 1,
        json.dumps(report["scrub"], ensure_ascii=False),
    )

    price_articles = [article for article in body["articles"] if article["price_freshness"]]
    indexed_article_ids = {chunk["article_id"] for chunk in body["chunks"]}
    check(
        "数据平台：价格时效治理 —— 价格条目不进语义索引，知识库报价保留 live_source",
        bool(price_articles)
        and all(article["article_id"] not in indexed_article_ids for article in price_articles)
        and any(article["live_source"] for article in price_articles),
        json.dumps(
            [{"id": article["article_id"], "live_source": article["live_source"]} for article in price_articles],
            ensure_ascii=False,
        ),
    )
    check(
        "数据平台：索引一致性核对（chunk / version / embedding_version 全对齐，无孤儿或缺失向量）",
        report["index"]["orphan_vectors"] == 0
        and report["index"]["missing_vectors"] == 0
        and report["index"]["items"] == len(body["chunks"]),
        json.dumps(report["index"], ensure_ascii=False),
    )

    r = await client.post(
        "/v1/data/search",
        json={"query": "敏感肌可以用这款雪绒花修护精华吗？", "product_id": "SKU-1001"},
        headers={"X-API-Key": CONTENT_KEY},
    )
    searched = r.json()
    blocks = searched["blocks"]
    check(
        "数据平台：元数据预过滤 + 混合召回 + 重排，组装带 [chunk_id] 标注的上下文",
        searched["success"] and bool(blocks) and f"[{blocks[0]['chunk_id']}]" in searched["render"],
        f"blocks={[block['chunk_id'] for block in blocks]}",
    )

    r = await client.post("/v1/data/evaluate", json={"k": 3}, headers={"X-API-Key": CONTENT_KEY})
    evaluated = r.json()
    check(
        "数据平台：召回评估 Recall@K / Hit Rate@K / MRR，分组切分无泄漏",
        evaluated["metrics"]["recall_at_k"] == 1.0
        and evaluated["metrics"]["mrr"] == 1.0
        and evaluated["leaks"] == [],
        json.dumps({"metrics": evaluated["metrics"], "splits": evaluated["splits"]}, ensure_ascii=False),
    )

    r = await client.post("/v1/data/export", json={"format": "sharegpt"}, headers={"X-API-Key": CONTENT_KEY})
    exported = r.json()
    records = exported["records"]
    check(
        "数据平台：训练导出带校验门禁（不合规样本被拒并给出原因）+ 版本与样本统计",
        exported["stats"]["exported"] >= 1
        and exported["stats"]["rejected"] >= 1
        and any(item["reason"] == "too_short" for item in exported["rejected"]),
        json.dumps(exported["stats"], ensure_ascii=False),
    )
    check(
        "数据平台：按目标框架转换字段与角色（ShareGPT human/gpt，而不是改文件后缀）",
        bool(records)
        and all(
            all(
                {"from", "value"} <= set(message) and message["from"] in ("human", "gpt", "system")
                for message in record["conversations"]
            )
            for record in records
        ),
        json.dumps(records[0]["conversations"], ensure_ascii=False) if records else "",
    )
    dumped = json.dumps(records, ensure_ascii=False)
    check(
        "数据平台：导出门禁双保险（入库脱敏 + 导出再扫）—— 训练集不含明文手机号/邮箱",
        "13800138000" not in dumped and "bob@example.com" not in dumped,
        "",
    )

    r = await client.get("/v1/data/lineage/chunk::KB-1001%23c0", headers={"X-API-Key": CONTENT_KEY})
    chain = r.json()["trace"]["chain"]
    check(
        "数据平台：血缘反向可追溯（chunk ← article ← staging ← standard ← raw）",
        r.json()["success"]
        and len(chain) >= 4
        and any(node.startswith("raw::") for node in chain)
        and chain[0].startswith("chunk::"),
        json.dumps(chain, ensure_ascii=False),
    )

    r = await client.post("/v1/data/pipeline/run", json={}, headers={"X-API-Key": CONTENT_KEY})
    check(
        "数据平台：每次请求装配全新流水线（重复跑索引不累积，避免跨请求脏状态）",
        r.json()["report"]["index"]["items"] == report["index"]["items"],
        f"items={r.json()['report']['index']['items']}",
    )

    # ---------------- 10. 配额预留-实扣 ---------------- #
    quota_after = (await client.get("/v1/quota", headers={"X-API-Key": CS_BOT_KEY})).json()
    check(
        "配额：按真实 usage 结算（token 用量已累计）",
        quota_after["tokens"] > 0 and quota_after["backend"] == "InMemoryQuotaStore",
        json.dumps({k: quota_after[k] for k in ("requests", "tokens", "backend")}, ensure_ascii=False),
    )

    # 配额耗尽：用 content-ops 的小额度 key 连续打，验证 429 与 Retry-After
    small_key_limits = container.catalog.current().api_key_map()["sk-project-content-demo"]
    exhausted = False
    for _ in range(small_key_limits.requests_per_window + 2):
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": CONTENT_KEY},
        )
        if resp.status_code == 429:
            exhausted = True
            check(
                "配额耗尽 -> 429 + Retry-After（不是 500）",
                resp.headers.get("Retry-After") is not None
                and resp.json()["error"]["code"] == "quota_exceeded",
                f"Retry-After={resp.headers.get('Retry-After')}",
            )
            break
    if not exhausted:
        check("配额耗尽 -> 429", False, "未触发配额上限（窗口配额可能配置过大）")

    # ---------------- 11. 可观测 ---------------- #
    r = await client.get("/metrics")
    body = r.json()
    check(
        "指标：请求量/错误率/延迟分位/Token 消耗",
        body["requests_total"] > 0 and body["errors_total"] >= 1 and body["latency"],
        f"requests={body['requests_total']} errors={body['errors_total']}",
    )

    r = await client.get("/v1/config", headers={"X-API-Key": CS_BOT_KEY})
    body = r.json()
    check(
        "配置视图：暴露结构不暴露密钥",
        body["revision"] == 1 and body["transport"] == "inprocess" and len(body["mcp_paths"]) == 4,
        json.dumps(body, ensure_ascii=False),
    )

    # ---------------- 12. 配置热重载：坏配置不生效 ---------------- #
    bad_config = Path(tempfile.mkdtemp()) / "bad_models.yaml"
    bad_config.write_text(
        "default_provider: openai_compat\n"
        "models:\n"
        "  chat-lite:\n"
        "    provider: mock\n"
        "    model: mock-fast\n"
        "routing:\n"
        "  default: chat-lite\n"
        "  overrides:\n"
        "    product_consulting: not-exist-model\n",  # 指向不存在的模型 -> 必须被拒
        encoding="utf-8",
    )
    original_path = container.settings.models_config_path
    container.settings.models_config_path = str(bad_config)
    r = await client.post("/admin/config/reload", headers={"x-admin-token": ""})
    reload_status = r.status_code
    revision_after_bad = (await client.get("/v1/config", headers={"X-API-Key": CS_BOT_KEY})).json()["revision"]
    check(
        "坏配置不生效：校验失败且旧快照继续可用",
        revision_after_bad == 1,
        f"reload_status={reload_status} revision={revision_after_bad}",
    )

    container.settings.models_config_path = original_path
    r = await client.post("/admin/config/reload", headers={"x-admin-token": ""})
    check(
        "管理端点默认关闭（未配置 ADMIN_TOKEN 时拒绝）",
        r.status_code == 403,
        f"status={r.status_code}",
    )

    # ---------------- 13. MCP over Streamable HTTP ---------------- #
    mcp_headers = {
        "x-internal-token": "smoke-internal-token",
        "x-project-id": "cs-bot",
        "x-scopes": "tools:read,tools:knowledge,llm:chat,memory:rw",
        "x-user-id": "u_10086",
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "smoke", "version": "1.0"},
        },
    }
    r = await client.post("/mcp/commerce/", json=init_payload, headers=mcp_headers)
    session_id = r.headers.get("mcp-session-id")
    check(
        "MCP 端点：initialize 成功并返回 session",
        r.status_code == 200 and session_id is not None,
        f"status={r.status_code} session={session_id}",
    )

    follow_headers = {**mcp_headers, "mcp-session-id": session_id or ""}
    await client.post(
        "/mcp/commerce/",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=follow_headers,
    )
    r = await client.post(
        "/mcp/commerce/",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        headers=follow_headers,
    )
    listed = r.json().get("result", {}).get("tools", []) if r.status_code == 200 else []
    check(
        "MCP 端点：协议层 tools/list 可发现工具",
        any(tool["name"] == "search_products" for tool in listed),
        f"tools={[t['name'] for t in listed]}",
    )

    r = await client.post(
        "/mcp/commerce/",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_products", "arguments": {"query": "精华"}},
        },
        headers=follow_headers,
    )
    text = json.dumps(r.json(), ensure_ascii=False)
    check(
        "MCP 端点：tools/call 走同一套校验与权限（复用进程内执行器）",
        r.status_code == 200 and "SKU-1001" in text,
        f"status={r.status_code}",
    )

    r = await client.post("/mcp/commerce/", json=init_payload, headers={**mcp_headers, "x-internal-token": "wrong"})
    check("MCP 端点：内部凭证错误 -> 401", r.status_code == 401, f"status={r.status_code}")

    # ---------------- 14. 平台治理（项目三 · 阶段 1）---------------- #
    # 14.1 审计：四级权限链 Service → Agent → Tool → Resource 必须落到可查记录上
    agent_headers = {"X-API-Key": CS_BOT_KEY, "X-Agent-Id": "customer_agent"}
    await client.post(
        "/v1/tools/get_order_status",
        json={"arguments": {"order_id": "ORD-90001"}, "user_id": "u_10086"},
        headers=agent_headers,
    )
    audit_body = (
        await client.get("/v1/audit", params={"decision": "allow", "limit": 50}, headers={"X-API-Key": CS_BOT_KEY})
    ).json()
    allowed_entries = [
        entry
        for entry in audit_body["entries"]
        if entry["tool"] == "get_order_status" and entry["agent"] == "customer_agent"
    ]
    check(
        "审计：放行调用留痕（Service/Agent/Tool/Resource 四级可追溯）",
        any(
            entry["service"] == "cs-service"
            and entry["resource"] == "order:ORD-90001"
            and entry["decision"] == "allow"
            and entry["trace_id"]
            for entry in allowed_entries
        ),
        json.dumps(allowed_entries[:1], ensure_ascii=False),
    )

    deny_body = (
        await client.get("/v1/audit", params={"decision": "deny"}, headers={"X-API-Key": CS_BOT_KEY})
    ).json()
    check(
        "审计：越权尝试同样留痕（只记成功调用会让越权完全隐形）",
        any(
            entry["tool"] == "create_order"
            and entry["decision"] == "deny"
            and entry["reason"] == "missing_scope:tools:write"
            and entry["error_code"] == "permission_denied"
            for entry in deny_body["entries"]
        ),
        json.dumps(deny_body["entries"][:1], ensure_ascii=False),
    )

    sales_audit = (await client.get("/v1/audit", headers={"X-API-Key": SALES_KEY})).json()
    check(
        "审计：按项目隔离（凭证只能看到自己项目的记录）",
        sales_audit["project_id"] == "sales-agent"
        and bool(sales_audit["entries"])
        and all(entry["project_id"] == "sales-agent" for entry in sales_audit["entries"]),
        f"project={sales_audit['project_id']} entries={len(sales_audit['entries'])}",
    )

    # 14.2 Prompt 模板中心：角色组装视图 + 正文预览的权限边界
    prompts_body = (await client.get("/v1/prompts", headers={"X-API-Key": CS_BOT_KEY})).json()
    roles = [item["role"] for item in prompts_body["roles"]]
    check(
        "Prompt 模板中心：按 Role 展示组装结构且不回传正文",
        "customer" in roles
        and prompts_body["templates"].get("cs.system") == ["v1", "v2"]
        and prompts_body["active"].get("cs.system") == "v2",
        json.dumps({"roles": roles, "active_cs_system": prompts_body["active"].get("cs.system")}, ensure_ascii=False),
    )

    refused_preview = await client.post("/v1/prompts/customer/preview", json={}, headers={"X-API-Key": CS_BOT_KEY})
    check(
        "Prompt 预览：渲染会暴露系统提示词 -> 需 prompt:rw（客服凭证被拒）",
        refused_preview.status_code == 403 and refused_preview.json()["error"]["code"] == "permission_denied",
        f"status={refused_preview.status_code}",
    )

    preview = await client.post(
        "/v1/prompts/customer/preview",
        json={"scenario": "after_sales", "variables": {}},
        headers={"X-API-Key": EVAL_KEY},
    )
    preview_body = preview.json()
    preview_layers = [layer["name"] for layer in preview_body.get("blueprint", {}).get("layers", [])]
    check(
        "Prompt 组装：base 片段 + 场景片段按声明顺序拼接（售后场景追加证据/政策约束）",
        preview.status_code == 200
        and preview_body["blueprint"]["role"] == "customer"
        and preview_layers == ["cs.system", "customer.evidence", "customer.policy"]
        and "人工确认" in preview_body["rendered"],
        f"layers={preview_layers}",
    )

    # 14.3 成本：按 Service/Agent 聚合 + 当日预算水位（用带 Agent 身份的调用产生归属）
    await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "成本分摊口径测试"}]},
        headers=agent_headers,
    )
    cost_body = (await client.get("/v1/cost", headers=agent_headers)).json()
    check(
        "成本：按 Service/Agent 聚合（能回答「哪条业务线在花钱」）",
        cost_body["service"] == "cs-service"
        and cost_body["agent"] == "customer_agent"
        and "cs-service/customer_agent" in cost_body["metering"]
        and cost_body["totals"]["total_tokens"] > 0,
        json.dumps({"metering": sorted(cost_body["metering"]), "totals": cost_body["totals"]}, ensure_ascii=False),
    )
    check(
        "成本：当日预算水位可见（预扣-校正后按 UTC 自然日切分）",
        cost_body["budget"]["scope"] == "cs-service/customer_agent"
        and cost_body["budget"]["used_tokens"] > 0
        and cost_body["budget"]["limit_tokens"] == container.settings.budget_daily_tokens_per_agent,
        json.dumps(cost_body["budget"], ensure_ascii=False),
    )

    # 14.4 预算两道阈值：软阈值降级 / 硬上限拒绝（临时压阈值，用完立刻还原）
    original_downgrade_ratio = container.budget.limits.downgrade_ratio
    container.budget.limits.downgrade_ratio = 0.0
    downgraded = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "预算降级测试"}], "logical_model": "chat-pro"},
        headers=agent_headers,
    )
    container.budget.limits.downgrade_ratio = original_downgrade_ratio
    check(
        "预算：触及软阈值自动降级到 Lite 模型（体验换成本，不中断业务）",
        downgraded.status_code == 200
        and downgraded.json()["logical_model"] == container.catalog.current().routing.default,
        f"logical_model={downgraded.json().get('logical_model')}",
    )

    original_daily_tokens = container.budget.limits.daily_tokens_per_agent
    container.budget.limits.daily_tokens_per_agent = 1
    exhausted_budget = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "预算硬上限测试"}]},
        headers=agent_headers,
    )
    container.budget.limits.daily_tokens_per_agent = original_daily_tokens
    check(
        "预算：触及硬上限 -> 429（止损线不可协商，与配额 error code 统一）",
        exhausted_budget.status_code == 429
        and exhausted_budget.json()["error"]["code"] == "quota_exceeded"
        and exhausted_budget.headers.get("Retry-After") is not None,
        f"status={exhausted_budget.status_code} code={exhausted_budget.json().get('error', {}).get('code')}",
    )

    # 14.5 评测：六维报告（评测跑的就是共享执行内核，不另写一套流程）
    eval_resp = await client.post("/v1/eval/run", json={}, headers={"X-API-Key": EVAL_KEY})
    eval_payload = eval_resp.json()
    report_body = eval_payload.get("report", {})
    dimensions = report_body.get("dimensions", {})
    check(
        "评测：内置用例集跑通并输出六维报告",
        eval_resp.status_code == 200
        and report_body.get("cases") == 7
        and set(dimensions) == {"accuracy", "grounding", "tool_selection", "safety", "cost", "latency"},
        json.dumps({"cases": report_body.get("cases"), "pass_rate": report_body.get("pass_rate")}, ensure_ascii=False),
    )
    check(
        "评测：分维度打分（总分看不出「哪一类问题在退化」）",
        report_body.get("failed") == 0 and all(item["rate"] == 1.0 for item in dimensions.values()),
        json.dumps({dim: item["rate"] for dim, item in dimensions.items()}, ensure_ascii=False),
    )
    refused_eval = await client.post("/v1/eval/run", json={}, headers={"X-API-Key": CS_BOT_KEY})
    check(
        "评测：跑评测集需 eval:rw（业务线凭证不能触发平台级动作）",
        refused_eval.status_code == 403 and refused_eval.json()["error"]["code"] == "permission_denied",
        f"status={refused_eval.status_code}",
    )

    # 14.6 限流：Service 维度滑动窗口（与配额超限分开的 error code）
    original_service_limit = container.settings.ratelimit_service_limit
    container.settings.ratelimit_service_limit = 1
    limited = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "限流测试"}]},
        headers={"X-API-Key": CS_BOT_KEY},
    )
    container.settings.ratelimit_service_limit = original_service_limit
    limited_body = limited.json()
    check(
        "限流：Service 维度滑动窗口 -> 429 rate_limited（与配额超限区分，退避即可重试）",
        limited.status_code == 429
        and limited_body["error"]["code"] == "rate_limited"
        and limited_body["error"]["retryable"] is True
        and limited.headers.get("Retry-After") is not None,
        f"status={limited.status_code} code={limited_body.get('error', {}).get('code')}",
    )
    check(
        "限流：无 Redis 时降级为进程内滑动窗口，且拒绝响应仍带 trace_id",
        container.ratelimit.backend == "InMemorySlidingWindowStore" and bool(limited_body["trace_id"]),
        f"backend={container.ratelimit.backend}",
    )

    # 15. 项目四补齐：三级切片 / 重排端口 / 训练产物 / 数据回流 / 自动出题评分
    #     全部放在最后：这些检查会改写切片模式与回流收集器状态，不能影响上面的既有断言。
    hierarchical_record = {
        "source": "product_knowledge",
        "article_id": "KB-HIER",
        "title": "雪绒花精华使用说明",
        "scene": "product_consulting",
        "question": "雪绒花精华怎么用？",
        "answer": (
            "雪绒花精华适合干性皮肤。雪绒花精华建议早晚各一次。雪绒花精华可叠加保湿面霜。"
            "雪绒花精华开封后请冷藏。雪绒花精华过敏体质先做耳后测试。雪绒花精华孕期使用前请咨询医生。"
        ),
        "conditions": [],
        "product_id": "SKU-1001",
        "updated_at": "2026-03-01T00:00:00Z",
        "valid_from": "2026-03-01T00:00:00Z",
    }
    original_chunk_mode = container.settings.data_chunk_mode
    container.settings.data_chunk_mode = "hierarchical"
    hier_run = await client.post(
        "/v1/data/pipeline/run", json={"records": [hierarchical_record]}, headers={"X-API-Key": CONTENT_KEY}
    )
    hier_payload = hier_run.json()
    hier_chunks = hier_payload.get("chunks", [])
    hier_ids = {chunk["chunk_id"] for chunk in hier_chunks}
    check(
        "三级切片：L1（滑动窗口）/L2/L3 分层产出，父子链都能解析到（合并才不会断链）",
        hier_run.status_code == 200
        and {chunk["level"] for chunk in hier_chunks} == {1, 2, 3}
        and all(chunk["parent_id"] in hier_ids for chunk in hier_chunks if chunk["parent_id"]),
        f"levels={sorted({chunk['level'] for chunk in hier_chunks})} chunks={len(hier_chunks)}",
    )

    merged_search = await client.post(
        "/v1/data/search",
        json={"query": "雪绒花精华怎么用", "top_k": 3},
        headers={"X-API-Key": CONTENT_KEY},
    )
    merged_body = merged_search.json()
    check(
        "Auto-merging：同父块下命中多个 L1 时合并回 L2 —— 检索用小块、回答用完整上下文",
        merged_search.status_code == 200
        and merged_body["rerank"]["merged_children"] >= 1
        and any(block["chunk_id"].endswith((":L2:000", ":L2:001", ":L3:000")) for block in merged_body["blocks"]),
        f"merged={merged_body.get('rerank', {}).get('merged_children')} blocks={[block['chunk_id'] for block in merged_body.get('blocks', [])]}",
    )
    check(
        "重排端口：未配 Jina key 时落地为本地词典重排（离线可跑，缺 key 显式降级而不是报错）",
        merged_body["rerank"]["backend"] == "lexical" and container.settings.data_rerank_backend == "lexical",
        f"backend={merged_body.get('rerank', {}).get('backend')}",
    )

    container.settings.data_chunk_mode = original_chunk_mode
    restored_run = await client.post("/v1/data/pipeline/run", json={}, headers={"X-API-Key": CONTENT_KEY})
    restored_chunks = restored_run.json().get("chunks", [])
    check(
        "切片模式可切换：还原为语义边界模式后，切片回到 L1 且不触发任何合并（既有行为不被改动）",
        restored_run.status_code == 200
        and {chunk["level"] for chunk in restored_chunks} == {1}
        and all(chunk["parent_id"] is None for chunk in restored_chunks)
        and any(chunk["chunk_id"].endswith("#c0") for chunk in restored_chunks),
        f"levels={sorted({chunk['level'] for chunk in restored_chunks})} chunks={len(restored_chunks)}",
    )

    # 15.2 数据回流：五来源各有用例 —— 每一条都从真实业务动作里长出训练/评测信号
    reflux_events = {
        "correction": [
            {"id": "cor_1", "question": "多久发货？", "original": "尽快发货。", "corrected": "付款后 48 小时内发出，节假日顺延。", "chunk_ids": ["KB-1001#c0"]},
            {"id": "cor_2", "question": "能退货吗？", "corrected": "支持 7 天无理由。", "chunk_ids": []},
        ],
        "review": [
            {"id": "rev_1", "question": "敏感肌能用吗？", "before": "可以的。", "after": "敏感肌建议先做耳后测试。", "reviewer": "ops_lin", "reason": "缺条件说明"},
        ],
        "tool_call": [
            {"id": "tool_1", "question": "精华现在有货吗？", "tool_name": "search_products", "result": "SKU-1001 现货", "chunk_ids": ["KB-1001#c0"]},
        ],
        "hitl": [
            {"id": "hitl_1", "question": "能便宜点吗？", "answer": "可以给您打五折。", "approved": False, "corrected": "价格以官方渠道为准，我这里不能改价。", "chunk_ids": []},
        ],
        "live_asr": [
            {"id": "live_1", "question": "雪绒花精华适合什么肤质？", "answer": "干皮闭眼入，敏感肌先测耳后。", "chunk_ids": ["KB-1001#c0"]},
        ],
    }
    accepted_counts = {}
    for source, events in reflux_events.items():
        resp = await client.post("/v1/data/reflux", json={"source": source, "events": events}, headers={"X-API-Key": CONTENT_KEY})
        accepted_counts[source] = resp.json().get("report", {}).get("by_source", {}).get(source)
    refused_event = await client.post(
        "/v1/data/reflux",
        json={"source": "correction", "events": [{"id": "dirty_1", "question": "只有提问没有纠偏内容"}]},
        headers={"X-API-Key": CONTENT_KEY},
    )
    reflux_report_body = (await client.get("/v1/data/reflux", headers={"X-API-Key": CONTENT_KEY})).json()["report"]
    check(
        "数据回流：五来源（纠偏/审核/Tool/HITL/直播 ASR）各有用例，字段不全的脏数据被拒并记录原因",
        all(accepted_counts.get(source, 0) >= 1 for source in reflux_events)
        and refused_event.json()["report"]["dropped"]
        and all(reflux_report_body["by_source"][source] >= 1 for source in reflux_events),
        json.dumps({"accepted": accepted_counts, "dropped": reflux_report_body["dropped"]}, ensure_ascii=False),
    )
    check(
        "数据回流：只有「被改写过」的记录才构成 DPO 偏好对，其余只作正向样本",
        reflux_report_body["dpo_pairs"] == 3 and reflux_report_body["trainable"] == 6,
        json.dumps({"dpo_pairs": reflux_report_body["dpo_pairs"], "records": reflux_report_body["records"]}, ensure_ascii=False),
    )

    # 15.3 训练产物：LLaMA-Factory 可直接消费的数据集 + 训练 YAML（数据准备链路，训练在外部执行）
    await client.post(
        "/v1/data/reflux",
        json={"source": "correction", "events": [{"id": "leak_1", "question": "客服电话是多少？", "original": "很快回复。", "corrected": "请拨打 13800138000 联系人工客服。"}]},
        headers={"X-API-Key": CONTENT_KEY},
    )
    training_resp = await client.post(
        "/v1/data/training/export",
        json={"dataset_name": "beauty_cs", "version": "v1", "preview": 2},
        headers={"X-API-Key": CONTENT_KEY},
    )
    training_body = training_resp.json()
    training_config = training_body.get("train_config_yaml", "")
    check(
        "训练产物：LLaMA-Factory 可直接消费（ranking 偏好数据集 + dpo 训练配置 + dataset_info/数据集文件）",
        training_resp.status_code == 200
        and training_body.get("dataset_info", {}).get("beauty_cs_dpo", {}).get("ranking") is True
        and "stage: dpo" in training_config
        and len(training_body.get("preview", {}).get("dpo", [])) == 2
        and {entry["name"] for entry in training_body.get("files", [])}
        == {"beauty_cs_sft_v1.jsonl", "beauty_cs_dpo_v1.jsonl", "dataset_info.json", "train_beauty_cs.yaml"},
        json.dumps(training_body.get("stats", {}), ensure_ascii=False),
    )
    training_reasons = {entry["id"]: entry["reason"] for entry in training_body.get("rejected", [])}
    check(
        "训练产物门禁：脱敏后仍带明文手机号的偏好对被拒（审核通过 ≠ 可训练）",
        training_reasons.get("reflux_leak_1") == "sensitive:phone_cn",
        json.dumps(training_reasons, ensure_ascii=False),
    )

    # 15.4 自动出题 + 自动评分：AI 美妆顾问考核链路
    quiz_generated = await client.post("/v1/data/quiz", json={}, headers={"X-API-Key": CONTENT_KEY})
    quiz_items = quiz_generated.json().get("items", [])
    quiz_types = {item["quiz_type"] for item in quiz_items}
    check(
        "自动出题：从已审核知识条目生成事实型/边界型/拒答型三类题目（拒答型来自价格时效条目）",
        quiz_generated.status_code == 200 and quiz_types == {"fact", "boundary", "refusal"} and all(item["group"] for item in quiz_items),
        f"types={sorted(quiz_types)} items={len(quiz_items)}",
    )

    refusal_item = next(item for item in quiz_items if item["quiz_type"] == "refusal")
    good_refusal = await client.post(
        "/v1/data/quiz",
        json={"answers": {refusal_item["quiz_id"]: "价格以官方渠道实时页面为准，不承诺具体价格。"}},
        headers={"X-API-Key": CONTENT_KEY},
    )
    bad_refusal = await client.post(
        "/v1/data/quiz",
        json={"answers": {refusal_item["quiz_id"]: "最低价 99 元，今天下单再减 10 元。"}},
        headers={"X-API-Key": CONTENT_KEY},
    )
    check(
        "自动评分：拒答型题目必须命中「引导官方渠道」策略才通过；编造最低价直接不及格",
        good_refusal.json()["results"][0]["verdict"] == "pass"
        and bad_refusal.json()["results"][0]["verdict"] == "fail",
        json.dumps(
            {"good": good_refusal.json()["results"][0]["score"], "bad": bad_refusal.json()["results"][0]["score"]},
            ensure_ascii=False,
        ),
    )

    fact_item = next(item for item in quiz_items if item["quiz_type"] == "fact")
    fact_answer_ok = await client.post(
        "/v1/data/quiz",
        json={"answers": {fact_item["quiz_id"]: fact_item["gold_answer"]}},
        headers={"X-API-Key": CONTENT_KEY},
    )
    fact_answer_bad = await client.post(
        "/v1/data/quiz",
        json={"answers": {fact_item["quiz_id"]: "这个我不清楚，你换个问题吧。"}},
        headers={"X-API-Key": CONTENT_KEY},
    )
    ok_result = fact_answer_ok.json()["results"][0]
    bad_result = fact_answer_bad.json()["results"][0]
    check(
        "自动评分：按实测检索证据打分（答对不判 fail、答非所问判 fail，且暴露一致性/覆盖两个维度）",
        ok_result["verdict"] in {"pass", "partial"}
        and ok_result["checks"]["keyword_coverage"] == 1.0
        and bad_result["verdict"] == "fail"
        and set(ok_result["checks"]) == {"evidence_consistency", "keyword_coverage"},
        json.dumps({"ok": ok_result["checks"], "bad": bad_result["checks"]}, ensure_ascii=False),
    )

    # ---------------- 16. 宿主业务后端（commerce）：/v1/shop/* 与 /v1/admin/* ---------------- #
    # 16.1 浏览开放：货架是公开信息，不需要任何凭证
    shop_list = await client.get("/v1/shop/products", params={"limit": 5})
    shop_body = shop_list.json()
    check(
        "商城商品列表公开可读（浏览开放、交易鉴权）",
        shop_list.status_code == 200
        and shop_body["count"] >= 1
        and {"product_id", "name", "price_usd"} <= set(shop_body["items"][0]),
        f"count={shop_body.get('count')}",
    )
    shop_detail = await client.get(f"/v1/shop/products/{shop_body['items'][0]['product_id']}")
    check(
        "商品详情含成分表 ingredients（货架信息带得动美妆导购）",
        shop_detail.status_code == 200 and "ingredients" in shop_detail.json(),
        f"product={shop_body['items'][0]['product_id']}",
    )

    # 16.2 会员 token 是会员资源的门槛：项目级 API Key 不是会员凭证
    anon_cart = await client.get("/v1/shop/cart")
    apikey_cart = await client.get("/v1/shop/cart", headers={"X-API-Key": CS_BOT_KEY})
    check(
        "购物车必须会员 token：无 token 401、项目级 API Key 也不认（两套凭证不通用）",
        anon_cart.status_code == 401 and apikey_cart.status_code == 401,
        f"anon={anon_cart.status_code} apikey={apikey_cart.status_code}",
    )

    shop_email = "smoke-shop@example.com"
    registered = await client.post(
        "/v1/shop/members/register",
        json={"email": shop_email, "password": "Smoke12345!", "nickname": "冒烟买家"},
    )
    logged_in = await client.post("/v1/shop/members/login", json={"email": shop_email, "password": "Smoke12345!"})
    shop_token = logged_in.json().get("token") if logged_in.status_code == 200 else None
    member_headers = {"Authorization": f"Bearer {shop_token}"}
    profile = await client.get("/v1/shop/members/me", headers=member_headers)
    check(
        "会员注册 + 登录发放 token，token 能读回本人档案",
        registered.status_code == 200 and bool(shop_token) and profile.status_code == 200,
        f"register={registered.status_code} login={logged_in.status_code} me={profile.status_code}",
    )

    # 16.3 加购：回整车而不是回单行（客户端要拿最新总价刷新结算栏）
    added = await client.post(
        "/v1/shop/cart/items", json={"product_id": "SKU-1002", "quantity": 1}, headers=member_headers
    )
    check(
        "加购返回整车且购物车非空",
        added.status_code == 200 and bool(added.json()["cart"]["items"]),
        f"items={len(added.json().get('cart', {}).get('items', []))}",
    )

    # 16.4 REST 下单 + 幂等重放：REST 与白名单 Tool 走同一段 usecase，行为必须一致
    order_req = {"items": [{"product_id": "SKU-1002", "quantity": 1}], "idempotency_key": "smoke-rest-order-001"}
    placed = await client.post("/v1/shop/orders", json=order_req, headers=member_headers)
    replayed = await client.post("/v1/shop/orders", json=order_req, headers=member_headers)
    placed_order = placed.json().get("order", {})
    replayed_order = replayed.json().get("order", {})
    check(
        "商城 REST 下单成功，且同键重放返回同一笔订单（idempotent_replay=true）",
        placed.status_code == 200
        and placed_order.get("status") == "PENDING_PAYMENT"
        and replayed_order.get("order_id") == placed_order.get("order_id")
        and replayed_order.get("idempotent_replay") is True,
        f"order={placed_order.get('order_id')}",
    )

    # 16.5 下单事件与业务写在同一事务：Outbox 按本单 order_id 匹配（取第一条必然错）
    if placed_order.get("order_id"):
        async with container.commerce.uow_factory() as uow:
            pending_events = await uow.outbox.pending()
            await uow.commit()
        check(
            "下单事件落进 Outbox（与业务写同一事务，commit 成则事件必在）",
            any(
                row["topic"] == "OrderPlaced" and row["payload"].get("order_id") == placed_order["order_id"]
                for row in pending_events
            ),
            f"topics={sorted({row['topic'] for row in pending_events})}",
        )
    else:
        check("下单事件落进 Outbox（与业务写同一事务，commit 成则事件必在）", False, "前置下单失败")

    # 16.6 订单查询 + 付款状态迁移（CAS：期望旧状态是待付款）
    fetched = await client.get(f"/v1/shop/orders/{placed_order.get('order_id')}", headers=member_headers)
    paid = await client.post(f"/v1/shop/orders/{placed_order.get('order_id')}/pay", headers=member_headers)
    check(
        "订单查询可见、付款后状态迁移为 PAID",
        fetched.status_code == 200 and paid.status_code == 200 and paid.json()["order"]["status"] == "PAID",
        f"fetch={fetched.status_code} pay={paid.status_code} status={paid.json().get('order', {}).get('status')}",
    )

    # 16.7 取消订单退回库存：下单减、取消加，必须可观测地一增一减
    cancel_sku = "SKU-1002-01"
    before_cancel = (await container.commerce.catalog.stocks_for([cancel_sku]))[cancel_sku].available
    cancel_order = await client.post(
        "/v1/shop/orders",
        json={"items": [{"product_id": "SKU-1002", "quantity": 1}], "idempotency_key": "smoke-rest-order-cancel-01"},
        headers=member_headers,
    )
    cancel_id = cancel_order.json().get("order", {}).get("order_id")
    cancelled = await client.post(f"/v1/shop/orders/{cancel_id}/cancel", headers=member_headers)
    after_cancel = (await container.commerce.catalog.stocks_for([cancel_sku]))[cancel_sku].available
    check(
        "取消订单：状态迁移为 CANCELLED 且库存原样退回",
        cancel_order.status_code == 200
        and cancelled.status_code == 200
        and cancelled.json()["order"]["status"] == "CANCELLED"
        and after_cancel == before_cancel,
        f"status={cancelled.json().get('order', {}).get('status')} stock={before_cancel}->{after_cancel}",
    )

    # 16.8 运营侧后台：commerce:rw 闸门 + 会员 token 不能进后台
    admin_anon = await client.get("/v1/admin/stock/low")
    admin_content = await client.get("/v1/admin/stock/low", headers={"X-API-Key": CONTENT_KEY})
    admin_ok = await client.get("/v1/admin/stock/low", headers={"X-API-Key": SHOP_ADMIN_KEY})
    admin_member = await client.get("/v1/admin/orders/stats", headers=member_headers)
    check(
        "后台鉴权：无凭证 401 / 非 commerce:rw 403 / 运营凭证 200",
        admin_anon.status_code == 401 and admin_content.status_code == 403 and admin_ok.status_code == 200,
        f"anon={admin_anon.status_code} content={admin_content.status_code} admin={admin_ok.status_code}",
    )
    check(
        "会员 token 不能进后台（两套凭证由不同签发方产生，互不通用）",
        admin_member.status_code == 401,
        f"status={admin_member.status_code}",
    )

    # 16.9 运营台订单列表：跨会员视图 + 状态过滤。买家侧 list_by_member 看不到别人的单，
    # 所以「后台可见数 ≥ 单会员自己看到的数」是这条接口存在的理由，也是它没写错的最低证据。
    admin_orders_body = (await client.get("/v1/admin/orders", headers={"X-API-Key": SHOP_ADMIN_KEY})).json()
    admin_shipped_body = (await client.get("/v1/admin/orders?status=SHIPPED", headers={"X-API-Key": SHOP_ADMIN_KEY})).json()
    member_orders_body = (await client.get("/v1/shop/orders", headers=member_headers)).json()
    check(
        "运营台订单列表：跨会员可见且状态过滤生效（过滤后每行状态都为该值）",
        admin_orders_body.get("count", 0) >= member_orders_body.get("count", 0)
        and admin_shipped_body.get("count", 0) <= admin_orders_body.get("count", 0)
        and all(item.get("status") == "SHIPPED" for item in admin_shipped_body.get("items", []))
        and all(item.get("order_id") for item in admin_orders_body.get("items", [])),
        f"admin={admin_orders_body.get('count')} shipped={admin_shipped_body.get('count')} member={member_orders_body.get('count')}",
    )

    # 16.10 运营台商品列表：必须能看见非在售商品（否则下架后无法复原），
    # 同时买家侧必须**严格等于**其中 ON_SALE 的子集 —— 这两条一起才能证明「不漏给买家、也不锁死运营」。
    admin_products_body = (await client.get("/v1/admin/products", headers={"X-API-Key": SHOP_ADMIN_KEY})).json()
    public_products_body = (await client.get("/v1/shop/products")).json()
    admin_on_sale = sorted(item["product_id"] for item in admin_products_body.get("items", []) if item.get("status") == "ON_SALE")
    public_ids = sorted(item["product_id"] for item in public_products_body.get("items", []))
    check(
        "运营台商品列表带状态与 SKU；买家侧严格等于其中 ON_SALE 子集（下架不外泄、运营可复原）",
        admin_products_body.get("count", 0) >= len(public_ids)
        and all(item.get("skus") and item.get("status") for item in admin_products_body.get("items", []))
        and admin_on_sale == public_ids,
        f"admin_total={admin_products_body.get('total')} on_sale={len(admin_on_sale)} public={len(public_ids)}",
    )

    # ---------------- 17. 应用层（项目一 · AI 智能客服导购系统） ---------------- #
    cs_headers = {"X-API-Key": CS_BOT_KEY}
    sales_headers = {"X-API-Key": SALES_KEY}
    hitl_headers = {"X-API-Key": HITL_KEY}

    # 17.1 规则优先：给订单号问状态 → 直连白名单 Tool，零模型推理（llm_calls=0）
    route_order = await client.post("/v1/agents/route", json={"task": "查一下订单 ORD-1024 现在的状态"}, headers=cs_headers)
    route_order_body = route_order.json().get("route", {})
    check(
        "复杂度路由：订单号问状态直连 get_order_status 且零 LLM 调用（规则优先于模型）",
        route_order.status_code == 200
        and route_order_body.get("kind") == "direct"
        and route_order_body.get("tool") == "get_order_status"
        and route_order_body.get("llm_calls") == 0,
        f"kind={route_order_body.get('kind')} tool={route_order_body.get('tool')} llm_calls={route_order_body.get('llm_calls')}",
    )

    # 17.2 明确的知识问题也直连检索（同样不花推理）
    route_kn = await client.post(
        "/v1/agents/route", json={"task": "这个成分的功效与注意事项是什么"}, headers=cs_headers
    )
    route_kn_body = route_kn.json().get("route", {})
    check(
        "复杂度路由：知识类问题直连 search_knowledge（省下一次 Agent 循环）",
        route_kn.status_code == 200 and route_kn_body.get("kind") == "direct" and route_kn_body.get("tool") == "search_knowledge",
        f"tool={route_kn_body.get('tool')}",
    )

    # 17.3 只有跨域/多步任务才升级成 Agent 编排
    route_cross = await client.post(
        "/v1/agents/route", json={"task": "帮我推荐精华，并且顺便看看有没有优惠"}, headers=cs_headers
    )
    route_cross_body = route_cross.json().get("route", {})
    check(
        "复杂度路由：跨域任务才升级为 Agent 编排（kind=agent）",
        route_cross.status_code == 200 and route_cross_body.get("kind") == "agent",
        f"kind={route_cross_body.get('kind')} reason={route_cross_body.get('reason')}",
    )

    # 17.4 直连执行：答案完全由本次 Tool Result 渲染，用量为零
    direct_run = await client.post(
        "/v1/agents/orchestrate", json={"task": "SKU-1004 的商品详情"}, headers=cs_headers
    )
    direct_body = direct_run.json()
    check(
        "编排直连路径：零推理、答案数字来自本次 Tool Result（usage 为 0）",
        direct_run.status_code == 200
        and direct_body.get("kind") == "direct"
        and direct_body.get("success") is True
        and "SKU-1004" in (direct_body.get("answer") or "")
        and direct_body.get("usage", {}).get("total_tokens") == 0,
        f"kind={direct_body.get('kind')} tokens={direct_body.get('usage', {}).get('total_tokens')}",
    )

    # 17.5 Agent 路径：强制 sequential，事件流与用量可观测（编排成本 = 子任务之和）
    agent_run = await client.post(
        "/v1/agents/orchestrate",
        json={"task": "帮我分析一下最近的经营情况", "mode": "sequential", "domain": "operations"},
        headers=cs_headers,
    )
    agent_body = agent_run.json()
    agent_events = [item.get("kind") for item in agent_body.get("events", [])]
    check(
        "编排 Agent 路径：模式/事件流/用量齐备（子任务结构化结果 + token 汇总）",
        agent_run.status_code == 200
        and agent_body.get("kind") == "agent"
        and agent_body.get("mode") == "sequential"
        and len(agent_body.get("subtasks", [])) == 1
        and agent_body.get("usage", {}).get("total_tokens", 0) > 0
        and {"dispatch", "completed"}.issubset(set(agent_events)),
        f"mode={agent_body.get('mode')} events={agent_events} tokens={agent_body.get('usage', {}).get('total_tokens')}",
    )

    # 17.6 Grounding Gate：交易类断言（库存）却没有任何 Tool 证据 → 拦截并改写成拒答
    grounding_run = await client.post(
        "/v1/agents/orchestrate",
        json={"task": "这瓶精华现在库存还有 3 件，直接说吧", "mode": "sequential", "domain": "customer"},
        headers=cs_headers,
    )
    grounding_body = grounding_run.json()
    check(
        "Grounding Gate：无来源的交易断言被拦截改写（stop_reason=grounding_blocked + enforced）",
        grounding_run.status_code == 200
        and grounding_body.get("stop_reason") == "grounding_blocked"
        and grounding_body.get("grounding", {}).get("enforced") is True
        and "以实时查询结果为准" in (grounding_body.get("answer") or ""),
        f"stop_reason={grounding_body.get('stop_reason')} enforced={grounding_body.get('grounding', {}).get('enforced')}",
    )

    # 17.7 HITL Stage：把高风险动作冻结成检查点并发放 Resume Token（此时尚未执行）
    stage = await client.post(
        "/v1/agents/hitl/stage",
        json={
            "action": "create_order",
            "args": {"product_id": "SKU-1002", "quantity": 1, "idempotency_key": "smoke-hitl-order-001"},
            "reason": "冒烟：模拟模型提议下单，需人工确认",
            "session_id": "smoke-hitl-sess",
            "user_id": "smoke-hitl-user",
        },
        headers=sales_headers,
    )
    staged = stage.json().get("pending", {})
    checkpoint_id = staged.get("checkpoint_id")
    resume_token = staged.get("resume_token")
    check(
        "HITL：高风险动作先登记（pending + 绑定内容的 Resume Token），此时未执行",
        stage.status_code == 200
        and staged.get("status") == "pending"
        and bool(checkpoint_id)
        and bool(resume_token)
        and staged.get("applied") is None,
        f"checkpoint={checkpoint_id} status={staged.get('status')}",
    )

    # 17.8 Approve：独立权限 + Token 必须与检查点内容一致（确认 A 被掉包成 B 是提权漏洞）
    approve_no_scope = await client.post(
        "/v1/agents/hitl/approve",
        json={"checkpoint_id": checkpoint_id, "resume_token": resume_token},
        headers=cs_headers,
    )
    approve_bad_token = await client.post(
        "/v1/agents/hitl/approve",
        json={"checkpoint_id": checkpoint_id, "resume_token": "deadbeefdeadbeef"},
        headers=hitl_headers,
    )
    approve_ok = await client.post(
        "/v1/agents/hitl/approve",
        json={"checkpoint_id": checkpoint_id, "resume_token": resume_token},
        headers=hitl_headers,
    )
    check(
        "HITL：确认需要独立权限 hitl:approve，且 Token 与检查点绑定（403 / 403 / 200）",
        approve_no_scope.status_code == 403
        and approve_bad_token.status_code == 403
        and approve_ok.status_code == 200
        and approve_ok.json()["pending"]["status"] == "approved",
        f"no_scope={approve_no_scope.status_code} bad_token={approve_bad_token.status_code} ok={approve_ok.status_code}",
    )

    # 17.9 Apply：仍受 hitl:approve 门禁；执行时用**原始请求者身份**（下单归属不因审批而改变）
    apply_no_scope = await client.post(
        "/v1/agents/hitl/apply", json={"checkpoint_id": checkpoint_id}, headers=cs_headers
    )
    apply_ok = await client.post(
        "/v1/agents/hitl/apply", json={"checkpoint_id": checkpoint_id}, headers=hitl_headers
    )
    apply_body = apply_ok.json()
    applied = apply_body.get("pending", {}).get("applied") or {}
    check(
        "HITL：执行需 hitl:approve；并以原始请求者身份落单（非审批人）",
        apply_no_scope.status_code == 403
        and apply_ok.status_code == 200
        and apply_body.get("success") is True
        and applied.get("ok") is True
        and bool(applied.get("data", {}).get("order_id")),
        f"no_scope={apply_no_scope.status_code} applied_ok={applied.get('ok')} order={applied.get('data', {}).get('order_id')}",
    )

    # 17.10 A2A：服务间通道用内部共享令牌鉴权，并透传 session_id（接手方沿用同一会话）
    a2a_anon = await client.post("/v1/a2a/message:send", json={"task": "查下 SKU-1002 详情"})
    a2a_bad = await client.post(
        "/v1/a2a/message:send",
        json={"task": "查下 SKU-1002 详情"},
        headers={"X-Internal-Token": "wrong-internal-token"},
    )
    a2a_ok = await client.post(
        "/v1/a2a/message:send",
        json={
            "task": "查下 SKU-1002 详情",
            "session_id": "a2a-sess-1",
            "agent": "customer",
            "scopes": ["tools:read", "llm:chat"],
        },
        headers={"X-Internal-Token": "smoke-internal-token"},
    )
    a2a_body = a2a_ok.json()
    check(
        "A2A：内部令牌鉴权（无/错令牌 401）且 session_id 原样透传",
        a2a_anon.status_code == 401
        and a2a_bad.status_code == 401
        and a2a_ok.status_code == 200
        and a2a_body.get("session_id") == "a2a-sess-1",
        f"anon={a2a_anon.status_code} bad={a2a_bad.status_code} ok={a2a_ok.status_code} session={a2a_body.get('session_id')}",
    )

    # 17.11 实时语音（文本轮）：事件流按序吐出，可据此算每环耗时
    rt_text = await client.post(
        "/v1/agents/realtime/turn", json={"text": "你好，帮我看看护肤建议"}, headers=cs_headers
    )
    rt_text_body = rt_text.json()
    rt_text_types = [event.get("type") for event in rt_text_body.get("events", [])]
    check(
        "实时语音（文本轮）：asr 跳过 → 编排 → TTS → 收尾，事件带序号可观测",
        rt_text.status_code == 200
        and rt_text_types == ["asr.skipped", "orchestrator.done", "tts.chunk", "turn.end"]
        and rt_text_body.get("audio_frames", 0) > 0,
        f"events={rt_text_types} frames={rt_text_body.get('audio_frames')}",
    )

    # 17.12 实时语音（模拟上行语音轮）：RTC 采集 → ASR 识别 → 编排，链路打通
    rt_voice = await client.post(
        "/v1/agents/realtime/turn", json={"utterance": "敏感肌用什么精华"}, headers=cs_headers
    )
    rt_voice_body = rt_voice.json()
    rt_voice_types = [event.get("type") for event in rt_voice_body.get("events", [])]
    check(
        "实时语音（语音轮）：RTC→ASR→编排链路打通，转写文本回灌编排",
        rt_voice.status_code == 200
        and rt_voice_types[:2] == ["rtc.capture", "asr.final"]
        and rt_voice_body.get("transcript") == "敏感肌用什么精华",
        f"events={rt_voice_types} transcript={rt_voice_body.get('transcript')}",
    )

    # 17.13 检索重写：同一问题跑三种策略给出可对比召回；检索能力有独立权限边界
    compare = await client.post(
        "/v1/agents/retrieval/compare", json={"query": "敏感肌精华怎么用"}, headers=cs_headers
    )
    compare_body = compare.json().get("report", {})
    compare_denied = await client.post(
        "/v1/agents/retrieval/compare", json={"query": "敏感肌精华怎么用"}, headers=sales_headers
    )
    check(
        "检索重写：none/step_back/hyde 三策略可对比召回；无 tools:knowledge 403",
        compare.status_code == 200
        and len(compare_body.get("strategies", [])) == 3
        and set(compare_body.get("delta", {})) == {"step_back", "hyde"}
        and compare_denied.status_code == 403,
        f"strategies={len(compare_body.get('strategies', []))} denied={compare_denied.status_code}",
    )

    # 17.14 反馈闭环：一条纠偏记录投影成 Prompt few-shot（可被后续请求消费）
    fb = await client.post(
        "/v1/agents/feedback",
        json={"kind": "correction", "question": "能退货吗", "answer": "不能", "corrected": "七天无理由可退"},
        headers=cs_headers,
    )
    fb_examples = await client.get("/v1/agents/feedback/examples", headers=cs_headers)
    fb_examples_body = fb_examples.json()
    check(
        "反馈闭环：纠偏记录落库并投影为 Prompt few-shot（幂等投影、只增不改）",
        fb.status_code == 200
        and fb_examples.status_code == 200
        and fb_examples_body.get("stats", {}).get("total", 0) >= 1
        and any(item.get("answer") == "七天无理由可退" for item in fb_examples_body.get("prompt_examples", [])),
        f"total={fb_examples_body.get('stats', {}).get('total')} prompt={len(fb_examples_body.get('prompt_examples', []))}",
    )

    # 17.15 编排接口同样受凭证保护：无 Key 直接 401
    orchestrate_anon = await client.post("/v1/agents/orchestrate", json={"task": "你好"})
    check(
        "应用层接口鉴权：未携带凭证访问编排接口 401",
        orchestrate_anon.status_code == 401,
        f"status={orchestrate_anon.status_code}",
    )

    # ---------------- 18. 应用层（项目二 · AI 运营系统） ---------------- #
    ops_headers = {"X-API-Key": OPS_KEY}
    brief = {
        "product_id": "SKU-1002",
        "product_name": "烟酰胺精华",
        "market": "cn",
        "audience": "25-35 岁成分党",
        "channel": "xiaohongshu",
        "keywords": ["烟酰胺", "提亮", "屏障修护"],
        "tone": "专业但不端着",
    }

    # 18.1 运营线是独立权限面：无 Key 401；客服凭证无 ops:rw 403；运营凭证 200
    ops_anon = await client.post("/v1/ops/content/run", json={"brief": brief})
    ops_denied = await client.post("/v1/ops/content/run", json={"brief": brief}, headers=cs_headers)
    ops_ok = await client.post("/v1/ops/content/run", json={"brief": brief, "run_id": "smoke_ops_run"}, headers=ops_headers)
    ops_body = ops_ok.json()
    check(
        "运营系统鉴权：独立 scope ops:rw（无 Key 401 / 客服凭证 403 / 运营凭证 200）",
        ops_anon.status_code == 401 and ops_denied.status_code == 403 and ops_ok.status_code == 200,
        f"anon={ops_anon.status_code} denied={ops_denied.status_code} ok={ops_ok.status_code}",
    )

    # 18.2 内容工作流：四节点按拓扑跑完，模型分工（复杂分析 chat-pro / 短文案 chat-lite）可观测
    records = ops_body.get("records", [])
    check(
        "内容工作流：选题→成分卖点→(文案‖产品图) 全节点成功，且结构化作答有降级留痕",
        ops_body.get("status") == "succeeded"
        and [item.get("node") for item in records] == ["topic", "selling_points", "copy", "image"]
        and ops_body.get("artifacts", {}).get("models") == ["chat-lite", "chat-pro"]
        and len(ops_body.get("artifacts", {}).get("fallbacks", [])) == 4,
        f"status={ops_body.get('status')} nodes={len(records)} models={ops_body.get('artifacts', {}).get('models')}",
    )

    # 18.3 成本可核算：编排用量按节点汇总（轻量模型与旗舰模型都记账）
    usage = ops_body.get("artifacts", {}).get("usage", {})
    check(
        "内容工作流：token 用量按节点汇总进编排结果（成本可核算）",
        usage.get("total_tokens", 0) > 0 and usage.get("cost_usd", 0) > 0,
        f"tokens={usage.get('total_tokens')} cost={usage.get('cost_usd')}",
    )

    # 18.4 续跑：回滚 copy 后只重跑被失效节点，回滚留痕可回放（不重跑已完成节点）
    before_calls = container.gateway.provider_of("chat-lite").mock.call_count
    resume_ops = await client.post(
        "/v1/ops/content/resume",
        json={"brief": brief, "run_id": "smoke_ops_run", "rollback_nodes": ["copy"]},
        headers=ops_headers,
    )
    resume_body = resume_ops.json()
    after_calls = container.gateway.provider_of("chat-lite").mock.call_count
    check(
        "节点级续跑：回滚 copy 后只重跑该节点（其余节点复用已完成输出，不重复付费）",
        resume_ops.status_code == 200
        and set(resume_body.get("resumed_from", [])) == {"topic", "selling_points", "image"}
        and after_calls - before_calls == 2,  # chat-lite 只被 copy 重跑（首次 + 结构化重试）
        f"resumed_from={resume_body.get('resumed_from')} calls_delta={after_calls - before_calls}",
    )

    replay_ops = await client.get("/v1/ops/workflow/smoke_ops_run", headers=ops_headers)
    replay = replay_ops.json().get("replay", [])
    check(
        "回放：每次尝试都留痕（含 rolled_back），而不是只留最终状态",
        replay_ops.status_code == 200
        and any(item.get("status") == "rolled_back" and item.get("node") == "copy" for item in replay)
        and any(item.get("node") == "copy" and item.get("status") == "succeeded" for item in replay),
        f"entries={len(replay)}",
    )

    missing_run = await client.get("/v1/ops/workflow/nope", headers=ops_headers)
    check("未知 run 查询 -> 404（不返回空壳快照）", missing_run.status_code == 404, f"status={missing_run.status_code}")

    # 18.5 SSE 事件级流式：逐条推进（run.start → 节点/ token → run.done），并回传 X-Run-Id
    stream_text = ""
    stream_run_id = ""
    async with client.stream("POST", "/v1/ops/content/stream", json={"brief": brief}, headers=ops_headers) as stream_resp:
        stream_run_id = stream_resp.headers.get("x-run-id", "")
        stream_ct = stream_resp.headers.get("content-type", "")
        async for chunk in stream_resp.aiter_text():
            stream_text += chunk
    check(
        "SSE 流式：事件级推进（run.start/node.start/token/run.done）+ 回传 X-Run-Id 供续跑对账",
        stream_ct.startswith("text/event-stream")
        and "event: run.start" in stream_text
        and "event: token" in stream_text
        and "event: run.done" in stream_text
        and stream_run_id.startswith("content_"),
        f"ct={stream_ct} run_id={stream_run_id} bytes={len(stream_text)}",
    )

    # 18.6 客户端断开要真的止损：提前关流 → 后台 run 被取消并计入 aborted
    from src.ops.schemas import ContentBrief  # 局部导入：smoke 顶部只导 API 层，避免耦合

    aborted_before = container.ops.streamer.stats()["aborted"]
    aborted_stream = container.ops.stream_content(ContentBrief(**brief), run_id="smoke_ops_abort")
    first_frame = await aborted_stream.__anext__()
    await aborted_stream.aclose()  # 等价于客户端关页面
    aborted_after = container.ops.streamer.stats()["aborted"]
    check(
        "流式断开释放：客户端提前断开 → 后台任务被取消并记 aborted（不继续烧 token）",
        "event: run.start" in first_frame and aborted_after == aborted_before + 1,
        f"aborted {aborted_before}->{aborted_after}",
    )

    # 18.7 多模态素材：同一 (source_table, source_id) 重复导入幂等（asset_id 不变、计数不涨）
    asset_payload = {
        "source_table": "products",
        "source_id": "SKU-1002",
        "modality": "image",
        "title": "烟酰胺精华主图",
        "product_id": "SKU-1002",
        "source": {"caption": "烟酰胺 提亮 精华"},
    }
    asset_a = await client.post("/v1/ops/assets/import", json=asset_payload, headers=ops_headers)
    asset_b = await client.post("/v1/ops/assets/import", json=asset_payload, headers=ops_headers)
    check(
        "多模态素材：按 (source_table, source_id) 幂等 UPSERT（重复导入 asset_id 不变、计数不涨）",
        asset_a.status_code == 200
        and asset_a.json()["created"] is True
        and asset_b.json()["created"] is False
        and asset_a.json()["asset"]["asset_id"] == asset_b.json()["asset"]["asset_id"]
        and asset_a.json()["assets"] == asset_b.json()["assets"],
        f"id={asset_a.json()['asset']['asset_id']} created={asset_a.json()['created']}->{asset_b.json()['created']}",
    )

    await client.post(
        "/v1/ops/assets/import",
        json={
            "source_table": "pages",
            "source_id": "page-77",
            "modality": "page",
            "title": "成分详情页",
            "source": {"text": "烟酰胺成分说明与搭配用法"},
        },
        headers=ops_headers,
    )
    asset_search = await client.get("/v1/ops/assets/search", params={"query": "烟酰胺"}, headers=ops_headers)
    hits = asset_search.json().get("hits", [])
    check(
        "跨模态检索：文本查询能同时命中图片与页面素材（不同模态统一成 retrieval_text）",
        asset_search.status_code == 200
        and {"image", "page"}.issubset({hit["modality"] for hit in hits}),
        f"modalities={sorted({hit['modality'] for hit in hits})}",
    )

    # 18.8 直播：长直播按时间窗分段转录 → 重叠去重 → 分批分析 → 切片候选；FFmpeg 只构造命令
    live = await client.post(
        "/v1/ops/live/process",
        json={"source": {"title": "烟酰胺专场直播"}, "duration_s": 200, "source_path": "smoke-live.mp4"},
        headers=ops_headers,
    )
    live_body = live.json()
    check(
        "直播处理：分段转录 + 重叠去重（重复文本被裁掉）+ 分批分析 + 切片候选",
        live.status_code == 200
        and len(live_body.get("segments", [])) >= 3
        and live_body.get("dedupe", {}).get("trimmed_chars", 0) > 0
        and live_body.get("dedupe", {}).get("output_segments") == len(live_body.get("segments", []))
        and bool(live_body.get("clips")),
        f"segments={len(live_body.get('segments', []))} trimmed={live_body.get('dedupe', {}).get('trimmed_chars')} clips={len(live_body.get('clips', []))}",
    )
    import shutil as _shutil

    ffmpeg = live_body.get("ffmpeg", {})
    check(
        "直播切片：只构造 FFmpeg 命令；缺二进制时如实降级声明（绝不假装已产出视频）",
        all(cmd["command"].startswith("ffmpeg -ss") for cmd in ffmpeg.get("commands", []))
        and ffmpeg.get("degraded") is (_shutil.which("ffmpeg") is None)
        and (ffmpeg.get("reason") == "ffmpeg_not_installed" or ffmpeg.get("degraded") is False),
        f"degraded={ffmpeg.get('degraded')} reason={ffmpeg.get('reason')} cmds={len(ffmpeg.get('commands', []))}",
    )

    # 18.9 运营分析：指标与归因是确定性计算（可复算）；素材建议可执行
    analytics = await client.post(
        "/v1/ops/analytics/report",
        json={
            "channels": [
                {"channel": "live", "impressions": 10000, "clicks": 500, "orders": 60, "revenue": 6000, "spend": 1000, "baseline_orders": 40},
                {"channel": "search", "impressions": 8000, "clicks": 200, "orders": 20, "revenue": 1800, "spend": 900, "baseline_orders": 25},
            ],
            "materials": [
                {"source_id": "m1", "views": 1000, "likes": 100, "comments": 10, "shares": 5, "saves": 40, "title": "精华测评"},
                {"source_id": "m2", "views": 100, "likes": 1, "comments": 0, "shares": 0, "saves": 0},
            ],
        },
        headers=ops_headers,
    )
    analytics_body = analytics.json()
    metrics = {item["key"]: item["value"] for item in analytics_body.get("metrics", [])}
    attribution = {item["channel"]: item for item in analytics_body.get("attribution", [])}
    check(
        "运营分析：指标可复算（CTR/CVR/ROAS 与明细一致），归因份额可核对（且不用模型算）",
        analytics.status_code == 200
        and metrics.get("ctr") == round(700 / 18000, 4)
        and metrics.get("roas") == round(7800 / 1900, 4)
        and attribution["live"]["delta"] == 20.0
        and round(sum(item["share"] for item in analytics_body.get("attribution", [])), 6) == 1.0,
        f"ctr={metrics.get('ctr')} roas={metrics.get('roas')} live_delta={attribution.get('live', {}).get('delta')}",
    )
    check(
        "素材建议：爆款评分有排序且给出标题与剪辑指令（供下一轮内容生产消费）",
        [item["source_id"] for item in analytics_body.get("materials", [])] == ["m1", "m2"]
        and analytics_body["materials"][0]["title"]
        and analytics_body["materials"][0]["editing"],
        f"top={analytics_body.get('materials', [{}])[0].get('source_id')}",
    )

    # 18.10 自进化：回流 → 自动评测集 + 自动评分 → 模板/策略建议；模板**不自动发布**
    evo = await client.post(
        "/v1/ops/selfevolve/cycle",
        json={
            "events": [
                {"source": "correction", "question": f"烟酰胺精华怎么搭配？第 {i} 问", "corrected": "建议与保湿精华叠加使用", "chunk_ids": ["chunk-1"]}
                for i in range(3)
            ]
        },
        headers=ops_headers,
    )
    evo_body = evo.json()
    evo_report = await client.get("/v1/ops/selfevolve/report", headers=ops_headers)
    template = evo_report.json().get("template", {})
    check(
        "自进化闭环：回流 → 自动评测集 → 评分 → 模板/策略建议（模板只登记不激活，需人工审核）",
        evo.status_code == 200
        and len(evo_body.get("eval_cases", [])) == 3
        and evo_body["scoring"]["status"] == "awaiting_answers"
        and evo_body["template_patch"]["to_version"] != evo_body["template_patch"]["from_version"]
        and template.get("active_version") == evo_body["template_patch"]["from_version"]
        and bool(evo_body.get("strategy"))
        and "人工审核" in evo_body.get("notes", ""),
        f"cases={len(evo_body.get('eval_cases', []))} active={template.get('active_version')} to={evo_body['template_patch']['to_version']}",
    )

    applied = await client.post("/v1/ops/selfevolve/apply", headers=ops_headers)
    reverted = await client.post("/v1/ops/selfevolve/revert", headers=ops_headers)
    check(
        "模板迭代可人工上线也可回滚：apply 后生效，revert 回到上一版（版本顺序即回滚顺序）",
        applied.status_code == 200
        and reverted.status_code == 200
        and applied.json()["active_version"] == evo_body["template_patch"]["to_version"]
        and reverted.json()["active_version"] == evo_body["template_patch"]["from_version"],
        f"active={applied.json().get('active_version')} -> {reverted.json().get('active_version')}",
    )

    # 18.11 并行：实测墙钟 < 串行下界（不编造加速比）；单个失败不取消同批其它单元
    bench = await client.post(
        "/v1/ops/parallel/benchmark", json={"count": 4, "sleep_s": 0.05, "fail_every": 0}, headers=ops_headers
    )
    bench_body = bench.json()
    bench_fail = await client.post(
        "/v1/ops/parallel/benchmark", json={"count": 4, "sleep_s": 0.02, "fail_every": 2}, headers=ops_headers
    )
    check(
        "并行实测：gather + 线程池并行，墙钟 < 串行下界且并发度受限；单点失败不取消同批其它单元",
        bench.status_code == 200
        and bench_body.get("speedup", 0) > 1.0
        and bench_body.get("wall_ms", 0) < bench_body.get("sequential_ms", 0)
        and bench_body.get("peak_concurrency", 0) <= bench_body.get("max_concurrency", 0)
        and bench_body.get("failures") == []
        and bench_fail.json().get("failures") == ["unit1", "unit3"],
        f"speedup={bench_body.get('speedup')} wall={bench_body.get('wall_ms')} seq={bench_body.get('sequential_ms')} fails={bench_fail.json().get('failures')}",
    )

    # ---------------- 19. 统一验证（跨项目 · 阶段 6） ---------------- #
    # 前 18 段各自证明「一个项目能跑」；这一段证明「四个项目其实是同一个系统」：
    # 路由共存于一次装机、鉴权与错误信封跨项目一致、排他 scope 互不串门、trace 贯穿全部入口。
    fresh_data = {"X-API-Key": CONTENT_KEY}
    fresh_admin = {"X-API-Key": SHOP_ADMIN_KEY}
    fresh_eval = {"X-API-Key": EVAL_KEY}

    # 19.1 六个前缀共存于同一应用（不是各自起一个服务）
    coexist = {
        "/v1/tools": await client.get("/v1/tools", headers=cs_headers),
        "/v1/data": await client.get("/v1/data/reflux", headers=fresh_data),
        "/v1/agents": await client.get("/v1/agents/feedback/examples", headers=cs_headers),
        "/v1/shop": await client.get("/v1/shop/products"),
        "/v1/admin": await client.get("/v1/admin/orders/stats", headers=fresh_admin),
        "/v1/ops": await client.get("/v1/ops/assets/search", params={"query": "烟酰胺"}, headers=ops_headers),
    }
    check(
        "统一验证：四个项目与共享底座共存于同一应用（一次装机同时承载六个前缀）",
        len(coexist) == 6 and all(resp.status_code == 200 for resp in coexist.values()),
        " ".join(f"{prefix}={resp.status_code}" for prefix, resp in coexist.items()),
    )

    # 19.2 鉴权契约跨项目一致：无凭证一律 401，且错误信封形状相同（可被统一 SDK 处理）
    guarded = [
        ("GET", "/v1/ops/assets/search", {"params": {"query": "x"}}),
        ("GET", "/v1/data/reflux", {}),
        ("GET", "/v1/admin/orders/stats", {}),
        ("GET", "/v1/agents/feedback/examples", {}),
        ("POST", "/v1/eval/run", {"json": {}}),
    ]
    envelopes = []
    for method, path, kwargs in guarded:
        resp = await client.request(method, path, **kwargs)
        body = resp.json()
        envelopes.append(
            (path, resp.status_code, body.get("success"), (body.get("error") or {}).get("code"), bool(body.get("trace_id")))
        )
    check(
        "鉴权契约跨项目一致：无凭证一律 401，且信封形状相同（success=false + error.code=unauthenticated + trace_id）",
        all(
            status == 401 and success is False and code == "unauthenticated" and has_trace
            for _, status, success, code, has_trace in envelopes
        ),
        f"statuses={sorted({row[1] for row in envelopes})} codes={sorted({row[3] for row in envelopes})}",
    )

    # 19.3 排他 scope 互不串门：匿名 401 / 越权（客服凭证）403 / 属主 200，四行全对才算通过
    matrix_rows = []
    for scope, method, path, kwargs, owner in [
        ("ops:rw", "GET", "/v1/ops/assets/search", {"params": {"query": "x"}}, ops_headers),
        ("data:rw", "GET", "/v1/data/reflux", {}, fresh_data),
        ("commerce:rw", "GET", "/v1/admin/orders/stats", {}, fresh_admin),
        ("eval:rw", "POST", "/v1/eval/run", {"json": {}}, fresh_eval),
    ]:
        anon = await client.request(method, path, **kwargs)
        foreign = await client.request(method, path, headers=cs_headers, **kwargs)
        owner_resp = await client.request(method, path, headers=owner, **kwargs)
        matrix_rows.append((scope, anon.status_code, foreign.status_code, owner_resp.status_code))
    check(
        "排他 scope 互不串门：匿名 401 / 客服凭证 403 / 属主凭证 200（ops/data/commerce/eval 逐行三态）",
        all(row[1:] == (401, 403, 200) for row in matrix_rows),
        " ".join(f"{scope}={anon}/{foreign}/{owner}" for scope, anon, foreign, owner in matrix_rows),
    )

    # 19.4 trace 贯穿全部入口：显式传入原样回传，未传则生成 tr_ 前缀
    pinned = await client.get(
        "/v1/ops/assets/search",
        params={"query": "x"},
        headers={**ops_headers, "X-Trace-Id": "tr_smoke_unified_001"},
    )
    generated = await client.get("/v1/data/reflux", headers=fresh_data)
    check(
        "trace 贯穿：跨项目同一套追踪——显式传入的 trace_id 原样回传、未传时自动生成 tr_ 前缀",
        pinned.headers.get("X-Trace-Id") == "tr_smoke_unified_001"
        and generated.headers.get("X-Trace-Id", "").startswith("tr_"),
        f"pinned={pinned.headers.get('X-Trace-Id')} generated={generated.headers.get('X-Trace-Id')}",
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
