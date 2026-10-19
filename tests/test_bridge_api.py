"""桥接层 HTTP 契约：鉴权、授权、参数二次校验、异常映射。

这些用例刻意走 HTTP 层而不是直接调函数 ——
「错误响应长什么样」「状态码是多少」本身就是对外契约的一部分，
只有从入口打进去才算真的验证了契约。
"""

from __future__ import annotations

from conftest import CONTENT_KEY, CS_KEY, SALES_KEY


async def test_auth_failures_return_structured_401(app_client):
    async with app_client() as (client, _):
        missing = await client.post("/v1/tools/search_products", json={"arguments": {"query": "精华"}})
        assert missing.status_code == 401
        body = missing.json()
        assert body["success"] is False
        assert body["error"]["code"] == "unauthenticated"
        # 401 也必须带 trace_id：调用方靠它来对账整条链路
        assert body["trace_id"]

        wrong = await client.post(
            "/v1/tools/search_products", json={"arguments": {"query": "精华"}}, headers={"X-API-Key": "sk-nope"}
        )
        assert wrong.status_code == 401
        assert wrong.json()["error"]["code"] == "unauthenticated"

        # Bearer 形式与 X-API-Key 等价
        bearer = await client.post(
            "/v1/tools/search_products",
            json={"arguments": {"query": "精华"}},
            headers={"Authorization": f"Bearer {CS_KEY}"},
        )
        assert bearer.status_code == 200 and bearer.json()["success"] is True


async def test_argument_validation_and_unknown_tool_are_recoverable(app_client):
    async with app_client() as (client, _):
        # 参数越界：由服务端二次校验拦下，绝不相信模型/调用方给的参数
        bad_args = await client.post(
            "/v1/tools/search_products",
            json={"arguments": {"query": "精华", "limit": 999}},
            headers={"X-API-Key": CS_KEY},
        )
        body = bad_args.json()
        assert bad_args.status_code == 200  # 工具级失败不是传输失败
        assert body["success"] is False and body["error"]["code"] == "invalid_arguments"
        assert body["error"]["retryable"] is False

        # 未知工具：可恢复（模型可以据此换工具），不能是 500
        unknown = await client.post(
            "/v1/tools/not_a_real_tool", json={"arguments": {}}, headers={"X-API-Key": CS_KEY}
        )
        assert unknown.status_code == 200
        assert unknown.json()["error"]["code"] == "not_found"


async def test_scope_and_ownership_are_enforced_server_side(app_client):
    async with app_client() as (client, _):
        # 鉴权（你是谁）≠ 授权（你能调什么）：content-ops 只有 llm:chat
        denied = await client.post(
            "/v1/tools/search_products", json={"arguments": {"query": "面霜"}}, headers={"X-API-Key": CONTENT_KEY}
        )
        err = denied.json()["error"]
        assert err["code"] == "permission_denied"
        # terminal 表示 Agent Loop 必须终止而不是继续重试/换参数
        assert err["terminal"] is True

        # 写操作需要独立 scope：客服凭证没有 tools:write
        order_body = {
            "arguments": {"product_id": "SKU-1001", "quantity": 1, "idempotency_key": "pytest-idem-001"},
            "user_id": "u_10086",
        }
        write_denied = await client.post("/v1/tools/create_order", json=order_body, headers={"X-API-Key": CS_KEY})
        assert write_denied.json()["error"]["code"] == "permission_denied"

        # 订单归属：参数合法 ≠ 有权访问别人的数据
        other = await client.post(
            "/v1/tools/get_order_status",
            json={"arguments": {"order_id": "ORD-90001"}, "user_id": "u_99999"},
            headers={"X-API-Key": CS_KEY},
        )
        assert other.json()["error"]["code"] == "permission_denied"

        mine = await client.post(
            "/v1/tools/get_order_status",
            json={"arguments": {"order_id": "ORD-90001"}, "user_id": "u_10086"},
            headers={"X-API-Key": CS_KEY},
        )
        assert mine.json()["success"] is True

        # 写操作幂等：同一幂等键重复调用返回首次结果
        first = await client.post("/v1/tools/create_order", json=order_body, headers={"X-API-Key": SALES_KEY})
        second = await client.post("/v1/tools/create_order", json=order_body, headers={"X-API-Key": SALES_KEY})
        first_body, second_body = first.json(), second.json()
        assert first_body["success"] is True
        assert first_body["data"]["order_id"] == second_body["data"]["order_id"]
        assert second_body["data"]["idempotent_replay"] is True


async def test_identity_comes_from_credential_not_request_body(app_client):
    async with app_client() as (client, _):
        # 请求体里声明的 project_id 一律忽略：身份只能由凭证推导
        r = await client.post(
            "/v1/tools/search_products",
            json={"arguments": {"query": "精华"}, "client_project_id": "sales-agent"},
            headers={"X-API-Key": CS_KEY},
        )
        assert r.json()["success"] is True

        quota = (await client.get("/v1/quota", headers={"X-API-Key": CS_KEY})).json()
        assert quota["project_id"] == "cs-bot"

        # trace_id 透传：入口给的 trace 必须原样回给调用方
        traced = await client.post(
            "/v1/tools/search_products",
            json={"arguments": {"query": "精华"}},
            headers={"X-API-Key": CS_KEY, "X-Trace-Id": "tr_pytest_001"},
        )
        assert traced.headers.get("x-trace-id") == "tr_pytest_001"
        assert traced.json()["trace_id"] == "tr_pytest_001"


async def test_agent_run_exposes_guardrails_and_cost_contract(app_client):
    from src.infra.llm.mock import calls, final

    async with app_client() as (client, container):
        mock = container.gateway.provider_of("chat-lite").mock
        mock.reset(
            [
                calls(("search_products", {"query": "精华"})),
                final("按证据回答：SKU-1001 适合敏感肌。"),
            ]
        )

        r = await client.post(
            "/v1/agent/run",
            json={"task": "推荐一款适合敏感肌的精华", "session_id": "sess-pytest", "user_id": "u_10086"},
            headers={"X-API-Key": CS_KEY},
        )
        body = r.json()
        assert body["success"] is True and body["stop_reason"] == "completed"
        assert body["steps"] >= 2 and len(body["tool_calls"]) == 1
        # 结构化结果契约三件套：answer / usage(成本依据) / transcript(排障与可视化)
        assert body["usage"]["total_tokens"] > 0
        assert any(step["type"] == "tool" for step in body["transcript"])


async def test_quota_is_reserved_and_settled_across_chat_calls(app_client):
    """一次真实调用必须同时改变 requests 与 tokens 两个维度，且成本被计入。"""
    from src.infra.llm.mock import final

    async with app_client() as (client, container):
        before = (await client.get("/v1/quota", headers={"X-API-Key": CS_KEY})).json()
        container.gateway.provider_of("chat-lite").mock.reset([final("已按证据回答。")])

        chat = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "适合敏感肌吗"}], "session_id": "sess-quota"},
            headers={"X-API-Key": CS_KEY},
        )
        assert chat.json()["success"] is True

        after = (await client.get("/v1/quota", headers={"X-API-Key": CS_KEY})).json()
        assert after["requests"] > before["requests"]
        assert after["tokens"] > before["tokens"]


async def test_admin_endpoint_is_closed_without_token(app_client):
    """未配置 ADMIN_TOKEN 时管理端点默认关闭：默认安全，而不是默认开放。"""
    async with app_client() as (client, _):
        r = await client.post("/admin/config/reload")
        assert r.status_code == 403

        view = (await client.get("/v1/config", headers={"X-API-Key": CS_KEY})).json()
        assert view["revision"] == 1
        # MCP 服务集群的挂载路径要能被运维看到（尾斜杠与真实可访问地址一致）
        assert all(path.startswith("/mcp/") for path in view["mcp_paths"])
