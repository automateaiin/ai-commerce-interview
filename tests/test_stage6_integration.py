"""阶段 6（统一验证）专项用例：架构守卫 + 跨项目凭证隔离。

这一段不新增业务功能，只把「四个项目其实是一个系统」变成**可执行的断言**：
    A 分层依赖方向（不准倒置）   B 唯一组合根   C 路由挂载完整   D 凭证隔离矩阵

为什么这些断言值得单独存在：
- 分层依赖一旦倒置（下层反向 import 上层），编译不会报错、单体测试也不会红，
  只有真的分仓/分服务时才会炸出来 —— 所以必须在 CI 里静态挡住。
- 『唯一组合根』同理：只要有人在业务模块里自己 new 一个 Service，单例边界就悄悄破了。
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from tests.conftest import CONTENT_KEY, CS_KEY, SHOP_ADMIN_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# 允许引用上层模块的两个文件：api.py 负责挂载各项目 router，container.py 是唯一组合根。
_COMPOSITION_ROOT = {Path("src/infra/api.py"), Path("src/infra/container.py")}

# 每层「不准反向依赖」的上层集合（对应 PROJECT.md 第 2 章的依赖方向图）
_FORBIDDEN_UPWARD = {
    "infra": {"agents", "ops", "commerce"},
    "harness": {"agents", "ops", "commerce"},
    "dataplatform": {"agents", "ops", "commerce"},
    "commerce": {"agents", "ops"},
    "agents": {"ops"},
    "ops": {"agents"},
}

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+src\.(agents|ops|commerce)\b")

# 四个项目各自独占的 scope（其余业务凭证一律不该拿到）
OPS_KEY = "sk-project-ops-demo"
EVAL_KEY = "sk-project-eval-demo"

# (scope, method, path, kwargs, 属主凭证)：每行做「匿名 401 / 越权 403 / 属主 200」三态
_SCOPE_MATRIX = [
    ("ops:rw", "GET", "/v1/ops/assets/search", {"params": {"query": "烟酰胺"}}, OPS_KEY),
    ("data:rw", "GET", "/v1/data/reflux", {}, CONTENT_KEY),
    ("commerce:rw", "GET", "/v1/admin/orders/stats", {}, SHOP_ADMIN_KEY),
    ("eval:rw", "POST", "/v1/eval/run", {"json": {}}, EVAL_KEY),
]


def test_layer_dependency_direction_is_one_way() -> None:
    """下层不准反向 import 上层：infra/harness/dataplatform 不得引用应用层与业务库。"""
    violations: list[str] = []
    for layer, forbidden in _FORBIDDEN_UPWARD.items():
        for path in (SRC / layer).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT)
            if rel in _COMPOSITION_ROOT:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                match = _IMPORT_RE.match(line)
                if match and match.group(1) in forbidden:
                    violations.append(f"{rel.as_posix()}:{lineno} → src.{match.group(1)}")
    assert not violations, "分层依赖被倒置（下层反向引用上层）: " + "; ".join(violations)


def test_only_container_composes_project_services() -> None:
    """唯一组合根：各项目的 build_*_service 只能在自己 factory 与 container 里被调用。"""
    builders = {
        "build_agents_service": {"src/agents/factory.py", "src/infra/container.py"},
        "build_ops_service": {"src/ops/factory.py", "src/infra/container.py"},
        "build_commerce_service": {"src/commerce/factory.py", "src/infra/container.py"},
    }
    seen: dict[str, set[str]] = defaultdict(set)
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for name in builders:
            if re.search(rf"\b{name}\s*\(", text):
                seen[name].add(rel)

    for name, allowed in builders.items():
        overflow = seen[name] - allowed
        assert not overflow, f"{name} 的装配点溢出组合根: {sorted(overflow)}"


def test_all_project_prefixes_are_mounted_on_one_app() -> None:
    """四个项目与共享底座挂在**同一个应用**上（不是各自起一个服务）。"""
    from src.infra.api import create_app

    app = create_app()
    # 用 OpenAPI 路径表而不是 app.routes：新版 FastAPI 会把 include_router 的结果包一层
    # `_IncludedRouter`，直接读 route.path 拿不到真实路径；schema 是稳定的对外契约。
    paths = sorted(app.openapi()["paths"])
    expected = [
        "/healthz",
        "/readyz",
        "/metrics",
        "/v1/tools",
        "/v1/chat/completions",
        "/v1/agent/run",
        "/v1/data/pipeline/run",
        "/v1/data/search",
        "/v1/agents/route",
        "/v1/a2a/message:send",
        "/v1/shop/products",
        "/v1/admin/orders/stats",
        "/v1/ops/content/run",
    ]
    missing = [needle for needle in expected if not any(p.startswith(needle) for p in paths)]
    assert not missing, f"路由未挂载到应用: {missing}"


async def test_unauthenticated_requests_share_one_error_envelope(app_client) -> None:
    """四项目的受保护端点：无凭证一律 401，且错误信封形状一致（可被统一 SDK 处理）。"""
    endpoints = [
        ("GET", "/v1/ops/assets/search", {"params": {"query": "x"}}),
        ("GET", "/v1/data/reflux", {}),
        ("GET", "/v1/admin/orders/stats", {}),
        ("GET", "/v1/agents/feedback/examples", {}),
        ("POST", "/v1/eval/run", {"json": {}}),
    ]
    async with app_client() as (client, _container):
        for method, path, kwargs in endpoints:
            response = await client.request(method, path, **kwargs)
            body = response.json()
            assert response.status_code == 401, f"{path} 期望 401，实得 {response.status_code}"
            assert body["success"] is False, f"{path} 信封 success 应为 False"
            assert body["error"]["code"] == "unauthenticated", f"{path} 错误码应为 unauthenticated"
            assert body.get("trace_id"), f"{path} 信封应带 trace_id"


async def test_exclusive_scopes_do_not_leak_across_projects(app_client) -> None:
    """排他 scope 互不串门：匿名 401 / 越权凭证 403 / 属主凭证 200，四行全对才通过。"""
    checked: list[tuple[str, int, int, int]] = []
    async with app_client() as (client, _container):
        for scope, method, path, kwargs, owner_key in _SCOPE_MATRIX:
            anon = await client.request(method, path, **kwargs)
            foreign = await client.request(method, path, headers={"X-API-Key": CS_KEY}, **kwargs)
            owner = await client.request(method, path, headers={"X-API-Key": owner_key}, **kwargs)
            checked.append((scope, anon.status_code, foreign.status_code, owner.status_code))

    wrong = [row for row in checked if row[1:] != (401, 403, 200)]
    assert not wrong, f"凭证隔离矩阵不符（期望 401/403/200）: {wrong}"
