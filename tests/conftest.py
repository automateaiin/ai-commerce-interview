"""测试公共夹具。

两个约定：
1) 测试环境必须在**导入应用之前**设好（配置在模块导入时就被读取）；
2) 每次跑测试用一个独立的临时库文件，保证用例之间不互相污染 ——
   依赖「上一条用例留下的数据」的测试是不可复现的。
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PROVIDER_MODE", "mock")  # 离线可跑，不依赖任何真实密钥
os.environ.setdefault("INTERNAL_SERVICE_TOKEN", "test-internal-token")
os.environ.setdefault("MCP_ALLOWED_HOSTS", "testserver")  # ASGI 直连下 Host=testserver
os.environ.setdefault("LOG_LEVEL", "ERROR")  # 用例里不刷日志，需要看链路时手动打开

# 每个测试进程用独立临时库；删掉上次残留，避免历史消息让断言失真
_TEST_DB = Path(tempfile.gettempdir()) / "ecom_infra_pytest.db"
_TEST_DB.unlink(missing_ok=True)
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_DB.as_posix()}")

# 业务库同样用独立临时文件：不污染仓库 data/ 目录，也不让上一轮订单影响本轮断言
_TEST_COMMERCE_DB = Path(tempfile.gettempdir()) / "ecom_commerce_pytest.db"
_TEST_COMMERCE_DB.unlink(missing_ok=True)
os.environ.setdefault("COMMERCE_DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_COMMERCE_DB.as_posix()}")
os.environ.setdefault("COMMERCE_MEMBER_TOKEN_SECRET", "pytest-member-token-secret")
os.environ.setdefault("COMMERCE_SWEEPER_INTERVAL_S", "0")  # 关掉后台清理：用例要的是确定性，不是后台时序


@asynccontextmanager
async def boot_app():
    """启动完整应用（含 lifespan），产出 (httpx.AsyncClient, Container)。"""
    from src.infra.api import create_app

    app = create_app()
    # ASGITransport 不会自动跑 lifespan，这里手动进入，保证 MCP 会话管理器已就绪
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30) as client:
            yield client, app.state.container


@pytest.fixture
def app_client():
    """返回「异步上下文管理器工厂」，用例里用 `async with app_client() as (client, container)`。

    刻意不做成 async fixture：async fixture 的 setup 与 teardown 可能落在不同 task 上，
    而 MCP 的 session_manager 内部是 anyio TaskGroup（要求 enter/exit 同一 task），
    会直接抛 "Attempted to exit cancel scope in a different task"。
    把启动与关闭都放进测试体内，才能保证是同一个 task。
    """
    return boot_app


CS_KEY = "sk-project-service-demo"
SALES_KEY = "sk-project-sales-demo"
CONTENT_KEY = "sk-project-content-demo"
SHOP_ADMIN_KEY = "sk-project-shopadmin-demo"  # 业务后端运营侧凭证（commerce:rw）
