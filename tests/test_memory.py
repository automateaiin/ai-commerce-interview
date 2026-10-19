"""会话记忆：messages（会话生命周期）+ user_profiles（用户生命周期）双表。

两条设计要点：
1) 读历史必须带 project_id —— 猜对 session_id 也读不到别的项目的数据；
2) 画像写入是**字段级合并**而不是整块覆盖 —— 否则两次不同来源的写入会把彼此抹掉。
"""

from __future__ import annotations

import pytest

from src.infra.core.errors import InfraError
from src.infra.memory.store import MemoryStore, build_context


@pytest.fixture
async def store() -> MemoryStore:
    from src.infra.core.settings import get_settings

    instance = MemoryStore.from_url(get_settings().resolved_database_url())
    await instance.init_schema()
    yield instance
    await instance.dispose()


async def test_history_is_read_back_in_chronological_order(store: MemoryStore):
    for role, content in (("user", "这款精华适合敏感肌吗"), ("assistant", "适合，成分温和")):
        await store.append_message(project_id="cs-bot", session_id="s-hist", role=role, content=content, user_id="u1")

    history = await store.recent_messages(project_id="cs-bot", session_id="s-hist", limit=10)
    # 返回顺序必须可以直接喂给模型：时间正序，不是倒序
    assert [item.role for item in history] == ["user", "assistant"]


async def test_history_is_isolated_per_project(store: MemoryStore):
    await store.append_message(project_id="cs-bot", session_id="s-iso", role="user", content="客服侧会话")
    await store.append_message(project_id="sales-agent", session_id="s-iso", role="user", content="销售侧会话")

    rows = await store.recent_messages(project_id="cs-bot", session_id="s-iso", limit=10)
    assert [row.content for row in rows] == ["客服侧会话"]


async def test_profile_upsert_merges_fields_instead_of_overwriting(store: MemoryStore):
    first = await store.upsert_profile(project_id="cs-bot", user_id="u-merge", patch={"skin_type": "敏感肌"})
    second = await store.upsert_profile(project_id="cs-bot", user_id="u-merge", patch={"budget_usd": 150})

    assert first["skin_type"] == "敏感肌"
    # 第二次写入不得把第一次的字段抹掉
    assert second == {"skin_type": "敏感肌", "budget_usd": 150}
    assert await store.get_profile(project_id="cs-bot", user_id="u-merge") == {
        "skin_type": "敏感肌",
        "budget_usd": 150,
    }


async def test_stale_profile_write_is_rejected_by_optimistic_lock(store: MemoryStore):
    await store.upsert_profile(project_id="cs-bot", user_id="u-lock", patch={"skin_type": "油性"})

    # 当前版本是 1，拿着过期的 version=0 来写必须失败：旧写入不能覆盖新值
    with pytest.raises(InfraError) as captured:
        await store.upsert_profile(
            project_id="cs-bot", user_id="u-lock", patch={"skin_type": "干性"}, expected_version=0
        )

    assert captured.value.detail["expected"] == 0
    assert captured.value.detail["current"] == 1
    assert (await store.get_profile(project_id="cs-bot", user_id="u-lock"))["skin_type"] == "油性"


async def test_context_bundle_bounds_history_by_token_budget(store: MemoryStore):
    for index in range(8):
        await store.append_message(
            project_id="cs-bot", session_id="s-budget", role="user", content=f"第{index}轮问题" * 8, user_id="u1"
        )
    history = await store.recent_messages(project_id="cs-bot", session_id="s-budget", limit=20)

    bundle = build_context(
        system_prompt="你是客服",
        history=history,
        current_user_message="最后一个问题",
        token_budget=60,
        recent_turns=12,
    )

    # 结构固定：system 开头、当前问题结尾
    assert bundle.messages[0]["role"] == "system"
    assert bundle.messages[-1]["content"] == "最后一个问题"
    # 超出预算的历史被丢弃而不是无限膨胀（丢了多少是可以观测的，不是静默截断）
    assert bundle.dropped_messages > 0
    assert bundle.used_tokens <= 60


async def test_context_bundle_starts_without_history_when_empty(store: MemoryStore):
    bundle = build_context(
        system_prompt="你是客服",
        history=[],
        current_user_message="你好",
        token_budget=1000,
        recent_turns=12,
    )
    assert [message["role"] for message in bundle.messages] == ["system", "user"]
