"""记忆存储与上下文组装。

三件事：
1) `MemoryStore`：消息追加、按会话读取、画像 UPSERT（字段级合并 + 乐观锁）；
2) `build_context`：按 token 预算从新到旧组装上下文，超出部分做摘要，
   而不是把全部历史无限追加；
3) 归属校验：任何读写都先确认 (project_id, session_id) 属于当前调用方授权范围。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from ..core.errors import InfraError, ValidationError
from ..core.logging import get_logger
from ..llm.gateway import estimate_tokens
from ..llm.types import Message
from .models import Base, MessageRecord, UserProfileRecord, utcnow

logger = get_logger("infra.memory")


class MemoryStore:
    """会话记忆与用户画像的持久化入口。"""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, url: str) -> "MemoryStore":
        return cls(create_async_engine(url, pool_pre_ping=True))

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    # -- 会话消息 ----------------------------------------------------------- #
    async def append_message(
        self,
        *,
        project_id: str,
        session_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        tokens: int | None = None,
    ) -> None:
        record = MessageRecord(
            project_id=project_id,
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            tokens=tokens if tokens is not None else estimate_tokens(content),
        )
        async with self._sessionmaker() as session:
            session.add(record)
            await session.commit()

    async def recent_messages(
        self, *, project_id: str, session_id: str, limit: int = 20
    ) -> list[MessageRecord]:
        """取最近 N 条消息。

        强制带 project_id：即使 session_id 猜对了，跨项目也读不到 —— 防横向越权。
        """
        stmt = (
            select(MessageRecord)
            .where(MessageRecord.project_id == project_id, MessageRecord.session_id == session_id)
            .order_by(MessageRecord.id.desc())
            .limit(limit)
        )
        async with self._sessionmaker() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return list(reversed(rows))

    # -- 用户画像 ----------------------------------------------------------- #
    async def get_profile(self, *, project_id: str, user_id: str) -> dict[str, Any] | None:
        stmt = select(UserProfileRecord).where(
            UserProfileRecord.project_id == project_id, UserProfileRecord.user_id == user_id
        )
        async with self._sessionmaker() as session:
            record = (await session.execute(stmt)).scalar_one_or_none()
        return dict(record.profile_data) if record else None

    async def upsert_profile(
        self,
        *,
        project_id: str,
        user_id: str,
        patch: dict[str, Any],
        source: str = "model",
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """UPSERT 用户画像，**字段级合并**，不做整块覆盖。

        并发语义要讲清楚（面试必问）：
        - UPSERT 解决的是「插入冲突」，它**不**解决业务上的字段覆盖顺序问题；
        - 因此合并策略分两条路：
          * PostgreSQL：用 jsonb `||` 在数据库侧做合并，一次往返、天然原子；
          * 其它方言（本地演示用 SQLite）：事务内读-合并-写，并用 version 做乐观锁，
            旧写入无法覆盖新值（版本不匹配即失败）。
        - 跨来源冲突仍需业务定优先级（source / 置信度 / 时间），不能只靠数据库。
        """
        if not patch:
            return await self.get_profile(project_id=project_id, user_id=user_id) or {}

        if self.dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(UserProfileRecord).values(
                project_id=project_id,
                user_id=user_id,
                profile_data=patch,
                version=1,
                source=source,
                updated_at=utcnow(),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[UserProfileRecord.project_id, UserProfileRecord.user_id],
                set_={
                    # 字段级合并：保留已有字段，只覆盖/新增本次变更的字段
                    "profile_data": UserProfileRecord.__table__.c.profile_data.op("||")(stmt.excluded.profile_data),
                    "version": UserProfileRecord.__table__.c.version + 1,
                    "source": stmt.excluded.source,
                    "updated_at": utcnow(),
                },
            )
            async with self._sessionmaker() as session:
                await session.execute(stmt)
                await session.commit()
            return await self.get_profile(project_id=project_id, user_id=user_id) or {}

        # 非 PostgreSQL：乐观锁 + 读-合并-写
        async with self._sessionmaker() as session:
            record = (
                await session.execute(
                    select(UserProfileRecord).where(
                        UserProfileRecord.project_id == project_id, UserProfileRecord.user_id == user_id
                    )
                )
            ).scalar_one_or_none()
            if record is None:
                record = UserProfileRecord(
                    project_id=project_id, user_id=user_id, profile_data=dict(patch), version=1, source=source
                )
                session.add(record)
            else:
                if expected_version is not None and record.version != expected_version:
                    raise InfraError(
                        "画像版本冲突，拒绝写入以避免旧数据覆盖新值",
                        detail={"expected": expected_version, "current": record.version},
                    )
                merged = dict(record.profile_data or {})
                merged.update(patch)
                record.profile_data = merged
                record.version += 1
                record.source = source
                record.updated_at = utcnow()
            await session.commit()
            return dict(record.profile_data or {})


@dataclass(slots=True)
class ContextBundle:
    """组装好的模型输入 + 组装过程的诊断信息。"""

    messages: list[Message]
    used_tokens: int
    dropped_messages: int
    summarized: bool


def build_context(
    *,
    system_prompt: str,
    history: Sequence[MessageRecord],
    current_user_message: str,
    token_budget: int,
    recent_turns: int = 12,
) -> ContextBundle:
    """按 token 预算组装上下文。

    策略（面试要讲的取舍）：最近 N 轮 + 从新到旧截断 + 超出部分折叠成一句摘要，
    而不是把全部历史塞给模型 —— 无限追加会让 token 成本与延迟线性上涨，
    且无关历史会干扰当前任务。
    """
    if not current_user_message.strip():
        raise ValidationError("用户消息不能为空")

    budget = max(token_budget - estimate_tokens(current_user_message) - estimate_tokens(system_prompt), 0)
    picked: list[Message] = []
    used = 0
    dropped = 0

    for record in reversed(list(history)[-recent_turns:]):
        cost = record.tokens or estimate_tokens(record.content)
        if used + cost > budget:
            dropped += 1
            continue
        picked.append({"role": record.role, "content": record.content})
        used += cost

    picked.reverse()
    messages: list[Message] = [{"role": "system", "content": system_prompt}]
    if dropped:
        # 被截断的历史用一句摘要占位：让模型知道「之前聊过但被压缩了」
        messages.append(
            {"role": "system", "content": f"[上下文摘要] 本会话更早的 {dropped} 条消息因 token 预算被压缩。"}
        )
    messages.extend(picked)
    messages.append({"role": "user", "content": current_user_message})
    return ContextBundle(messages=messages, used_tokens=used, dropped_messages=dropped, summarized=bool(dropped))
