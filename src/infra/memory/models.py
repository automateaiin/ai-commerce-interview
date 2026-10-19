"""记忆服务的数据模型（PostgreSQL / SQLite 通用）。

两张表对应两种生命周期（面试必问 session_id 与 user_id 的差别）：
- messages：**会话生命周期**。一次对话的短期记忆，按 (project_id, session_id) 隔离。
- user_profiles：**用户生命周期**。跨会话的长期信息，唯一约束 (project_id, user_id)。

所有查询都必须带 project_id 条件 —— 这是多项目隔离在数据层的落点：
不是「客户端传了 project_id 就信」，而是「服务端按授权范围构造查询条件」。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MessageRecord(Base):
    """会话消息。tokens 字段存估算值，用于上下文预算与成本归因。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        # 覆盖「按项目+会话取最近消息」这条最高频查询
        Index("ix_messages_project_session_created", "project_id", "session_id", "created_at"),
    )


class UserProfileRecord(Base):
    """用户画像。唯一约束是 UPSERT 能成立的前提。"""

    __tablename__ = "user_profiles"

    project_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    profile_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uq_user_profiles_project_user"),)
