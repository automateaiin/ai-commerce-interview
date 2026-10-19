"""Session Rehydrate + 摘要记忆（复用 infra/memory）。

「记住用户」不是把全部历史塞进 Prompt 就行，那会让 token 成本随会话线性上涨。
这里的策略是三层：
1) **近期轮次优先**：最近的对话最能决定当前意图；
2) **超预算折叠**：被截断的历史折叠成一句摘要占位，让模型知道「之前聊过但被压缩」；
3) **画像单独存**：肤质、过敏史这类稳定事实走用户画像（UPSERT、字段级合并），
   不跟逐轮对话混在一起 —— 否则每轮都要从对话里重新推断。

隔离是硬约束：`project_id` 强制带上，`scope`（project+user）作为围栏隔离键，
保证不同用户的上下文不会互相污染。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.infra.core.logging import get_logger
from src.infra.memory.store import ContextBundle, MemoryStore, build_context

from .prompt_guard import PromptGuard

logger = get_logger("agents.session")


@dataclass(slots=True)
class SessionSnapshot:
    """一次会话回填的结果：可喂给模型的上下文 + 诊断信息 + 隔离键。"""

    bundle: ContextBundle
    scope: str
    user_profile: dict[str, Any] = field(default_factory=dict)
    guard_hits: list[str] = field(default_factory=list)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.bundle.messages

    def to_payload(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "used_tokens": self.bundle.used_tokens,
            "dropped_messages": self.bundle.dropped_messages,
            "summarized": self.bundle.summarized,
            "profile_keys": sorted(self.user_profile),
            "guard_hits": self.guard_hits,
        }


class SessionManager:
    def __init__(
        self,
        *,
        memory: MemoryStore,
        guard: PromptGuard,
        token_budget: int,
        recent_turns: int,
    ) -> None:
        self._memory = memory
        self._guard = guard
        self._token_budget = token_budget
        self._recent_turns = recent_turns

    async def rehydrate(
        self,
        *,
        project_id: str,
        session_id: str,
        user_id: str | None,
        current_message: str,
        system_prompt: str,
    ) -> SessionSnapshot:
        history = await self._memory.recent_messages(
            project_id=project_id, session_id=session_id, limit=self._recent_turns * 2
        )
        profile: dict[str, Any] = {}
        if user_id:
            profile = await self._memory.get_profile(project_id=project_id, user_id=user_id) or {}

        bundle = build_context(
            system_prompt=self._guard.harden(system_prompt),
            history=history,
            current_user_message=current_message,
            token_budget=self._token_budget,
            recent_turns=self._recent_turns,
        )
        # 只对「可能被注入」的历史文本做扫描留痕；命中不阻断，交给围栏兜底
        hits: list[str] = []
        for record in history:
            hits.extend(self._guard.scan(record.content).hits)
        if hits:
            logger.warning(
                "agents.session.guard_hits",
                extra={"event": "session", "status": "warn", "reason": ",".join(sorted(set(hits)))},
            )
        return SessionSnapshot(
            bundle=bundle,
            scope=self.scope(project_id=project_id, user_id=user_id),
            user_profile=profile,
            guard_hits=sorted(set(hits)),
        )

    async def persist(
        self,
        *,
        project_id: str,
        session_id: str,
        user_id: str | None,
        user_message: str,
        assistant_message: str | None,
    ) -> None:
        await self._memory.append_message(
            project_id=project_id, session_id=session_id, role="user", content=user_message, user_id=user_id
        )
        if assistant_message:
            await self._memory.append_message(
                project_id=project_id, session_id=session_id, role="assistant", content=assistant_message, user_id=user_id
            )

    async def update_profile(
        self, *, project_id: str, user_id: str, patch: dict[str, Any], source: str = "model"
    ) -> dict[str, Any]:
        return await self._memory.upsert_profile(project_id=project_id, user_id=user_id, patch=patch, source=source)

    @staticmethod
    def scope(*, project_id: str, user_id: str | None) -> str:
        """隔离键：同项目 + 同用户才算同一段上下文。"""
        return f"project:{project_id}|user:{user_id or 'anon'}"


__all__ = ["SessionManager", "SessionSnapshot"]
