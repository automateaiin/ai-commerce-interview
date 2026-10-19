"""知识检索工具与记忆工具。

这两个工具是「RAG 服务」和「Memory 服务」对外的 MCP 接口。
注意职责边界：
- RAG 工具只负责「把相关证据取回来」，它不决定整个任务流程，也不生成最终回答；
- Memory 工具只负责「存取会话与用户记忆」，不应无条件把全部历史暴露给模型；
- 两者都从 ToolContext 取 project_id，绝不信任调用方传来的项目标识。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..memory.store import MemoryStore
from ..rag.service import KnowledgeService
from .base import FunctionTool, ToolContext, ToolRisk

logger = get_logger("infra.tools.knowledge")


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(min_length=1, description="检索问题或关键词，例如「敏感肌精华成分」「美国站退换政策」")
    product_id: str | None = Field(default=None, description="限定商品 ID，可为空表示全库检索")
    market: str | None = Field(default=None, description="限定站点市场，例如 US")
    top_k: int = Field(default=3, ge=1, le=10, description="返回片段数量")


class SessionHistoryArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=50, description="读取最近多少条会话消息")


class UpdateProfileArgs(BaseModel):
    patch: dict[str, Any] = Field(
        description="要合并写入的画像字段，例如 {\"skin_type\": \"敏感肌\", \"budget_usd\": 150}"
    )
    source: str = Field(default="agent", description="数据来源标识，用于冲突时的优先级判断与审计")


def build_knowledge_tools(knowledge: KnowledgeService) -> list[FunctionTool]:
    async def _search_knowledge(args: SearchKnowledgeArgs, ctx: ToolContext) -> dict[str, Any]:
        result = await knowledge.search(
            args.query, top_k=args.top_k, product_id=args.product_id, market=args.market
        )
        return {
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "source": chunk.source,
                    "product_id": chunk.product_id,
                    "score": chunk.score,
                }
                for chunk in result.chunks
            ],
            "has_evidence": result.has_evidence,
            # 诊断信息回传：让上层能区分「检索没召回」和「召回了但答错」
            "diagnostics": {
                "considered": result.considered,
                "filtered_expired": result.filtered_expired,
                "filtered_metadata": result.filtered_metadata,
            },
        }

    return [
        FunctionTool(
            name="search_knowledge",
            description=(
                "从商品知识库检索成分、功效、使用方法、注意事项与政策类证据。"
                "回答事实性问题前应先调用它；返回结果含来源与有效期过滤情况，"
                "若 has_evidence=false 说明没有可用证据，不要凭记忆编造。"
            ),
            args_model=SearchKnowledgeArgs,
            handler=_search_knowledge,
            requires_scope="tools:knowledge",
            risk=ToolRisk.READ_ONLY,
            timeout_s=5.0,
        )
    ]


def build_memory_tools(memory: MemoryStore) -> list[FunctionTool]:
    async def _get_history(args: SessionHistoryArgs, ctx: ToolContext) -> dict[str, Any]:
        if not ctx.session_id:
            raise ValidationError("缺少 session_id，无法读取会话历史")
        records = await memory.recent_messages(
            project_id=ctx.project_id, session_id=ctx.session_id, limit=args.limit
        )
        return {
            "messages": [
                {"role": record.role, "content": record.content, "tokens": record.tokens}
                for record in records
            ],
            "count": len(records),
        }

    async def _update_profile(args: UpdateProfileArgs, ctx: ToolContext) -> dict[str, Any]:
        if not ctx.user_id:
            raise ValidationError("缺少 user_id，无法更新用户画像")
        if not args.patch:
            raise ValidationError("patch 不能为空")
        profile = await memory.upsert_profile(
            project_id=ctx.project_id, user_id=ctx.user_id, patch=args.patch, source=args.source
        )
        return {"user_id": ctx.user_id, "profile": profile}

    return [
        FunctionTool(
            name="get_conversation_history",
            description=(
                "读取当前会话最近的消息历史。当需要确认用户之前说过什么、"
                "或在多轮任务中需要回顾上下文时调用。"
            ),
            args_model=SessionHistoryArgs,
            handler=_get_history,
            requires_scope="memory:rw",
            risk=ToolRisk.READ_ONLY,
            timeout_s=3.0,
        ),
        FunctionTool(
            name="update_user_profile",
            description=(
                "合并更新用户长期画像（如肤质、预算、偏好品牌）。"
                "字段级合并，不会覆盖未提及的已有字段；"
                "只应保存业务明确允许保存的信息，不要保存隐私或推测内容。"
            ),
            args_model=UpdateProfileArgs,
            handler=_update_profile,
            requires_scope="memory:rw",
            risk=ToolRisk.WRITE,
            timeout_s=5.0,
            idempotent=True,
            parallel_safe=False,
        ),
    ]
