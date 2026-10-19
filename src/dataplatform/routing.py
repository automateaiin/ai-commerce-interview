"""用途分流 + 价格与时效治理 —— 最容易出事、也最容易被深挖的一环。

核心判断（面试要能一句话说清）：
**RAG 负责「可更新、可引用的事实」，SFT 负责「稳定的表达风格与处理模式」。**
把两者混为一谈，就会出现「微调后语气更像品牌客服，但事实错误变多」——
因为易变事实（价格、库存、活动）被写进了模型参数里。

价格治理是这条原则的极端案例：
- 历史聊天里的价格**不是当前报价**，所以带价格的条目被标记 `price_freshness`，
  只保留结构化引用与 `live_source`（实时报价走可信业务接口），**不进语义索引**；
- 商品知识里的价格同理：可结构化保存，但不作为长期语义知识被检索。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from ..infra.core.logging import get_logger
from ..infra.core.tracing import stage
from .chunking import MODE_HIERARCHICAL, Chunker
from .schemas import (
    KnowledgeArticle,
    KnowledgeChunk,
    KnowledgeStatus,
    SamplePurpose,
    SourceKind,
    StagingConversation,
    TrainingSample,
    speaker_role,
)

logger = get_logger("data.routing")

PRODUCT_FACT_HINTS = ("成分", "功效", "适用", "肤质", "使用方法", "规格", "容量", "产地", "保质期", "搭配", "配方")
PRICE_HINTS = ("价格", "售价", "元", "￥", "$", "促销", "活动价", "折扣", "优惠", "券", "限时")


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _looks_price_related(text: str, meta: dict[str, Any]) -> bool:
    if meta.get("price") is not None and str(meta.get("price")).strip():
        return True
    return sum(1 for hint in PRICE_HINTS if hint in text) >= 2


def _looks_factual(text: str, meta: dict[str, Any]) -> bool:
    if meta.get("question") and meta.get("title"):
        return True
    return any(hint in text for hint in PRODUCT_FACT_HINTS)


class Router:
    """审核通过的 staging → 知识文章/切片 或 SFT 候选。"""

    def __init__(self, chunker: Chunker | None = None) -> None:
        self._chunker = chunker or Chunker()

    def route(
        self, staging: Sequence[StagingConversation]
    ) -> tuple[list[KnowledgeArticle], list[KnowledgeChunk], list[TrainingSample], dict[str, Any]]:
        articles: list[KnowledgeArticle] = []
        chunks: list[KnowledgeChunk] = []
        samples: list[TrainingSample] = []
        price_related = 0

        with stage("data.routing", staging=len(staging)):
            for index, item in enumerate(staging):
                meta = item.messages[0].meta if item.messages else {}
                text = item.text
                article_id = (
                    item.messages[0].source_id
                    if item.messages and item.messages[0].source == SourceKind.PRODUCT_KNOWLEDGE
                    else f"art_{item.staging_id}"
                )
                price_flag = _looks_price_related(text, meta)
                if price_flag:
                    price_related += 1

                if price_flag or _looks_factual(text, meta):
                    article = KnowledgeArticle(
                        article_id=article_id,
                        title=str(meta.get("title") or self._infer_title(text)),
                        scene=str(meta.get("scene") or item.scene or "general"),
                        question=str(meta.get("question") or self._infer_question(item)),
                        answer=str(meta.get("answer") or text),
                        conditions=list(meta.get("conditions") or []),
                        confidence=0.9 if meta.get("question") else 0.75,
                        source_staging_ids=[item.staging_id],
                        product_id=meta.get("product_id") or item.product_id,
                        market=meta.get("market"),
                        tags=list(meta.get("tags") or []),
                        valid_from=_parse_date(meta.get("valid_from")),
                        valid_to=_parse_date(meta.get("valid_to")),
                        version=int(meta.get("version") or 1),
                        status=KnowledgeStatus.ACTIVE,
                        price_freshness=price_flag,
                        live_source=meta.get("price_live_source") if price_flag else None,
                    )
                    articles.append(article)
                    # 带时效价格的知识**不进语义索引**：宁可回答时查实时源，也不让过期价被检索
                    if not price_flag:
                        chunks.extend(self._build_chunks(article, start_index=len(chunks)))
                else:
                    samples.append(
                        TrainingSample(
                            sample_id=f"sft_{item.staging_id}",
                            messages=self._to_messages(item),
                            purpose=SamplePurpose.SFT_STYLE,
                            source_staging_ids=[item.staging_id],
                            scene=item.scene,
                            meta={"reviewer": item.reviewer, "source": item.messages[0].source.value if item.messages else None},
                        )
                    )

        stats = {
            "articles": len(articles),
            "chunks": len(chunks),
            "sft_samples": len(samples),
            "price_related_excluded": price_related,
        }
        logger.info(
            "data.routing.done",
            extra={"event": "data_routing", "status": "ok", "reason": f"articles={len(articles)},chunks={len(chunks)},sft={len(samples)}"},
        )
        return articles, chunks, samples, stats

    def _build_chunks(self, article: KnowledgeArticle, *, start_index: int) -> list[KnowledgeChunk]:
        body = article.answer
        if article.conditions:
            body = f"{body}\n适用条件：{'；'.join(article.conditions)}"
        common = {
            "article_id": article.article_id,
            "product_id": article.product_id,
            "scene": article.scene,
            "market": article.market,
            "tags": list(article.tags),
            "version": article.version,
            "valid_from": article.valid_from,
            "valid_to": article.valid_to,
        }
        if self._chunker.mode == MODE_HIERARCHICAL:
            # 三级模式：L1/L2/L3 全部入索引，L1 负责精准召回，L2/L3 供 Auto-merging 回填上下文
            return [
                KnowledgeChunk(
                    chunk_id=node.node_id,
                    chunk_index=node.index,
                    text=node.text,
                    level=node.level,
                    parent_id=node.parent_id,
                    **common,
                )
                for node in self._chunker.hierarchical(body, prefix=article.article_id)
            ]

        built: list[KnowledgeChunk] = []
        for offset, piece in enumerate(self._chunker.chunk_document(body)):
            built.append(
                KnowledgeChunk(
                    chunk_id=f"{article.article_id}#c{offset}",
                    chunk_index=offset,
                    text=piece,
                    **common,
                )
            )
        return built

    @staticmethod
    def _to_messages(item: StagingConversation) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for message in item.messages:
            role = speaker_role(message.speaker)
            messages.append({"role": role, "content": message.content})
        if item.after is not None:
            # 审核编辑是针对「客服回答」的修正：保留原始提问，用 after 覆盖回答。
            # 注意不能直接用 item.text —— 它本身已优先取 after，会造成一问一答重复。
            question = next(
                (message.content for message in item.messages if speaker_role(message.speaker) == "user"),
                item.before or "",
            )
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": item.after}]
        return messages

    @staticmethod
    def _infer_title(text: str) -> str:
        return text.strip().splitlines()[0][:40] if text.strip() else "未命名知识"

    @staticmethod
    def _infer_question(item: StagingConversation) -> str:
        for message in item.messages:
            if speaker_role(message.speaker) == "user":
                return message.content
        return item.messages[0].meta.get("question") or "（无明确问法）"
