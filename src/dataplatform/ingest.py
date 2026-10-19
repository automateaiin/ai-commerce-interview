"""接入与标准化：按来源分解析器，映射成统一字段。

面试要点（最容易被追问的三个点）：
1) **为什么按来源分解析器**：聊天导出、商品知识、直播 ASR 的字段完全不同，
   硬塞进一个「万能解析」会变成一堆 if；分解析器后每类都能单独测试与演进。
2) **为什么映射失败要保留原始标识**：群聊里发送人映射（wxid → 内部用户）可能失败，
   这时必须保留 `raw_speaker` 并把 parse_status 标成 partial，**绝不猜测身份** ——
   猜错身份会把别人的话记到错误的人头上，是隐私事故。
3) **为什么统一时间**：多来源时间格式不一（epoch 秒/毫秒、带 Z、无时区），
   不统一会让按时间窗切片和有效期过滤全部失效。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from ..infra.core.errors import ValidationError
from ..infra.core.logging import get_logger
from ..infra.core.tracing import stage
from .schemas import ParseStatus, RawRecord, SourceKind, StandardMessage

logger = get_logger("data.ingest")

# 内部用户标识映射表（生产来自组织通讯录；此处为离线示例）
SPEAKER_MAP: dict[str, str] = {
    "wxid_alice": "cs_agent_alice",
    "wxid_bob": "customer_bob",
    "wxid_carol": "cs_agent_carol",
}


def normalize_ts(value: Any) -> datetime:
    """把多种时间表示统一成带时区的 UTC datetime。

    支持：ISO 字符串（含 Z）、epoch 秒、epoch 毫秒、datetime 对象。
    解析不了就抛错 —— 宁可让这条数据解析失败，也不要写入一个错误时间（会导致有效期判断全错）。
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:  # 毫秒时间戳
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValidationError(f"无法解析时间: {value}", detail={"value": str(value)}) from exc
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise ValidationError(f"不支持的时间类型: {type(value).__name__}")


def _resolve_speaker(raw: str | None) -> tuple[str | None, str | None, ParseStatus]:
    """解析发送人。映射失败保留原始标识 + partial，不做猜测。"""
    if not raw:
        return None, None, ParseStatus.PARTIAL
    mapped = SPEAKER_MAP.get(raw)
    if mapped is None:
        return raw, raw, ParseStatus.PARTIAL
    return mapped, raw, ParseStatus.OK


def parse_chat_export(record: dict[str, Any]) -> tuple[RawRecord, list[StandardMessage]]:
    """企业聊天导出：群聊/私聊消息。"""
    raw = RawRecord(source=SourceKind.CHAT_EXPORT, source_id=str(record.get("id") or record.get("msg_id")), payload=dict(record))
    speaker, raw_speaker, status = _resolve_speaker(record.get("sender") or record.get("speaker"))
    session_id = str(record.get("session_id") or record.get("chat_id") or "")
    if not session_id:
        raise ValidationError("聊天记录缺少 session_id", detail={"source_id": raw.source_id})
    content = str(record.get("text") or record.get("content") or "")
    message = StandardMessage(
        source=SourceKind.CHAT_EXPORT,
        source_id=raw.source_id,
        session_id=session_id,
        speaker=speaker,
        raw_speaker=raw_speaker,
        ts=normalize_ts(record.get("timestamp") or record.get("ts")),
        content=content,
        parse_status=status,
        meta={"channel": record.get("channel"), "is_group": bool(record.get("is_group"))},
    )
    return raw, [message]


def parse_product_knowledge(record: dict[str, Any]) -> tuple[RawRecord, list[StandardMessage]]:
    """商品 / 活动知识：结构化字段直接进 meta，正文作为 content。"""
    raw = RawRecord(
        source=SourceKind.PRODUCT_KNOWLEDGE,
        source_id=str(record.get("article_id") or record.get("id")),
        payload=dict(record),
    )
    answer = str(record.get("answer") or record.get("content") or "")
    message = StandardMessage(
        source=SourceKind.PRODUCT_KNOWLEDGE,
        source_id=raw.source_id,
        session_id=f"kb::{raw.source_id}",
        speaker="knowledge_base",
        ts=normalize_ts(record.get("updated_at") or record.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        content=answer,
        meta={
            "title": record.get("title"),
            "scene": record.get("scene"),
            "question": record.get("question"),
            "conditions": list(record.get("conditions") or []),
            "product_id": record.get("product_id"),
            "market": record.get("market"),
            "tags": list(record.get("tags") or []),
            "valid_from": record.get("valid_from"),
            "valid_to": record.get("valid_to"),
            # 价格字段单独标记：历史价格绝不能被当成长期知识
            "price": record.get("price"),
            "price_live_source": record.get("price_live_source"),
        },
    )
    return raw, [message]


def parse_asr_transcript(record: dict[str, Any]) -> tuple[RawRecord, list[StandardMessage]]:
    """直播 ASR 转写：按片段切段，保留时间戳与商品/场次关联。"""
    raw = RawRecord(
        source=SourceKind.LIVE_ASR,
        source_id=str(record.get("segment_id") or record.get("id")),
        payload=dict(record),
    )
    speaker, raw_speaker, status = _resolve_speaker(record.get("speaker"))
    start_seconds = float(record.get("start_s") or 0.0)
    base = normalize_ts(record.get("started_at") or datetime.now(timezone.utc).isoformat())
    message = StandardMessage(
        source=SourceKind.LIVE_ASR,
        source_id=raw.source_id,
        session_id=str(record.get("session_id") or record.get("live_id") or ""),
        speaker=speaker,
        raw_speaker=raw_speaker,
        ts=datetime.fromtimestamp(base.timestamp() + start_seconds, tz=timezone.utc),
        content=str(record.get("text") or ""),
        parse_status=status,
        meta={
            "start_s": start_seconds,
            "product_ref": record.get("product_ref"),
            "asset_id": record.get("asset_id"),
        },
    )
    return raw, [message]


_PARSERS: dict[SourceKind, Callable[[dict[str, Any]], tuple[RawRecord, list[StandardMessage]]]] = {
    SourceKind.CHAT_EXPORT: parse_chat_export,
    SourceKind.PRODUCT_KNOWLEDGE: parse_product_knowledge,
    SourceKind.LIVE_ASR: parse_asr_transcript,
}


class Ingestor:
    """按来源分派解析器。新增来源只需注册一个解析器，不改调用方。"""

    def __init__(self, parsers: dict[SourceKind, Callable[..., Any]] | None = None) -> None:
        self._parsers = dict(parsers or _PARSERS)

    def register(self, kind: SourceKind, parser: Callable[..., Any]) -> None:
        self._parsers[kind] = parser

    def parse(self, records: Sequence[dict[str, Any]]) -> tuple[list[RawRecord], list[StandardMessage]]:
        """逐一解析；单条解析失败**不影响整批**，失败条目计入 failed 计数并写告警日志。"""
        raws: list[RawRecord] = []
        messages: list[StandardMessage] = []
        failed = 0
        with stage("data.ingest", records=len(records)):
            for record in records:
                kind_value = record.get("source")
                try:
                    kind = SourceKind(kind_value)
                except ValueError:
                    failed += 1
                    logger.warning(
                        "data.ingest.unknown_source",
                        extra={"event": "data_ingest", "status": "error", "reason": str(kind_value)},
                    )
                    continue
                try:
                    raw, parsed = self._parsers[kind](record)
                except ValidationError as exc:
                    failed += 1
                    logger.warning(
                        "data.ingest.parse_failed",
                        extra={"event": "data_ingest", "status": "error", "error_code": exc.code},
                    )
                    continue
                raws.append(raw)
                messages.extend(parsed)

        partial = sum(1 for item in messages if item.parse_status is ParseStatus.PARTIAL)
        logger.info(
            "data.ingest.done",
            extra={
                "event": "data_ingest",
                "status": "ok",
                "reason": f"raw={len(raws)},messages={len(messages)},partial={partial},failed={failed}",
            },
        )
        return raws, messages
