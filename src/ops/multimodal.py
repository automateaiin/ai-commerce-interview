"""多模态素材知识化：OCR / VLM / ASR 端口 → retrieval_text → Embedding → rag_assets。

这一模块解决的问题（面试主线）：
运营侧的素材是**异构**的 —— 商品图、详情页、直播回放、切片、短视频，四种模态各自
无法直接进检索。做法是给每种模态配一个**抽取端口**，把它的内容统一成一段可检索文本
（`retrieval_text`），再走 infra 的 `Embedder` 向量化。这样「文本查询能命中图片/视频」，
检索链路对上层只有一种形态。

为什么要端口（而不是直接调云服务）：
OCR/VLM/ASR 都是**外部能力**，换供应商、离线降级、加缓存都发生在端口之后。
本仓库给的是离线实现（确定性、可复现），生产把端口换成真实服务即可，上层零改动。

幂等键为什么是 `(source_table, source_id)`：
同一条业务记录可能被流水线反复处理（重跑、补数据、消息重投）。若每次导入都新建一份资产，
检索库里会慢慢堆满**近重复副本**，召回被稀释、去重也失效。所以 upsert 命中已有资产时
**更新内容、保持 asset_id 不变**：下游引用 asset_id 的地方不会因为一次重跑而断链。

诚实边界：向量落在**进程内**（生产应落 pgvector）；离线哈希向量无语义泛化能力，
跨模态检索靠的是 retrieval_text 里的**同词命中**，不是真正的图文跨模态对齐 —— 见 PROJECT.md 第 8 章。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

from src.infra.core.errors import NotFoundError, ValidationError
from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage
from src.infra.rag.service import Embedder, HashingEmbedder

from .schemas import AssetHit, AssetModality, ContentAsset

logger = get_logger("ops.multimodal")


# --------------------------------------------------------------------------- #
# 抽取端口（离线实现 + 可替换真实服务）
# --------------------------------------------------------------------------- #
class OcrPort(Protocol):
    name: str

    async def extract(self, source: dict[str, Any]) -> str: ...


class VlmPort(Protocol):
    name: str

    async def describe(self, source: dict[str, Any]) -> str: ...


class AsrPort(Protocol):
    name: str

    async def transcribe(self, source: dict[str, Any], *, window: tuple[float, float] | None = None) -> str: ...


def _seeded_text(source: dict[str, Any], *, fallback: str) -> str:
    """离线抽取的统一兜底：优先用调用方给的文本，否则用标题/来源拼一段确定性文本。

    确定性很重要：同一条素材重复导入必须得到**同样的** retrieval_text，
    否则向量会漂移，upsert 的幂等也就名存实亡。
    """
    for key in ("text", "caption", "transcript", "title"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    label = source.get("title") or source.get("file") or source.get("source_id") or fallback
    tags = " ".join(str(item) for item in source.get("tags") or [])
    return f"{label} {tags}".strip() or fallback


class OfflineOcr:
    """离线 OCR：从 `text` 字段或标题生成确定性文本，并标注为离线等价实现。"""

    name = "offline_ocr"

    async def extract(self, source: dict[str, Any]) -> str:
        return f"[ocr:offline] {_seeded_text(source, fallback='页面文本')}"


class OfflineVlm:
    """离线 VLM：把图片/视频**描述**为文本（真实实现是视觉模型理解画面）。

    诚实边界：这里不真的看图，只是把调用方给的 caption/标题整理成可检索文本。
    """

    name = "offline_vlm"

    async def describe(self, source: dict[str, Any]) -> str:
        visual = source.get("caption") or source.get("visual") or source.get("product_name")
        subject = visual or source.get("title") or source.get("source_id") or "商品素材"
        return f"[vlm:offline] 画面主体：{subject}；用途：{source.get('usage') or '商品展示'}"


_ASR_SCRIPTS = (
    "开场介绍产品亮点",
    "讲解核心成分",
    "演示使用手法",
    "对比同类竞品",
    "回答评论区问题",
    "引导下单转化",
    "分享真实使用反馈",
    "强调限时权益",
)
"""离线 ASR 的确定性话术库：按桶轮换。

为什么要轮换而不是所有桶共用一句话：如果每段文本都是同一句话的复制，
那么「近重复去重」会把整段误判成重复而丢弃；轮换后只有在**真正重叠的桶**上
才会出现字面重复，去重逻辑检验的才是它该检验的东西。
"""


class OfflineAsr:
    """离线 ASR：直播/视频语音转写。支持按时间窗请求（直播分段转录复用同一端口）。

    实现要点：按 15s 全局网格分桶产出确定性短语。相邻窗口若共享某个桶
    （即窗口有重叠），该桶的文本会在两段里各出现一次 —— 这正是真实 ASR
    「重叠区被转录两遍」的等价物，供上层做重叠去重。
    """

    name = "offline_asr"
    bucket_s = 15

    async def transcribe(self, source: dict[str, Any], *, window: tuple[float, float] | None = None) -> str:
        base = _seeded_text(source, fallback="直播口播内容")
        if window is None:
            return f"[asr:offline] {base}"
        start, end = window
        first = int(start // self.bucket_s) * self.bucket_s
        phrases: list[str] = []
        bucket = first
        while bucket < end:
            if bucket + self.bucket_s > start:
                slot = (bucket // self.bucket_s) % len(_ASR_SCRIPTS)
                phrases.append(f"{bucket}秒{bucket + self.bucket_s}秒{_ASR_SCRIPTS[slot]}（{base}）")
            bucket += self.bucket_s
        return "".join(phrases)


# --------------------------------------------------------------------------- #
# 素材资产库
# --------------------------------------------------------------------------- #
def _asset_id(source_table: str, source_id: str) -> str:
    """asset_id 由幂等键派生 → 同一业务记录永远得到同一个 asset_id。"""
    digest = hashlib.blake2b(f"{source_table}:{source_id}".encode("utf-8"), digest_size=8).hexdigest()
    return f"asset_{digest}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _tokens(text: str) -> set[str]:
    compact = "".join(text.split()).lower()
    return {compact[index : index + 2] for index in range(len(compact) - 1)} or {compact}


class AssetStore:
    """多模态素材资产库（rag_assets 的进程内等价实现）。

    三件事：端口抽取 → 向量化 → 幂等 UPSERT。容量有上界（超限淘汰最旧），
    淘汰只影响检索面，不影响已导入记录的正确性。
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        ocr: OcrPort | None = None,
        vlm: VlmPort | None = None,
        asr: AsrPort | None = None,
        max_assets: int = 500,
        embed_version: str = "hash-256-v1",
    ) -> None:
        self._embedder = embedder or HashingEmbedder()
        self._ocr = ocr or OfflineOcr()
        self._vlm = vlm or OfflineVlm()
        self._asr = asr or OfflineAsr()
        self._max = max(1, int(max_assets))
        self._embed_version = embed_version
        # key = (source_table, source_id)；OrderedDict 保证淘汰顺序 = 写入顺序
        self._assets: OrderedDict[tuple[str, str], ContentAsset] = OrderedDict()
        self._vectors: dict[str, list[float]] = {}
        self._upserts = 0
        self._created = 0

    @property
    def size(self) -> int:
        return len(self._assets)

    def stats(self) -> dict[str, Any]:
        by_modality = {item.value: 0 for item in AssetModality}
        for asset in self._assets.values():
            by_modality[asset.modality.value] += 1
        return {
            "assets": self.size,
            "created": self._created,
            "upserts": self._upserts,
            "max_assets": self._max,
            "by_modality": by_modality,
            "embed_version": self._embed_version,
        }

    async def _extract(self, modality: AssetModality, source: dict[str, Any]) -> tuple[str, str]:
        """按模态分派到对应端口，返回 (retrieval_text, extractor_name)。"""
        if modality is AssetModality.PAGE:
            return await self._ocr.extract(source), self._ocr.name
        if modality is AssetModality.IMAGE:
            return await self._vlm.describe(source), self._vlm.name
        if modality is AssetModality.VIDEO:
            return await self._vlm.describe(source), self._vlm.name
        if modality is AssetModality.LIVE_CLIP:
            return await self._asr.transcribe(source), self._asr.name
        raise ValidationError(f"不支持的素材模态: {modality}", detail={"modality": str(modality)})

    async def upsert(
        self,
        *,
        source_table: str,
        source_id: str,
        modality: AssetModality | str,
        title: str = "",
        product_id: str | None = None,
        market: str | None = None,
        tags: Sequence[str] = (),
        source: dict[str, Any] | None = None,
        retrieval_text: str | None = None,
    ) -> tuple[ContentAsset, bool]:
        """导入一份素材。返回 (asset, created)。

        `created=False` 表示命中已有资产（幂等更新）：asset_id 不变、assets 计数不变、
        `updated_at` 前移；`created_at` 保持首次导入的时间，便于对账「这条素材什么时候进来的」。
        """
        if not source_table or not source_id:
            raise ValidationError("素材导入必须提供 source_table 与 source_id")
        mode = AssetModality(modality) if not isinstance(modality, AssetModality) else modality
        payload = dict(source or {})

        with_ = stage("ops.asset.extract", modality=mode.value)
        async with with_:
            if retrieval_text and retrieval_text.strip():
                text, extractor = retrieval_text.strip(), "caller_supplied"
            else:
                text, extractor = await self._extract(mode, {**payload, "title": title or payload.get("title")})

            if not text.strip():
                raise ValidationError("素材抽取结果为空，拒绝写入", detail={"source_id": source_id})

            vector = (await self._embedder.embed([text]))[0]

        key = (source_table, source_id)
        existing = self._assets.get(key)
        now = datetime.now(timezone.utc)
        if existing is not None:
            # 命中幂等键：更新内容、保持 asset_id 与 created_at 不变，仅前移 updated_at
            asset = existing.model_copy(
                update={
                    "modality": mode,
                    "retrieval_text": text,
                    "title": title or existing.title,
                    "product_id": product_id or existing.product_id,
                    "market": market or existing.market,
                    "tags": list(tags) or existing.tags,
                    "extractor": extractor,
                    "updated_at": now,
                }
            )
            self._assets.move_to_end(key)
            self._upserts += 1
            created = False
        else:
            asset = ContentAsset(
                asset_id=_asset_id(source_table, source_id),
                source_table=source_table,
                source_id=source_id,
                modality=mode,
                retrieval_text=text,
                title=title,
                product_id=product_id,
                market=market,
                tags=list(tags),
                extractor=extractor,
                embed_version=self._embed_version,
                created_at=now,
                updated_at=now,
            )
            self._assets[key] = asset
            self._created += 1
            created = True

        self._vectors[asset.asset_id] = vector
        self._evict()
        logger.info(
            "ops.asset.upsert",
            extra={
                "event": "ops_asset",
                "status": "ok" if created else "updated",
                "changed_keys": asset.asset_id,
                "reason": f"modality={mode.value},extractor={extractor}",
            },
        )
        return asset, created

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        product_id: str | None = None,
        market: str | None = None,
    ) -> list[AssetHit]:
        """跨模态检索：向量相似度 + 关键词命中，可选按商品/市场过滤。"""
        if not query.strip():
            raise ValidationError("素材检索 query 不能为空")
        if not self._assets:
            return []
        query_vector = (await self._embedder.embed([query]))[0]
        query_terms = {term for term in _tokens(query) if len(term) >= 2}

        hits: list[AssetHit] = []
        for asset in self._assets.values():
            if product_id and asset.product_id not in (None, product_id):
                continue
            if market and asset.market not in (None, market):
                continue
            vector = self._vectors.get(asset.asset_id)
            if vector is None:
                continue
            vector_score = _cosine(query_vector, vector)
            haystack = f"{asset.title} {asset.retrieval_text} {' '.join(asset.tags)}".lower()
            keyword_score = len({t for t in query_terms if t in haystack}) / max(len(query_terms), 1)
            score = 0.6 * vector_score + 0.4 * keyword_score
            hits.append(AssetHit(asset=asset, score=round(score, 4)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        logger.info(
            "ops.asset.search",
            extra={"event": "ops_asset", "status": "ok", "hits": len(hits[:top_k]), "top_k": top_k},
        )
        return hits[:top_k]

    def get(self, asset_id: str) -> ContentAsset:
        for asset in self._assets.values():
            if asset.asset_id == asset_id:
                return asset
        raise NotFoundError(f"素材不存在: {asset_id}", detail={"asset_id": asset_id})

    def _evict(self) -> None:
        while len(self._assets) > self._max:
            (table, source), evicted = self._assets.popitem(last=False)
            self._vectors.pop(evicted.asset_id, None)
            logger.info(
                "ops.asset.evict",
                extra={"event": "ops_asset", "status": "skipped", "reason": f"capacity:{table}:{source}"},
            )


__all__ = [
    "AsrPort",
    "AssetStore",
    "OfflineAsr",
    "OfflineOcr",
    "OfflineVlm",
    "OcrPort",
    "VlmPort",
]
