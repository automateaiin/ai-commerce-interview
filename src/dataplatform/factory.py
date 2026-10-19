"""数据平台装配入口：配置 → 组件。

与 harness/factory 同样的理由：数据层各组件（切片器/索引/检索器）参数全部来自配置，
工厂是唯一允许「配置字符串/数值 → 结构化组件」的地方，
这样单测可以只替换工厂入参，而不用改动算法本身。
"""

from __future__ import annotations

from src.infra.core.settings import Settings
from src.infra.rag.service import HashingEmbedder

from .chunking import Chunker
from .dedupe import Deduper
from .filters import FilterChain
from .index import IndexConfig, VectorIndex
from .ingest import Ingestor
from .pipeline import DataPipeline
from .rerank import build_reranker
from .retrieval import Retriever
from .scrub import Scrubber


def build_index(settings: Settings) -> VectorIndex:
    embedding_version = f"hash-{settings.data_embed_dim}-v1"
    return VectorIndex(
        embedder=HashingEmbedder(dim=settings.data_embed_dim),
        config=IndexConfig(embedding_version=embedding_version),
    )


def build_retriever(settings: Settings, index: VectorIndex) -> Retriever:
    return Retriever(
        index,
        rerank_candidates=settings.data_rerank_candidates,
        context_char_budget=settings.data_context_char_budget,
        reranker=build_reranker(settings),
        merge_ratio=settings.data_merge_ratio,
    )


def build_chunker(settings: Settings) -> Chunker:
    return Chunker(
        time_window_s=settings.data_chunk_time_window_s,
        max_tokens=settings.data_chunk_max_tokens,
        overlap_tokens=settings.data_chunk_overlap_tokens,
        mode=settings.data_chunk_mode,
        l1_window=settings.data_chunk_l1_window,
        l2_group=settings.data_chunk_l2_group,
        l3_group=settings.data_chunk_l3_group,
        merge_ratio=settings.data_merge_ratio,
    )


def build_pipeline(settings: Settings) -> DataPipeline:
    """按配置装配数据流水线。"""
    return DataPipeline(
        ingestor=Ingestor(),
        deduper=Deduper(threshold=0.42),
        scrubber=Scrubber(),
        filter_chain=FilterChain(),
        chunker=build_chunker(settings),
        index=build_index(settings),
        reranker=build_reranker(settings),
        merge_ratio=settings.data_merge_ratio,
    )


__all__ = ["build_chunker", "build_index", "build_pipeline", "build_retriever"]
