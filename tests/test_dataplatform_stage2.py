"""项目四（阶段 2）新增能力的分层单元测试。

覆盖四块新增：三级切片 + Auto-merging、重排端口、训练数据工程、回流与出题评分。
风格与 `test_dataplatform.py` 一致：**纯函数 → 组件 → 编排** 三层递进，全部离线可跑。
"""

from __future__ import annotations

from datetime import datetime, timezone

import yaml

import pytest

from src.dataplatform.chunking import (
    MODE_HIERARCHICAL,
    ChunkNode,
    Chunker,
    auto_merge,
    split_sentences,
)
from src.dataplatform.index import VectorIndex
from src.dataplatform.quiz import AutoGrader, Expect, QuizBank, QuizItem, QuizType
from src.dataplatform.reflux import RefluxCollector, RefluxSource
from src.dataplatform.rerank import LexicalReranker, build_reranker
from src.dataplatform.retrieval import Retriever
from src.dataplatform.schemas import (
    KnowledgeArticle,
    KnowledgeChunk,
    SamplePurpose,
    SourceKind,
    StagingConversation,
    StandardMessage,
    TrainingSample,
)
from src.dataplatform.training import DpoBuilder, build_llamafactory
from src.infra.core.settings import Settings
from src.infra.rag.service import HashingEmbedder

SENTENCES = [
    "雪绒花精华适合干性皮肤。",
    "雪绒花精华建议早晚各一次。",
    "雪绒花精华可叠加保湿面霜。",
    "雪绒花精华开封后请冷藏。",
    "雪绒花精华过敏体质先做耳后测试。",
    "雪绒花精华孕期使用前请咨询医生。",
]


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    level: int = 1,
    parent_id: str | None = None,
    index: int = 0,
    product_id: str | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        article_id=chunk_id.split(":")[0].split("#")[0],
        chunk_index=index,
        text=text,
        product_id=product_id,
        level=level,
        parent_id=parent_id,
    )


def make_sample(sample_id: str, question: str, answer: str) -> TrainingSample:
    return TrainingSample(
        sample_id=sample_id,
        messages=[{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        purpose=SamplePurpose.SFT_STYLE,
    )


def make_article(article_id: str, *, price_freshness: bool = False, conditions: list[str] | None = None) -> KnowledgeArticle:
    return KnowledgeArticle(
        article_id=article_id,
        title=f"{article_id} 使用说明",
        scene="product_consulting",
        question=f"{article_id} 怎么用？",
        answer=f"{article_id} 建议早晚各一次，敏感肌先做局部测试。",
        conditions=conditions or [],
        product_id="SKU-1001",
        price_freshness=price_freshness,
    )


# --------------------------------------------------------------------------- #
# 1. 三级切片 + Auto-merging
# --------------------------------------------------------------------------- #
def test_split_sentences_keeps_delimiters() -> None:
    sentences = split_sentences("第一句。第二句！第三句？")
    assert sentences == ["第一句。", "第二句！", "第三句？"]


def test_hierarchical_builds_three_levels_with_intact_lineage() -> None:
    chunker = Chunker(mode=MODE_HIERARCHICAL, l1_window=2, l2_group=3, l3_group=6, max_tokens=220)
    nodes = chunker.hierarchical("".join(SENTENCES), prefix="KB-T")

    assert {node.level for node in nodes} == {1, 2, 3}
    l1 = [node for node in nodes if node.level == 1]
    # 句子滑动窗口（step=1）：6 句 → 6 个 L1 窗口，相邻窗口有重叠
    assert len(l1) == len(SENTENCES)
    assert l1[0].text == SENTENCES[0] + SENTENCES[1]
    assert l1[1].text == SENTENCES[1] + SENTENCES[2]
    # 父子链必须都能在本次产出里被解析到，否则合并时会找不到父块
    ids = {node.node_id for node in nodes}
    assert all(node.parent_id in ids for node in nodes if node.parent_id)
    assert l1[0].parent_id == "KB-T:L2:000"


def test_hierarchical_enforces_token_cap_and_keeps_parent_anchor() -> None:
    long_text = "很长的句子没有标点" * 200  # 单句超长 → 必须被再切
    chunker = Chunker(mode=MODE_HIERARCHICAL, max_tokens=220, overlap_tokens=40)
    nodes = chunker.hierarchical(long_text, prefix="KB-L")

    assert nodes and all(node.tokens <= 220 for node in nodes)
    # 超限父块被切开时，首片必须保留原 node_id —— 否则子块的 parent_id 会指向不存在的节点
    assert any(node.node_id == "KB-L:L2:000" for node in nodes)
    assert all(node.parent_id is None or any(n.node_id == node.parent_id for n in nodes) for node in nodes)


def test_auto_merge_merges_only_when_coverage_reaches_ratio() -> None:
    nodes = [
        ChunkNode(node_id="p", level=2, text="父块正文"),
        ChunkNode(node_id="a", level=1, text="a", parent_id="p"),
        ChunkNode(node_id="b", level=1, text="b", parent_id="p"),
    ]
    merged, reasons = auto_merge(nodes, ["a", "b"], ratio=0.5)
    assert [node.node_id for node in merged] == ["p"]
    assert reasons and "合并" in reasons[0]

    sparse, _ = auto_merge(nodes, ["a"], ratio=0.6)
    assert [node.node_id for node in sparse] == ["a"]
    # 父块本身已被召回时不再合并（否则会把父块替换成自己）
    kept, _ = auto_merge(nodes, ["p", "a"], ratio=0.5)
    assert "p" in {node.node_id for node in kept}


async def test_retriever_merges_children_into_parent_context() -> None:
    index = VectorIndex(embedder=HashingEmbedder(dim=64))
    chunks = [
        make_chunk(f"A:L1:{position:03d}", SENTENCES[position] + SENTENCES[min(position + 1, len(SENTENCES) - 1)], index=position, parent_id=f"A:L2:{position // 3:03d}")
        for position in range(len(SENTENCES))
    ]
    for group in range(2):
        span = "".join(SENTENCES[group * 3 : group * 3 + 3])
        chunks.append(make_chunk(f"A:L2:{group:03d}", span, level=2, parent_id="A:L3:000", index=group))
    chunks.append(make_chunk("A:L3:000", "".join(SENTENCES), level=3))
    await index.upsert(chunks)

    report = await Retriever(index, rerank_candidates=10).search("雪绒花精华怎么用", top_k=3)

    assert report.merged_children >= 1
    assert any(item.chunk_id.startswith("A:L2:") for item in report.evidence)


async def test_semantic_mode_never_merges() -> None:
    index = VectorIndex(embedder=HashingEmbedder(dim=64))
    await index.upsert([make_chunk(f"B#c{position}", SENTENCES[position], index=position) for position in range(4)])

    report = await Retriever(index, rerank_candidates=10).search("雪绒花精华怎么用", top_k=3)

    assert report.merged_children == 0
    assert all(item.chunk_id.startswith("B#c") for item in report.evidence)


# --------------------------------------------------------------------------- #
# 2. 重排端口
# --------------------------------------------------------------------------- #
async def test_lexical_reranker_rewards_entity_and_penalizes_missing_entity() -> None:
    chunks = [
        make_chunk("c_hit", "适用于 SKU-1001 的精华，早晚各一次。"),
        make_chunk("c_miss", "一般护肤建议：注意保湿与防晒。"),
    ]
    for chunk in chunks:
        chunk.score = 0.5

    outcome = await LexicalReranker().score("SKU-1001 精华怎么用", chunks)

    assert outcome.backend == "lexical"
    assert outcome.scores["c_hit"] == pytest.approx(0.75)
    assert outcome.scores["c_miss"] == pytest.approx(0.35)
    assert any("缺少 query 实体" in reason for reason in outcome.reasons)


def test_build_reranker_degrades_to_local_when_key_missing() -> None:
    assert build_reranker(Settings()).backend == "lexical"
    assert build_reranker(Settings(data_rerank_backend="jina", data_jina_api_key=None)).backend == "lexical"


# --------------------------------------------------------------------------- #
# 3. 训练数据工程（DPO 偏好对 + LLaMA-Factory 产物）
# --------------------------------------------------------------------------- #
def make_staging(staging_id: str, question: str, answer: str, *, after: str | None = None) -> StagingConversation:
    ts = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    return StagingConversation(
        staging_id=staging_id,
        source=SourceKind.CHAT_EXPORT,
        session_id="sess_1",
        messages=[
            StandardMessage(source=SourceKind.CHAT_EXPORT, source_id=f"{staging_id}_q", session_id="sess_1", speaker="customer_bob", ts=ts, content=question),
            StandardMessage(source=SourceKind.CHAT_EXPORT, source_id=f"{staging_id}_a", session_id="sess_1", speaker="cs_agent", ts=ts, content=answer),
        ],
        after=after,
        scene="product_consulting",
    )


def test_dpo_builder_pairs_review_edits_and_rejects_useless_pairs() -> None:
    items = [
        make_staging("stg_1", "多久发货？", "亲，48 小时内发货哦。", after="亲，付款后 48 小时内发出（节假日顺延），可在订单页查看物流。"),
        make_staging("stg_2", "能退货吗？", "支持 7 天无理由。", after="支持 7 天无理由。"),  # 未改动 → 丢弃
        make_staging("stg_3", "敏感肌能用吗？", "可以的。"),  # 无 after → 不是编辑留痕
    ]

    pairs = DpoBuilder().from_reviews(items)

    assert len(pairs) == 1
    assert pairs[0].source == "review_edit"
    assert pairs[0].rejected == "亲，48 小时内发货哦。"
    assert pairs[0].chosen.startswith("亲，付款后")


def test_build_llamafactory_emits_consumable_dataset_and_config() -> None:
    samples = [make_sample("s1", "多久发货？", "付款后 48 小时内发出，节假日顺延。")]
    pairs = DpoBuilder().from_reviews(
        [make_staging("stg_1", "多久发货？", "尽快发货。", after="付款后 48 小时内发出，节假日顺延。")]
    )

    artifact = build_llamafactory(samples, pairs, dataset_name="beauty_cs", version="v1")

    # ranking: true 才会被 LLaMA-Factory 当作偏好数据集，这是可训练性的关键字段
    assert artifact.dataset_info["beauty_cs_dpo"]["ranking"] is True
    assert artifact.dataset_info["beauty_cs_dpo"]["columns"]["chosen"] == "chosen"
    config = yaml.safe_load(artifact.train_config_yaml)
    assert config["stage"] == "dpo" and config["dataset"] == "beauty_cs_dpo"
    assert set(artifact.files()) == {"beauty_cs_sft_v1.jsonl", "beauty_cs_dpo_v1.jsonl", "dataset_info.json", "train_beauty_cs.yaml"}
    assert artifact.stats["dpo_exported"] == 1 and artifact.stats["sft_exported"] == 1


def test_training_gate_rejects_short_and_sensitive_pairs() -> None:
    builder = DpoBuilder()
    pairs = [
        *builder.from_corrections([{"id": "ok", "prompt": "多久发货？", "chosen": "48 小时内发出。", "rejected": "很快发货。"}]),
        *builder.from_corrections([{"id": "bad", "prompt": "联系方式？", "chosen": "请拨打 13800138000 咨询。", "rejected": "回复很快。"}]),
    ]

    artifact = build_llamafactory([make_sample("s1", "多久发货？", "48 小时内发出，节假日顺延。")], pairs)

    reasons = {entry["id"]: entry["reason"] for entry in artifact.rejected}
    assert reasons.get("dpo_cor_bad") == "sensitive:phone_cn"
    assert artifact.stats["dpo_exported"] == 1


# --------------------------------------------------------------------------- #
# 4. 回流（五来源 → 训练 / 评测）
# --------------------------------------------------------------------------- #
def test_reflux_collects_all_five_sources_and_projects_them() -> None:
    collector = RefluxCollector()
    collector.add(RefluxSource.CORRECTION, id="c1", question="多久发货？", original="尽快发货。", corrected="付款后 48 小时内发出。", chunk_ids=["KB-1:L1:000"])
    collector.add(RefluxSource.REVIEW, id="r1", question="能退货吗？", before="支持 7 天。", after="支持 7 天无理由退货。")
    collector.add(RefluxSource.TOOL_CALL, id="t1", question="我的订单到哪了？", tool_name="get_order_status", result="已出库", chunk_ids=["KB-2:L1:000"])
    collector.add(RefluxSource.HITL, id="h1", question="敏感肌能用吗？", answer="可以。", approved=False, corrected="敏感肌需先做局部测试。", chunk_ids=["KB-3:L1:000"])
    collector.add(RefluxSource.LIVE_ASR, id="l1", question="雪绒花精华怎么用？", answer="早晚各一次，叠加面霜更好。", chunk_ids=["KB-4:L1:000"])
    dropped = collector.add(RefluxSource.CORRECTION, id="bad", question="缺少纠偏内容")

    assert dropped is None
    report = collector.report()
    assert report["by_source"] == {source.value: 1 for source in RefluxSource}
    assert report["dpo_pairs"] == 3  # 纠偏 / 审核编辑 / HITL 改写过
    assert any("missing" in reason for reason in report["dropped"])
    assert len(collector.to_training()) == 5
    assert len(collector.to_golden()) == 4
    assert all(item.group for item in collector.to_golden())


def test_reflux_review_without_change_yields_no_preference_pair() -> None:
    collector = RefluxCollector()
    collector.add(RefluxSource.REVIEW, id="r1", question="多久发货？", before="48 小时内发出。", after="48 小时内发出。")

    assert collector.to_dpo_pairs() == []
    assert len(collector.to_training()) == 1


# --------------------------------------------------------------------------- #
# 5. 出题与自动评分
# --------------------------------------------------------------------------- #
def test_quiz_bank_generates_three_question_types() -> None:
    articles = [
        make_article("KB-A", price_freshness=True, conditions=["敏感肌需先做局部测试", "孕期请咨询医生"]),
        make_article("KB-B"),
    ]
    chunks = [make_chunk(f"KB-A#c0", "正文", index=0), make_chunk("KB-B#c0", "正文", index=0)]

    items = QuizBank().generate(articles, chunks=chunks)

    kinds = {item.quiz_type for item in items}
    assert kinds == {QuizType.FACT, QuizType.BOUNDARY, QuizType.REFUSAL}
    refusal = next(item for item in items if item.quiz_type is QuizType.REFUSAL)
    assert refusal.expect is Expect.REFUSAL and refusal.gold_chunk_ids == ["KB-A#c0"]


def test_grader_enforces_refusal_strategy_and_grounded_scoring() -> None:
    grader = AutoGrader()
    refusal = _refusal_item()
    good = grader.grade(refusal, "价格以官方渠道实时页面为准，不承诺最低价")
    bad = grader.grade(refusal, "最低价 99 元，今天下单还能再减 10 元")
    assert good.verdict == "pass" and good.score == 1.0
    assert bad.verdict == "fail" and bad.score == 0.0

    items = QuizBank().generate([make_article("KB-A")], chunks=[make_chunk("KB-A#c0", "正文")])
    fact = next(item for item in items if item.quiz_type is QuizType.FACT)
    grounded = grader.grade(fact, fact.gold_answer, evidence_texts=[fact.gold_answer])
    off_track = grader.grade(fact, "这个我不清楚，您换个问题吧。", evidence_texts=[fact.gold_answer])
    assert grounded.verdict == "pass"
    assert off_track.verdict == "fail" and any("覆盖不足" in reason or "不一致" in reason for reason in off_track.reasons)


def _refusal_item() -> QuizItem:
    """拒答型题目在评分器中不依赖证据，构造一个最小实例即可。"""
    items = QuizBank().generate([make_article("KB-R", price_freshness=True)])
    return next(item for item in items if item.quiz_type is QuizType.REFUSAL)
