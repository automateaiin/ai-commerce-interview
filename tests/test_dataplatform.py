"""项目三数据平台：数据治理链路的分层单元测试。

分三层验证，和真实排障顺序一致：
1) **纯函数层**（去重度量、脱敏、切片、过滤规则、角色映射）—— 快速、可回归；
2) **组件层**（索引一致性、检索组装、评估、导出门禁、审核状态机）—— 验证契约而非实现；
3) **编排层**（在 fixtures 上跑整条流水线）—— 验证阶段之间的**顺序与耦合**没被改坏，
   例如「价格不进索引」「近重复删整轮」「脱敏在入库就生效」这类跨模块约束。

刻意不用真实密钥/网络：全部离线可跑，保证 CI 里稳定。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.dataplatform.chunking import Chunker, estimate_tokens
from src.dataplatform.dedupe import Deduper, content_hash, exact_key, jaccard
from src.dataplatform.evaluation import Evaluator, GoldenItem, grouped_split, leak_check
from src.dataplatform.export import Processor
from src.dataplatform.factory import build_pipeline
from src.dataplatform.filters import FilterChain
from src.dataplatform.fixtures import GOLDEN_ITEMS, RAW_RECORDS
from src.dataplatform.index import IndexConfig, VectorIndex
from src.dataplatform.ingest import Ingestor, normalize_ts
from src.dataplatform.retrieval import Retriever
from src.dataplatform.review import ReviewQueue
from src.dataplatform.routing import Router
from src.dataplatform.schemas import (
    Evidence,
    KnowledgeChunk,
    ParseStatus,
    SamplePurpose,
    SourceKind,
    StandardMessage,
    StagingConversation,
    TrainingSample,
    speaker_role,
)
from src.dataplatform.scrub import Scrubber
from src.infra.core.errors import ValidationError
from src.infra.core.settings import get_settings
from src.infra.rag.service import HashingEmbedder

TS = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)


def make_message(
    content: str,
    *,
    speaker: str | None = "customer_bob",
    session_id: str = "sess_1",
    source_id: str | None = None,
    parse_status: ParseStatus = ParseStatus.OK,
    ts: datetime = TS,
) -> StandardMessage:
    return StandardMessage(
        source=SourceKind.CHAT_EXPORT,
        source_id=source_id or f"m_{abs(hash(content)) % 10**6}",
        session_id=session_id,
        speaker=speaker,
        ts=ts,
        content=content,
        parse_status=parse_status,
    )


def make_staging(
    texts: list[str],
    *,
    staging_id: str = "stg_test",
    speakers: tuple[str, str] = ("customer_bob", "cs_agent_alice"),
) -> StagingConversation:
    messages = [make_message(text, speaker=speakers[index % 2]) for index, text in enumerate(texts)]
    return StagingConversation(
        staging_id=staging_id,
        source=SourceKind.CHAT_EXPORT,
        session_id="sess_1",
        messages=messages,
    )


def make_chunk(chunk_id: str, text: str, *, product_id: str = "SKU-1001") -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        article_id=f"art_{chunk_id}",
        chunk_index=0,
        text=text,
        product_id=product_id,
        scene="chat",
    )


def make_sample(sample_id: str, messages: list[dict[str, str]]) -> TrainingSample:
    return TrainingSample(
        sample_id=sample_id,
        messages=messages,
        purpose=SamplePurpose.SFT_STYLE,
        source_staging_ids=["stg_1"],
        scene="chat",
    )


def tiny_index() -> VectorIndex:
    return VectorIndex(embedder=HashingEmbedder(dim=64), config=IndexConfig())


# --------------------------------------------------------------------------- #
# 1. 纯函数层
# --------------------------------------------------------------------------- #
def test_speaker_role_maps_who_to_role_not_literal_name() -> None:
    """角色判定必须按「谁说的」映射，而不是拿 speaker 字面量去比 customer/user。

    这正是踩过的坑：speaker 实际是 customer_bob / cs_agent_alice，
    早期用 `speaker in ("customer","user")` 判定，导致所有消息都被当成 assistant
    （切片语义边界失效、SFT 样本 role_order 全被拒）。
    """
    assert speaker_role("customer_bob") == "user"
    assert speaker_role("cs_agent_alice") == "assistant"
    assert speaker_role("wxid_alice") == "assistant"  # 未映射的原始标识不能猜成客户
    assert speaker_role(None) == "assistant"


def test_bigram_jaccard_separates_rewrites_from_unrelated_text() -> None:
    """近重复用字符 bigram Jaccard：改写版相似度显著高于无关文本，阈值才站得住。

    （SimHash Hamming 实测区分度不足：相似对 17~18 bit、无关对 ~33 bit，窗口太窄。）
    """
    original = "雪绒花修护精华适合敏感肌使用，成分含积雪草与神经酰胺"
    rewrite = "雪绒花修护精华适合敏感肌使用，含积雪草与神经酰胺成分"
    unrelated = "物流一般三天发货，支持七天无理由退换"

    assert jaccard(original, original) == 1.0
    assert jaccard(original, rewrite) >= 0.42
    assert jaccard(original, unrelated) < 0.1


def test_exact_key_is_session_time_content_triple() -> None:
    """精确去重的键必须含 session + 时间 + 内容哈希：只按内容去重会误杀跨会话的同句问候。"""
    first = make_message("您好")
    same = make_message("您好")
    other_session = make_message("您好", session_id="sess_2")
    other_time = make_message("您好", ts=datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc))

    assert exact_key(first) == exact_key(same)
    assert exact_key(first) != exact_key(other_session)
    assert exact_key(first) != exact_key(other_time)
    assert content_hash("abc") == content_hash("abc")


def test_deduper_exact_removes_message_level_duplicates() -> None:
    messages = [make_message("物流多久发货？"), make_message("物流多久发货？"), make_message("支持退换吗？")]
    report = Deduper().exact(messages)
    assert report.exact_duplicates == 1
    assert len(report.kept) == 2


def test_deduper_near_is_unit_level_and_keeps_first_occurrence() -> None:
    """近重复在**语义单元级**判定：删的是「整轮」，不是半轮 —— 否则会把问答对拆成孤儿。"""
    units = [
        ("u1", "雪绒花修护精华适合敏感肌使用吗"),
        ("u2", "雪绒花修护精华适合敏感肌使用吗？"),
        ("u3", "物流一般几天能到"),
    ]
    kept, pairs = Deduper(threshold=0.42).near(units)
    assert kept == ["u1", "u3"]
    assert ("u1", "u2") in pairs


def test_scrubber_replaces_values_with_typed_placeholders() -> None:
    """脱敏：命中值替换成占位符，日志里只出现**命中类型**，不出现原值。"""
    raw = "联系方式 13800138000 或 bob@example.com，卡号 6222021234567890123，密钥 sk-abcdef123456"
    result = Scrubber().scrub(raw)

    assert "13800138000" not in result.text
    assert "bob@example.com" not in result.text
    assert "6222021234567890123" not in result.text
    assert "sk-abcdef123456" not in result.text
    assert {"email", "phone_cn", "bank_card", "api_key"} <= set(result.hits)
    # 19 位卡号不能被手机号规则二次命中（规则顺序敏感：卡号先于手机号）
    assert result.hits.get("phone_cn", 0) == 1


def test_normalize_ts_handles_iso_epoch_ms_and_datetime_but_rejects_garbage() -> None:
    """时间归一化要吃掉三种来源格式：ISO 串 / epoch（含毫秒）/ datetime —— 但垃圾输入直接报错，不猜。"""
    assert normalize_ts("2026-03-01T10:00:00Z") == TS
    assert normalize_ts(TS) == TS
    assert normalize_ts(int(TS.timestamp() * 1000)) == TS  # 毫秒
    assert normalize_ts(int(TS.timestamp())) == TS  # 秒
    with pytest.raises(ValidationError):
        normalize_ts("不是时间")


def test_split_turns_starts_new_turn_on_customer_message() -> None:
    """切片先用语义边界：客户消息开启新一轮，时间窗只做初筛兜底。"""
    messages = [
        make_message("物流多久发货？", speaker="customer_bob"),
        make_message("一般三天内发出", speaker="cs_agent_alice"),
        make_message("支持退换吗？", speaker="customer_bob"),
        make_message("支持七天无理由退换", speaker="cs_agent_alice"),
    ]
    turns = Chunker().split_turns(messages)
    assert len(turns) == 2
    assert turns[0].messages[0].content.startswith("物流")
    assert turns[1].messages[0].content.startswith("支持退换")


def test_split_text_respects_token_cap() -> None:
    """超长正文要按 token 上限切开，避免语义在切口处断裂。"""
    chunker = Chunker(max_tokens=40, overlap_tokens=8)
    text = "".join(f"第{i}条：本店承诺七天无理由退换，客服全程跟进。 " for i in range(40))
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    assert all(estimate_tokens(chunk) <= 40 for chunk in chunks)


# --------------------------------------------------------------------------- #
# 2. 组件层
# --------------------------------------------------------------------------- #
def test_ingest_unmapped_sender_keeps_raw_identity_and_marks_partial() -> None:
    """身份映射失败 → 标 partial 并保留原始标识，绝不猜一个身份出来（宁缺勿错）。"""
    records = [
        {
            "source": "chat_export",
            "id": "chat_unknown",
            "session_id": "sess_F",
            "sender": "wxid_ghost",
            "timestamp": "2026-03-01T10:00:00Z",
            "text": "帮我看看这个订单",
        }
    ]
    _, messages = Ingestor().parse(records)

    assert len(messages) == 1
    assert messages[0].parse_status is ParseStatus.PARTIAL
    assert messages[0].raw_speaker == "wxid_ghost"  # 原始标识保留给人工核对
    assert messages[0].speaker != "cs_agent_ghost"  # 不允许臆造成某个客服身份


def test_ingest_one_broken_record_does_not_break_the_batch() -> None:
    """单条脏数据只影响它自己：整批不能被一条异常记录带崩（否则重跑成本极高）。"""
    records = [
        {
            "source": "chat_export",
            "id": "chat_ok",
            "session_id": "sess_1",
            "sender": "wxid_bob",
            "timestamp": "2026-03-01T10:00:00Z",
            "text": "物流多久发货？",
        },
        {"source": "chat_export", "id": "chat_broken", "session_id": "sess_1"},  # 缺 sender / timestamp
    ]
    _, messages = Ingestor().parse(records)

    assert [message.content for message in messages] == ["物流多久发货？"]


def test_filter_chain_reports_reason_per_rule_and_routes_uncertain_to_review() -> None:
    """过滤链每条规则都要给得出原因；不确定的一律转人工审核，而不是自动采信或直接删。"""
    messages = [
        make_message("物流多久发货？", source_id="ok"),
        make_message("嗯", source_id="short"),
        make_message("已加入群聊", source_id="sys"),
        make_message("本店最低价 加微信 私聊下单 扫码进群", source_id="ad"),
        make_message("正常文本\ufffd乱码", source_id="moji"),
        make_message("身份未知的一条", speaker="wxid_ghost", parse_status=ParseStatus.PARTIAL, source_id="review"),
    ]
    report = FilterChain().run(messages)

    assert len(report.kept) == 1
    assert {"too_short", "system_notice", "ad_noise", "mojibake"} <= set(report.rule_hits)
    assert len(report.review_flagged) == 1
    assert len(report.rejected) == 4
    # 每条被拒都带 rule_id / reason，便于统计误删率
    assert all(decision.reason for _, decision in report.rejected)
    assert report.hit_rate("mojibake", len(messages)) == round(1 / len(messages), 4)


def test_review_reject_requires_reason_and_edit_keeps_before_after() -> None:
    """驳回必须给理由；编辑留痕 before/after，且编辑后仍需再次通过才生效。"""
    queue = ReviewQueue()
    item = make_staging(["物流多久发货？", "一般三天内发出"])
    queue.add(item)

    with pytest.raises(ValidationError):
        queue.reject(item.staging_id, reviewer="alice", reason="")

    queue.reject(item.staging_id, reviewer="alice", reason="答非所问")
    assert item.reason == "答非所问"

    queue.add(item)
    queue.edit(item.staging_id, reviewer="alice", new_text="一般三到五天发出", reason="口径修正")
    assert item.before == "物流多久发货？\n一般三天内发出"
    assert item.after == "一般三到五天发出"
    assert item.status.value == "pending"  # 编辑不等于放行

    queue.approve(item.staging_id, reviewer="bob")
    assert queue.stats()["approved"] >= 1


def test_router_excludes_price_from_index_and_keeps_live_source() -> None:
    """价格是时效性事实：不进语义索引（避免检索到过期价），只保留结构化引用 + live_source。"""
    staging = make_staging(["现在多少钱？", "活动价 199 元，立减 50"])
    staging.product_id = "SKU-1001"
    # 结构化字段挂在消息 meta 上（Router 从 messages[0].meta 读来源元数据）
    staging.messages[0].meta = {"price_live_source": "pricing-api:/sku/SKU-1001/price", "question": "这款多少钱？"}

    articles, chunks, samples, _ = Router().route([staging])

    assert len(articles) == 1
    assert articles[0].price_freshness is True
    assert articles[0].live_source == "pricing-api:/sku/SKU-1001/price"
    assert chunks == []
    assert not samples


def test_router_routes_factual_turn_to_article_and_style_turn_to_sft() -> None:
    """用途分流：可引用的事实（成分/功效）进 RAG；稳定的表达风格（口径话术）进 SFT。"""
    factual = make_staging(["这款精华成分安全吗？", "成分含积雪草与神经酰胺，适合敏感肌，功效是修护屏障"])
    style = make_staging(["送货快吗？", "亲，一般三天内送到哦，放心啦～"])

    articles, chunks, samples, _ = Router().route([factual, style])

    assert len(articles) == 1
    assert chunks and chunks[0].article_id == articles[0].article_id
    assert len(samples) == 1
    assert samples[0].purpose is SamplePurpose.SFT_STYLE
    assert [message["role"] for message in samples[0].messages] == ["user", "assistant"]


async def test_index_tracks_consistency_and_deletes_to_avoid_orphan_vectors() -> None:
    """一致性核对：主表与向量必须对得上；下架内容要连向量一起删，否则留下检不到也用不上的孤儿。"""
    index = tiny_index()
    chunk = make_chunk("c1", "雪绒花修护精华适合敏感肌使用")
    await index.upsert([chunk])

    assert index.size == 1
    assert index.consistency() == {"items": 1, "vectors": 1, "orphan_vectors": 0, "missing_vectors": 0}

    index.delete(["c1"])
    # 主表与向量同步归零：没有只删一半导致的孤儿向量
    assert index.consistency() == {"items": 0, "vectors": 0, "orphan_vectors": 0, "missing_vectors": 0}


async def test_search_applies_metadata_filter_before_ann() -> None:
    """过滤在前、ANN 在后：先把范围收敛到该商品的切片，再做向量召回。"""
    index = tiny_index()
    await index.upsert(
        [
            make_chunk("c1", "雪绒花修护精华适合敏感肌使用", product_id="SKU-1001"),
            make_chunk("c2", "雪绒花修护精华适合敏感肌使用", product_id="SKU-2001"),
        ]
    )
    report = await Retriever(index).search("雪绒花修护精华敏感肌", top_k=5, product_id="SKU-1001")

    assert report.evidence
    assert all(item.chunk_id == "c1" for item in report.evidence)
    assert report.filtered_metadata >= 1  # SKU-2001 的切片在召回前就被元数据预过滤挡掉了


def test_assemble_dedupes_and_truncates_with_explicit_flag() -> None:
    """上下文组装要有预算意识：去重、按字符预算裁剪，并且**显式**告知被截断。"""
    blocks = [
        Evidence(chunk_id="c1", article_id="a1", text="x" * 60, source="vector", score=0.9),
        Evidence(chunk_id="c1", article_id="a1", text="x" * 60, source="keyword", score=0.5),
        Evidence(chunk_id="c2", article_id="a2", text="y" * 60, source="vector", score=0.8),
    ]
    roomy = Retriever(tiny_index(), context_char_budget=500).assemble(blocks)
    tight = Retriever(tiny_index(), context_char_budget=100).assemble(blocks)

    assert [item.chunk_id for item in roomy.blocks] == ["c1", "c2"]
    assert roomy.dropped_duplicates == 1
    assert roomy.truncated is False
    assert [item.chunk_id for item in tight.blocks] == ["c1"]
    assert tight.truncated is True


def test_grouped_split_never_leaks_a_group_across_splits() -> None:
    """分组切分：同一 group（同一会话/商品）必须整体进同一个 split，否则评估会虚高。"""
    items = [GoldenItem(question=f"q{i}", gold_chunk_ids=[f"c{i}"], group=f"g{i // 2}") for i in range(6)]
    splits = grouped_split(items)

    groups = {name: {item.group for item in entries} for name, entries in splits.items()}
    assert groups["train"] == {"g0"}
    assert groups["valid"] == {"g1"}
    assert groups["test"] == {"g2"}
    assert sum(len(entries) for entries in splits.values()) == len(items)

    # 分组数不足时如实返回空集，而不是复制其它 split 来撑数（自欺的泄漏）
    thin = grouped_split([GoldenItem(question="q", gold_chunk_ids=["c"], group="g0")])
    assert thin["test"] == []


def test_leak_check_flags_near_duplicate_questions_across_splits() -> None:
    """泄漏自检：训练集与验证集里出现近乎同题的问句，等于把答案泄给了验证集。"""
    splits = {
        "train": [GoldenItem(question="雪绒花修护精华适合敏感肌使用吗", gold_chunk_ids=["c1"], group="g1")],
        "valid": [GoldenItem(question="雪绒花修护精华适合敏感肌使用吗？", gold_chunk_ids=["c2"], group="g2")],
    }
    assert leak_check(splits) == ["train~valid"]
    assert leak_check({"train": splits["train"]}) == []


def test_export_gate_rejects_each_reason_and_keeps_counts() -> None:
    """导出门禁：不合规样本拒收并给原因；「审核通过」不等于「每条都能训练」。"""
    valid = make_sample("s_valid", [{"role": "user", "content": "物流多久发货"}, {"role": "assistant", "content": "三天内"}])
    too_short = make_sample("s_short", [{"role": "user", "content": "在吗"}])
    wrong_order = make_sample("s_order", [{"role": "assistant", "content": "您好"}, {"role": "user", "content": "你好"}])
    duplicate = make_sample("s_dup", [{"role": "user", "content": "物流多久发货"}, {"role": "assistant", "content": "三天内"}])
    sensitive = make_sample(
        "s_sensitive",
        [{"role": "user", "content": "怎么联系"}, {"role": "assistant", "content": "请拨打 13800138000"}],
    )

    artifact = Processor(version="t1").export([valid, too_short, wrong_order, duplicate, sensitive])

    assert artifact.stats["exported"] == 1
    assert artifact.stats["rejected"] == 4
    assert {issue.reason for issue in artifact.rejected} == {"too_short", "role_order", "duplicate", "sensitive"}


def test_export_converts_fields_per_target_framework() -> None:
    """不同训练框架要按目标结构转换字段与角色，而不是改个文件后缀。"""
    sample = make_sample("s1", [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}])
    processor = Processor(system_prompt="你是电商客服")

    sharegpt = processor.export([sample], fmt="sharegpt").records[0]
    assert sharegpt["conversations"][0] == {"from": "system", "value": "你是电商客服"}
    assert [message["from"] for message in sharegpt["conversations"][1:]] == ["human", "gpt"]

    alpaca = processor.export([sample], fmt="alpaca").records[0]
    assert alpaca["instruction"] == "问" and alpaca["output"] == "答"

    openai = processor.export([sample], fmt="openai").records[0]
    assert [message["role"] for message in openai["messages"]] == ["user", "assistant"]


# --------------------------------------------------------------------------- #
# 3. 编排层：在 fixtures 上跑整条流水线
# --------------------------------------------------------------------------- #
async def test_pipeline_returns_structured_result_and_error_distinguishes_stages() -> None:
    """一次编排返回结构化结果：不只看最终答案，还要能按阶段定位问题。"""
    pipeline = build_pipeline(get_settings())
    result = await pipeline.run(RAW_RECORDS, reviewer="auto_policy", export_format="sharegpt", export_batch="t1")

    assert {"ingest", "dedupe", "scrub", "filter", "review", "routing", "index", "lineage"} <= set(result.report)
    assert result.dataset.raw and result.dataset.messages and result.dataset.staging
    assert result.dataset.articles and result.dataset.chunks and result.dataset.sft_samples
    assert result.retriever is not None and result.export is not None


async def test_pipeline_end_to_end_enforces_cross_module_invariants() -> None:
    """整条流水线的价值在于**阶段顺序**：脱敏先于导出、去重先于审核、价格不进索引。

    这里只断言跨模块约束与统计口径，不把断言绑死在易变的文案上。
    """
    pipeline = build_pipeline(get_settings())
    result = await pipeline.run(RAW_RECORDS, reviewer="auto_policy", export_format="sharegpt", export_batch="t1")

    report = result.report
    dataset = result.dataset

    # 接入：三类来源混排都能解析
    assert report["ingest"]["raw"] >= 10

    # 精确去重在消息级；近重复在语义单元级删「整轮」——被删单元不会以半轮形式残留
    assert report["dedupe"]["exact_duplicates"] >= 1
    assert report["dedupe"]["near_duplicates"] >= 1
    queued = {item.staging_id for item in pipeline.review.all()}
    removed = [pair[1] for pair in report["dedupe"]["near_duplicate_pairs"]]
    assert removed and all(staging_id not in queued for staging_id in removed)

    # 过滤：格式无效的直接剔除，不确定的进人工审核
    assert report["filter"]["rejected"] >= 3
    assert report["review"]["pending"] >= 1
    assert report["scrub"]["total"] >= 1

    # 索引一致性：向量数与切片数一致、无孤儿
    assert report["index"]["items"] == len(dataset.chunks)
    assert report["index"]["orphan_vectors"] == 0

    # 价格时效治理：价格条目一律不进语义索引
    indexed_articles = {chunk.article_id for chunk in dataset.chunks}
    price_articles = [article for article in dataset.articles if article.price_freshness]
    assert price_articles
    assert all(article.article_id not in indexed_articles for article in price_articles)

    # 训练导出：门禁生效，且脱敏在入库阶段已把明文个人信息清掉（双保险）
    exported = Processor().export(dataset.sft_samples, fmt="sharegpt")
    assert exported.stats["exported"] >= 1
    assert exported.stats["rejected"] >= 1
    assert "13800138000" not in exported.to_jsonl()

    # 血缘：能从切片反向追到原始记录
    chain = result.lineage.trace(f"chunk::{dataset.chunks[0].chunk_id}")["chain"]
    assert any(node.startswith("raw::") for node in chain)

    # 评估：召回与排序在标准集上达标
    metrics = await Evaluator(result.retriever).evaluate([GoldenItem(**item) for item in GOLDEN_ITEMS], k=3)
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
