"""阶段 5（项目二 · AI 运营系统）专项用例。

分组对应 PROJECT.md 6.2 的九条验收：
A 图定义与拓扑   B 续跑/回滚/回放   C 结构化输出（重试/兜底/接地）   D 模型分工降本
E 并行实测        F SSE 流式与断开释放   G 多模态素材幂等与跨模态检索
H 直播分段/去重/FFmpeg 降级   I 运营分析与归因   J 自进化闭环   K 接口层端到端
"""

from __future__ import annotations

import json

import pytest

from src.infra.core.errors import ValidationError
from src.infra.core.settings import Settings
from src.infra.llm.types import LLMResponse, Usage
from src.infra.prompts.registry import build_default_registry
from src.infra.rag.service import HashingEmbedder, KnowledgeService
from src.ops.analytics import OpsAnalytics
from src.ops.factory import build_ops_service
from src.ops.live import LiveProcessor, dedupe_overlaps
from src.ops.multimodal import AssetStore, OfflineAsr
from src.ops.parallel import ParallelRunner, ParallelUnit
from src.ops.schemas import AssetModality, ContentBrief
from src.ops.selfevolve import SelfEvolveLoop
from src.ops.workflow.graph import (
    Node,
    NodeContext,
    WorkflowDefinitionError,
    WorkflowGraph,
    WorkflowRunner,
)
from src.ops.workflow.checkpointer import Checkpointer
from src.ops.workflow.nodes import build_content_graph, topic_node
from src.ops.workflow.schemas import TopicPlan

from tests.conftest import CS_KEY

OPS_KEY = "sk-project-ops-demo"

BRIEF = ContentBrief(
    product_id="SKU-1002",
    product_name="烟酰胺精华",
    market="cn",
    audience="25-35 岁成分党",
    channel="xiaohongshu",
    keywords=["烟酰胺", "提亮", "屏障修护"],
    tone="专业但不端着",
)


# --------------------------------------------------------------------------- #
# 测试替身：可脚本化网关（无 emitter 走 complete，有 emitter 走 stream）
# --------------------------------------------------------------------------- #
class StubGateway:
    """按脚本逐次返回文本的假网关；脚本耗尽后返回最后一个。"""

    def __init__(self, script: list[str] | None = None, *, default: str = "not-json-at-all") -> None:
        self.script = list(script or [])
        self.default = default
        self.calls: list[dict] = []

    def _next(self) -> str:
        return self.script.pop(0) if self.script else self.default

    async def complete(self, messages, *, logical_model=None, temperature=None, **kwargs) -> LLMResponse:
        self.calls.append({"model": logical_model, "messages": messages, "stream": False})
        return LLMResponse(
            text=self._next(),
            usage=Usage(prompt_tokens=11, completion_tokens=7, cost_usd=0.002),
            finish_reason="stop",
        )

    async def stream(self, messages, *, logical_model=None, temperature=None, **kwargs):
        from src.infra.llm.types import Chunk

        self.calls.append({"model": logical_model, "messages": messages, "stream": True})
        text = self._next()
        for index in range(0, len(text), 8):
            yield Chunk(delta_text=text[index : index + 8])
        yield Chunk(finish_reason="stop", usage=Usage(prompt_tokens=11, completion_tokens=7, cost_usd=0.002))


def _registry(settings: Settings | None = None):
    settings = settings or Settings()
    return build_default_registry(settings.resolved_prompts_config_path())


def _ctx(gateway, *, prompts=None, knowledge=None, brief=BRIEF, run_id="run_unit", node="topic", model="chat-pro", **extra):
    settings = extra.pop("settings", None) or Settings()
    return NodeContext(
        gateway=gateway,
        knowledge=knowledge or KnowledgeService(embedder=HashingEmbedder()),
        prompts=prompts or _registry(settings),
        brief=brief,
        run_id=run_id,
        parallel=extra.pop("parallel", None),
        settings=settings,
        emitter=extra.pop("emitter", None),
        node=node,
        logical_model=model,
    )


def _outline(gateway, *, prompts=None, settings=None):
    """跑完整内容图（同步模式）。"""
    settings = settings or Settings()
    prompts = prompts or _registry(settings)
    graph = build_content_graph(settings)
    runner = WorkflowRunner(
        graph=graph,
        checkpointer=Checkpointer(),
        ctx_template=_ctx(gateway, prompts=prompts, settings=settings),
        max_attempts=settings.ops_max_attempts,
        structured_retries=settings.ops_structured_retries,
    )
    return runner, "run_outline"


def _topic_json() -> str:
    return json.dumps({"topics": ["屏障修护场景种草", "熬夜党提亮"], "rationale": "卖点与人群匹配"}, ensure_ascii=False)


def _points_json(evidence="ev-1") -> str:
    return json.dumps(
        {
            "points": [
                {"claim": "烟酰胺帮助提亮肤色", "evidence": [evidence], "compliance": "ok"},
            ]
        },
        ensure_ascii=False,
    )


# =========================================================================== #
# A. 图定义与拓扑
# =========================================================================== #
def test_content_graph_topology_puts_copy_and_image_in_same_layer():
    graph = build_content_graph(Settings())
    graph.validate()  # 未知依赖 / 成环都会被挡在跑之前
    layers = [[node.name for node in layer] for layer in graph.layers()]
    assert layers == [["topic"], ["selling_points"], ["copy", "image"]]
    copy_node = graph.get("copy")
    image_node = graph.get("image")
    assert copy_node.depends_on == ("selling_points",)
    assert image_node.depends_on == ("selling_points",)
    assert image_node.optional is True


def test_content_graph_model_split_uses_catalog_logical_models():
    settings = Settings()
    graph = build_content_graph(settings)
    assert graph.get("topic").logical_model == settings.ops_analysis_model
    assert graph.get("selling_points").logical_model == settings.ops_analysis_model
    assert graph.get("copy").logical_model == settings.ops_copy_model
    assert graph.get("image").logical_model == settings.ops_copy_model


def test_graph_detects_unknown_dependency_and_cycle():
    graph = WorkflowGraph("bad")
    graph.add(Node("a", handler=_noop, depends_on=("ghost",)))
    with pytest.raises(WorkflowDefinitionError):
        graph.validate()

    cycle = WorkflowGraph("cycle")
    cycle.add(Node("x", handler=_noop, depends_on=("y",)))
    cycle.add(Node("y", handler=_noop, depends_on=("x",)))
    with pytest.raises(WorkflowDefinitionError):
        cycle.validate()


def test_graph_descendants_cover_transitive_dependents():
    graph = build_content_graph(Settings())
    assert graph.descendants("selling_points") == {"copy", "image"}
    assert graph.descendants("copy") == set()
    assert graph.upstream("copy") == {"topic", "selling_points"}


async def _noop(inputs, ctx):  # pragma: no cover - 仅用于图定义校验
    return {}


# =========================================================================== #
# B. 续跑 / 回滚 / 回放
# =========================================================================== #
async def test_run_records_every_node_and_marks_offline_fallbacks():
    gateway = StubGateway(default="[mock:chat] 这不是 JSON")
    runner, run_id = _outline(gateway)
    result = await runner.run(run_id)

    assert result.status == "succeeded"
    assert [record.node for record in result.records] == ["topic", "selling_points", "copy", "image"]
    # 离线默认答案不是 JSON → 四个节点都走确定性兜底，且原因被如实记录
    assert result.artifacts["fallbacks"]  # 原因被记录，不是静默兜底
    assert {item["node"] for item in result.artifacts["fallbacks"]} == {"topic", "selling_points", "copy", "image"}
    assert result.artifacts["usage"]["total_tokens"] == 4 * 2 * 18  # 每节点 2 次尝试（首次 + 结构化重试）


async def test_rollback_invalidates_node_and_downstream_then_resume_completes():
    settings = Settings()
    graph = build_content_graph(settings)
    checkpointer = Checkpointer()
    run_id = "run_resume"

    def _runner(gateway):
        return WorkflowRunner(
            graph=graph,
            checkpointer=checkpointer,
            ctx_template=_ctx(gateway, settings=settings),
            max_attempts=settings.ops_max_attempts,
            structured_retries=settings.ops_structured_retries,
        )

    first_gateway = StubGateway(default="[mock] 非 JSON")
    first = await _runner(first_gateway).run(run_id)
    assert first.status == "succeeded"
    assert len(first_gateway.calls) == 4 * 2  # 四节点 × (首次 + 结构化重试)

    # 回滚卖点及其下游：只失效单节点会让下游拿着旧输出继续用（版本混用）
    invalidated = checkpointer.rollback(run_id, graph.descendants("selling_points") | {"selling_points"}, reason="unit")
    assert set(invalidated) == {"selling_points", "copy", "image"}

    resume_gateway = StubGateway(default="[mock] 非 JSON")
    resumed = await _runner(resume_gateway).run(run_id, resume=True)
    assert resumed.status == "succeeded"
    assert resumed.resumed_from == ["topic"]
    # 只重跑被失效的三个节点（3 × 2 次调用），已完成节点没有重跑
    assert len(resume_gateway.calls) == 3 * 2
    rollback_marks = [item for item in checkpointer.replay(run_id) if item["status"] == "rolled_back"]
    assert {item["node"] for item in rollback_marks} == {"selling_points", "copy", "image"}


async def test_only_selected_node_reruns_when_downstream_untouched():
    settings = Settings()
    graph = build_content_graph(settings)
    checkpointer = Checkpointer()
    run_id = "run_partial"

    def _runner(gateway):
        return WorkflowRunner(graph=graph, checkpointer=checkpointer, ctx_template=_ctx(gateway, settings=settings))

    first_gateway = StubGateway(default="[mock] 非 JSON")
    await _runner(first_gateway).run(run_id)
    checkpointer.rollback(run_id, {"image"}, reason="regen image only")

    resume_gateway = StubGateway(default="[mock] 非 JSON")
    resumed = await _runner(resume_gateway).run(run_id, resume=True)
    assert sorted(resumed.resumed_from) == ["copy", "selling_points", "topic"]
    assert len(resume_gateway.calls) == 2  # 只有 image 重跑（同层其它节点不受牵连）


def test_checkpointer_snapshot_reports_missing_run():
    snapshot = Checkpointer().snapshot("nope")
    assert snapshot["found"] is False


# =========================================================================== #
# C. 结构化输出：重试 / 兜底 / 接地
# =========================================================================== #
async def test_structured_output_retries_with_reason_and_then_succeeds():
    gateway = StubGateway(["[mock] 忘记返回 JSON", _topic_json()])
    ctx = _ctx(gateway, settings=Settings())
    outcome = await topic_node({}, ctx)

    assert "_fallback_reason" not in outcome
    assert outcome["topics"] == ["屏障修护场景种草", "熬夜党提亮"]
    assert len(gateway.calls) == 2  # 第一次不合法 → 带失败原因重问一次
    assert gateway.calls[1]["messages"][2]["role"] == "assistant"  # 回填了上一次的原文
    assert "不合法" in gateway.calls[1]["messages"][3]["content"]
    assert outcome["usage"]["prompt_tokens"] == 22  # 两次调用的用量都计入成本


async def test_structured_output_falls_back_deterministically_when_retries_exhausted():
    gateway = StubGateway(default="[mock] 永远不是 JSON")
    ctx = _ctx(gateway, settings=Settings())
    outcome = await topic_node({}, ctx)

    assert outcome["_fallback_reason"]
    assert outcome["topics"]  # 兜底仍产出可用结构，不是空壳
    assert "离线兜底" in outcome["rationale"]
    assert len(gateway.calls) == Settings().ops_structured_retries + 1


async def test_selling_points_enforce_grounding_drops_unretrievable_evidence():
    gateway = StubGateway([json.dumps({"topics": ["t"], "rationale": "r"}, ensure_ascii=False), _points_json("chunk_fake")])
    settings = Settings()
    graph = build_content_graph(settings)
    runner = WorkflowRunner(graph=graph, checkpointer=Checkpointer(), ctx_template=_ctx(gateway, settings=settings))
    result = await runner.run("run_grounding")

    points = result.artifacts["selling_points"]
    assert points["ungrounded_count"] == 1
    assert points["points"][0]["evidence"] == []  # 引用不到的 chunk_id 被当场抹掉
    assert points["points"][0]["compliance"] == "ungrounded"


async def test_selling_points_keeps_grounded_evidence():
    knowledge = KnowledgeService(embedder=HashingEmbedder())
    found = await knowledge.search("烟酰胺 提亮", top_k=3)
    assert found.chunks  # 知识库里有可引用的切片
    real_id = found.chunks[0].chunk_id

    gateway = StubGateway(
        [json.dumps({"topics": ["t"], "rationale": "r"}, ensure_ascii=False), _points_json(real_id)]
    )
    settings = Settings()
    graph = build_content_graph(settings)
    runner = WorkflowRunner(
        graph=graph, checkpointer=Checkpointer(), ctx_template=_ctx(gateway, knowledge=knowledge, settings=settings)
    )
    result = await runner.run("run_grounded_ok")
    points = result.artifacts["selling_points"]
    assert points["ungrounded_count"] == 0
    assert points["points"][0]["compliance"] == "ok"
    assert points["points"][0]["evidence"] == [real_id]


# =========================================================================== #
# D. 模型分工降本（在真实 run 里可观测）
# =========================================================================== #
async def test_artifacts_record_both_logical_models_used_in_one_run():
    gateway = StubGateway(default="[mock] 非 JSON")
    runner, run_id = _outline(gateway)
    result = await runner.run(run_id)
    assert result.artifacts["models"] == ["chat-lite", "chat-pro"]
    used = {call["model"] for call in gateway.calls}
    assert used == {"chat-pro", "chat-lite"}


# =========================================================================== #
# E. 并行实测
# =========================================================================== #
async def test_parallel_runner_reports_measured_speedup_and_peak():
    runner = ParallelRunner(max_concurrency=2)
    units = [
        ParallelUnit(name=f"u{i}", fn=lambda: __import__("time").sleep(0.05), args=(), mode="thread")
        for i in range(4)
    ]
    report = await runner.run(units)
    assert report.peak_concurrency == 2  # 并发度受信号量约束，不是无上限 fan-out
    assert report.max_concurrency == 2
    assert report.wall_ms < report.sequential_ms
    assert report.speedup > 1.5
    assert report.failures == []


async def test_parallel_runner_does_not_cancel_siblings_on_failure():
    runner = ParallelRunner(max_concurrency=3)

    def _boom():
        raise RuntimeError("intentional")

    units = [
        ParallelUnit(name="ok1", fn=lambda: 1, args=(), mode="thread"),
        ParallelUnit(name="bad", fn=_boom, args=(), mode="thread"),
        ParallelUnit(name="ok2", fn=lambda: 2, args=(), mode="thread"),
    ]
    report = await runner.run(units)
    assert report.failures == ["bad"]  # 一个失败不被当作整批失败
    assert sorted(item.value for item in report.outcomes if item.ok) == [1, 2]
    assert report.outcomes[1].error.startswith("RuntimeError")


async def test_parallel_runner_rejects_wrong_batch_dependency():
    runner = ParallelRunner()
    with pytest.raises(ValidationError):
        await runner.run([ParallelUnit(name="a", fn=lambda: 1, args=(), depends_on=("ghost",))])
    with pytest.raises(ValidationError):
        await runner.run(
            [
                ParallelUnit(name="b", fn=lambda: 1, args=(), depends_on=("a",)),
                ParallelUnit(name="a", fn=lambda: 1, args=()),
            ]
        )


# =========================================================================== #
# F. SSE 流式 + 客户端断开释放
# =========================================================================== #
async def test_stream_emits_ordered_events_and_run_done(app_client):
    async with app_client() as (_client, container):
        frames = [frame async for frame in container.ops.stream_content(BRIEF, run_id="run_sse")]
    events = [json.loads(frame.split("data: ", 1)[1].strip()) for frame in frames]
    names = [item["event"] for item in events]

    assert names[0] == "run.start"
    assert names[-1] == "run.done"
    assert names.count("node.start") == 4
    assert "token" in names and "llm.start" in names
    tokens = [item["delta"] for item in events if item["event"] == "token"]
    assert "".join(tokens)  # token 是增量，拼起来能还原文本
    assert events[-1]["status"] == "succeeded"


async def test_stream_abort_on_client_disconnect_releases_background_run(app_client):
    async with app_client() as (_client, container):
        before = container.ops.streamer.stats()["aborted"]
        stream = container.ops.stream_content(BRIEF, run_id="run_abort")
        first = await stream.__anext__()
        assert "run.start" in first
        await stream.aclose()  # 模拟客户端断开
        after = container.ops.streamer.stats()
    assert after["aborted"] == before + 1
    assert after["runs"] >= 1


# =========================================================================== #
# G. 多模态素材：幂等 UPSERT + 跨模态检索
# =========================================================================== #
async def test_asset_upsert_is_idempotent_on_source_key():
    store = AssetStore(embedder=HashingEmbedder(), max_assets=10)
    first, created_a = await store.upsert(
        source_table="products",
        source_id="SKU-1002",
        modality=AssetModality.IMAGE,
        title="烟酰胺精华主图",
        product_id="SKU-1002",
        source={"caption": "烟酰胺精华 提亮"},
    )
    second, created_b = await store.upsert(
        source_table="products",
        source_id="SKU-1002",
        modality=AssetModality.IMAGE,
        title="烟酰胺精华主图（改）",
        product_id="SKU-1002",
        source={"caption": "烟酰胺精华 提亮"},
    )
    assert created_a is True and created_b is False
    assert first.asset_id == second.asset_id  # 同一条业务记录永远得到同一个 id，下游引用不断链
    assert store.size == 1
    assert second.updated_at >= first.updated_at and second.created_at == first.created_at


async def test_asset_store_search_hits_across_modalities():
    store = AssetStore(embedder=HashingEmbedder(), max_assets=10)
    await store.upsert(
        source_table="products",
        source_id="SKU-1002",
        modality=AssetModality.IMAGE,
        title="精华主图",
        source={"caption": "烟酰胺 提亮 精华"},
    )
    await store.upsert(
        source_table="pages",
        source_id="page-77",
        modality=AssetModality.PAGE,
        title="详情页",
        source={"text": "烟酰胺成分说明与用法"},
    )
    await store.upsert(
        source_table="live",
        source_id="live-9",
        modality=AssetModality.LIVE_CLIP,
        title="直播切片",
        source={"text": "主播讲解烟酰胺搭配建议"},
    )
    hits = await store.search("烟酰胺", top_k=5)
    assert hits
    assert {hit.asset.modality for hit in hits} >= {AssetModality.IMAGE, AssetModality.PAGE}


async def test_asset_store_evicts_oldest_beyond_capacity():
    store = AssetStore(embedder=HashingEmbedder(), max_assets=2)
    for index in range(3):
        await store.upsert(
            source_table="products",
            source_id=f"SKU-{index}",
            modality=AssetModality.IMAGE,
            title=f"图{index}",
            source={"caption": f"商品{index}"},
        )
    assert store.size == 2
    assert store.stats()["created"] == 3


async def test_asset_import_requires_source_key():
    store = AssetStore(embedder=HashingEmbedder())
    with pytest.raises(ValidationError):
        await store.upsert(source_table="", source_id="SKU-x", modality=AssetModality.PAGE)
    with pytest.raises(ValidationError):
        await store.upsert(source_table="products", source_id="", modality=AssetModality.PAGE)


# =========================================================================== #
# H. 直播：分段 / 重叠去重 / FFmpeg 降级
# =========================================================================== #
async def test_live_windows_and_overlap_dedupe():
    processor = LiveProcessor(gateway=StubGateway(), asr=OfflineAsr(), window_s=90, overlap_s=15, batch_size=3)
    segments = await processor.transcribe({"title": "烟酰胺专场"}, duration_s=180)
    assert [(item.start_s, item.end_s) for item in segments] == [(0.0, 90.0), (75.0, 165.0), (150.0, 180.0)]

    kept, report = dedupe_overlaps(segments, overlap_s=15)
    assert report["trimmed_chars"] > 0  # 重叠区被裁掉，不再是重复文本
    assert report["output_segments"] == len(kept) == 3
    assert len(kept[0].text) > len(kept[1].text)  # 后一段被裁短
    assert report["dropped_segments"] == []  # 只是重叠，不是重复内容，不能整段丢掉


def test_dedupe_drops_only_when_window_fully_covered():
    from src.ops.schemas import LiveSegment

    identical = LiveSegment(index=0, start_s=0.0, end_s=90.0, text="开场介绍产品亮点" * 20)
    covered = LiveSegment(index=1, start_s=75.0, end_s=165.0, text="开场介绍产品亮点" * 20)
    kept, report = dedupe_overlaps([identical, covered], overlap_s=15)
    assert report["dropped_segments"] == [1]  # 整段被上一窗覆盖 → 丢弃并并入前一段时间区间
    assert len(kept) == 1 and kept[0].end_s == 165.0


async def test_live_process_reports_ffmpeg_degradation_without_executing():
    import shutil

    processor = LiveProcessor(
        gateway=StubGateway(), asr=OfflineAsr(), window_s=90, overlap_s=15, batch_size=3, analysis_model="chat-pro"
    )
    result = await processor.process({"title": "烟酰胺专场"}, duration_s=180)
    payload = result.to_payload()

    expected_degraded = shutil.which("ffmpeg") is None
    assert payload["ffmpeg"]["degraded"] is expected_degraded
    if expected_degraded:
        assert payload["ffmpeg"]["reason"] == "ffmpeg_not_installed"
    assert payload["ffmpeg"]["commands"] and payload["ffmpeg"]["commands"][0]["command"].startswith("ffmpeg -ss")
    assert payload["analyses"] and payload["clips"]
    assert payload["dedupe"]["trimmed_chars"] > 0


async def test_live_batch_analysis_falls_back_on_invalid_json():
    processor = LiveProcessor(gateway=StubGateway(default="[mock] 不是 JSON"), asr=OfflineAsr(), batch_size=2)
    segments = await processor.transcribe({"title": "x"}, duration_s=45)
    analyses, usage = await processor.analyze(segments)
    assert len(analyses) == 1
    assert analyses[0].highlight_index is not None
    assert analyses[0].batch_index == 0
    assert usage["prompt_tokens"] > 0


async def test_live_batch_analysis_accepts_valid_json_and_drops_out_of_range_index():
    payload = json.dumps({"summary": "本批讲成分", "selling_points": ["提亮"], "highlight_index": 99}, ensure_ascii=False)
    processor = LiveProcessor(gateway=StubGateway([payload]), asr=OfflineAsr(), batch_size=3)
    segments = await processor.transcribe({"title": "x"}, duration_s=45)
    analyses, _usage = await processor.analyze(segments)
    assert analyses[0].summary == "本批讲成分"
    assert analyses[0].highlight_index is None  # 越界索引被丢弃，不让它污染切片


# =========================================================================== #
# I. 运营分析与归因
# =========================================================================== #
def test_analytics_metrics_and_recomputable_attribution():
    payload = {
        "channels": [
            {"channel": "live", "impressions": 10000, "clicks": 500, "orders": 60, "revenue": 6000, "spend": 1000, "baseline_orders": 40},
            {"channel": "search", "impressions": 8000, "clicks": 200, "orders": 20, "revenue": 1800, "spend": 900, "baseline_orders": 25},
        ]
    }
    report = OpsAnalytics().report(payload)
    metrics = {item["key"]: item["value"] for item in report["metrics"]}
    assert metrics["ctr"] == round(700 / 18000, 4)
    assert metrics["cvr"] == round(80 / 700, 4)
    assert metrics["roas"] == round(7800 / 1900, 4)

    attribution = {item["channel"]: item for item in report["attribution"]}
    assert attribution["live"]["delta"] == 20.0
    assert attribution["search"]["delta"] == -5.0
    assert round(sum(item["share"] for item in report["attribution"]), 6) == 1.0  # 仅正向增量参与分摊
    assert "delta = orders - baseline_orders" in attribution["live"]["basis"]


def test_analytics_material_suggestions_are_ranked_and_actionable():
    report = OpsAnalytics().report(
        {
            "channels": [],
            "materials": [
                {"source_id": "m1", "views": 1000, "likes": 100, "comments": 10, "shares": 5, "saves": 40, "title": "精华测评"},
                {"source_id": "m2", "views": 100, "likes": 1, "comments": 0, "shares": 0, "saves": 0},
            ],
        }
    )
    materials = report["materials"]
    assert [item["source_id"] for item in materials] == ["m1", "m2"]
    assert 0.0 <= materials[1]["viral_score"] <= materials[0]["viral_score"] <= 1.0
    assert materials[0]["title"].startswith("精华测评")
    assert materials[0]["editing"]


def test_analytics_requires_some_input():
    with pytest.raises(ValidationError):
        OpsAnalytics().report({"channels": [], "materials": []})


# =========================================================================== #
# J. 自进化闭环
# =========================================================================== #
def _events(count: int = 3) -> list[dict]:
    return [
        {
            "source": "correction",
            "id": f"e{index}",
            "question": f"烟酰胺精华怎么搭配？第 {index} 问",
            "corrected": "建议与保湿精华叠加使用",
            "chunk_ids": ["chunk-1"],
            "product_id": "SKU-1002",
        }
        for index in range(count)
    ]


def _answers(case_ids, *, refusal: bool) -> dict[str, str]:
    if refusal:
        return {case_id: "价格以官方渠道实时页面为准" for case_id in case_ids}
    return {case_id: "建议与保湿精华叠加使用" for case_id in case_ids}


def test_selfevolve_builds_eval_cases_and_scores_them():
    loop = SelfEvolveLoop(prompts=_registry())
    counts = loop.ingest(_events(3), default_source="correction")
    assert counts["correction"] == 3
    cases = loop.build_eval_cases()
    assert len(cases) == 3
    assert all(case.gold_chunk_ids == ["chunk-1"] for case in cases)  # 金标切片随记录一起带上

    answers = _answers([case.case_id for case in cases], refusal=False)
    evidence = {case.case_id: [case.gold_answer] for case in cases}
    scoring = loop.score(cases, answers, evidence)
    assert scoring["graded"] == 3
    assert scoring["pass_rate"] == 1.0


def test_selfevolve_prices_question_becomes_refusal_case():
    loop = SelfEvolveLoop(prompts=_registry())
    loop.ingest(
        [
            {
                "source": "correction",
                "question": "烟酰胺精华现在最低价多少钱？",
                "corrected": "价格以官方渠道实时页面为准",
                "chunk_ids": ["chunk-1"],
            }
        ]
    )
    cases = loop.build_eval_cases()
    assert cases[0].case_type == "refusal"
    scoring = loop.score(cases, {cases[0].case_id: "价格以官方渠道实时页面为准"}, None)
    assert scoring["pass_rate"] == 1.0  # 拒答策略命中


def test_selfevolve_cycle_proposes_unpublished_patch_and_strategy():
    prompts = _registry()
    loop = SelfEvolveLoop(prompts=prompts)
    active_before = prompts.active_version("content.system")
    cycle = loop.cycle(_events(4), default_source="correction")

    payload = cycle.to_payload()
    assert payload["scoring"]["status"] == "awaiting_answers"  # 没给答案就明确降级，不假装评了分
    patch = payload["template_patch"]
    assert patch["template"] == "content.system"
    assert patch["to_version"] != patch["from_version"]
    assert prompts.active_version("content.system") == active_before  # 关键：只登记，不自动发布
    assert payload["strategy"]
    assert "人工审核" in payload["notes"]

    from src.ops.schemas import TemplatePatch

    applied = loop.apply_patch(TemplatePatch(**patch))
    assert applied["active_version"] == patch["to_version"]
    assert loop.revert()["active_version"] == patch["from_version"]  # 版本顺序即回滚顺序


def test_selfevolve_passing_score_patch_mentions_target_and_cost():
    loop = SelfEvolveLoop(prompts=_registry())
    loop.ingest(_events(3), default_source="correction")
    cases = loop.build_eval_cases()
    answers = _answers([case.case_id for case in cases], refusal=False)
    evidence = {case.case_id: [case.gold_answer] for case in cases}
    scoring = loop.score(cases, answers, evidence)
    patch = loop.propose_template_patch(scoring, cases)
    assert "达标" in patch.reason
    assert "成本" in patch.reason


def test_selfevolve_low_score_triggers_stronger_patch_and_strategy():
    loop = SelfEvolveLoop(prompts=_registry())
    loop.ingest(_events(3), default_source="correction")
    cases = loop.build_eval_cases()
    bad = {case.case_id: "完全跑题的答案，与检索证据无关" for case in cases}
    scoring = loop.score(cases, bad, None)
    assert scoring["pass_rate"] < 0.7

    patch = loop.propose_template_patch(scoring, cases)
    assert "未达标" in patch.reason
    names = {item.name for item in loop.propose_strategy(scoring)}
    assert "retrieval.top_k" in names


# =========================================================================== #
# K. 接口层端到端
# =========================================================================== #
def test_factory_wires_all_ops_components():
    service = build_ops_service(
        settings=Settings(),
        gateway=StubGateway(),
        knowledge=KnowledgeService(embedder=HashingEmbedder()),
        prompts=_registry(),
    )
    assert set(service.stats()) == {"workflow", "streaming", "assets", "parallel"}
    assert service.graph.get("topic").name == "topic"
    assert service.ctx_template.brief.product_id == "__unset__"  # 占位符，请求时被替换


async def test_ops_endpoints_enforce_ops_scope(app_client):
    body = {"brief": BRIEF.model_dump()}
    async with app_client() as (client, _container):
        anon = await client.post("/v1/ops/content/run", json=body)
        assert anon.status_code == 401

        # 客服线的 key 没有 ops:rw，不能触发运营内容生产
        denied = await client.post("/v1/ops/content/run", json=body, headers={"X-API-Key": CS_KEY})
        assert denied.status_code == 403

        ok = await client.post("/v1/ops/content/run", json=body, headers={"X-API-Key": OPS_KEY})
        assert ok.status_code == 200
        payload = ok.json()
        assert payload["success"] is True and payload["status"] == "succeeded"


async def test_content_run_then_snapshot_then_resume_endpoint(app_client):
    headers = {"X-API-Key": OPS_KEY}
    async with app_client() as (client, _container):
        run = await client.post(
            "/v1/ops/content/run",
            json={"brief": BRIEF.model_dump(), "run_id": "run_api_resume"},
            headers=headers,
        )
        assert run.status_code == 200

        snapshot = await client.get("/v1/ops/workflow/run_api_resume", headers=headers)
        assert snapshot.status_code == 200
        assert snapshot.json()["run"]["found"] is True

        resumed = await client.post(
            "/v1/ops/content/resume",
            json={"brief": BRIEF.model_dump(), "run_id": "run_api_resume", "rollback_nodes": ["copy"]},
            headers=headers,
        )
        assert resumed.status_code == 200
        body = resumed.json()
        assert set(body["resumed_from"]) == {"topic", "selling_points", "image"}  # 只重跑被回滚的 copy

        after = await client.get("/v1/ops/workflow/run_api_resume", headers=headers)
        replay = after.json()["replay"]
        rolled = {item["node"] for item in replay if item["status"] == "rolled_back"}
        assert rolled == {"copy"}  # 回滚留痕可回放
        assert any(item["node"] == "copy" and item["status"] == "succeeded" for item in replay)

        missing = await client.get("/v1/ops/workflow/nope", headers=headers)
        assert missing.status_code == 404


async def test_content_resume_requires_run_id(app_client):
    async with app_client() as (client, _container):
        resp = await client.post(
            "/v1/ops/content/resume", json={"brief": BRIEF.model_dump()}, headers={"X-API-Key": OPS_KEY}
        )
        assert resp.status_code == 400


async def test_content_stream_endpoint_returns_sse(app_client):
    async with app_client() as (client, _container):
        headers = {"X-API-Key": OPS_KEY}
        async with client.stream(
            "POST", "/v1/ops/content/stream", json={"brief": BRIEF.model_dump()}, headers=headers
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["x-run-id"].startswith("content_")
            text = "".join([chunk async for chunk in response.aiter_text()])
    assert "event: run.start" in text and "event: run.done" in text
    assert "event: token" in text


async def test_asset_endpoints_round_trip(app_client):
    headers = {"X-API-Key": OPS_KEY}
    async with app_client() as (client, _container):
        imported = await client.post(
            "/v1/ops/assets/import",
            json={
                "source_table": "products",
                "source_id": "SKU-1002",
                "modality": "image",
                "title": "烟酰胺精华主图",
                "product_id": "SKU-1002",
                "source": {"caption": "烟酰胺 提亮 精华"},
            },
            headers=headers,
        )
        assert imported.status_code == 200 and imported.json()["created"] is True
        asset_id = imported.json()["asset"]["asset_id"]

        again = await client.post(
            "/v1/ops/assets/import",
            json={
                "source_table": "products",
                "source_id": "SKU-1002",
                "modality": "image",
                "title": "烟酰胺精华主图",
                "product_id": "SKU-1002",
                "source": {"caption": "烟酰胺 提亮 精华"},
            },
            headers=headers,
        )
        assert again.json()["created"] is False and again.json()["asset"]["asset_id"] == asset_id

        search = await client.get("/v1/ops/assets/search", params={"query": "烟酰胺"}, headers=headers)
        assert search.status_code == 200
        assert search.json()["hits"]


async def test_live_analytics_and_selfevolve_endpoints(app_client):
    headers = {"X-API-Key": OPS_KEY}
    async with app_client() as (client, _container):
        live = await client.post(
            "/v1/ops/live/process",
            json={"source": {"title": "烟酰胺专场"}, "duration_s": 120},
            headers=headers,
        )
        assert live.status_code == 200
        assert live.json()["segments"] and live.json()["ffmpeg"]["commands"]

        analytics = await client.post(
            "/v1/ops/analytics/report",
            json={"channels": [{"channel": "live", "impressions": 100, "clicks": 10, "orders": 2, "revenue": 200, "spend": 50, "baseline_orders": 1}]},
            headers=headers,
        )
        assert analytics.status_code == 200
        assert analytics.json()["metrics"]

        cycle = await client.post("/v1/ops/selfevolve/cycle", json={"events": _events(2)}, headers=headers)
        assert cycle.status_code == 200
        assert cycle.json()["template_patch"]["template"] == "content.system"

        report = await client.get("/v1/ops/selfevolve/report", headers=headers)
        assert report.status_code == 200 and report.json()["cycles"]

        applied = await client.post("/v1/ops/selfevolve/apply", headers=headers)
        assert applied.status_code == 200
        reverted = await client.post("/v1/ops/selfevolve/revert", headers=headers)
        assert reverted.status_code == 200


async def test_parallel_benchmark_endpoint_reports_speedup(app_client):
    async with app_client() as (client, _container):
        resp = await client.post(
            "/v1/ops/parallel/benchmark",
            json={"count": 4, "sleep_s": 0.05, "fail_every": 0},
            headers={"X-API-Key": OPS_KEY},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["speedup"] > 1.0
        assert body["peak_concurrency"] <= body["max_concurrency"]
        assert body["failures"] == []

        failing = await client.post(
            "/v1/ops/parallel/benchmark",
            json={"count": 4, "sleep_s": 0.02, "fail_every": 2},
            headers={"X-API-Key": OPS_KEY},
        )
        assert failing.json()["failures"] == ["unit1", "unit3"]


async def test_ops_stats_expose_bounded_state(app_client):
    async with app_client() as (_client, container):
        stats = container.ops.stats()
    assert set(stats) == {"workflow", "streaming", "assets", "parallel"}
    assert stats["assets"]["max_assets"] >= 1
    assert stats["workflow"]["runs"] >= 0
