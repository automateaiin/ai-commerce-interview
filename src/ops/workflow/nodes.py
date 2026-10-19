"""运营内容的四个节点：选题 → 成分卖点 → 文案 / 产品图。

设计要点（面试三条主线）：

1) **结构化输出是节点契约，不是「顺手 json.loads」**：
   每个节点声明自己的 Pydantic 模型；模型输出先解析再校验，不合法就带**失败原因**重试，
   重试仍不合法则回退到确定性兜底并记录 `fallback_reason`。
   —— 兜底不是「假装成功」，run 的 artifacts 里能看到哪些节点是兜底产出的。

2) **模型分工降本**：
   选题与成分卖点是「分析型」任务（要综合证据、定策略）→ 用旗舰逻辑模型；
   文案与产品图提示词是「生成型短任务」→ 用轻量逻辑模型。
   逻辑模型名来自 infra 的模型目录，节点不写死物理模型，换供应商不改这里。

3) **依赖显式注入**：下游节点的 prompt 里能看到上游的**结构化字段**（不是上游原文），
   这就是「Agent 间上下文传递」在工作流里的等价物：不声明依赖，下游什么都看不到。

诚实边界：产品图只产出提示词 + 把「出图」这一步丢到线程池（离线实现不真实渲染）；
成分卖点的 sop 检索走 infra 的离线哈希向量，与真实 embedding 的差距见 PROJECT.md 第 8 章。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.infra.core.logging import get_logger
from src.infra.llm.types import LLMResponse, Usage

from .graph import Node, NodeContext, WorkflowGraph
from .schemas import CopyDraft, ImageBrief, SellingPoint, SellingPoints, TopicPlan

logger = get_logger("ops.workflow.nodes")

_CONTENT_SCENARIO = "copywriting"
_JSON_CONTRACT = (
    "\n\n【输出契约】只返回一个 JSON 对象：不要 Markdown 代码块、不要前后解释、不要注释。\n"
    "必须严格匹配以下字段（多字段、少字段、类型不符都判为不合法）：\n{schema}\n"
)


@dataclass(slots=True)
class StructuredOutcome:
    """结构化输出结果：payload 落库、fallback_reason 供排障、logical_model 供成本归集。"""

    payload: dict[str, Any]
    logical_model: str
    fallback_reason: str | None = None


def _strip_fences(text: str) -> str:
    """剥掉模型常见的 ```json 包裹 —— 这是最容易让 json.loads 失败的形态。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _schema_hint(schema: type) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)


async def _call_model(
    ctx: NodeContext,
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float | None,
) -> LLMResponse:
    """取一次模型输出。

    - **同步模式**（无 emitter）：一次 `complete`，拿完整文本；
    - **流式模式**（SSE 注入了 emitter）：走 `stream`，边收边把 token 推给前端，
      最后把拼装好的文本与 usage 交回给同一段结构化校验逻辑。

    同一个节点代码在两种模式下都成立 —— 不需要为「流式」单独维护一份生成逻辑，
    否则两条路径的 prompt 与校验迟早会漂移。

    诚实边界：`gateway.stream` 只做单次调用（流式不可整请求重试，会重复内容），
    因此它不参与预算预扣；生产应基于最后一帧 usage 补结算。
    """
    if ctx.emitter is None:
        return await ctx.gateway.complete(messages, logical_model=model, temperature=temperature)

    parts: list[str] = []
    finish = "stop"
    usage: Usage | None = None
    async for chunk in ctx.gateway.stream(messages, logical_model=model, temperature=temperature):
        if chunk.delta_text:
            parts.append(chunk.delta_text)
            await ctx.emit("token", {"delta": chunk.delta_text})
        if chunk.finish_reason:
            finish = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
    return LLMResponse(text="".join(parts), usage=usage or Usage(), finish_reason=finish)


async def generate_structured(
    ctx: NodeContext,
    *,
    schema: type,
    instruction: str,
    payload_hint: dict[str, Any],
    fallback: Callable[[], Any],
    temperature: float | None = None,
) -> StructuredOutcome:
    """模型调用 + 结构化校验 + 重试 + 确定性兜底。

    重试不是「原样再问一次」：第二次会把**上一次的失败原因**和模型自己的原文一起回填，
    让模型知道要修哪里；这是结构化输出能真正收敛的关键。
    """
    model = ctx.logical_model or ""
    retries = int(getattr(getattr(ctx, "settings", None), "ops_structured_retries", 1) or 0)
    system = ctx.prompts.compose("content", scenario=_CONTENT_SCENARIO, max_steps=1)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system + _JSON_CONTRACT.format(schema=_schema_hint(schema))},
        {"role": "user", "content": json.dumps(payload_hint, ensure_ascii=False) + "\n\n任务：" + instruction},
    ]

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    last_reason: str | None = None
    for attempt in range(retries + 1):
        await ctx.emit("llm.start", {"attempt": attempt + 1})
        response = await _call_model(ctx, messages, model=model, temperature=temperature)
        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens
        usage["cost_usd"] += response.usage.cost_usd
        text = response.text or ""
        try:
            parsed = schema.model_validate_json(_strip_fences(text))
        except Exception as exc:  # noqa: BLE001 - 校验失败要转成「可重试」的显式原因
            last_reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            logger.warning(
                "ops.structured.invalid",
                extra={
                    "event": "ops_structured",
                    "status": "invalid",
                    "node": ctx.node,
                    "attempt": attempt + 1,
                    "reason": last_reason,
                },
            )
            if attempt < retries:
                messages = [
                    messages[0],
                    messages[1],
                    {"role": "assistant", "content": text[:2000]},
                    {
                        "role": "user",
                        "content": (
                            f"上一次输出不合法：{last_reason}。请只返回符合契约的 JSON，不要任何其它内容。"
                        ),
                    },
                ]
                continue
            break
        return StructuredOutcome(payload={**parsed.model_dump(), "usage": usage}, logical_model=model)

    built = fallback()
    payload = {**built.model_dump(), "usage": usage, "_fallback_reason": last_reason or "structured_output_invalid"}
    logger.warning(
        "ops.structured.fallback",
        extra={
            "event": "ops_structured",
            "status": "degraded",
            "node": ctx.node,
            "reason": payload["_fallback_reason"],
        },
    )
    return StructuredOutcome(
        payload=payload, logical_model=model, fallback_reason=payload["_fallback_reason"]
    )


# --------------------------------------------------------------------------- #
# 确定性兜底构造器（离线可复现；只有在模型输出不合法时才被使用）
# --------------------------------------------------------------------------- #
def topic_fallback(brief: Any) -> TopicPlan:
    seeds = list(brief.keywords) or [brief.product_name or brief.product_id]
    topics = [f"{brief.product_name or brief.product_id}｜{seed} 场景化种草" for seed in seeds[:3]]
    return TopicPlan(topics=topics, rationale="离线兜底：按需求单关键词生成候选选题（模型输出未通过结构化校验）")


def selling_points_fallback(brief: Any, evidence_ids: list[str]) -> SellingPoints:
    seeds = list(brief.keywords)
    points = [
        SellingPoint(
            claim=f"{seed} 的核心作用与适用场景",
            evidence=evidence_ids[:1],
            compliance="ok" if evidence_ids else "ungrounded",
        )
        for seed in seeds[:3]
    ] or [SellingPoint(claim=f"{brief.product_name or brief.product_id} 基础功效说明", evidence=[], compliance="ungrounded")]
    return SellingPoints(points=points)


def copy_fallback(brief: Any, points: list[dict[str, Any]]) -> CopyDraft:
    claim = (points[0].get("claim") if points else "") or "成分实力派"
    return CopyDraft(
        headline=f"{brief.product_name or brief.product_id}｜{claim}",
        body=(
            f"为{brief.audience}准备的{brief.product_name or brief.product_id}："
            f"{claim}。{brief.tone}口吻，适配{brief.channel}渠道。"
        ),
        hashtags=list(brief.keywords)[:5] or ["美妆", "成分党"],
        cta="点击了解更多",
    )


def image_fallback(brief: Any) -> ImageBrief:
    return ImageBrief(
        prompt=(
            f"高端美妆产品摄影：{brief.product_name or brief.product_id} 置于纯净背景，"
            f"柔和自然光，突出质感与{brief.audience}的使用场景，商业级构图"
        ),
        aspect="portrait_4_3",
        negative="低分辨率, 文字水印, 变形瓶身",
    )


def _enforce_grounding(points: list[dict[str, Any]], allowed_ids: list[str]) -> tuple[list[dict[str, Any]], int]:
    """接地校验：卖点引用的证据必须真的来自本次检索，否则降级为 ungrounded。

    这是「不让模型编证据」的落地手段：模型可以引用 chunk_id，但引用不到的会被当场抹掉，
    而不是留着一个看起来很像真的假证据进下游素材。
    """
    allowed = set(allowed_ids)
    cleaned: list[dict[str, Any]] = []
    ungrounded = 0
    for point in points:
        evidence = [item for item in list(point.get("evidence") or []) if item in allowed]
        compliance = point.get("compliance") or "ok"
        if not evidence:
            compliance = "ungrounded"
            ungrounded += 1
        cleaned.append({**point, "evidence": evidence, "compliance": compliance})
    return cleaned, ungrounded


# --------------------------------------------------------------------------- #
# 四个节点
# --------------------------------------------------------------------------- #
async def topic_node(inputs: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    outcome = await generate_structured(
        ctx,
        schema=TopicPlan,
        instruction="为这个产品产出 2-3 个内容选题，并说明为什么选它们（结合人群与渠道）。",
        payload_hint={
            "product_id": ctx.brief.product_id,
            "product_name": ctx.brief.product_name,
            "market": ctx.brief.market,
            "audience": ctx.brief.audience,
            "channel": ctx.brief.channel,
            "keywords": ctx.brief.keywords,
        },
        fallback=lambda: topic_fallback(ctx.brief),
    )
    return outcome.payload


async def selling_points_node(inputs: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    topic = inputs.get("topic") or {}
    query = " ".join(
        [ctx.brief.product_name or ctx.brief.product_id, *ctx.brief.keywords, *(topic.get("topics") or [])[:1]]
    ).strip()
    retrieval = await ctx.knowledge.search(query, top_k=3, min_score=0.0)
    evidence_ids = [chunk.chunk_id for chunk in retrieval.chunks]
    excerpts = [{"chunk_id": chunk.chunk_id, "title": chunk.title, "text": chunk.text[:120]} for chunk in retrieval.chunks]

    outcome = await generate_structured(
        ctx,
        schema=SellingPoints,
        instruction=(
            "提炼 2-4 条成分/功效卖点。每条卖点必须绑定上面给出的证据 chunk_id（evidence 字段），"
            "没有证据支撑的卖点标 compliance=ungrounded。"
        ),
        payload_hint={
            "product": ctx.brief.product_name or ctx.brief.product_id,
            "audience": ctx.brief.audience,
            "topic": topic,
            "evidence": excerpts,
        },
        fallback=lambda: selling_points_fallback(ctx.brief, evidence_ids),
    )
    points, ungrounded = _enforce_grounding(list(outcome.payload.get("points") or []), evidence_ids)
    outcome.payload["points"] = points
    outcome.payload["evidence_chunks"] = evidence_ids
    outcome.payload["ungrounded_count"] = ungrounded
    if ungrounded and not outcome.fallback_reason:
        logger.info(
            "ops.selling_points.ungrounded",
            extra={
                "event": "ops_node",
                "status": "degraded",
                "node": ctx.node,
                "node_count": ungrounded,
                "reason": "evidence_not_retrieved",
            },
        )
    return outcome.payload


async def copy_node(inputs: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    points = inputs.get("selling_points") or {}
    outcome = await generate_structured(
        ctx,
        schema=CopyDraft,
        instruction="基于卖点写一条短文案：标题 + 正文 + 话题标签 + 行动号召，控制在渠道可读长度内。",
        payload_hint={
            "product": ctx.brief.product_name or ctx.brief.product_id,
            "channel": ctx.brief.channel,
            "tone": ctx.brief.tone,
            "audience": ctx.brief.audience,
            "selling_points": points.get("points"),
        },
        fallback=lambda: copy_fallback(ctx.brief, list(points.get("points") or [])),
        temperature=0.7,
    )
    return outcome.payload


def _render_image_blocking(prompt: str, aspect: str) -> dict[str, Any]:
    """「出图」这一步是**阻塞**调用（真实实现是调用外部渲染服务 / 写文件），
    所以它必须走线程池，不能直接 await 在事件循环里 —— 否则整层并行会被它拖住。"""
    time.sleep(0.05)
    return {"render_status": "offline_placeholder", "bytes": len(prompt.encode("utf-8")), "aspect": aspect}


async def image_node(inputs: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    points = inputs.get("selling_points") or {}
    outcome = await generate_structured(
        ctx,
        schema=ImageBrief,
        instruction="产出产品图的生成提示词（正/负面提示词 + 画幅），用于外部出图服务。",
        payload_hint={
            "product": ctx.brief.product_name or ctx.brief.product_id,
            "audience": ctx.brief.audience,
            "market": ctx.brief.market,
            "selling_points": points.get("points"),
        },
        fallback=lambda: image_fallback(ctx.brief),
    )
    if ctx.parallel is not None:
        render = await ctx.parallel.run_blocking(
            "image.render", _render_image_blocking, outcome.payload["prompt"], outcome.payload.get("aspect") or ""
        )
        outcome.payload["render"] = render
    return outcome.payload


# --------------------------------------------------------------------------- #
# 图装配
# --------------------------------------------------------------------------- #
def build_content_graph(settings: Any) -> WorkflowGraph:
    """运营内容工作流：选题 → 成分卖点 → (文案 ‖ 产品图)。

    注意 `copy` 与 `image` 的 depends_on 都只有 `selling_points` —— 它们不互相依赖，
    所以 Runner 会把它们放进同一拓扑层并行执行；这不是「碰巧的顺序」，是图定义的结果。
    """
    graph = WorkflowGraph("ops.content")
    graph.add(
        Node(
            "topic",
            topic_node,
            kind="llm",
            logical_model=settings.ops_analysis_model,
            llm_report=True,
        )
    )
    graph.add(
        Node(
            "selling_points",
            selling_points_node,
            depends_on=("topic",),
            kind="llm",
            logical_model=settings.ops_analysis_model,
            llm_report=True,
        )
    )
    graph.add(
        Node(
            "copy",
            copy_node,
            depends_on=("selling_points",),
            kind="llm",
            logical_model=settings.ops_copy_model,
            llm_report=True,
        )
    )
    graph.add(
        Node(
            "image",
            image_node,
            depends_on=("selling_points",),
            kind="llm",
            logical_model=settings.ops_copy_model,
            llm_report=True,
            optional=True,
        )
    )
    return graph


__all__ = [
    "StructuredOutcome",
    "build_content_graph",
    "copy_node",
    "generate_structured",
    "image_node",
    "selling_points_node",
    "topic_node",
]
