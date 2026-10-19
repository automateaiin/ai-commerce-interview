"""项目二装配：由 Settings + 既有基建构造出 OpsService。

装配顺序就是依赖顺序：
    内容图 → Checkpointer / 并行器 / 流式器 → 素材库 → 直播处理器 → 分析器 → 自进化闭环
        → 门面（把上面这些收成一个出口）

依赖方向：本模块只依赖 `infra`（模型网关 / 检索 / 提示词中心）与 `dataplatform`（回流收集器 / 评分器），
不反向依赖 `agents` 或 `commerce` —— 运营系统是独立业务线，不是项目一的附属。
"""

from __future__ import annotations

from typing import Any

from src.infra.core.settings import Settings

from .analytics import OpsAnalytics
from .live import LiveProcessor
from .multimodal import AssetStore
from .parallel import ParallelRunner
from .selfevolve import SelfEvolveLoop
from .service import OpsService
from .streaming import OpsStreamer
from .schemas import ContentBrief
from .workflow.checkpointer import Checkpointer
from .workflow.graph import NodeContext
from .workflow.nodes import build_content_graph


def build_ops_service(
    *,
    settings: Settings,
    gateway: Any,
    knowledge: Any,
    prompts: Any,
) -> OpsService:
    """装配运营系统门面。

    `ctx_template` 里的 `brief` / `run_id` / `emitter` 都是**请求级**参数，
    这里只放一个占位符，每次请求由 `OpsService._runner_for` 用 `dataclasses.replace` 覆盖 ——
    这样节点代码不必感知「我这次跑的是哪个需求单、是不是流式」。
    """
    graph = build_content_graph(settings)
    checkpointer = Checkpointer()
    parallel = ParallelRunner(max_concurrency=settings.ops_parallel_max_concurrency)
    prompts_registry = prompts

    ctx_template = NodeContext(
        gateway=gateway,
        knowledge=knowledge,
        prompts=prompts_registry,
        brief=ContentBrief(product_id="__unset__"),
        run_id="__unset__",
        parallel=parallel,
        settings=settings,
        emitter=None,
    )

    return OpsService(
        graph=graph,
        checkpointer=checkpointer,
        streamer=OpsStreamer(),
        parallel=parallel,
        assets=AssetStore(
            max_assets=settings.ops_asset_max_assets,
            embed_version=settings.ops_asset_embed_version,
        ),
        live=LiveProcessor(
            gateway=gateway,
            window_s=settings.ops_live_window_s,
            overlap_s=settings.ops_live_overlap_s,
            batch_size=settings.ops_live_batch_size,
            analysis_model=settings.ops_analysis_model,
        ),
        analytics=OpsAnalytics(),
        selfevolve=SelfEvolveLoop(
            prompts=prompts_registry,
            max_events=settings.ops_selfevolve_max_events,
        ),
        ctx_template=ctx_template,
        max_attempts=settings.ops_max_attempts,
        structured_retries=settings.ops_structured_retries,
    )


__all__ = ["build_ops_service"]
