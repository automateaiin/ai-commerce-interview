"""【项目二】AI 运营系统（应用层）。

定位：运营内容自动生产 · 多模态素材知识化 · 直播智能处理 · 运营数据分析与自进化。

依赖方向：ops → harness（执行内核） → infra（平台层）；ops → dataplatform（素材回流与知识）。

模块（阶段 5 交付，见 PROJECT.md 第 6.2 节）：
- ``workflow``      运营内容工作流：graph（节点+边）+ checkpointer（节点级持久化/回滚/重试）+ nodes + schemas
- ``streaming``     SSE + 事件级流式输出（客户端断开释放后台任务）
- ``parallel``      asyncio.gather + run_in_executor 并行生成（含实测加速比）
- ``multimodal``    多模态素材知识化：rag_assets + OCR/VLM/ASR 端口 → retrieval_text（UPSERT 幂等）
- ``live``          直播分段转录 + 重叠去重 + 分批分析（FFmpeg 仅构造命令）
- ``analytics``     运营指标 + 可复算归因 + 素材爆款评分/标题/剪辑建议
- ``selfevolve``    回流 → 自动评测集 / 自动评分 → 模板与策略迭代建议（不自动发布）
- ``schemas``       跨模块契约
- ``service``       门面 OpsService
- ``factory``       由 Settings 装配门面

装配入口：``src.infra.container`` 调用 ``factory.build_ops_service`` 构造 ``OpsService``，
挂在 ``container.ops`` 上，由 ``router.py`` 暴露 ``/v1/ops/*``（scope ``ops:rw``）。
"""
