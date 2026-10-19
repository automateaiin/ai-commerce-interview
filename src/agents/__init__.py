"""【项目一】AI 智能客服导购系统（应用层）。

定位：面向消费者（Customer Agent）与品牌运营（Operations Agent）的双端 Agent，
由 Orchestrator 按任务复杂度统一路由；简单查询直连白名单 Tool，复杂跨域任务才进 Agent 推理。

依赖方向：agents → harness（执行内核） → infra（平台层）；agents → commerce（业务数据源）。

模块（阶段 4 已交付，见 PROJECT.md 第 6.1 节）：
- ``schemas``               编排契约：AgentInvocation / SubTaskResult / RoutePlan / OrchestrationResult
- ``orchestrator``          复杂度路由（规则优先 / 模型兜底 + 计划逐项校验）：单域查询直连 Tool，跨域才分派 Agent
- ``customer_agent``        Customer Agent：肤质测试 / 成分咨询 / 方案推荐 / 订单售后
- ``operations_agent``      Operations Agent：经营分析 / 质检 / 卖点 / 运营报告
- ``orchestration``         Sequential / Fan-out-Fan-in / Handoff + OrchestrationMode Protocol + Registry
- ``grounding``             交易类回答绑定 Tool Result 或知识来源
- ``prompt_guard``          分层 Prompt 治理 + fencing 清洗
- ``session``               Session Rehydrate + 摘要记忆
- ``a2a``                   Specialist Agent ``/message:send`` + session_id 透传
- ``hitl``                  Stage→Approve→Apply + checkpoint + Resume Token
- ``realtime``              RTC→ASR→Orchestrator→LLM→TTS（端口 + Mock 实现）
- ``retrieval_strategy``    Step-Back / HyDE 查询重写
- ``feedback``              纠偏 / 优质回复 / 失败案例 → Prompt 与 RAG few-shot
- ``service``               门面：一次装配所有子能力
- ``router``                ``/v1/agents/*`` 与 ``/v1/a2a/message:send``
- ``factory``               装配

接线：由 ``src/infra/container.py`` 组合根装配，``src/infra/api.py`` 挂载 router。
验证：``tests/test_agents_stage4.py``（41 条）+ ``scripts/smoke.py`` 第 17 段（15 项）。
"""
