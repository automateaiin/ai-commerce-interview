"""桥接层对外契约（Pydantic 模型）。

对外响应统一为「信封」结构，成功与失败形状一致：
    {"success": bool, "data": ..., "error": {"code","message","retryable"}, "trace_id": "...", "duration_ms": ...}

这样调用方（业务系统 / Agent）不需要为失败写一套分支解析，
而且每个响应都带 trace_id —— 出问题时可以直接拿这个 ID 去日志系统查完整链路。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数（服务端会二次校验）")
    session_id: str | None = Field(default=None, max_length=128, description="会话 ID（桥接层从请求体取）")
    user_id: str | None = Field(default=None, max_length=128, description="已由业务侧完成映射的用户标识")
    client_project_id: str | None = Field(
        default=None,
        description="【已废弃】客户端声明的项目标识。服务端一律忽略，仅用于演示「不可信输入」的埋点。",
    )


class ToolCallResponse(BaseModel):
    success: bool
    tool: str
    data: Any | None = None
    error: dict[str, Any] | None = None
    trace_id: str
    duration_ms: float


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    logical_model: str | None = Field(default=None, description="逻辑模型名，例如 chat-lite / chat-pro")
    task: str | None = Field(
        default=None, description="任务类型，用于路由选模型：product_consulting / complex_reasoning ..."
    )
    session_id: str | None = Field(default=None, description="传入则自动读写会话记忆")
    user_id: str | None = Field(default=None, description="传入则加载用户画像作为上下文")
    max_output_tokens: int | None = Field(default=None, ge=1, le=8192)


class ChatCompletionResponse(BaseModel):
    success: bool
    text: str | None = None
    logical_model: str
    model: str
    provider: str
    usage: dict[str, Any]
    finish_reason: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str
    duration_ms: float
    stages: list[dict[str, Any]] = Field(default_factory=list, description="各阶段耗时，用于定位瓶颈")


class AgentRunRequest(BaseModel):
    task: str = Field(min_length=1, description="给子 Agent 的任务描述")
    session_id: str | None = None
    user_id: str | None = None
    logical_model: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=12)
    tool_names: list[str] | None = Field(default=None, description="限定可用工具（越小越安全）")
    confirm_high_risk: bool = Field(
        default=False,
        description="人工已确认高风险动作（如下单）后才可置 True；否则这类调用会被确认门拦下并交接",
    )


class AgentRunResponse(BaseModel):
    success: bool
    answer: str | None = None
    steps: int
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    stop_reason: str
    trace_id: str
    duration_ms: float
    transcript: list[dict[str, Any]] = Field(default_factory=list, description="结构化执行轨迹")


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False


class QuotaView(BaseModel):
    project_id: str
    backend: str
    requests: float
    tokens: float
    limits: dict[str, int]


class ConfigView(BaseModel):
    revision: int
    logical_models: list[str]
    routing_default: str
    mcp_paths: list[str]
    transport: str


# ---- 平台治理（项目三）---------------------------------------------------- #
class EvalRunRequest(BaseModel):
    """跑一轮评测。用例集默认内置（离线可复现），可按 case_id 过滤。"""

    suite_id: str = Field(default="default", max_length=64)
    case_ids: list[str] | None = Field(default=None, description="只跑指定用例，便于回归时缩小范围")
    max_cases: int | None = Field(default=None, ge=1, le=50)


class PromptPreviewRequest(BaseModel):
    """预览某角色的组装结果：按 Role 拼片段并渲染，用于核对上线前的 Prompt 内容。"""

    scenario: str | None = Field(default=None, max_length=64)
    variables: dict[str, Any] = Field(default_factory=dict)


# ---- 数据平台（项目三）---------------------------------------------------- #
ExportFormat = Literal["sharegpt", "alpaca", "openai"]
SamplePurposeLiteral = Literal["knowledge", "sft_style", "excluded"]


class DataPipelineRunRequest(BaseModel):
    """跑一条数据流水线：原始记录 → 治理 → 审核 → 资产 → 索引 → 导出 → 血缘。

    records 缺省时用内置示例脏数据 —— 让接口开箱可演示，且回归结果稳定可比。
    """

    records: list[dict[str, Any]] | None = Field(
        default=None, description="原始记录（三类来源混排）；缺省用内置示例数据"
    )
    reviewer: str | None = Field(default="auto_policy", description="自动审核人标识；传 null 则全部留在待审核")
    export_format: ExportFormat | None = Field(default=None, description="同时导出训练集（SFT）")
    export_batch: str | None = Field(default=None, description="导出批次号，写入血缘")
    training_export: bool = Field(default=False, description="同时生成 LLaMA-Factory 训练产物（SFT + DPO + YAML）")
    training_dataset: str = Field(default="beauty_cs", description="训练数据集名（写入 dataset_info 与 YAML）")


class DataSearchRequest(BaseModel):
    """检索知识切片。product_id / scene 是元数据预过滤，先过滤再做向量召回。"""

    query: str = Field(min_length=1)
    product_id: str | None = None
    scene: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class DataEvaluateRequest(BaseModel):
    """在标准证据标注集上评估检索质量；items 缺省用内置的黄金问题集。"""

    items: list[dict[str, Any]] | None = Field(default=None, description="[{question, gold_chunk_ids, group, ...}]")
    k: int = Field(default=3, ge=1, le=20)


class DataExportRequest(BaseModel):
    """把最近一次流水线产出的 SFT 样本按目标训练框架导出（带校验门禁）。"""

    format: ExportFormat = "sharegpt"
    purpose: SamplePurposeLiteral | None = Field(
        default="sft_style", description="只导出该用途的样本；null 表示不过滤"
    )


class DataTrainingExportRequest(BaseModel):
    """生成 LLaMA-Factory 可直接消费的训练产物（SFT + DPO 数据集 + 训练 YAML）。

    诚实边界：本仓库只做数据准备与验证链路，训练在外部环境执行。
    """

    dataset_name: str = Field(default="beauty_cs", min_length=1)
    version: str = Field(default="v1", min_length=1)
    preview: int = Field(default=3, ge=0, le=20, description="每条数据集返回的预览记录数（避免响应过大）")


class DataRefluxRequest(BaseModel):
    """写入一条回流事件（五来源之一）。字段要求由来源决定，缺字段会被拒绝并记录原因。"""

    source: str = Field(description="correction | review | tool_call | hitl | live_asr")
    events: list[dict[str, Any]] = Field(min_length=1)


class DataQuizRequest(BaseModel):
    """自动出题 / 自动评分：不传 answers 只出题；传了则按实际检索证据打分。"""

    answers: dict[str, str] | None = Field(default=None, description="{quiz_id: 被考核答案}")
    top_k: int = Field(default=3, ge=1, le=10)
