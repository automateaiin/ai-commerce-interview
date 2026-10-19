"""运营工作流子包：图定义 + 节点级检查点 + 内容节点。

对外只暴露这几样：图（`WorkflowGraph`）、执行器（`WorkflowRunner`）、检查点（`Checkpointer`）、
节点装配入口（`build_content_graph`）。节点内部实现（含结构化输出与兜底）不是公共接口。
"""

from .checkpointer import Checkpointer
from .graph import (
    Node,
    NodeContext,
    NodeFailedError,
    WorkflowDefinitionError,
    WorkflowGraph,
    WorkflowRunner,
)
from .nodes import build_content_graph
from .schemas import (
    CopyDraft,
    ImageBrief,
    NodeAttempt,
    NodeRecord,
    NodeStatus,
    SellingPoint,
    SellingPoints,
    TopicPlan,
    WorkflowResult,
)

__all__ = [
    "Checkpointer",
    "CopyDraft",
    "ImageBrief",
    "Node",
    "NodeAttempt",
    "NodeContext",
    "NodeFailedError",
    "NodeRecord",
    "NodeStatus",
    "SellingPoint",
    "SellingPoints",
    "TopicPlan",
    "WorkflowDefinitionError",
    "WorkflowGraph",
    "WorkflowResult",
    "WorkflowRunner",
    "build_content_graph",
]
