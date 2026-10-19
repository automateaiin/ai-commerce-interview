"""项目一门面：把编排与各子系统收成一个对象。

组合根（`infra/container.py`）只需要装配它、路由只需要用它，
而「谁依赖谁」全部在 `factory.py` 里一次讲清楚 —— 这是本仓库统一的门面 + 工厂惯例。
"""

from __future__ import annotations

from .a2a import A2AService
from .customer_agent import CustomerAgent
from .feedback import FeedbackCollector
from .hitl import HITLManager
from .operations_agent import OperationsAgent
from .orchestrator import Orchestrator
from .prompt_guard import PromptGuard
from .realtime import MockRTCSession, RealtimePipeline
from .retrieval_strategy import QueryRewriter
from .session import SessionManager


class AgentService:
    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        customer: CustomerAgent,
        operations: OperationsAgent,
        sessions: SessionManager,
        hitl: HITLManager,
        realtime: RealtimePipeline,
        rtc: MockRTCSession,
        feedback: FeedbackCollector,
        retrieval: QueryRewriter,
        a2a: A2AService,
        guard: PromptGuard,
    ) -> None:
        self.orchestrator = orchestrator
        self.customer = customer
        self.operations = operations
        self.sessions = sessions
        self.hitl = hitl
        self.realtime = realtime
        self.rtc = rtc
        self.feedback = feedback
        self.retrieval = retrieval
        self.a2a = a2a
        self.guard = guard


__all__ = ["AgentService"]
