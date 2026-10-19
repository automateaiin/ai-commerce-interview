"""SSE + 事件级流式输出。

为什么工作流要能「流式」（面试常问「一次生成不就完了」）：
运营内容工作流里，出选题、出卖点、写文案是**串行**的三个模型调用，一次请求可能十几秒。
如果只在最后一次性返回，用户在十几秒里看不到任何进展；而**事件级**流式能把
「哪个节点开始跑了、正在吐第几个 token、哪个节点回退了」实时推出去 —— 用户看到的是过程，
排障时看到的也是过程（哪一步卡住一眼可见）。

事件级 vs token 级的分工：
- `token` 事件是**增量**（delta），前端自己拼接；
- 其它事件（`node.start` / `node.done` / `run.done` …）是**状态**，带完整字段。
两者混在一个 SSE 流里，靠 `event:` 字段分流。

客户端断开为什么要显式释放（这里是最容易被忽略的一环）：
SSE 是个长连接，用户一关页面，连接就断了，但**后台的模型调用不会自己停** ——
它会继续烧 token、继续占线程。所以生成器被 `aclose()` 时必须在 finally 里取消后台任务，
把「断开」变成一次真实的止损。本实现把它记成 `aborted` 计数，供监控看「有多少请求被中途放弃」。

诚实边界：流式路径走 `gateway.stream`，它是**单次调用、不重试**（一旦有 chunk 发出，
重试就会造成内容重复），因此也不参与预算预扣 —— 生产应基于最后一帧 usage 补结算。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, AsyncIterator, Callable

from src.infra.core.errors import ValidationError
from src.infra.core.logging import get_logger

from .workflow.graph import NodeContext, WorkflowRunner
from .workflow.schemas import WorkflowResult

logger = get_logger("ops.streaming")

RunnerFactory = Callable[[Callable[[dict[str, Any]], Any]], WorkflowRunner]
"""入参是一个 emitter，返回一个把该 emitter 注入进节点上下文的 Runner。

之所以是「工厂」而不是「一个 Runner 实例」：每次请求的 run_id、brief、emitter 都不同，
Runner 必须按请求构造；但流式的能力（队列、取消、计数）又是跨请求复用的。
"""


def format_sse(payload: dict[str, Any]) -> str:
    """把一个事件字典序列化成 SSE 帧。

    `event:` 用事件名，`data:` 用**整条 JSON**（含 node/run_id），
    这样前端既能按事件名订阅，也能从 data 里拿到归属信息 —— 两者都在，才不会二选一。
    """
    name = str(payload.get("event") or "message")
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {name}\ndata: {data}\n\n"


class OpsStreamer:
    """把一个工作流 run 转成 SSE 事件流的适配器。

    实现要点：emitter 就是往 `asyncio.Queue` 里放事件的异步闭包；runner 与消费循环
    用 `asyncio.wait(..., FIRST_COMPLETED)` 交错推进 —— 谁先就绪就先处理谁，
    既不会「等 run 跑完才开始推」，也不会「只推事件却把 run 挂着」。
    """

    def __init__(self, *, make_runner: RunnerFactory | None = None, max_events: int = 20_000) -> None:
        self._make_runner = make_runner
        self._max_events = max(1, max_events)
        self._runs = 0
        self._aborts = 0
        self._events = 0
        self._last_result: WorkflowResult | None = None

    @property
    def last_result(self) -> WorkflowResult | None:
        """最近一次跑完的 run 结果（用于「流式跑完后仍要拿到落库记录」的场景）。"""
        return self._last_result

    def stats(self) -> dict[str, Any]:
        return {"runs": self._runs, "aborted": self._aborts, "events": self._events}

    async def stream(
        self, run_id: str, *, resume: bool = False, make_runner: RunnerFactory | None = None
    ) -> AsyncIterator[str]:
        """流式执行一个 run，逐条 yield SSE 帧。

        调用方（FastAPI 的 StreamingResponse）在客户端断开时会 `aclose()` 本生成器，
        这里的 finally 负责取消后台任务并记账。

        `make_runner` 允许**按请求**提供 runner 工厂（run_id / brief 是请求级参数），
        未提供时用构造时注入的那一个。
        """
        factory = make_runner or self._make_runner
        if factory is None:
            raise ValidationError("OpsStreamer 未提供 runner 工厂")

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def emitter(payload: dict[str, Any]) -> None:
            queue.put_nowait(payload)

        runner = factory(emitter)
        run_task = asyncio.create_task(runner.run(run_id, resume=resume))
        self._runs += 1
        self._last_result = None
        drained = False

        try:
            while True:
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({getter, run_task}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    payload = getter.result()
                    self._events += 1
                    yield format_sse(payload)
                    if self._events >= self._max_events:
                        # 护栏：事件量有上界，避免异常情况把流无限拉长
                        getter.cancel()
                        logger.warning(
                            "stream.event_limit",
                            extra={"event": "ops_stream", "status": "degraded", "run_id": run_id,
                                   "changed_keys": self._max_events},
                        )
                        break
                    continue
                # run 先结束：清空队列里剩余事件后收尾
                getter.cancel()
                while not queue.empty():
                    self._events += 1
                    yield format_sse(queue.get_nowait())
                if run_task.exception() is not None:
                    # run 以异常收尾：把错误也作为一帧推给客户端，而不是让流静默截断
                    exc = run_task.exception()
                    code = getattr(exc, "code", type(exc).__name__)
                    self._events += 1
                    yield format_sse(
                        {"event": "run.error", "run_id": run_id, "code": code, "message": str(exc)[:200]}
                    )
                drained = True
                break
        except GeneratorExit:
            # 客户端断开（aclose）；真正的资源释放交给 finally
            raise
        finally:
            if run_task.done():
                if not run_task.cancelled() and run_task.exception() is None:
                    self._last_result = run_task.result()
                else:
                    # 把异常取出来，避免 "exception was never retrieved" 噪声
                    run_task.exception()
            else:
                # 客户端断开 或 消费者提前退出：取消后台 run，释放模型调用与线程
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
                self._aborts += 1
                logger.warning(
                    "stream.aborted",
                    extra={
                        "event": "ops_stream",
                        "status": "aborted",
                        "run_id": run_id,
                        "reason": "client_disconnected",
                        "changed_keys": self._events,
                    },
                )
            if drained:
                logger.info(
                    "stream.done",
                    extra={"event": "ops_stream", "status": "ok", "run_id": run_id, "changed_keys": self._events},
                )

    def inject_emitter(self, ctx_template: NodeContext, emitter: Callable[[dict[str, Any]], Any]) -> NodeContext:
        """便捷方法：把 emitter 注入节点上下文模板（节点代码本身无需感知流式）。"""
        return replace(ctx_template, emitter=emitter)


__all__ = ["OpsStreamer", "RunnerFactory", "format_sse"]
