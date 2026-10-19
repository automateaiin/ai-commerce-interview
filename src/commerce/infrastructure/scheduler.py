"""进程内周期调度器：订单超时取消 sweeper 的宿主。

为什么不引 Celery（M1）：
- 业务上只有「定时扫过期订单」这一类任务，引一个 broker + worker + 结果后端
  换来的运维成本和收益完全不成比例；
- `api.py` 的 lifespan 已经天然提供了应用的启动/关闭时机，把周期性任务挂在这里
  可以让「应用关闭 → 任务立即停止」成为必然，而不是「worker 还活着但 DB 已经关了」。

三个必须做对的细节：
1. **异常绝不能打死循环**：一次 DB 抖动不能让 sweeper 永久停摆，因此吞异常并记日志；
2. **优雅关闭**：用 `asyncio.Event` 而不是裸 `cancel()`，让正在执行的那一轮跑完再退出，
   避免「取消到一半、库存回滚了一半」；
3. **首轮延迟启动**：不用 `sleep(interval)` 打头，否则刚启动就空跑一轮。
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from ...infra.core.logging import get_logger

logger = get_logger("commerce.scheduler")


class PeriodicScheduler:
    """单任务周期调度器。`interval_s <= 0` 表示禁用（显式关闭，而不是静默不跑）。"""

    def __init__(
        self,
        name: str,
        *,
        interval_s: float,
        job: Callable[[], Awaitable[None]],
    ) -> None:
        self._name = name
        self._interval = interval_s
        self._job = job
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._runs = 0
        self._failures = 0

    @property
    def enabled(self) -> bool:
        return self._interval > 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def stats(self) -> dict[str, object]:
        return {"name": self._name, "enabled": self.enabled, "running": self.running, "runs": self._runs, "failures": self._failures}

    async def _loop(self) -> None:
        while not self._stop.is_set():
            self._runs += 1
            try:
                await self._job()
            except asyncio.CancelledError:  # 关闭时的正常路径，必须原样上抛
                raise
            except Exception:  # noqa: BLE001 - 单轮失败不能让周期性任务永久停摆
                self._failures += 1
                logger.warning(
                    "scheduler.job_failed",
                    extra={"event": "scheduler", "job": self._name, "runs": self._runs, "status": "error"},
                    exc_info=True,
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue

    async def start(self) -> None:
        if not self.enabled or self.running:
            logger.info(
                "scheduler.disabled" if not self.enabled else "scheduler.already_running",
                extra={"event": "startup", "job": self._name, "enabled": self.enabled, "interval_s": self._interval},
            )
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name=f"scheduler:{self._name}")
        logger.info(
            "scheduler.started",
            extra={"event": "startup", "job": self._name, "interval_s": self._interval, "status": "ok"},
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=max(self._interval, 5) + 5)
        except asyncio.TimeoutError:  # pragma: no cover - 只会在任务卡死时触发
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        finally:
            self._task = None
            logger.info(
                "scheduler.stopped",
                extra={"event": "shutdown", "job": self._name, "runs": self._runs, "status": "ok"},
            )


__all__ = ["PeriodicScheduler"]
