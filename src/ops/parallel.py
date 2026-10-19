"""并行生成：`asyncio.gather` + `run_in_executor`。

两种并行的分工必须讲清楚（面试常问「为什么不用线程池跑全部」）：

- **异步单元**（模型调用、检索）走 `asyncio`：它们是在等网络，`await` 就能重叠；
- **阻塞单元**（出图、写文件、跑 FFmpeg 这类本地重活）必须走 `run_in_executor`：
  `await` 一个阻塞函数等于把整个事件循环卡住，同层的其它节点会一起被拖慢，
  「并行」变成「一起变慢」。

三条护栏：
1) 最大并发有上界（信号量）—— fan-out 不是无限并发，否则一次请求能把线程池/连接池打满；
2) 同批单元必须声明**批内依赖**并且顺序正确，依赖没满足的单元不允许混进同一批；
3) 收益用**实测**说话：分别给出墙钟耗时与「逐个串行下界」（各单元耗时之和），
   不编造加速比。

诚实边界：这里测到的加速比来自**本地阻塞步骤**的真实并发；模型侧的延迟取决于供应商网络，
本仓库的离线度量不含这一部分，因此不能拿这里的数字去宣称线上端到端加速比。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable

from src.infra.core.errors import ValidationError
from src.infra.core.logging import get_logger

logger = get_logger("ops.parallel")


@dataclass(slots=True)
class ParallelUnit:
    """一个可并行的工作单元。mode=thread 表示走线程池（阻塞实现）。"""

    name: str
    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    mode: str = "thread"
    depends_on: tuple[str, ...] = ()


@dataclass(slots=True)
class UnitOutcome:
    name: str
    ok: bool
    value: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    mode: str = "thread"

    def to_payload(self) -> dict[str, Any]:
        return {
            "unit": self.name,
            "ok": self.ok,
            "mode": self.mode,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "value": self.value if self.ok else None,
        }


@dataclass(slots=True)
class ParallelReport:
    """并行执行报告：包含**实测**耗时与串行下界，用于证明「并行真的快」。"""

    outcomes: list[UnitOutcome] = field(default_factory=list)
    wall_ms: float = 0.0
    sequential_ms: float = 0.0
    peak_concurrency: int = 0
    max_concurrency: int = 1

    @property
    def speedup(self) -> float:
        if self.wall_ms <= 0:
            return 1.0
        return round(self.sequential_ms / self.wall_ms, 3)

    @property
    def failures(self) -> list[str]:
        return [item.name for item in self.outcomes if not item.ok]

    def to_payload(self) -> dict[str, Any]:
        return {
            "units": [item.to_payload() for item in self.outcomes],
            "wall_ms": round(self.wall_ms, 2),
            "sequential_ms": round(self.sequential_ms, 2),
            "speedup": self.speedup,
            "peak_concurrency": self.peak_concurrency,
            "max_concurrency": self.max_concurrency,
            "failures": self.failures,
            "note": "加速比来自本地阻塞步骤的实测；不含模型侧网络延迟。",
        }


class ParallelRunner:
    """并行执行器：同层/同批单元并发，并发度有上界。"""

    def __init__(self, *, max_concurrency: int = 4) -> None:
        self._max = max(1, int(max_concurrency))
        self._running = 0
        self._peak = 0

    @property
    def max_concurrency(self) -> int:
        return self._max

    async def run_blocking(self, name: str, fn: Callable[..., Any], *args: Any) -> Any:
        """把**阻塞**函数丢到默认线程池执行。返回函数结果，异常原样抛出。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(fn, *args))

    async def run(self, units: Iterable[ParallelUnit]) -> ParallelReport:
        """并发执行一批单元，返回含实测耗时的报告。

        批内单元各自可能失败：这里**不**因为一个失败就取消其它单元
        （已发出的调用/已付的成本不该被白扔掉），失败如实记进报告由调用方决策。
        """
        batch = list(units)
        self._validate_batch(batch)
        semaphore = asyncio.Semaphore(self._max)
        loop = asyncio.get_running_loop()
        self._peak = 0

        async def execute(unit: ParallelUnit) -> UnitOutcome:
            async with semaphore:
                self._running += 1
                self._peak = max(self._peak, self._running)
                started = time.perf_counter()
                try:
                    if unit.mode == "thread":
                        value = await loop.run_in_executor(None, partial(unit.fn, *unit.args))
                    else:
                        value = await unit.fn(*unit.args)
                    elapsed = (time.perf_counter() - started) * 1000
                    return UnitOutcome(name=unit.name, ok=True, value=value, duration_ms=elapsed, mode=unit.mode)
                except Exception as exc:  # noqa: BLE001 - 失败要留痕，不能吞掉
                    elapsed = (time.perf_counter() - started) * 1000
                    logger.warning(
                        "ops.parallel.unit_failed",
                        extra={
                            "event": "ops_parallel",
                            "status": "failed",
                            "node": unit.name,
                            "reason": f"{type(exc).__name__}: {exc}"[:160],
                            "duration_ms": round(elapsed, 2),
                        },
                    )
                    return UnitOutcome(
                        name=unit.name,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}"[:160],
                        duration_ms=elapsed,
                        mode=unit.mode,
                    )
                finally:
                    self._running -= 1

        started = time.perf_counter()
        outcomes = await asyncio.gather(*(execute(unit) for unit in batch))
        wall_ms = (time.perf_counter() - started) * 1000
        report = ParallelReport(
            outcomes=list(outcomes),
            wall_ms=wall_ms,
            # 串行下界 = 各单元耗时之和（串行执行时不会有任何重叠）
            sequential_ms=sum(item.duration_ms for item in outcomes),
            peak_concurrency=self._peak,
            max_concurrency=self._max,
        )
        logger.info(
            "ops.parallel.done",
            extra={
                "event": "ops_parallel",
                "status": "ok",
                "changed_keys": len(report.outcomes),
                "duration_ms": round(report.wall_ms, 2),
                "reason": f"speedup={report.speedup}, peak={report.peak_concurrency}",
            },
        )
        return report

    @staticmethod
    def _validate_batch(batch: list[ParallelUnit]) -> None:
        """批内依赖必须存在且**声明顺序正确**：不允许「依赖后面的单元」。"""
        if not batch:
            return
        names = [unit.name for unit in batch]
        if len(set(names)) != len(names):
            raise ValidationError("并行单元重名", detail={"units": names})
        position = {name: index for index, name in enumerate(names)}
        for unit in batch:
            for dep in unit.depends_on:
                if dep not in position:
                    raise ValidationError(
                        "并行单元依赖不在同一批内，应改为分层串行",
                        detail={"unit": unit.name, "missing_dependency": dep},
                    )
                if position[dep] >= position[unit.name]:
                    raise ValidationError(
                        "并行单元依赖顺序错误（依赖必须在被依赖者之后）",
                        detail={"unit": unit.name, "out_of_order_dependency": dep},
                    )


__all__ = ["ParallelReport", "ParallelRunner", "ParallelUnit", "UnitOutcome"]
