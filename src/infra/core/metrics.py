"""轻量指标聚合。

刻意不引入 Prometheus 客户端：面试项目里更重要的是「指标口径是什么」，
而不是依赖体积。这里只做进程内聚合，字段与真实监控系统对齐，
后续要接 Prometheus 只需要把 snapshot() 映射成 gauge/counter 即可。

口径说明（被追问时要能讲）：
- 延迟用 P95 而不是平均值：平均值会被少量极慢请求掩盖，
  而 P95 直接对应「用户感知的慢」。
- 错误率按「归一化错误码」统计，而不是按 HTTP 状态码，
  因为上游限流和本服务限流都是 429，但归因完全不同。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

_MAX_SAMPLES_PER_KEY = 2000


class MetricsRegistry:
    """进程内指标注册表。单事件循环下操作是安全的（无跨线程共享）。"""

    def __init__(self) -> None:
        self._counters: Counter[tuple[str, str]] = Counter()
        self._latency: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._tokens: Counter[tuple[str, str]] = Counter()
        self._cost: dict[str, float] = defaultdict(float)
        # Metering：按「Service/Agent」聚合的调用量与成本明细。
        # 与 token/cost 聚合的区别在于维度：这里回答的是「哪个业务线的哪个 Agent 花了多少钱」，
        # 而 provider/model 维度回答的是「哪个模型贵」—— 两个问题必须分别可查。
        self._metering: dict[str, dict[str, Any]] = {}

    # -- 写入 --------------------------------------------------------------- #
    def incr(self, kind: str, key: str, value: int = 1) -> None:
        self._counters[(kind, key)] += value

    def observe_latency(self, kind: str, key: str, duration_ms: float) -> None:
        samples = self._latency[(kind, key)]
        samples.append(duration_ms)
        if len(samples) > _MAX_SAMPLES_PER_KEY:
            # 简易蓄水池：保留最近的样本，控制内存
            del samples[: len(samples) - _MAX_SAMPLES_PER_KEY]

    def observe_tokens(self, provider: str, model: str, prompt: int, completion: int, cost_usd: float = 0.0) -> None:
        self._tokens[("prompt", f"{provider}/{model}")] += prompt
        self._tokens[("completion", f"{provider}/{model}")] += completion
        self._cost[f"{provider}/{model}"] += cost_usd

    def observe_metering(
        self,
        *,
        service: str,
        agent: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """按 Service/Agent 聚合一次模型调用。

        这是「成本按业务线/Agent 分摊」的原始数据 —— 没有它，只知道总花费，
        一旦某个 Agent 跑飞就只能靠猜。
        """
        bucket = self._metering.setdefault(
            f"{service}/{agent}",
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms_total": 0.0,
                "models": {},
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += prompt_tokens + completion_tokens
        bucket["cost_usd"] += cost_usd
        bucket["latency_ms_total"] += latency_ms
        by_model = bucket["models"].setdefault(model, {"calls": 0, "total_tokens": 0, "cost_usd": 0.0})
        by_model["calls"] += 1
        by_model["total_tokens"] += prompt_tokens + completion_tokens
        by_model["cost_usd"] += cost_usd

    # -- 读取 --------------------------------------------------------------- #
    @staticmethod
    def _percentile(samples: list[float], q: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
        return round(ordered[index], 2)

    def snapshot(self) -> dict[str, Any]:
        requests_total = sum(v for (kind, _), v in self._counters.items() if kind == "request")
        errors_total = sum(v for (kind, _), v in self._counters.items() if kind == "error")
        latency = {
            f"{kind}:{key}": {
                "count": len(samples),
                "p50_ms": self._percentile(samples, 0.5),
                "p95_ms": self._percentile(samples, 0.95),
                "max_ms": round(max(samples), 2) if samples else 0.0,
            }
            for (kind, key), samples in self._latency.items()
        }
        return {
            "requests_total": requests_total,
            "errors_total": errors_total,
            "error_rate": round(errors_total / requests_total, 4) if requests_total else 0.0,
            "counters": {f"{kind}:{key}": value for (kind, key), value in self._counters.items()},
            "latency": latency,
            "tokens": {f"{kind}:{key}": value for (kind, key), value in self._tokens.items()},
            "cost_usd": {key: round(value, 6) for key, value in self._cost.items()},
            "metering": self.metering_snapshot(),
        }

    def metering_snapshot(self) -> dict[str, Any]:
        """按 Service/Agent 的用量与成本汇总（供 /metrics 与成本端点复用）。"""
        by_key: dict[str, Any] = {}
        for key, bucket in self._metering.items():
            calls = max(bucket["calls"], 1)
            by_key[key] = {
                "calls": bucket["calls"],
                "prompt_tokens": bucket["prompt_tokens"],
                "completion_tokens": bucket["completion_tokens"],
                "total_tokens": bucket["total_tokens"],
                "cost_usd": round(bucket["cost_usd"], 6),
                "avg_latency_ms": round(bucket["latency_ms_total"] / calls, 2),
                "models": {
                    name: {
                        "calls": item["calls"],
                        "total_tokens": item["total_tokens"],
                        "cost_usd": round(item["cost_usd"], 6),
                    }
                    for name, item in bucket["models"].items()
                },
            }
        return {
            "by_service_agent": by_key,
            "totals": {
                "calls": sum(item["calls"] for item in by_key.values()),
                "total_tokens": sum(item["total_tokens"] for item in by_key.values()),
                "cost_usd": round(sum(item["cost_usd"] for item in by_key.values()), 6),
            },
        }


metrics = MetricsRegistry()
