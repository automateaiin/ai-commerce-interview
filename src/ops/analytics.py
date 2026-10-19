"""运营分析：可复算的指标 + 渠道归因 + 素材侧可执行建议。

面试要点：**为什么归因不用大模型算**。
归因输出会被用来「追加预算 / 砍掉某渠道」——它是**决策依据**，所以必须满足两条：
① 可复算（同样输入永远得到同样输出，能被业务方拿着原始明细核对）；
② 可解释（每个数字都能指回公式，而不是一句「模型认为 A 渠道贡献最大」）。
大模型负责的是**表达**（标题、文案、剪辑点的措辞），而不是**计算**。
把这两件事混在一起，是运营系统最常见、也最难解释的坑。

模块只做只读聚合（渠道数据 + 素材表现），不改业务状态：
- 指标：CTR / CVR / ROAS / AOV；
- 归因：相对基线的**增量**与**占比**，`basis` 字段写明公式；
- 素材建议：爆款评分（启发式加权）+ 标题 + 剪辑建议 —— 直接作为下一轮内容生产的 brief 输入。

诚实边界：爆款评分是**启发式加权**，不是训练出来的排序模型；权重写在代码里、可被业务方质疑与调整，
这比给一个不透明的分数更诚实。见 PROJECT.md 第 8 章。
"""

from __future__ import annotations

from typing import Any, Sequence

from src.infra.core.errors import ValidationError
from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage

from .schemas import Attribution, MaterialSuggestion, Metric

logger = get_logger("ops.analytics")

# 爆款评分的启发式权重（显式写死，便于被质疑、被调整）
_ENGAGEMENT_WEIGHTS = {"views": 0.30, "likes": 0.25, "saves": 0.20, "comments": 0.15, "shares": 0.10}


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


class OpsAnalytics:
    """运营分析器：输入渠道与素材明细，输出指标、归因、素材建议。"""

    def __init__(self, *, top_materials: int = 5) -> None:
        self._top_materials = max(1, int(top_materials))

    def report(self, payload: dict[str, Any]) -> dict[str, Any]:
        channels = payload.get("channels") or []
        materials = payload.get("materials") or []
        if not isinstance(channels, list) or not isinstance(materials, list):
            raise ValidationError("channels 与 materials 必须是数组")
        if not channels and not materials:
            raise ValidationError("运营分析至少需要 channels 或 materials 之一")

        with_ = stage("ops.analytics.report", channels=len(channels), materials=len(materials))
        with with_:
            metrics = self._metrics(channels)
            attribution = self._attribution(channels)
            suggestions = self._materials(materials, payload.get("market"))

        return {
            "metrics": [item.to_payload() for item in metrics],
            "attribution": [item.to_payload() for item in attribution],
            "materials": [item.to_payload() for item in suggestions],
            "notes": "指标与归因均为可复算的确定性计算；素材建议为启发式评分，供内容工作流消费。",
        }

    # -- 指标 --------------------------------------------------------------- #
    def _metrics(self, channels: Sequence[dict[str, Any]]) -> list[Metric]:
        impressions = sum(_num(item.get("impressions")) for item in channels)
        clicks = sum(_num(item.get("clicks")) for item in channels)
        orders = sum(_num(item.get("orders")) for item in channels)
        revenue = sum(_num(item.get("revenue")) for item in channels)
        spend = sum(_num(item.get("spend")) for item in channels)

        metrics = [
            Metric(key="ctr", label="整体点击率", value=_safe_div(clicks, impressions), unit="ratio"),
            Metric(key="cvr", label="整体转化率", value=_safe_div(orders, clicks), unit="ratio"),
            Metric(key="roas", label="整体投产比", value=_safe_div(revenue, spend), unit="ratio"),
            Metric(key="aov", label="客单价", value=_safe_div(revenue, orders), unit="currency"),
            Metric(key="revenue", label="总营收", value=round(revenue, 4), unit="currency"),
            Metric(key="orders", label="总订单", value=round(orders, 4), unit="count"),
        ]
        for item in channels:
            channel = str(item.get("channel") or "unknown")
            metrics.append(
                Metric(
                    key=f"ctr.{channel}",
                    label=f"{channel} 点击率",
                    value=_safe_div(_num(item.get("clicks")), _num(item.get("impressions"))),
                    unit="ratio",
                )
            )
            metrics.append(
                Metric(
                    key=f"roas.{channel}",
                    label=f"{channel} 投产比",
                    value=_safe_div(_num(item.get("revenue")), _num(item.get("spend"))),
                    unit="ratio",
                )
            )
        return metrics

    # -- 归因 --------------------------------------------------------------- #
    def _attribution(self, channels: Sequence[dict[str, Any]]) -> list[Attribution]:
        """相对基线的增量归因：delta = orders - baseline_orders；share = delta / Σ正向增量。"""
        deltas: list[tuple[str, float, float]] = []  # (channel, delta, orders)
        for item in channels:
            channel = str(item.get("channel") or "unknown")
            orders = _num(item.get("orders"))
            baseline = _num(item.get("baseline_orders"))
            deltas.append((channel, orders - baseline, baseline))

        total_positive = sum(max(delta, 0.0) for _, delta, _ in deltas)
        results: list[Attribution] = []
        for channel, delta, baseline in deltas:
            share = _safe_div(max(delta, 0.0), total_positive)
            results.append(
                Attribution(
                    channel=channel,
                    metric_key="orders",
                    delta=round(delta, 4),
                    share=share,
                    basis=f"delta = orders - baseline_orders；share = max(delta,0) / Σ max(delta,0)（共 {len(deltas)} 渠道）",
                )
            )
        logger.info(
            "ops.analytics.attribution",
            extra={"event": "ops_analytics", "status": "ok", "changed_keys": len(results)},
        )
        return results

    # -- 素材建议 ----------------------------------------------------------- #
    def _materials(self, materials: Sequence[dict[str, Any]], market: str | None) -> list[MaterialSuggestion]:
        if not materials:
            return []
        peaks = {key: max((_num(item.get(key)) for item in materials), default=0.0) for key in _ENGAGEMENT_WEIGHTS}
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in materials:
            score = 0.0
            for key, weight in _ENGAGEMENT_WEIGHTS.items():
                peak = peaks.get(key) or 0.0
                score += weight * (_num(item.get(key)) / peak if peak else 0.0)
            scored.append((round(min(score, 1.0), 4), item))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("source_id") or "")))
        suggestions: list[MaterialSuggestion] = []
        for score, item in scored[: self._top_materials]:
            source_id = str(item.get("source_id") or item.get("id") or "unknown")
            suggestions.append(
                MaterialSuggestion(
                    source_id=source_id,
                    viral_score=score,
                    title=self._title(item, source_id, market),
                    editing=self._editing(item, score),
                )
            )
        logger.info(
            "ops.analytics.materials",
            extra={"event": "ops_analytics", "status": "ok", "changed_keys": len(suggestions)},
        )
        return suggestions

    def _title(self, item: dict[str, Any], source_id: str, market: str | None) -> str:
        """确定性标题建议：按主导互动信号选钩子类型，避免同分素材给出同一句话。"""
        subject = str(item.get("title") or item.get("product_name") or f"素材{source_id}")
        dominant = max(
            _ENGAGEMENT_WEIGHTS,
            key=lambda key: _num(item.get(key)) / (_ENGAGEMENT_WEIGHTS[key] or 1.0),
        )
        hooks = {
            "views": "热门话题切入",
            "likes": "情绪共鸣向",
            "saves": "实用收藏向",
            "comments": "争议互动向",
            "shares": "社交传播向",
        }
        suffix = f"｜{market}市场" if market else ""
        return f"{subject}｜{hooks[dominant]}{suffix}"

    def _editing(self, item: dict[str, Any], score: float) -> list[str]:
        """剪辑建议：前 3 秒结构 + 时长 + 复用策略，都是可直接执行的指令。"""
        suggestions = ["前 3 秒直给核心卖点，不要铺垫"]
        if score >= 0.6:
            suggestions.append("已跑量，做 15s 短切版本投流，保留原爆点结构")
        else:
            suggestions.append("补一段真实使用场景提升完播率")
        if _num(item.get("views")):
            suggestions.append("截取评论区高频问题作为二次创作选题")
        return suggestions


__all__ = ["OpsAnalytics"]
