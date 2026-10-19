"""自进化闭环：回流事件 → 自动评测集 → 自动评分 → 模板 / 策略迭代建议。

这条链路是「AI 运营系统」区别于「写死脚本」的关键，也是面试最容易被追问的一环：

1) **数据哪里来**：不是靠人工标注，而是靠业务运行中自然产生的**回流事件** ——
   人工纠正（correction）、审核改稿（review）、工具调用（tool_call）、
   人工审批（hitl）、直播口播（live_asr）。这些事件本来就是运营的日常动作，
   顺手沉淀就是现成的评测与训练语料（本模块复用项目三的 `RefluxCollector`）。

2) **怎么自动出题**：把「有提问 + 有权威答案（或金标切片）」的回流记录投影成评测题。
   关键细节是**带 group**：同一条业务线/同一个商品的问题必须同组，切分时整组进同一侧，
   否则同一商品的问题一半进评测集、一半当参考答案，就是**数据泄漏**，评测分虚高。

3) **怎么自动评分**：复用项目三的启发式 `AutoGrader`（证据一致性 + 关键覆盖 + 拒答策略）。
   它是启发式不是裁判 —— 所以分数用于**发现趋势**（哪类问题在退化），不用于对个人定责。

4) **迭代建议的边界（最重要）**：本模块**只产出建议，不自动发布**。
   模板迭代注册为新版本但**不激活**，active 仍是旧版本，必须人工审核后手动 activate；
   策略迭代（比如检索 top_k）也只是写进建议清单。
   理由：自动改生产提示词 = 把「系统稳定性」交给一次未经复核的模型输出，风险不对称。

诚实边界：离线环境下评测分来自启发式评分；真实闭环需接入线上真实答案与检索证据。见 PROJECT.md 第 8 章。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.dataplatform.quiz import AutoGrader, Expect, QuizItem, QuizType
from src.dataplatform.reflux import RefluxCollector, RefluxSource
from src.infra.core.logging import get_logger
from src.infra.core.tracing import stage

from .schemas import EvalCase, StrategyIteration, TemplatePatch

logger = get_logger("ops.selfevolve")

_PASS_TARGET = 0.7
_TARGET_TEMPLATE = "content.system"
_BASE_TEMPLATE = (
    "你是高端美妆电商的内容运营助手。生成选题、卖点与文案时必须依据检索到的证据；"
    "证据不足时明确说明无法确认，不得编造成分、功效或价格。"
)


def _looks_like_refusal(question: str) -> bool:
    return any(marker in question for marker in ("多少钱", "价格", "最低价", "优惠"))


@dataclass(slots=True)
class EvolutionCycle:
    """一次自进化循环的产物（全部是建议，不改变线上行为）。"""

    cycle_id: int
    sources: dict[str, int] = field(default_factory=dict)
    eval_cases: list[EvalCase] = field(default_factory=list)
    scoring: dict[str, Any] = field(default_factory=dict)
    template_patch: TemplatePatch | None = None
    strategy: list[StrategyIteration] = field(default_factory=list)
    notes: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "sources": dict(self.sources),
            "eval_cases": [case.model_dump() for case in self.eval_cases],
            "scoring": self.scoring,
            "template_patch": self.template_patch.to_payload() if self.template_patch else None,
            "strategy": [item.to_payload() for item in self.strategy],
            "notes": self.notes,
        }


class SelfEvolveLoop:
    """自进化闭环：采集回流 → 出评测题 → 评分 → 产出模板/策略建议。"""

    def __init__(
        self,
        *,
        prompts: Any,
        reflux: RefluxCollector | None = None,
        grader: AutoGrader | None = None,
        max_events: int = 500,
        target_template: str = _TARGET_TEMPLATE,
    ) -> None:
        self._prompts = prompts
        self._reflux = reflux or RefluxCollector(max_records=max(1, max_events))
        self._grader = grader or AutoGrader()
        self._target = target_template
        self._cycles: list[EvolutionCycle] = []
        self._ensure_template()

    def _ensure_template(self) -> None:
        """目标模板必须存在。缺失时注册一个基线（YAML 不可用的离线环境也能工作）。"""
        if self._target not in self._prompts.names():
            self._prompts.register(self._target, _BASE_TEMPLATE, version="v1", note="selfevolve:auto_base")

    # -- 采集 ---------------------------------------------------------------- #
    def ingest(self, events: Sequence[dict[str, Any]], *, default_source: str = "review") -> dict[str, int]:
        """把回流事件按来源分组归一化后写入收集器；脏数据被拒绝并计数，不静默吞掉。"""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in events:
            item = dict(raw)
            source = str(item.pop("source", default_source) or default_source)
            grouped.setdefault(source, []).append(item)
        counts: dict[str, int] = {}
        for source, payloads in grouped.items():
            records = self._reflux.add_many(RefluxSource(source), payloads)
            counts[source] = len(records)
        logger.info(
            "ops.selfevolve.ingest",
            extra={"event": "ops_selfevolve", "status": "ok", "changed_keys": len(events)},
        )
        return counts

    # -- 出题 ---------------------------------------------------------------- #
    def build_eval_cases(self) -> list[EvalCase]:
        """回流记录 → 评测题。只有「有提问 + 有答案」的记录能出题。"""
        cases: list[EvalCase] = []
        for record in self._reflux.records():
            if not record.question or not record.answer:
                continue
            case_type = "refusal" if _looks_like_refusal(record.question) else "fact"
            cases.append(
                EvalCase(
                    case_id=f"reflux_{record.record_id}",
                    question=record.question,
                    gold_answer=record.answer,
                    gold_chunk_ids=list(record.gold_chunk_ids),
                    group=record.group,
                    case_type=case_type,
                    source=record.source.value,
                )
            )
        return cases

    # -- 评分 ---------------------------------------------------------------- #
    def score(self, cases: Sequence[EvalCase], answers: dict[str, str] | None, evidence: dict[str, Sequence[str]] | None) -> dict[str, Any]:
        """对候选答案自动评分。没有候选答案时显式降级（只产出评测集，不假装评了分）。"""
        if not answers:
            return {
                "status": "awaiting_answers",
                "graded": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
                "results": [],
                "notes": "已产出自动评测集；缺少待评答案，未执行自动评分。",
            }
        items = [
            QuizItem(
                quiz_id=case.case_id,
                quiz_type=QuizType(case.case_type),
                question=case.question,
                gold_answer=case.gold_answer,
                gold_chunk_ids=list(case.gold_chunk_ids),
                group=case.group,
                expect=Expect.REFUSAL if case.case_type == "refusal" else Expect.GROUNDED,
            )
            for case in cases
            if case.case_type in ("fact", "boundary", "refusal")
        ]
        report = self._grader.grade_batch(items, answers, evidence=evidence)
        report["status"] = "graded"
        return report

    # -- 迭代建议 ------------------------------------------------------------ #
    def propose_template_patch(self, scoring: dict[str, Any], cases: Sequence[EvalCase]) -> TemplatePatch:
        """产出模板迭代**建议**：注册为新版本但**不激活**，等人工审核。"""
        current = self._prompts.get(self._target)
        next_version = f"v{len(self._prompts.versions(self._target)) + 1}"

        graded = int(scoring.get("graded") or 0)
        pass_rate = float(scoring.get("pass_rate") or 0.0)
        if graded and pass_rate < _PASS_TARGET:
            extra = "回答必须先引用检索证据；缺少证据时明确说明无法确认，不得凭记忆补充成分或功效。"
            reason = f"自动评分未达标：pass_rate={pass_rate:.2f} < {_PASS_TARGET}（共 {graded} 题）"
        elif graded:
            extra = "回答先给结论再给依据，控制在 120 字以内，保留证据引用。"
            reason = f"自动评分达标（pass_rate={pass_rate:.2f}），建议收紧输出结构与长度以降低成本"
        else:
            extra = "回答必须引用检索证据，缺少证据时明确说明无法确认。"
            reason = f"基于 {len(cases)} 条回流评测题产出的基线约束（尚未评分）"

        suggested = f"{current.template}\n{extra}"
        self._prompts.register(
            self._target,
            suggested,
            version=next_version,
            note=f"selfevolve:cycle:{reason}",
            activate=False,  # 关键：只登记，不激活
        )
        return TemplatePatch(
            template=self._target,
            from_version=current.version,
            to_version=next_version,
            reason=reason,
            suggested_text=extra,
        )

    def propose_strategy(self, scoring: dict[str, Any]) -> list[StrategyIteration]:
        """策略迭代建议：改的是「怎么生成/怎么检索」，不是某一句话。"""
        graded = int(scoring.get("graded") or 0)
        pass_rate = float(scoring.get("pass_rate") or 0.0)
        iterations: list[StrategyIteration] = []
        if graded and pass_rate < _PASS_TARGET:
            iterations.append(
                StrategyIteration(
                    name="retrieval.top_k",
                    before=3,
                    after=5,
                    reason=f"pass_rate={pass_rate:.2f} 偏低，扩大召回候选以提升证据覆盖",
                )
            )
            iterations.append(
                StrategyIteration(
                    name="llm.logical_model",
                    before="chat-lite",
                    after="chat-pro",
                    reason="复杂问答失败率偏高，升级到旗舰模型",
                )
            )
        else:
            iterations.append(
                StrategyIteration(
                    name="retrieval.top_k",
                    before=3,
                    after=3,
                    reason="当前表现达标，保持召回规模，优先用短文案模型降本",
                )
            )
        return iterations

    # -- 应用 / 回滚（需人工触发） ------------------------------------------- #
    def apply_patch(self, patch: TemplatePatch) -> dict[str, Any]:
        """人工审核通过后**手动**激活新版本（本方法不会被 cycle 自动调用）。"""
        self._prompts.activate(patch.template, patch.to_version)
        logger.info(
            "ops.selfevolve.patch_activated",
            extra={"event": "ops_selfevolve", "status": "ok", "reason": patch.reason, "changed_keys": patch.to_version},
        )
        return {"template": patch.template, "active_version": self._prompts.active_version(patch.template)}

    def revert(self) -> dict[str, Any]:
        """回滚到上一版（版本顺序即回滚顺序）。"""
        item = self._prompts.rollback(self._target)
        return {"template": self._target, "active_version": item.version}

    # -- 编排 ---------------------------------------------------------------- #
    def cycle(
        self,
        events: Sequence[dict[str, Any]],
        *,
        answers: dict[str, str] | None = None,
        evidence: dict[str, Sequence[str]] | None = None,
        default_source: str = "review",
    ) -> EvolutionCycle:
        sources = self.ingest(events, default_source=default_source)
        with_ = stage("ops.selfevolve.cycle", events=len(events))
        with with_:
            cases = self.build_eval_cases()
            scoring = self.score(cases, answers, evidence)
            patch = self.propose_template_patch(scoring, cases)
            strategy = self.propose_strategy(scoring)
        cycle = EvolutionCycle(
            cycle_id=len(self._cycles) + 1,
            sources=sources,
            eval_cases=cases,
            scoring=scoring,
            template_patch=patch,
            strategy=strategy,
            notes="只产出建议：模板已注册为新版本但未激活，需人工审核后 apply_patch 生效。",
        )
        self._cycles.append(cycle)
        logger.info(
            "ops.selfevolve.cycle.done",
            extra={"event": "ops_selfevolve", "status": "ok", "changed_keys": cycle.cycle_id},
        )
        return cycle

    def report(self) -> dict[str, Any]:
        return {
            "reflux": self._reflux.report(),
            "template": {
                "name": self._target,
                "active_version": self._prompts.active_version(self._target),
                "versions": self._prompts.versions(self._target),
            },
            "cycles": [cycle.to_payload() for cycle in self._cycles],
        }


__all__ = ["EvolutionCycle", "SelfEvolveLoop"]
