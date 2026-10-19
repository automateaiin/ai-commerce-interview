"""训练数据导出：格式转换 + **校验门禁** + 版本与样本统计。

面试要点：
- ShareGPT / Alpaca / OpenAI JSONL 只是不同训练框架的数据组织格式，
  关键是**按目标框架转换字段与角色结构**，而不是改文件后缀；
- 导出前必须过校验门禁：字段完整、角色顺序合法、无空样本、无重复、**无敏感信息**；
- 文件带版本号与样本统计，「审核通过」不等于「每条都能训练」——
  还要按训练目标筛选、查分布（低代表性、过时事实、潜在偏差都要剔除）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..infra.core.errors import ValidationError
from ..infra.core.logging import get_logger
from .dedupe import content_hash
from .schemas import SamplePurpose, TrainingSample

logger = get_logger("data.export")

SUPPORTED_FORMATS = ("sharegpt", "alpaca", "openai")


@dataclass(slots=True)
class ValidationIssue:
    index: int
    reason: str
    detail: str = ""


@dataclass(slots=True)
class ExportArtifact:
    format: str
    version: str
    records: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        return len(self.records)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(record, ensure_ascii=False) for record in self.records)


class Processor:
    """格式转换 + 校验门禁 + 版本标注。"""

    def __init__(
        self,
        *,
        version: str = "v1",
        system_prompt: str | None = None,
        verifier: Any | None = None,
    ) -> None:
        self._version = version
        self._system = system_prompt
        # 导出门禁复用平台的脱敏规则集（可以是独立实例，两次扫描互不影响）
        self._verifier = verifier

    def export(
        self,
        samples: Sequence[TrainingSample],
        *,
        fmt: str = "sharegpt",
        purpose: SamplePurpose | None = SamplePurpose.SFT_STYLE,
    ) -> ExportArtifact:
        if fmt not in SUPPORTED_FORMATS:
            raise ValidationError(f"不支持的导出格式: {fmt}", detail={"supported": list(SUPPORTED_FORMATS)})

        artifact = ExportArtifact(format=fmt, version=self._version)
        seen: set[str] = set()
        candidates = [
            sample for sample in samples if purpose is None or sample.purpose is purpose
        ]

        for index, sample in enumerate(candidates):
            issue = self._validate(index, sample, seen)
            if issue is not None:
                artifact.rejected.append(issue)
                continue
            seen.add(content_hash(json.dumps(sample.messages, ensure_ascii=False)))
            artifact.records.append(self._convert(sample, fmt))

        artifact.stats = {
            "version": self._version,
            "format": fmt,
            "input_samples": len(samples),
            "exported": len(artifact.records),
            "rejected": len(artifact.rejected),
            "duplicates": sum(1 for issue in artifact.rejected if issue.reason == "duplicate"),
            "sensitive_blocked": sum(1 for issue in artifact.rejected if issue.reason == "sensitive"),
        }
        logger.info(
            "data.export.done",
            extra={
                "event": "data_export",
                "status": "ok",
                "reason": f"fmt={fmt},exported={len(artifact.records)},rejected={len(artifact.rejected)}",
            },
        )
        return artifact

    # -- 校验门禁 ---------------------------------------------------------- #
    def _validate(self, index: int, sample: TrainingSample, seen: set[str]) -> ValidationIssue | None:
        messages = sample.messages
        if len(messages) < 2:
            return ValidationIssue(index, "too_short", "样本至少需要一问一答")
        if any(not str(message.get("content") or "").strip() for message in messages):
            return ValidationIssue(index, "empty_content")
        if messages[0].get("role") != "user":
            return ValidationIssue(index, "role_order", "角色必须以 user 开头")
        expected = ["user", "assistant"]
        for position, message in enumerate(messages):
            if message.get("role") != expected[position % 2]:
                return ValidationIssue(index, "role_order", "角色必须 user/assistant 交替")
        fingerprint = content_hash(json.dumps(messages, ensure_ascii=False))
        if fingerprint in seen:
            return ValidationIssue(index, "duplicate")
        # 导出门禁：最后一次敏感值扫描（入库时脱敏 + 导出时再扫 = 双保险）
        joined = "\n".join(str(message.get("content")) for message in messages)
        from .scrub import Scrubber

        hit = (self._verifier or Scrubber()).scan(joined)
        if hit:
            return ValidationIssue(index, "sensitive", ",".join(sorted(hit)))
        return None

    # -- 格式转换 ---------------------------------------------------------- #
    def _convert(self, sample: TrainingSample, fmt: str) -> dict[str, Any]:
        known = {
            "source": sample.source_staging_ids,
            "scene": sample.scene,
            "version": self._version,
        }
        if fmt == "sharegpt":
            records: list[dict[str, str]] = []
            if self._system:
                records.append({"from": "system", "value": self._system})
            records.extend(
                {
                    "from": "human" if message["role"] == "user" else "gpt",
                    "value": message["content"],
                }
                for message in sample.messages
            )
            return {"id": sample.sample_id, "conversations": records, **known}
        if fmt == "alpaca":
            return {
                "id": sample.sample_id,
                "instruction": sample.messages[0]["content"],
                "input": "",
                "output": sample.messages[-1]["content"],
                **known,
            }
        return {"id": sample.sample_id, "messages": list(sample.messages), **known}
