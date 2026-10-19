"""LLaMA-Factory 数据集与训练配置产物。

面试要点（这是「AI 数据闭环」里的最后一公里，也是最容易被吹过的地方）：
- 产出物必须是**目标框架可直接消费**的形态：ShareGPT 结构的 SFT 数据集、
  `ranking: true` 的 DPO 数据集、以及把二者串起来的 `dataset_info.json` 与训练 YAML；
- 训练数据也要过**同一套门禁**（空/过短/重复/敏感），因为「审核通过」≠「可训练」；
- 诚实边界：本仓库只做**数据准备与验证链路**，训练在外部环境执行 ——
  仓库里没有 GPU、也不安装 LLaMA-Factory 运行时，这是硬约束。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import yaml

from ...infra.core.logging import get_logger
from ..dedupe import content_hash
from ..export import Processor
from ..scrub import Scrubber
from ..schemas import SamplePurpose, TrainingSample
from .dpo import DpoPair

logger = get_logger("data.training.dataset")

DEFAULT_BASE_MODEL = "Qwen2.5-7B-Instruct"


@dataclass(slots=True)
class LlamafactoryArtifact:
    """可直接落盘的数据集 + 配置产物（本仓库不执行训练）。"""

    dataset_name: str
    version: str
    sft_file: str
    dpo_file: str
    sft_jsonl: str
    dpo_jsonl: str
    dataset_info: dict[str, Any]
    train_config_yaml: str
    sft_records: list[dict[str, Any]] = field(default_factory=list)
    dpo_records: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def files(self) -> dict[str, str]:
        """文件名 → 文本内容（供调用方一次性落盘）。"""
        return {
            self.sft_file: self.sft_jsonl,
            self.dpo_file: self.dpo_jsonl,
            "dataset_info.json": json.dumps(self.dataset_info, ensure_ascii=False, indent=2),
            f"train_{self.dataset_name}.yaml": self.train_config_yaml,
        }


def _to_dpo_record(pair: DpoPair) -> dict[str, Any]:
    """LLaMA-Factory 的 DPO 记录：conversations（共享前缀）+ chosen / rejected 两条回答。"""
    return {
        "id": pair.pair_id,
        "conversations": [{"from": "human", "value": pair.prompt}],
        "chosen": {"from": "gpt", "value": pair.chosen},
        "rejected": {"from": "gpt", "value": pair.rejected},
        "source": pair.source,
    }


def _validate_pairs(pairs: Sequence[DpoPair], *, verifier: Scrubber | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """偏好对门禁：过短 / 空 / 重复 / 敏感。逐条给出拒绝原因，与 SFT 导出同一口径。"""
    scrubber = verifier or Scrubber()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        record = _to_dpo_record(pair)
        if min(len(pair.prompt), len(pair.chosen), len(pair.rejected)) < 4:
            rejected.append({"id": pair.pair_id, "reason": "too_short"})
            continue
        fingerprint = content_hash(json.dumps(record, ensure_ascii=False))
        if fingerprint in seen:
            rejected.append({"id": pair.pair_id, "reason": "duplicate"})
            continue
        hit = scrubber.scan(f"{pair.prompt}\n{pair.chosen}\n{pair.rejected}")
        if hit:
            rejected.append({"id": pair.pair_id, "reason": f"sensitive:{','.join(sorted(hit))}"})
            continue
        seen.add(fingerprint)
        accepted.append(record)
    return accepted, rejected


def build_llamafactory(
    samples: Sequence[TrainingSample],
    pairs: Sequence[DpoPair],
    *,
    dataset_name: str = "beauty_cs",
    version: str = "v1",
    base_model: str = DEFAULT_BASE_MODEL,
    sft_format: str = "sharegpt",
    verifier: Scrubber | None = None,
) -> LlamafactoryArtifact:
    """把 SFT 样本与 DPO 偏好对组装成 LLaMA-Factory 数据集 + 训练配置。"""
    processor = Processor(version=version, verifier=verifier)
    sft = processor.export(samples, fmt=sft_format, purpose=SamplePurpose.SFT_STYLE)
    dpo_records, dpo_rejected = _validate_pairs(pairs, verifier=verifier)

    sft_file = f"{dataset_name}_sft_{version}.jsonl"
    dpo_file = f"{dataset_name}_dpo_{version}.jsonl"
    dataset_info = {
        f"{dataset_name}_sft": {
            "file_name": sft_file,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        },
        f"{dataset_name}_dpo": {
            "file_name": dpo_file,
            "formatting": "sharegpt",
            "ranking": True,  # 关键：ranking=true 才会被 LLaMA-Factory 当作偏好数据集
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        },
    }

    config = {
        "stage": "dpo",
        "do_train": True,
        "model_name_or_path": base_model,
        "dataset": f"{dataset_name}_dpo",
        "dataset_dir": "data",
        "template": "qwen",
        "finetuning_type": "lora",
        "lora_target": "all",
        # 偏好训练超参：beta 控制与参考模型的偏离程度，过大会退化成死板复读
        "pref_beta": 0.1,
        "pref_loss": "sigmoid",
        "learning_rate": 5.0e-5,
        "num_train_epochs": 3.0,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "cutoff_len": 2048,
        "bf16": True,
        "output_dir": f"saves/{dataset_name}/dpo_{version}",
        "logging_steps": 10,
        "save_steps": 100,
    }

    artifact = LlamafactoryArtifact(
        dataset_name=dataset_name,
        version=version,
        sft_file=sft_file,
        dpo_file=dpo_file,
        sft_jsonl=sft.to_jsonl(),
        dpo_jsonl="\n".join(json.dumps(record, ensure_ascii=False) for record in dpo_records),
        dataset_info=dataset_info,
        train_config_yaml=yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        sft_records=list(sft.records),
        dpo_records=dpo_records,
        rejected=[*[{"id": f"sft#{issue.index}", "reason": issue.reason} for issue in sft.rejected], *dpo_rejected],
    )
    artifact.stats = {
        "version": version,
        "dataset_name": dataset_name,
        "sft_exported": len(sft.records),
        "dpo_exported": len(dpo_records),
        "rejected": len(artifact.rejected),
        "sft_rejected": len(sft.rejected),
        "dpo_rejected": len(dpo_rejected),
        "base_model": base_model,
        "stage": "dpo",
    }
    logger.info(
        "data.training.dataset.built",
        extra={
            "event": "data_training",
            "status": "ok",
            "reason": f"sft={len(sft.records)},dpo={len(dpo_records)},rejected={len(artifact.rejected)}",
        },
    )
    return artifact


__all__ = ["DEFAULT_BASE_MODEL", "LlamafactoryArtifact", "build_llamafactory"]
