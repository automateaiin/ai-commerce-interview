"""训练数据工程（项目四 · 阶段 2）。

只做**数据准备与验证链路**，不在仓库里跑 GPU 训练 —— 这是必须说清的诚实边界：
- 本包产出 LLaMA-Factory **可直接消费**的数据集文件 + `dataset_info.json` + 训练 YAML 配置；
- 真正的训练在外部环境执行（本机无 GPU / 无 LLaMA-Factory 运行时）。

两个子模块：
- `dpo`     偏好对构造：审核编辑（before/after）与人工纠偏是最真实的偏好信号来源；
- `dataset` 数据集与配置产物：ShareGPT 结构的 SFT + ranking 结构的 DPO + 训练 YAML。
"""

from __future__ import annotations

from .dataset import LlamafactoryArtifact, build_llamafactory
from .dpo import DpoBuilder, DpoPair

__all__ = ["DpoBuilder", "DpoPair", "LlamafactoryArtifact", "build_llamafactory"]
