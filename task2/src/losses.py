"""损失：软标签交叉熵、类别/样本加权、广义交叉熵 GCE。

报告口径：
    L = -(1/N) Σ_i Σ_c y_soft[i,c] · log p[i,c]
y_soft 来自标签平滑、MixUp 或质量权重调整后的目标；样本质量权重在 MixUp **之前**
施加到来源标签上，避免混合后统一赋权（src/train.py 中体现）。

GCE（第七轮 C，q=0.7）：
    L = Σ_c y_c · (1 - p_c^q) / q ，再乘 6^q 使均匀预测（p=1/6）处的梯度尺度与
普通交叉熵一致，避免"梯度整体变小"被误当成有效学习率变化。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NUM_CLASSES


def soft_ce(logits: torch.Tensor, target: torch.Tensor,
            sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """target 为 (B, C) 概率分布；sample_weight 为 (B,) 逐样本权重。"""
    logp = F.log_softmax(logits, dim=1)
    loss = -(target * logp).sum(dim=1)
    if sample_weight is not None:
        w = sample_weight.to(loss.dtype)
        denom = w.sum().clamp_min(1e-8)
        return (loss * w).sum() / denom
    return loss.mean()


def gce(logits: torch.Tensor, target: torch.Tensor, q: float = 0.7,
        sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    p = F.softmax(logits, dim=1).clamp_min(1e-6)
    per_class = (1.0 - p.pow(q)) / q
    scale = float(NUM_CLASSES) ** q
    loss = (target * per_class).sum(dim=1) * scale
    if sample_weight is not None:
        w = sample_weight.to(loss.dtype)
        return (loss * w).sum() / w.sum().clamp_min(1e-8)
    return loss.mean()


class Criterion(nn.Module):
    """按配置选择 CE / GCE；target 既可是 one-hot 概率也可是类别下标。"""

    def __init__(self, q: float = 0.0, label_smoothing: float = 0.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.q = float(q)
        self.smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        if target.dim() == 1:
            target = F.one_hot(target, NUM_CLASSES).to(logits.dtype)
            if self.smoothing > 0:
                target = target * (1 - self.smoothing) + self.smoothing / NUM_CLASSES
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.dtype)
            target = target * cw.view(1, -1)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if self.q > 0:
            return gce(logits, target, self.q, sample_weight)
        return soft_ce(logits, target, sample_weight)
