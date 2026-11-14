"""损失函数：热力图高斯标签 + 坐标回归（smoothL1 / wing / l1）+ 可见性分类。"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import soft_argmax


# --------------------------------------------------------------------------- #
def make_heatmap_target(
    xy_norm: torch.Tensor, h: int, w: int, sigma: float = 2.0
) -> torch.Tensor:
    """把归一化坐标 (B,2) 变成 (B,1,h,w) 的高斯热力图。

    网格坐标约定与 soft_argmax 一致：坐标 x 对应网格位置 `x*w - 0.5`。
    """
    b = xy_norm.shape[0]
    dev, dt = xy_norm.device, xy_norm.dtype
    tx = xy_norm[:, 0] * w - 0.5
    ty = xy_norm[:, 1] * h - 0.5
    gy = torch.arange(h, device=dev, dtype=dt).view(1, h, 1)
    gx = torch.arange(w, device=dev, dtype=dt).view(1, 1, w)
    d2 = (gx - tx.view(b, 1, 1)) ** 2 + (gy - ty.view(b, 1, 1)) ** 2
    return torch.exp(-d2 / (2.0 * sigma * sigma)).unsqueeze(1)


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, target)
    return F.mse_loss(torch.sigmoid(logits), target)


def coord_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "smoothl1", **kw) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "wing":
        return wing_loss(pred, target, w=kw.get("wing_w", 0.05), e=kw.get("wing_e", 0.1))
    beta = float(kw.get("smoothl1_beta", 0.01))
    return F.smooth_l1_loss(pred, target, beta=beta)


def wing_loss(pred: torch.Tensor, target: torch.Tensor, w: float = 0.05, e: float = 0.1) -> torch.Tensor:
    """Wing loss（归一化坐标单位）：小误差用对数放大梯度，大误差退化为 L1，对离群点鲁棒。"""
    d = (pred - target).abs()
    c = w * (1.0 - math.log(1.0 + w / e))
    out = torch.where(d < w, w * torch.log1p(d / e), d - c)
    return out.mean()


# --------------------------------------------------------------------------- #
class FoveaLoss(nn.Module):
    """总损失 = 热力图 + soft-argmax 坐标 + coord_head 坐标 +（可选）可见性。"""

    def __init__(self, cfg: Dict):
        super().__init__()
        lc = cfg["loss"]
        self.mode_hm = lc["heatmap"]
        self.w_hm = float(lc["w_heatmap"])
        self.w_soft = float(lc["w_soft"])
        self.w_coord = float(lc["w_coord"])
        self.w_vis = float(lc["w_vis"])
        self.kind = lc["coord"]
        self.kw = {
            "smoothl1_beta": float(lc["smoothl1_beta"]),
            "wing_w": float(lc["wing_w"]),
            "wing_e": float(lc["wing_e"]),
        }
        self.sigma = float(cfg["heatmap"]["sigma"])
        self.beta = float(cfg["heatmap"]["beta"])

    def forward(self, pred: Dict[str, Optional[torch.Tensor]], xy: torch.Tensor,
                vis: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hm = pred["heatmap"]
        _, _, h, w = hm.shape
        target = make_heatmap_target(xy, h, w, self.sigma)

        parts: Dict[str, torch.Tensor] = {}
        parts["hm"] = heatmap_loss(hm, target, self.mode_hm)

        # soft-argmax 坐标（对 sigmoid 后的概率图做质量归一化期望）
        xy_soft = soft_argmax(torch.sigmoid(hm), self.beta)
        parts["soft"] = coord_loss(xy_soft, xy, self.kind, **self.kw)

        total = self.w_hm * parts["hm"] + self.w_soft * parts["soft"]

        if pred.get("xy_head") is not None and self.w_coord > 0:
            parts["coord"] = coord_loss(pred["xy_head"], xy, self.kind, **self.kw)
            total = total + self.w_coord * parts["coord"]

        if pred.get("vis_logit") is not None and vis is not None and self.w_vis > 0:
            parts["vis"] = F.binary_cross_entropy_with_logits(pred["vis_logit"], vis)
            total = total + self.w_vis * parts["vis"]

        parts["total"] = total
        return parts
