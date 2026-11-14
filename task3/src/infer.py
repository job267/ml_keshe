"""推理工具：对 Dataset 全量预测并计算指标（训练验证与最终预测共用）。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import FoveaDataset, patch_to_global
from .metrics import compute_metrics
from .models import predict_tta, select_xy


@torch.no_grad()
def run_inference(
    model: nn.Module,
    ds: FoveaDataset,
    cfg: Dict,
    device: torch.device,
    tta: Optional[bool] = None,
    source: Optional[str] = None,
    tta_spec: Optional[Dict] = None,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """返回 {id: (x, y) 归一化全图坐标} 与指标字典（指标用原始像素计算）。

    tta / tta_spec 二选一：tta=True 是纯翻转；tta_spec 可指定多尺度+平移（见 models.predict_tta）。
    默认从 cfg['tta'] 读取 TTA 配置。
    """
    model.eval()
    source = source or cfg["model"].get("coord_source", "soft")
    if tta_spec is None:
        spec = dict(cfg.get("tta") or {})
        spec.setdefault("flip", True)
    else:
        spec = dict(tta_spec)
    if tta is False:            # 调用方显式关闭翻转（多尺度/平移仍按 spec 执行）
        spec["flip"] = False
    elif tta is True:
        spec["flip"] = True
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    preds: Dict[str, Tuple[float, float]] = {}
    P, G, WH, VIS, VP = [], [], [], [], []
    for b in dl:
        x = b["image"].to(device, non_blocking=True)
        out = predict_tta(model, x, spec)
        xy = select_xy(out, source)
        if ds.task == "stage2":
            xy = patch_to_global(xy, b["crop"].to(device))
            gt = b["xy_global"]
        else:
            gt = b["xy"]
        xy = xy.float().cpu().numpy()
        P.append(xy)
        G.append(gt.numpy())
        WH.append(b["wh"].numpy())
        VIS.append(b["vis"].numpy())
        if out.get("vis_prob") is not None:
            VP.append(out["vis_prob"].float().cpu().numpy())
        for i, sid in enumerate(b["id"]):
            preds[sid] = (float(xy[i, 0]), float(xy[i, 1]))

    P, G, WH, VIS = np.concatenate(P), np.concatenate(G), np.concatenate(WH), np.concatenate(VIS)
    VP = np.concatenate(VP) if VP else None
    return preds, compute_metrics(P, G, WH, vis=VIS, vis_prob=VP)
