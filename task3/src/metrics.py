"""评价指标。

赛题指标说明
------------
提交文件把坐标拆成相邻两行（`{id}_Fovea_X` / `{id}_Fovea_Y`），
然后对 **所有行** 的数值统一计算 RMSE：

    RMSE = sqrt( mean_over_all_rows( (pred - gt)^2 ) )
         = sqrt( ( Σ dx_i^2 + Σ dy_i^2 ) / (2N) )

也就是说 X 和 Y 的误差被合并成一个数（不是分别算 RMSE 再平均）。
本模块的 `competition_rmse` 严格按此实现；其它指标（像素 L2、SR@k）仅用于诊断。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def to_pixels(xy_norm: np.ndarray, wh: np.ndarray) -> np.ndarray:
    """(N,2) 归一化坐标 -> (N,2) 原始像素坐标。wh: (N,2) = [w, h]。"""
    return np.asarray(xy_norm, dtype=np.float64) * np.asarray(wh, dtype=np.float64)


def apply_visibility(
    xy_norm: np.ndarray,
    vis_prob: Optional[np.ndarray] = None,
    vis_thr: float = 0.5,
    gt_vis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """按"不可见 -> (0,0)"规则改写预测。

    * vis_prob 给出时：概率低于阈值视为不可见；
    * gt_vis 给出时：GT 不可见的样本，预测强制为 (0,0)（保证指标口径正确）。
    """
    pred = np.asarray(xy_norm, dtype=np.float64).copy()
    if vis_prob is not None:
        pred[np.asarray(vis_prob) < vis_thr] = 0.0
    if gt_vis is not None:
        pred[np.asarray(gt_vis) <= 0] = 0.0
    return pred


def competition_rmse(dx: np.ndarray, dy: np.ndarray) -> float:
    """赛题口径 RMSE（像素）。dx/dy 为逐样本的像素误差。"""
    dx = np.asarray(dx, dtype=np.float64).ravel()
    dy = np.asarray(dy, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(np.concatenate([dx ** 2, dy ** 2]))))


def compute_metrics(
    pred_norm: np.ndarray,
    gt_norm: np.ndarray,
    wh: np.ndarray,
    vis: Optional[np.ndarray] = None,
    vis_prob: Optional[np.ndarray] = None,
    vis_thr: float = 0.5,
) -> Dict[str, float]:
    """返回赛题 RMSE + 常用诊断指标（单位：原始像素）。"""
    pred_norm = apply_visibility(pred_norm, vis_prob, vis_thr, vis)
    gt = np.asarray(gt_norm, dtype=np.float64)
    wh = np.asarray(wh, dtype=np.float64)
    p = to_pixels(pred_norm, wh)
    g = to_pixels(gt, wh)
    dx, dy = p[:, 0] - g[:, 0], p[:, 1] - g[:, 1]
    l2 = np.sqrt(dx ** 2 + dy ** 2)
    return {
        "rmse": competition_rmse(dx, dy),
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "sr50": float((l2 <= 50).mean()),
        "sr100": float((l2 <= 100).mean()),
        "sr200": float((l2 <= 200).mean()),
        "n": int(len(l2)),
    }


def format_metrics(m: Dict[str, float]) -> str:
    return (
        f"RMSE={m['rmse']:.2f}px | meanL2={m['mean_l2']:.2f} | "
        f"SR@50={m['sr50']:.2f} SR@100={m['sr100']:.2f} SR@200={m['sr200']:.2f} | n={m['n']}"
    )
