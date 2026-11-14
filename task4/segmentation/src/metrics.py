"""评价指标（全部基于 numpy bool 数组）。

与竞赛一致，主指标为 Dice::

    Dice = 2 * |P ∩ G| / (|P| + |G|)
"""

import numpy as np


def _as_bool(x) -> np.ndarray:
    return np.asarray(x).astype(bool)


def dice_coef(pred, gt, eps: float = 1e-7) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    inter = np.logical_and(p, g).sum()
    return float((2.0 * inter + eps) / (p.sum() + g.sum() + eps))


def iou_coef(pred, gt, eps: float = 1e-7) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((inter + eps) / (union + eps))


def sensitivity(pred, gt, eps: float = 1e-7) -> float:
    """召回率：血管被正确识别的比例。"""
    p, g = _as_bool(pred), _as_bool(gt)
    tp = np.logical_and(p, g).sum()
    return float((tp + eps) / (g.sum() + eps))


def specificity(pred, gt, eps: float = 1e-7) -> float:
    """背景被正确识别的比例。"""
    p, g = _as_bool(pred), _as_bool(gt)
    tn = np.logical_and(~p, ~g).sum()
    return float((tn + eps) / ((~g).sum() + eps))


def accuracy(pred, gt, eps: float = 1e-7) -> float:
    p, g = _as_bool(pred), _as_bool(gt)
    return float((p == g).mean())


def evaluate_binary(pred, gt) -> dict:
    """一次性返回常用指标。"""
    return {
        "dice": dice_coef(pred, gt),
        "iou": iou_coef(pred, gt),
        "se": sensitivity(pred, gt),
        "sp": specificity(pred, gt),
        "acc": accuracy(pred, gt),
    }
