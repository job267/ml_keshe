"""指标：Accuracy、Macro F1、逐类报告、混淆矩阵、Wilson 区间与配对 bootstrap。

报告要求同时给出"点估计 + 条件区间"，并区分开发验证集、折外集合与实际提交，
因此本模块所有函数都显式接收标签与预测，不隐含任何数据来源。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import CLASSES, NUM_CLASSES


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict]:
    out = []
    for c in range(NUM_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append({"class": CLASSES[c], "precision": p, "recall": r, "f1": f1,
                    "support": int((y_true == c).sum())})
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    m = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


def wilson_ci(correct: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(center - half), float(center + half))


def paired_bootstrap_diff(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                          groups: Sequence[int] | None = None, n_boot: int = 2000,
                          seed: int = 42) -> Dict[str, float]:
    """两个预测器的准确率差 (b - a) 的配对 bootstrap 区间；按重复组整组重采样。"""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        diffs[b] = (pred_b[idx] == y_true[idx]).mean() - (pred_a[idx] == y_true[idx]).mean()
    observed = (pred_b == y_true).mean() - (pred_a == y_true).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"observed": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_positive": float((diffs > 0).mean())}


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy(y_true, y_pred)
    lo, hi = wilson_ci(int((y_true == y_pred).sum()), len(y_true))
    return {"accuracy": acc, "macro_f1": macro_f1(y_true, y_pred),
            "wilson_ci": [lo, hi], "n": int(len(y_true)),
            "per_class": per_class_report(y_true, y_pred),
            "confusion": confusion(y_true, y_pred).tolist()}
