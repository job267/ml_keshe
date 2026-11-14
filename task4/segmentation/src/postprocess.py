"""后处理：概率图 -> 二值掩膜。

训练时的整图验证与最终推理共用这里的实现，保证"验证分数 = 实际提交效果"。
"""

import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """剔除面积小于 ``min_size`` 的连通域。"""
    if min_size <= 0:
        return mask
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[lab == i] = 1
    return out.astype(bool)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == idx)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """填充掩膜内部空洞（flood fill 背景法）。"""
    m = mask.astype(np.uint8)
    if m[0, 0] == 1:
        return mask
    flood = (m * 255).astype(np.uint8)
    ff = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return (((m * 255) | holes) > 0)


def postprocess(prob: np.ndarray, fov=None, threshold: float = 0.5,
                min_size: int = 0, do_fill_holes: bool = False) -> np.ndarray:
    """概率图后处理，返回 bool 掩膜。

    Parameters
    ----------
    prob : (H,W) float, 血管概率
    fov : (H,W) bool 或 None, 视场掩膜；视场外一律判为背景
    threshold : 二值化阈值
    min_size : 小连通域面积阈值（像素）
    do_fill_holes : 是否填洞
    """
    prob = np.asarray(prob, dtype=np.float32)
    mask = prob > float(threshold)
    if fov is not None:
        mask &= np.asarray(fov).astype(bool)
    if min_size > 0:
        mask = remove_small_components(mask, min_size)
    if do_fill_holes:
        mask = fill_holes(mask)
    return mask.astype(bool)
