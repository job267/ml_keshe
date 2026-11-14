"""输入通道构造。

眼底图血管与背景的对比度在 **绿色通道** 最高（红色通道接近饱和、
蓝色通道噪声大），因此主流做法都是基于绿色通道做增强。

本模块支持三种通道，由 ``Config.channels`` 组合：:

    green     绿色通道原图，归一化到 [0,1]
    clahe     绿色通道 + CLAHE 局部对比度增强，归一化到 [0,1]
    highpass  绿色通道减去大尺度高斯背景（血管残差），范围约 [-1,1]

注意：CLAHE 在 **整图** 上计算（训练时对整图缓存后切 patch），
这样训练与推理的统计特性保持一致。
"""

import cv2
import numpy as np

# 归一化到 [0,1] 的通道名（其余通道视为无界，不做 gamma 扰动）
BOUNDED_CHANNELS = ("green", "gray", "clahe", "red", "blue")


def _clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return op.apply(gray)


def to_channels(rgb: np.ndarray, channel_names=("clahe", "green", "highpass"),
                clahe_clip: float = 2.0, clahe_grid: int = 8,
                highpass_sigma: float = 12.0) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float32 (C,H,W)。

    Parameters
    ----------
    rgb : np.ndarray
        HWC uint8 RGB 图像。
    channel_names : sequence of str
        组成输入的各通道名称。
    """
    green = rgb[..., 1].astype(np.float32) / 255.0

    cache = {}
    out = []
    for name in channel_names:
        name = name.lower()
        if name in cache:
            out.append(cache[name])
            continue
        if name == "green":
            ch = green
        elif name in ("gray",):
            ch = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        elif name == "red":
            ch = rgb[..., 0].astype(np.float32) / 255.0
        elif name == "blue":
            ch = rgb[..., 2].astype(np.float32) / 255.0
        elif name == "clahe":
            ch = _clahe(rgb[..., 1], clahe_clip, clahe_grid).astype(np.float32) / 255.0
        elif name == "highpass":
            # 大尺度背景估计后相减，突出细血管
            bg = cv2.GaussianBlur(green, (0, 0), float(highpass_sigma))
            ch = green - bg
        else:
            raise ValueError(f"未知通道: {name}")
        cache[name] = ch
        out.append(ch)

    return np.stack(out, axis=0).astype(np.float32)


def normalize(x: np.ndarray, mean, std) -> np.ndarray:
    """按通道做 (x - mean) / std。x 为 (C,H,W)。"""
    m = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    return (x - m) / s


def scaled_rgb(rgb, scale):
    """统一训练与推理的 RGB 缩放器和取整方式。"""
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale 必须 >0")
    if scale == 1.0:
        return rgb
    h, w = rgb.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def prepare_rgb(rgb, cfg, scale=1.0, normalized=True):
    """唯一的 RGB -> 通道管线；训练切 patch 前保留未归一化通道。"""
    img = to_channels(scaled_rgb(rgb, scale), cfg.channel_list, cfg.clahe_clip, cfg.clahe_grid)
    return normalize(img, cfg.norm_mean_list, cfg.norm_std_list) if normalized else img


def bounded_mask(channel_names) -> np.ndarray:
    """返回每个通道是否取值在 [0,1]，用于安全地施加 gamma 扰动。"""
    return np.array([name.lower() in BOUNDED_CHANNELS for name in channel_names], dtype=bool)
