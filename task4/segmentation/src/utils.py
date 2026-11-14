"""通用工具：随机种子、设备选择、图像读写、视场（FOV）提取。"""

import os
import random
import time

import cv2
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 可复现性
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministic: bool = False):
    """固定随机种子。

    ``deterministic=True`` 会开启 cudnn 确定性算法（结果可比特级复现，
    但卷积会慢 2~3 倍），日常训练建议保持默认 False。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# 图像 IO（用 imdecode 以兼容中文路径）
# --------------------------------------------------------------------------- #
def imread_rgb(path: str) -> np.ndarray:
    """读取为 RGB uint8 HWC。"""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imread_gray(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"无法读取图像: {path}")
    return img


def imwrite(path: str, img: np.ndarray):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"无法编码图像: {path}")
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    buf.tofile(path)


# --------------------------------------------------------------------------- #
# 视场（FOV）提取：眼底图圆形区域
# --------------------------------------------------------------------------- #
def fov_mask(rgb: np.ndarray, thresh: int = 10, ksize: int = 7,
             fill_holes_flag: bool = True) -> np.ndarray:
    """由眼底图估计有效成像区域（圆形视场）。

    返回 bool 数组，True 表示位于视场内。
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (gray > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # 只保留最大连通域（去掉边缘噪点）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == idx).astype(np.uint8)

    # 填洞（暗病灶可能让视场出现空洞）
    if fill_holes_flag and m[0, 0] == 0:
        flood = (m * 255).astype(np.uint8)
        ff_mask = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        m = ((m * 255) | holes) > 0
        m = m.astype(np.uint8)

    return m.astype(bool)


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #
class AverageMeter:
    """滑动平均记录器。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

    def __str__(self):
        s = self.elapsed()
        return f"{int(s // 60)}m{s % 60:04.1f}s"


def list_ids(image_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """列出目录下的图片 id（文件名去扩展名），按数值升序。"""
    ids = []
    for name in os.listdir(image_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in exts:
            ids.append(stem)

    def _key(s):
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(ids, key=_key)


def parse_id_list(s: str):
    """'17,18,19' -> ['17','18','19']（保持字符串形式，便于拼路径）。"""
    return [tok.strip() for tok in str(s).split(",") if tok.strip()]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
