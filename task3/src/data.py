"""数据层：样本索引、K 折划分、图像缓存、Dataset 与增强。

数据事实（本赛题）：
    * 训练集 80 张，测试集 20 张，眼底彩照，分辨率 2992x2000（少量 1956x1934）。
    * 标签 fovea_localization_train_GT.csv 的列: data, Fovea_X, Fovea_Y（原始像素坐标）。
    * 若中心凹不可见则坐标为 (0,0)。当前训练集中所有样本都可见，
      但框架保留了 `vis` 字段与可选的可见性分类头，方便后续接带负样本的数据。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import resolve_path

cv2.setNumThreads(0)  # 避免与 DataLoader worker 争抢线程

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 样本
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    id: str                     # 4 位图像名，例如 "0001"
    path: Path
    w: int                      # 原始宽
    h: int                      # 原始高
    x: float = 0.0              # 原始像素坐标（测试集为 0）
    y: float = 0.0
    vis: int = 1                # 1=可见, 0=不可见（标签为 0,0）
    has_gt: bool = False        # 是否有真实标签（测试集为 False）

    @property
    def xy_norm(self) -> Tuple[float, float]:
        return (self.x / self.w, self.y / self.h)


def _find_image(img_dir: Path, stem: str, exts: Sequence[str] = (".jpg", ".png", ".jpeg")) -> Path:
    for e in exts:
        p = img_dir / f"{stem}{e}"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到图像: {img_dir / stem}.*")


def _image_size(path: Path) -> Tuple[int, int]:
    """只读文件头拿宽高，不做完整解码。"""
    with Image.open(path) as im:
        return im.size


def load_train_samples(cfg: Dict) -> List[Sample]:
    root = resolve_path(cfg["data"]["root"])
    gt = pd.read_csv(resolve_path(cfg["data"]["gt_csv"]))
    samples: List[Sample] = []
    for _, r in gt.iterrows():
        sid = f"{int(r['data']):04d}"
        path = _find_image(root / "train", sid)
        w, h = _image_size(path)
        x, y = float(r["Fovea_X"]), float(r["Fovea_Y"])
        vis = 0 if (x == 0 and y == 0) else 1
        samples.append(Sample(sid, path, w, h, x, y, vis, has_gt=True))
    samples.sort(key=lambda s: s.id)
    return samples


def load_test_samples(cfg: Dict) -> List[Sample]:
    """测试集顺序以 sample_submission.csv 为准。"""
    root = resolve_path(cfg["data"]["root"])
    tmpl = load_submission_template(cfg)
    ids: List[str] = []
    for image_id in tmpl["image_ids"]:
        sid = _id_from_submission_key(image_id)
        if sid not in ids:
            ids.append(sid)
    samples = []
    for sid in ids:
        path = _find_image(root / "test", sid)
        w, h = _image_size(path)
        samples.append(Sample(sid, path, w, h))
    return samples


def _id_from_submission_key(image_id: str) -> str:
    """'81_Fovea_X' -> '0081'。"""
    stem = image_id.rsplit("_", 1)[0].split("_")[0]
    return f"{int(stem):04d}"


def load_submission_template(cfg: Dict) -> Dict:
    """读取 sample_submission.csv，保留其行顺序，供 predict.py 原样回填。"""
    path = resolve_path(cfg["data"]["sample_submission"])
    df = pd.read_csv(path)
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError(f"提交模板列数异常: {cols}")
    image_col, value_col = cols[0], cols[1]
    keys = df[image_col].astype(str).tolist()
    rows = []
    for k in keys:
        stem, coord = k.rsplit("_", 1)
        rows.append((f"{int(stem.split('_')[0]):04d}", coord.upper()))
    return {
        "path": path,
        "columns": cols,
        "image_col": image_col,
        "value_col": value_col,
        "keys": keys,
        "rows": rows,
        "image_ids": [f"{sid}_Fovea_{c}" for sid, c in rows],
    }


def split_indices(n: int, n_folds: int, seed: int, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    """确定性 K 折划分（不依赖 sklearn）。"""
    if n_folds <= 1:
        idx = np.arange(n)
        return idx, idx
    perm = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(perm, n_folds)
    val = np.sort(folds[fold % n_folds])
    train = np.sort(np.concatenate([f for i, f in enumerate(folds) if i != fold % n_folds]))
    return train, val


# --------------------------------------------------------------------------- #
# 图像缓存（每个进程独立，按字节预算做整体清空）
# --------------------------------------------------------------------------- #
_WORKERS_OK: Optional[bool] = None


def dataloader_workers_ok() -> bool:
    """探测当前环境能否创建 DataLoader 子进程。

    某些受限环境（Windows 沙箱 / 无权限）不允许创建命名管道，多进程 DataLoader
    会抛 PermissionError。这里直接用 multiprocessing.Queue（内部就是命名管道）探测，
    失败则自动降级为 num_workers=0。
    """
    global _WORKERS_OK
    if _WORKERS_OK is None:
        try:
            import multiprocessing as mp

            q = mp.Queue(maxsize=1)
            q.close()
            q.join_thread()
            _WORKERS_OK = True
        except Exception as e:  # pragma: no cover
            print(f"[warn] 无法创建 DataLoader 子进程 ({type(e).__name__}: {e})，自动改用 num_workers=0")
            _WORKERS_OK = False
    return _WORKERS_OK


_CACHE: Dict[Tuple[str, int], np.ndarray] = {}
_CACHE_BYTES = 0


def clear_image_cache() -> None:
    global _CACHE_BYTES
    _CACHE.clear()
    _CACHE_BYTES = 0


def imread_cached(path: Path, decode_scale: int = 1, cache_mb: float = 300) -> np.ndarray:
    global _CACHE_BYTES
    key = (str(path), int(decode_scale))
    img = _CACHE.get(key)
    if img is not None:
        return img
    flag = {
        1: cv2.IMREAD_COLOR,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
        8: cv2.IMREAD_REDUCED_COLOR_8,
    }.get(int(decode_scale), cv2.IMREAD_COLOR)
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"cv2 读取失败: {path}")
    budget = max(0.0, cache_mb) * 1024 * 1024
    if _CACHE and _CACHE_BYTES + img.nbytes > budget:
        clear_image_cache()
    if cache_mb > 0:
        _CACHE[key] = img
        _CACHE_BYTES += img.nbytes
    return img


# --------------------------------------------------------------------------- #
# 几何 / 光度增强
# --------------------------------------------------------------------------- #
def _rand_affine(
    img: np.ndarray, pt: Tuple[float, float], cfg_aug: Dict
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """随机 旋转-缩放-平移；同时把关键点变换到新坐标系。"""
    h, w = img.shape[:2]
    ang = random.uniform(-cfg_aug["rotate"], cfg_aug["rotate"])
    s0, s1 = cfg_aug["scale"]
    sc = random.uniform(s0, s1)
    tmax = cfg_aug["translate"]
    tx = random.uniform(-tmax, tmax) * w
    ty = random.uniform(-tmax, tmax) * h
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    p = M @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    return out, (float(p[0]), float(p[1]))


def _sample_photo_params(cfg_aug: Dict) -> Dict:
    """一次性采样一组光度增强参数。

    拆成"采样参数 + 应用参数"两步，是为了让 **patch 与其对应的全局缩略图共享同一套
    颜色风格**（阶段二 6 通道输入时用），否则同一张图的两个视图色彩不一致会互相干扰。
    """
    p: Dict = {}

    g = float(cfg_aug.get("channel_gain", 0.0))
    p["gains"] = np.exp(np.random.normal(0.0, g, size=3)).astype(np.float32) if g > 0 else None

    p_ill = float(cfg_aug.get("illumination_p", 0.0))
    p["illum"] = (random.uniform(-1, 1), random.uniform(-1, 1)) if (p_ill > 0 and random.random() < p_ill) else None

    hue = float(cfg_aug.get("hue", 0.0))
    p["hue"] = random.uniform(-hue, hue) * 360.0 if hue > 0 else 0.0
    s0, s1 = cfg_aug.get("saturation", (1.0, 1.0))
    p["sat"] = random.uniform(s0, s1) if (s0 != 1.0 or s1 != 1.0) else 1.0

    p["bc"] = (
        (1.0 + random.uniform(-cfg_aug["contrast"], cfg_aug["contrast"]),
         random.uniform(-cfg_aug["brightness"], cfg_aug["brightness"]))
        if cfg_aug["brightness"] > 0 else None
    )

    g0, g1 = cfg_aug["gamma"]
    p["gamma"] = random.uniform(g0, g1) if (g1 > 0 and random.random() < 0.5) else 1.0

    p_clahe = float(cfg_aug.get("clahe_p", 0.0))
    p["clahe"] = random.uniform(1.0, 3.0) if (p_clahe > 0 and random.random() < p_clahe) else None

    # ---- 更强的色彩增强（默认关闭，见 configs/v4/stage1_strongaug.yaml）----
    lab_shift = float(cfg_aug.get("lab_shift", 0.0))
    ls0, ls1 = cfg_aug.get("lab_scale", (1.0, 1.0))
    if lab_shift > 0 or ls0 != 1.0 or ls1 != 1.0:
        p["lab"] = (
            np.array([random.uniform(-lab_shift, lab_shift) for _ in range(3)], dtype=np.float32) * np.array([1.0, 8.0, 8.0], np.float32),
            np.array([random.uniform(ls0, ls1) for _ in range(3)], dtype=np.float32),
        )
    else:
        p["lab"] = None

    p["gray"] = random.random() < float(cfg_aug.get("gray_p", 0.0))
    p["ch_drop"] = random.randrange(3) if random.random() < float(cfg_aug.get("channel_dropout_p", 0.0)) else -1
    p["ch_perm"] = list(np.random.permutation(3).tolist()) if random.random() < float(cfg_aug.get("channel_shuffle_p", 0.0)) else None

    p["blur"] = random.choice((3, 5)) if random.random() < cfg_aug["blur_p"] else 0
    p["noise"] = 0.02 if random.random() < cfg_aug["noise_p"] else 0.0
    return p


def _apply_photo(img: np.ndarray, p: Dict) -> np.ndarray:
    """把 `_sample_photo_params` 采样到的参数作用到 float32 [0,1] 的 RGB 图上。"""
    img = img.copy()

    if p["gains"] is not None:
        img = img * p["gains"]
        np.clip(img, 0.0, 1.0, out=img)

    if p["illum"] is not None:
        h, w = img.shape[:2]
        ax, ay = p["illum"]
        gx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        gy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        grad = 1.0 + 0.22 * (ax * gx + ay * gy)
        img = img * grad[..., None]
        np.clip(img, 0.0, 1.0, out=img)

    if p["hue"] != 0.0 or p["sat"] != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)          # float32: H∈[0,360], S,V∈[0,1]
        if p["hue"] != 0.0:
            hsv[..., 0] = np.mod(hsv[..., 0] + p["hue"], 360.0)
        if p["sat"] != 1.0:
            hsv[..., 1] = np.clip(hsv[..., 1] * p["sat"], 0.0, 1.0)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if p["bc"] is not None:
        a, b = p["bc"]
        img = img * a + b
        np.clip(img, 0.0, 1.0, out=img)

    if p["gamma"] != 1.0:
        img = np.power(img, p["gamma"], dtype=np.float32)

    if p["clahe"] is not None:
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=p["clahe"], tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # Lab 通道均值/方差抖动（模拟不同染色与白平衡，比 HSV 抖动更"整图一致"）
    if p["lab"] is not None:
        shift, scale = p["lab"]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)          # float32: L∈[0,100], a,b∈[-127,127]
        mean = lab.reshape(-1, 3).mean(axis=0)
        lab = (lab - mean) * scale + mean + shift
        lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
        lab[..., 1:] = np.clip(lab[..., 1:], -127.0, 127.0)
        img = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)

    if p["gray"]:
        gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        img = np.stack([gray, gray, gray], axis=2)

    if p["ch_drop"] >= 0:
        img = img.copy()
        img[..., p["ch_drop"]] = 0.0

    if p["ch_perm"] is not None:
        img = img[..., p["ch_perm"]]

    if p["blur"]:
        k = p["blur"]
        img = cv2.GaussianBlur(img, (k, k), 0)

    if p["noise"] > 0:
        img = img + np.random.normal(0.0, p["noise"], size=img.shape).astype(np.float32)
        np.clip(img, 0.0, 1.0, out=img)
    return img


def _photometric(img: np.ndarray, cfg_aug: Dict) -> np.ndarray:
    """img: float32 [0,1] 的 RGB 图 —— 采样并应用一组光度增强。

    设计原则（眼底图）：
      * **色彩通道可以大胆扰动**——不同相机/白平衡/染色差异很大，而中心凹的判据是
        "黄斑更暗、无血管"的**亮度**特征，与色相无关，所以色彩扰动不会破坏标签；
      * **亮度对比度只做温和扰动**（brightness/contrast/gamma 幅度不大），
        否则等于把唯一的判别证据抹掉。

    顺序：通道增益 -> 光照不均 -> HSV 色相/饱和度 -> 亮度/对比度 -> gamma -> CLAHE
          -> Lab 统计量 -> 灰度 -> 通道丢弃/打乱 -> 模糊/噪声
    """
    return _apply_photo(img, _sample_photo_params(cfg_aug))


def _to_tensor(img_rgb_float: np.ndarray) -> torch.Tensor:
    """(H,W,C) float32 -> (C,H,W) 归一化张量。C 可以是 3 或 6（6 通道时后 3 通道同样归一化）。"""
    t = torch.from_numpy(np.ascontiguousarray(img_rgb_float.transpose(2, 0, 1)))
    c = t.shape[0]
    reps = max(1, c // 3)
    mean = torch.as_tensor(np.tile(IMAGENET_MEAN, reps)[:c]).view(c, 1, 1)
    std = torch.as_tensor(np.tile(IMAGENET_STD, reps)[:c]).view(c, 1, 1)
    return (t - mean) / std


def _crop_patch(img: np.ndarray, cx: float, cy: float, s: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """按归一化中心与边长裁切；越界用 0 填充（眼底图边缘本来就是黑的）。

    返回 patch 与其在（可能已填充的）图中的像素矩形 (x0, y0, x1, y1)。
    """
    ch, cw = img.shape[:2]
    x0 = int(round((cx - s / 2.0) * cw))
    x1 = int(round((cx + s / 2.0) * cw))
    y0 = int(round((cy - s / 2.0) * ch))
    y1 = int(round((cy + s / 2.0) * ch))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - cw), max(0, y1 - ch)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FoveaDataset(Dataset):
    """两种模式：

    task='stage1'：整图缩放到 data.img_size，输出整图上的归一化坐标。
    task='stage2'：以 center（训练时=GT+抖动，验证/预测时=stage1 预测）为中心裁 patch，
                   输出 patch 内的归一化坐标 + 全局归一化坐标（用于评估）。
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        cfg: Dict,
        train: bool = False,
        task: str = "stage1",
        centers: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        assert task in ("stage1", "stage2")
        self.samples = list(samples)
        self.cfg = cfg
        self.train = train
        self.task = task
        self.centers = centers
        self.dcfg = cfg["data"]
        self.acfg = cfg["aug"]
        self.s2cfg = cfg["stage2"]
        self.use_context = bool(self.s2cfg.get("use_context", False)) and task == "stage2"
        self._warned = False

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def _load_full(self, s: Sample) -> np.ndarray:
        return imread_cached(s.path, self.dcfg["decode_scale"], self.dcfg["cache_mb"])

    def __getitem__(self, i: int) -> Dict:
        s = self.samples[i]
        img = self._load_full(s)
        gt = np.array(s.xy_norm, dtype=np.float32)

        if self.task == "stage1":
            return self._get_stage1(s, img, gt)
        return self._get_stage2(s, img, gt)

    # ------------------------------------------------------------------ #
    def _get_stage1(self, s: Sample, img: np.ndarray, gt: np.ndarray) -> Dict:
        tw, th = self.dcfg["img_size"]
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        pt = (float(gt[0]) * tw, float(gt[1]) * th)

        if self.train:
            img, pt = _rand_affine(img, pt, self.acfg)
            if random.random() < self.acfg["hflip"]:
                img = img[:, ::-1]
                # 坐标约定 x∈[0, W]（归一化再除以 W），镜像即 W - x
                pt = (tw - pt[0], pt[1])

        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        if self.train:
            rgb = _photometric(rgb, self.acfg)

        xy = np.array([np.clip(pt[0] / tw, 0.0, 1.0), np.clip(pt[1] / th, 0.0, 1.0)], dtype=np.float32)
        return {
            "image": _to_tensor(rgb),
            "xy": torch.from_numpy(xy),
            "vis": torch.tensor(float(s.vis)),
            "wh": torch.tensor([float(s.w), float(s.h)]),
            "id": s.id,
        }

    # ------------------------------------------------------------------ #
    def _get_stage2(self, s: Sample, img: np.ndarray, gt: np.ndarray) -> Dict:
        s2 = self.s2cfg
        patch_size = float(s2["patch_size"])
        out_size = int(s2["patch_resize"])
        ch, cw = img.shape[:2]

        if self.train:
            sigma = float(s2["jitter_sigma"])
            if random.random() < float(s2["big_jitter_prob"]):
                sigma = float(s2["big_jitter_sigma"])
            cy = cx = 0.0
            for _ in range(8):  # 保证 GT 落在 patch 内部（留 5% 余量）
                cx = gt[0] + random.gauss(0.0, sigma)
                cy = gt[1] + random.gauss(0.0, sigma)
                if 0.05 < (gt[0] - (cx - patch_size / 2)) / patch_size < 0.95 and \
                   0.05 < (gt[1] - (cy - patch_size / 2)) / patch_size < 0.95:
                    break
        else:
            if self.centers is not None and s.id in self.centers:
                cx, cy = self.centers[s.id]
            else:
                if not self._warned:
                    print(f"[warn] stage2 无中心点缓存，退化为以 GT 为中心裁切（仅限调试）")
                    self._warned = True
                cx, cy = float(gt[0]), float(gt[1])

        patch, (x0, y0, x1, y1) = _crop_patch(img, cx, cy, patch_size)
        # 坐标一律按"名义裁切框"(cx, cy, patch_size)换算，不按取整后的像素矩形，
        # 这样 patch_to_global 与之严格互逆（取整只带来 <0.5px 的重采样误差）。
        ox, oy = cx - patch_size / 2.0, cy - patch_size / 2.0
        local = np.array(
            [(float(gt[0]) - ox) / patch_size, (float(gt[1]) - oy) / patch_size], dtype=np.float32
        )
        patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        pt = (float(local[0]) * out_size, float(local[1]) * out_size)

        if self.train:
            patch, pt = _rand_affine(patch, pt, self.acfg)
            if random.random() < self.acfg["hflip"]:
                patch = patch[:, ::-1]
                pt = (out_size - pt[0], pt[1])

        rgb = patch[:, :, ::-1].astype(np.float32) / 255.0
        if self.train:
            # 同一张图的 patch 与全局缩略图共用一套光度参数（否则两个视图色彩打架）
            photo = _sample_photo_params(self.acfg)
            rgb = _apply_photo(rgb, photo)

        # 6 通道输入：patch RGB + 整图缩略图 RGB（当全局上下文用，弥补除视盘之外的先验）
        if self.use_context:
            ctx = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
            ctx_rgb = ctx[:, :, ::-1].astype(np.float32) / 255.0
            if self.train:
                ctx_rgb = _apply_photo(ctx_rgb, photo)
            rgb = np.concatenate([rgb, ctx_rgb], axis=2)

        patch_xy = np.array(
            [np.clip(pt[0] / out_size, 0.0, 1.0), np.clip(pt[1] / out_size, 0.0, 1.0)], dtype=np.float32
        )
        return {
            "image": _to_tensor(rgb),
            "xy": torch.from_numpy(patch_xy),          # 训练损失使用：patch 内坐标
            "xy_global": torch.from_numpy(gt.copy()),  # 评估使用：全局坐标
            "crop": torch.tensor([cx, cy, patch_size], dtype=torch.float32),
            "vis": torch.tensor(float(s.vis)),
            "wh": torch.tensor([float(s.w), float(s.h)]),
            "id": s.id,
        }


# --------------------------------------------------------------------------- #
# patch 内坐标 -> 全局坐标
# --------------------------------------------------------------------------- #
def patch_to_global(patch_xy: torch.Tensor, crop: torch.Tensor) -> torch.Tensor:
    """patch_xy: (B,2) ∈[0,1] patch 内归一化坐标; crop: (B,3) = cx, cy, size。"""
    cx, cy, s = crop[:, 0:1], crop[:, 1:2], crop[:, 2:3]
    ox = cx - s / 2.0
    oy = cy - s / 2.0
    return torch.cat([ox + patch_xy[:, 0:1] * s, oy + patch_xy[:, 1:2] * s], dim=1)
