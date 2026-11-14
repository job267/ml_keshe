"""数据集定义。

* 训练集：整图预处理后缓存到内存，每个 epoch 随机采样 ``patches_per_epoch`` 个 patch；
* 验证/推理集：返回整图（不做 resize），由 engine 负责 padding 与前向。

数据以 ``records``（``[{'id','image','label'}, ...]``）的形式传入，
因此同一套代码既能训 DRIVE（``train/image`` + ``train/label``），
也能训外部数据（STARE/CHASE/HRF，多个目录合并），见 :func:`build_external_records`。
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import augment as A
from . import preprocess as P
from .utils import fov_mask, imread_gray, imread_rgb, list_ids

IMG_EXTS = ("", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm")


def resolve_file(dirpath, stem):
    """在目录里找 ``stem`` + 任意常见图像扩展名的文件（png/jpg/tif/ppm...）。"""
    for ext in IMG_EXTS:
        p = os.path.join(dirpath, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def build_records(image_dir, label_dir):
    """按标签目录列举样本，返回 ``[{'id','image','label'}]``。"""
    ids = [i for i in list_ids(label_dir) if resolve_file(image_dir, i)]
    if not ids:
        raise RuntimeError(f"{label_dir} 下没有可用样本（或与 {image_dir} 无交集）")
    return [{"id": i,
             "image": resolve_file(image_dir, i),
             "label": resolve_file(label_dir, i)} for i in ids]


def build_external_records(roots):
    """合并多个外部数据目录，每个目录需含 ``image/`` 与 ``label/`` 子目录。

    id 会加上目录名前缀（如 ``stare/im0001``），避免不同数据集重名。
    """
    records = []
    for root in roots:
        root = root.rstrip("/\\")
        name = os.path.basename(root)
        recs = build_records(os.path.join(root, "image"), os.path.join(root, "label"))
        for r in recs:
            r["id"] = f"{name}/{r['id']}"
            r["dataset"] = name
        records.extend(recs)
    if not records:
        raise RuntimeError(f"外部数据为空: {roots}")
    return records


def discover_ids(cfg):
    """训练集可用 id（图像与标签文件名求交集）。"""
    return [r["id"] for r in build_records(cfg.train_image_dir, cfg.train_label_dir)]


def filter_records(records, ids):
    keep = set(ids)
    return [r for r in records if r["id"] in keep]


def split_ids(all_ids, val_ids_str):
    val = [i.strip() for i in (val_ids_str or "").split(",") if i.strip()]
    val = [v for v in val if v in all_ids]
    train = [i for i in all_ids if i not in set(val)]
    return train, val


class PatchDataset(Dataset):
    """训练用 patch 数据集（DRIVE 或外部数据通用）。"""

    def __init__(self, cfg, records, cache=None):
        self.cfg = cfg
        self.records = list(records)
        self.ids = [r["id"] for r in self.records]
        self.patches = int(cfg.patches_per_epoch)
        self.rng = np.random.RandomState(cfg.seed)
        self.bounded = P.bounded_mask(cfg.channel_list)

        self.cache = cache if cache is not None else {}
        # 内存优化：--cache_dtype float16 时整图缓存用半精度（800 张外部图可省一半内存），
        # 取 patch 前再转回 float32；float16 的量化误差约 1e-3，对训练无可见影响。
        self.cache_dtype = np.float16 if str(getattr(cfg, "cache_dtype", "float32")) == "float16" else None
        for r in self.records:
            if r["id"] not in self.cache:
                pair = _load_pair(cfg, r)
                if self.cache_dtype is not None:
                    pair = (pair[0].astype(self.cache_dtype), pair[1].astype(self.cache_dtype))
                self.cache[r["id"]] = pair
        ratios = [float(self.cache[r["id"]][1].mean()) for r in self.records]
        self.vessel_ratio = float(np.mean(ratios)) if ratios else 0.0

        if not 0 <= cfg.thin_center_prob <= cfg.vessel_center_prob <= 1:
            raise ValueError("需要 0 <= thin_center_prob <= vessel_center_prob <= 1")
        self.thin_centers = {}
        if cfg.thin_center_prob > 0:
            from skimage.morphology import skeletonize
            for rid in self.ids:
                if rid in self.thin_centers:
                    continue
                mask = self.cache[rid][1] > 0.5
                radius = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
                thin = skeletonize(mask) & (radius <= cfg.thin_max_radius * cfg.scale)
                self.thin_centers[rid] = np.column_stack(np.nonzero(thin))

        # ---- 强增强参数：-1 表示跟随 --strong_aug 预设 ----
        self.aug = _resolve_aug(cfg)
        self.donors = []
        if self.aug["copy_paste_p"] > 0:
            self._build_donors(n=12)

    def _pair(self, rid):
        """取出缓存的整图并转回 float32（缓存可能是 float16）。"""
        img, mask = self.cache[rid]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        return img, mask

    def _build_donors(self, n=12, size=64):
        """收集若干"血管密集"的小块，供 copy_paste 使用。"""
        for _ in range(n):
            idx = self.rng.randint(len(self.records))
            img, mask = self._pair(self.records[idx]["id"])
            p, m = A.sample_patch(img, mask, size, self.rng,
                                  vessel_center_prob=1.0, scale_range=(1.0, 1.0))
            if m.sum() >= 12:
                self.donors.append((p, m))

    def __len__(self):
        return self.patches

    def __getitem__(self, index):
        cfg = self.cfg
        idx = self.rng.randint(len(self.records))
        img, mask = self._pair(self.records[idx]["id"])

        center = None
        vessel_prob = cfg.vessel_center_prob
        if cfg.thin_center_prob > 0:
            centers = self.thin_centers[self.records[idx]["id"]]
            if self.rng.rand() < cfg.thin_center_prob and len(centers):
                center = centers[self.rng.randint(len(centers))]
            elif len(centers):
                vessel_prob = ((cfg.vessel_center_prob - cfg.thin_center_prob)
                               / (1 - cfg.thin_center_prob)) if cfg.thin_center_prob < 1 else 0

        img, mask = A.sample_patch(
            img, mask, cfg.patch_size, self.rng,
            vessel_center_prob=vessel_prob,
            scale_range=cfg.scale_range_tuple, center=center)

        donor = self.donors[self.rng.randint(len(self.donors))] if self.donors else None
        img, mask = A.augment_pair(
            img, mask, self.rng, bounded=self.bounded,
            rot_angle=cfg.rot_angle, donor=donor,
            gain=cfg.aug_gain, bias=cfg.aug_bias,
            noise=cfg.aug_noise, gamma=cfg.aug_gamma, **self.aug)

        img = P.normalize(img, cfg.norm_mean_list, cfg.norm_std_list)
        sample = (torch.from_numpy(img.astype(np.float32)),
                  torch.from_numpy(mask[None].astype(np.float32)))
        if cfg.model == 'unet_aux':
            from skimage.morphology import skeletonize
            skeleton = skeletonize(mask > .5)[None].astype(np.float32)
            return (*sample, {'skeleton': torch.from_numpy(skeleton)})
        return sample


class ImageDataset(Dataset):
    """验证/推理用整图数据集（返回原始尺寸，不 resize）。"""

    def __init__(self, cfg, ids, with_label=True, image_dir=None, label_dir=None,
                 records=None):
        self.cfg = cfg
        self.with_label = with_label
        self.bounded = P.bounded_mask(cfg.channel_list)
        if records is None:
            image_dir = image_dir or cfg.train_image_dir
            if with_label:
                label_dir = label_dir or cfg.train_label_dir
                records = build_records(image_dir, label_dir)
            else:  # 无标签推理：只按图像目录列举，不要求存在同名标签
                records = [{"id": i, "image": resolve_file(image_dir, i), "label": None}
                           for i in list_ids(image_dir)]
        by_id = {r["id"]: r for r in records}
        self.ids = [i for i in ids if i in by_id]
        self.records = [by_id[i] for i in self.ids]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        cfg = self.cfg
        rec = self.records[index]
        rgb = imread_rgb(rec["image"])
        img = P.to_channels(rgb, cfg.channel_list, cfg.clahe_clip, cfg.clahe_grid)
        img = P.normalize(img, cfg.norm_mean_list, cfg.norm_std_list)
        fov = fov_mask(rgb)

        item = {
            "image": torch.from_numpy(img.astype(np.float32)),
            "rgb": torch.from_numpy(rgb.copy()),
            "id": rec["id"],
            "fov": torch.from_numpy(fov.astype(np.float32)),
            "shape": torch.tensor(img.shape[1:], dtype=torch.int32),
        }
        if self.with_label:
            lab = imread_gray(rec["label"])
            mask = (lab > cfg.label_threshold).astype(np.float32)
            item["label"] = torch.from_numpy(mask)
        return item


def _load_pair(cfg, record):
    """读取一张训练图及其标签，预处理为 (C,H,W) float32 与 (H,W) float32。

    ``cfg.scale`` 不为 1 时先把原图与标签按该倍率缩放（图像 CUBIC / 缩小 AREA，
    标签 NEAREST 保持二值）。放大后再切 patch，等于让网络在"血管更粗的像素尺度"上
    学习，同时 patch 覆盖的原图范围不变（即上下文不变、分辨率提高）。
    """
    rgb = imread_rgb(record["image"])
    sf = float(getattr(cfg, "scale", 1.0) or 1.0)
    lab = imread_gray(record["label"])
    if sf != 1.0:
        h, w = rgb.shape[:2]
        nh, nw = max(1, int(round(h * sf))), max(1, int(round(w * sf)))
        lab = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)
    img = P.prepare_rgb(rgb, cfg, sf, normalized=False)
    mask = (lab > cfg.label_threshold).astype(np.float32)
    return img, mask


def _resolve_aug(cfg):
    """把 -1 的增强开关解析成具体强度：跟随 --strong_aug 预设，否则关闭。"""
    strong = bool(cfg.strong_aug)

    def pick(v, key):
        v = float(v)
        if v >= 0:
            return v
        return float(A.STRONG_AUG.get(key, 0.0)) if strong else 0.0

    return {
        "elastic": pick(cfg.aug_elastic, "elastic"),
        "cutout": pick(cfg.aug_cutout, "cutout"),
        "contrast": pick(cfg.aug_contrast, "contrast"),
        "cgap": pick(cfg.aug_cgap, "cgap"),
        "sharpen": pick(cfg.aug_sharpen, "sharpen"),
        "copy_paste_p": pick(cfg.aug_copy_paste, "copy_paste"),
    }
