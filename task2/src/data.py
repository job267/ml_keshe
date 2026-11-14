"""数据管线：扫描、重复组审计、分组分层划分、增强、MixUp 与随机遮挡。

设计要点（对应报告第 2 节与第十二轮）：
1. **重复组**：完全相同像素或完全左右镜像视为同一组（md5 of raw uint8 bytes），
   划分按组进行，保证同一内容的镜像副本不会跨训练/验证。
2. **分组分层划分**：按类别比例把"组"整体分配到 train / val / holdout，
   目标张数由 data.val_size 与 data.holdout_size 控制（第一轮 2848 / 2836）。
3. **质量权重**：常量图权重 0，低对比度与标签冲突组温和降权，再按组规模的平方根平衡，
   最后归一化到均值 1；第八轮 C 的保守版本只对折外高置信度不一致样本降权。
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from . import ARTIFACTS, CLASSES, CLASS_TO_IDX, DATA_DIR, NUM_CLASSES


# --------------------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------------------
def list_train(root: str | Path | None = None) -> List[Tuple[Path, int]]:
    root = Path(root or DATA_DIR)
    out: List[Tuple[Path, int]] = []
    for cls in CLASSES:
        d = root / "train" / cls
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")):
            out.append((p, CLASS_TO_IDX[cls]))
    if not out:
        raise FileNotFoundError(f"没有在 {root / 'train'} 下找到类别子目录（{CLASSES}）")
    return out


def list_test(root: str | Path | None = None) -> List[Path]:
    root = Path(root or DATA_DIR)
    d = root / "test"
    if not d.is_dir():
        raise FileNotFoundError(f"缺少测试目录：{d}")
    return sorted(p for p in d.glob("*.jpg"))


def _pixels(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def audit_duplicates(paths: Sequence[Path], cache: Path | None = None) -> Dict:
    """按像素哈希把完全相同/镜像相同的图像并成组，返回组信息与统计。"""
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    hashes: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(paths):
        arr = _pixels(p)
        hashes[arr.tobytes()].append(i)
        hashes[np.ascontiguousarray(np.fliplr(arr)).tobytes()].append(i)

    uf = _UnionFind(len(paths))
    for idxs in hashes.values():
        if len(idxs) > 1:
            for j in idxs[1:]:
                uf.union(idxs[0], j)

    group_ids = [uf.find(i) for i in range(len(paths))]
    sizes = defaultdict(int)
    for g in group_ids:
        sizes[g] += 1
    info = {
        "n_images": len(paths),
        "group_ids": group_ids,
        "n_groups": len(sizes),
        "n_extra_images": sum(max(0, s - 1) for s in sizes.values()),
        "group_sizes": {str(k): v for k, v in sizes.items()},
    }
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(info), encoding="utf-8")
    return info


def audit_summary(samples: Sequence[Tuple[Path, int]], group_ids: Sequence[int],
                  test_paths: Sequence[Path] | None = None) -> Dict:
    """类别分布、标签冲突组、常量图、低对比度图、与测试集的共享像素组。"""
    per_class = [0] * NUM_CLASSES
    for _, y in samples:
        per_class[y] += 1

    by_group: Dict[int, List[int]] = defaultdict(list)
    for (_, y), g in zip(samples, group_ids):
        by_group[g].append(y)
    conflict_groups = sum(1 for ys in by_group.values() if len(set(ys)) > 1)

    const_imgs, low_imgs = [], []
    for p, _ in samples:
        arr = _pixels(p)
        s = float(arr.std())
        if s == 0:
            const_imgs.append(p.name)
        if s < 5:            # 报告口径：std<5 含常量图（11 常量 + 2 低对比 = 13）
            low_imgs.append(p.name)

    shared = {}
    if test_paths:
        test_hash = set()
        for p in test_paths:
            arr = _pixels(p)
            test_hash.add(arr.tobytes())
            test_hash.add(np.ascontiguousarray(np.fliplr(arr)).tobytes())
        shared_imgs = []
        for (p, _) in samples:
            arr = _pixels(p)
            if arr.tobytes() in test_hash or np.ascontiguousarray(np.fliplr(arr)).tobytes() in test_hash:
                shared_imgs.append(p.name)
        shared = {"train_images_matching_test": len(shared_imgs),
                  "groups": len({g for (p, _), g in zip(samples, group_ids) if p.name in set(shared_imgs)})}

    return {
        "classes": list(CLASSES),
        "per_class": {CLASSES[i]: per_class[i] for i in range(NUM_CLASSES)},
        "n_train": len(samples),
        "conflict_groups": conflict_groups,
        "constant_images": len(const_imgs),
        "low_contrast_images": len(low_imgs),
        "constant_examples": const_imgs[:10],
        "low_contrast_examples": low_imgs[:10],
        "shared_with_test": shared,
    }


# --------------------------------------------------------------------------------------
# 划分
# --------------------------------------------------------------------------------------
def grouped_stratified_split(samples: Sequence[Tuple[Path, int]], group_ids: Sequence[int],
                             val_size: int, holdout_size: int, seed: int = 42) -> Dict[str, List[int]]:
    """按组分层：先把 holdout 抽满，再抽 val，其余进 train。"""
    rng = random.Random(seed)
    sizes: Dict[int, int] = defaultdict(int)
    for g in group_ids:
        sizes[g] += 1

    # 每个"组"归到组内出现最多的类别（组内标签冲突时按多数）
    label_of_group: Dict[int, int] = {}
    votes: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for (_, y), g in zip(samples, group_ids):
        votes[g][y] += 1
    for g, v in votes.items():
        label_of_group[g] = max(v.items(), key=lambda kv: kv[1])[0]

    groups_by_class: Dict[int, List[int]] = defaultdict(list)
    for g, y in label_of_group.items():
        groups_by_class[y].append(g)
    for y in groups_by_class:
        groups_by_class[y].sort()
        rng.shuffle(groups_by_class[y])

    total = len(samples)
    class_counts = {y: sum(sizes[g] for g in groups_by_class[y]) for y in range(NUM_CLASSES)}

    def quota(target: int) -> Dict[int, int]:
        return {y: int(round(target * class_counts.get(y, 0) / max(1, total))) for y in range(NUM_CLASSES)}

    def take(target: int, exclude: set[int]) -> set[int]:
        """按类别配额整组抽取，直到该类达标或该类别组用完。"""
        picked: set[int] = set()
        q = quota(target)
        got = defaultdict(int)
        for y in range(NUM_CLASSES):
            for g in groups_by_class[y]:
                if got[y] >= q[y]:
                    break
                if g in exclude or g in picked:
                    continue
                picked.add(g)
                got[y] += sizes[g]
        # 配额未满时，用任意剩余组补齐（保持总量接近目标）
        cur = sum(sizes[g] for g in picked)
        if cur < target:
            pool = [g for y in range(NUM_CLASSES) for g in groups_by_class[y]
                    if g not in exclude and g not in picked]
            rng.shuffle(pool)
            for g in pool:
                if cur >= target:
                    break
                picked.add(g)
                cur += sizes[g]
        # 整组抽取会整块越过目标，若能找到恰好等于超出量的组就退回，保证配额精确
        while cur > target:
            overshoot = cur - target
            exact = [g for g in picked if sizes[g] == overshoot]
            if not exact:
                break
            g = max(exact)
            picked.remove(g)
            cur -= sizes[g]
        return picked

    hold_groups = take(int(holdout_size), set())
    val_groups = take(int(val_size), hold_groups)

    splits = {"train": [], "val": [], "holdout": []}
    for idx, g in enumerate(group_ids):
        if g in hold_groups:
            splits["holdout"].append(idx)
        elif g in val_groups:
            splits["val"].append(idx)
        else:
            splits["train"].append(idx)
    return splits


def group_kfold(samples: Sequence[Tuple[Path, int]], group_ids: Sequence[int], k: int = 3,
                seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """按组做 k 折（用于折外研究），折内类别比例尽量一致。"""
    rng = random.Random(seed)
    groups = sorted(set(group_ids))
    rng.shuffle(groups)
    fold_of_group = {g: i % k for i, g in enumerate(groups)}
    folds = []
    for f in range(k):
        tr = [i for i, g in enumerate(group_ids) if fold_of_group[g] != f]
        te = [i for i, g in enumerate(group_ids) if fold_of_group[g] == f]
        folds.append((tr, te))
    return folds


# --------------------------------------------------------------------------------------
# 质量权重
# --------------------------------------------------------------------------------------
def compute_quality_weights(samples: Sequence[Tuple[Path, int]], group_ids: Sequence[int],
                            mode: str = "audit", oof_probs: Optional[np.ndarray] = None,
                            oof_labels: Optional[np.ndarray] = None,
                            conf_thr: float = 0.95, gap_thr: float = 0.8,
                            weight: float = 0.5, per_class_cap: float = 0.01) -> np.ndarray:
    w = np.ones(len(samples), dtype=np.float64)
    sizes = defaultdict(int)
    for g in group_ids:
        sizes[g] += 1

    if mode == "conservative_oof":
        if oof_probs is None or oof_labels is None:
            raise ValueError("conservative_oof 模式需要折外概率与标签")
        pred = oof_probs.argmax(1)
        conf = oof_probs.max(1)
        same = oof_probs[np.arange(len(pred)), oof_labels]
        bad = (pred != oof_labels) & (conf >= conf_thr) & ((conf - same) >= gap_thr)
        rng = np.random.default_rng(0)
        for c in range(NUM_CLASSES):
            idx = np.where(bad & (oof_labels == c))[0]
            cap = int(math.ceil(per_class_cap * (oof_labels == c).sum()))
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            w[idx] = weight
    else:
        for i, (p, _y) in enumerate(samples):
            arr = _pixels(p)
            s = float(arr.std())
            if s == 0:
                w[i] = 0.0
            elif s < 5:
                w[i] = 0.6

    # 冲突组温和降权
    labs = defaultdict(set)
    for (_, y), g in zip(samples, group_ids):
        labs[g].add(y)
    for i, g in enumerate(group_ids):
        if len(labs[g]) > 1 and w[i] > 0:
            w[i] = min(w[i], 0.6)

    for i, g in enumerate(group_ids):
        w[i] = w[i] / math.sqrt(max(1, sizes.get(g, 1)))
    m = w.mean()
    return w / m if m > 0 else w


# --------------------------------------------------------------------------------------
# 数据集与增强
# --------------------------------------------------------------------------------------
class RandomErasing:
    """随机遮挡：面积占比 2%–12%，宽高比 0.5–2，逐图独立触发。"""

    def __init__(self, prob: float = 0.25, area: Sequence[float] = (0.02, 0.12),
                 ratio: Sequence[float] = (0.5, 2.0), fill: float = 0.0):
        self.prob, self.area, self.ratio, self.fill = prob, area, ratio, fill

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return img
        _, h, w = img.shape
        for _ in range(10):
            area = random.uniform(*self.area) * h * w
            ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            eh = int(round(math.sqrt(area * ratio)))
            ew = int(round(math.sqrt(area / ratio)))
            if eh < h and ew < w:
                y = random.randint(0, h - eh)
                x = random.randint(0, w - ew)
                img[:, y:y + eh, x:x + ew] = self.fill
                break
        return img


def _strength(v: float, s: float) -> float:
    return v * s


def build_transforms(cfg: dict, train: bool) -> T.Compose:
    size = int(cfg["data"]["img_size"])
    strength = float(cfg["aug"].get("strength", 1.0))
    tf: List = []
    if train:
        tf.append(T.RandomAffine(degrees=_strength(float(cfg["aug"].get("rotation", 12.0)), strength),
                                 scale=tuple(cfg["aug"].get("scale", [0.9, 1.0]))))
        if float(cfg["aug"].get("hflip", 0.5)) > 0:
            tf.append(T.RandomHorizontalFlip(float(cfg["aug"]["hflip"]) * min(1.0, strength)))
        b = _strength(float(cfg["aug"].get("brightness", 0.2)), strength)
        c = _strength(float(cfg["aug"].get("contrast", 0.2)), strength)
        if b > 0 or c > 0:
            tf.append(T.ColorJitter(brightness=b, contrast=c))
    tf.append(T.Resize((size, size), antialias=True))
    tf.append(T.ToTensor())
    if bool(cfg["model"].get("pretrained")) or bool(cfg["data"].get("grayscale_to_rgb")):
        tf.append(T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
        tf.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        tf.append(T.Normalize([0.5], [0.5]))
    er = cfg["aug"].get("erasing", {})
    if train and er.get("enabled"):
        tf.append(RandomErasing(er.get("prob", 0.25), er.get("area", [0.02, 0.12]),
                                er.get("ratio", [0.5, 2.0])))
    return T.Compose(tf)


class FERDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], tf, weights: Optional[np.ndarray] = None):
        self.samples = list(samples)
        self.tf = tf
        self.weights = weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        p, y = self.samples[i]
        with Image.open(p) as im:
            img = im.convert("L")
            img = self.tf(img)
        w = float(self.weights[i]) if self.weights is not None else 1.0
        return img, y, i, w


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int = 4,
                drop_last: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=workers > 0)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def soft_targets(y: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    t = torch.full((y.size(0), NUM_CLASSES), smoothing / NUM_CLASSES, device=y.device)
    t.scatter_add_(1, y.view(-1, 1), torch.full_like(t[:, :1], 1.0 - smoothing))
    return t


def mixup_targets(ta: torch.Tensor, tb: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * ta + (1 - lam) * tb


def class_weights_from_counts(samples: Sequence[Tuple[Path, int]]) -> torch.Tensor:
    cnt = np.bincount([y for _, y in samples], minlength=NUM_CLASSES).astype(np.float64)
    w = cnt.sum() / (NUM_CLASSES * np.maximum(cnt, 1))
    return torch.tensor(w, dtype=torch.float32)


def prepare_data(cfg: dict, seed: int | None = None):
    """返回 (samples, group_ids, splits, weights, meta)；结果可缓存到 artifacts/。"""
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        from . import ROOT
        root = ROOT / root
    samples = list_train(root)
    cache = ARTIFACTS / "dup_index.json" if cfg["data"].get("cache_dup_index") else None
    dup = audit_duplicates([p for p, _ in samples], cache)
    group_ids = dup["group_ids"]
    seed = int(seed if seed is not None else cfg["exp"]["seed"])
    splits = grouped_stratified_split(samples, group_ids, int(cfg["data"]["val_size"]),
                                      int(cfg["data"]["holdout_size"]), seed)
    weights = None
    if cfg["train"].get("quality_weights"):
        weights = compute_quality_weights(samples, group_ids, cfg["train"].get("quality_mode", "audit"))
    meta = {"dup": {k: v for k, v in dup.items() if k != "group_ids"},
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "seed": seed}
    return samples, group_ids, splits, weights, meta
