"""把下载到的原始外部数据整理成统一结构：

    data_external/
    ├── _raw/            # 原样下载（不再改动）
    ├── stare/{image,label}/   stare_im0001.png ...
    ├── chase/{image,label}/   chase_Image_01L.png ...
    ├── hrf/{image,label}/     hrf_01_dr.png ...
    └── meta.json        # 每个数据集的张数、分辨率、血管占比

统一约定（与 DRIVE 一致）：图像 PNG、掩膜 PNG、血管=255 / 背景=0、文件名去重前缀。

用法::

    python scripts/prepare_external.py
    python scripts/prepare_external.py --only stare
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_external", "_raw")
OUT = os.path.join(ROOT, "data_external")

RAW_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp")

# 各数据集：标签文件名规则（给 stem 加的后缀）
LABEL_SUFFIX = {
    "stare": ".ah",       # im0001.ppm -> im0001.ah.ppm
    "chase": "_1stHO",    # Image_01L.png -> Image_01L_1stHO.png
    "hrf": "",            # 01_dr.JPG -> 01_dr.tif
}


def find_file(dirpath, stem):
    for ext in RAW_EXTS:
        p = os.path.join(dirpath, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_mask(path):
    """读标签并统一成 0/255 的 (H,W) uint8。"""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == bool:                 # CHASE 的 mode "1"
        arr = arr.astype(np.uint8) * 255
    elif arr.max() <= 1:                  # 0/1 编码
        arr = (arr > 0).astype(np.uint8) * 255
    else:
        arr = ((arr > 127).astype(np.uint8)) * 255
    return arr


def process(key, verbose=True):
    raw_imgs = sorted(glob.glob(os.path.join(RAW, key, "*", "input", "*")))
    if not raw_imgs:
        print(f"[{key}] 没有原始文件，跳过")
        return None

    out_img = os.path.join(OUT, key, "image")
    out_lab = os.path.join(OUT, key, "label")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    n_ok, n_skip, sizes, ratios = 0, 0, set(), []
    for ip in raw_imgs:
        split = os.path.basename(os.path.dirname(os.path.dirname(ip)))
        stem = stem_of(ip)
        lp = find_file(os.path.join(RAW, key, split, "label"), stem + LABEL_SUFFIX.get(key, ""))
        if lp is None:
            if verbose:
                print(f"  [缺少标签] {ip}")
            continue

        name = f"{key}_{stem}"
        oi = os.path.join(out_img, name + ".png")
        ol = os.path.join(out_lab, name + ".png")
        if os.path.exists(oi) and os.path.exists(ol):
            n_skip += 1
        else:
            rgb = np.array(Image.open(ip).convert("RGB"))
            mask = load_mask(lp)
            if rgb.shape[:2] != mask.shape:
                raise RuntimeError(f"尺寸不一致 {ip} {rgb.shape} vs {mask.shape}")
            Image.fromarray(rgb).save(oi)          # PNG 无损
            Image.fromarray(mask).save(ol)
            n_ok += 1
        sizes.add(Image.open(oi).size)
        ratios.append(float(load_mask(ol).mean() / 255.0))

    info = {
        "dataset": key,
        "count": len(ratios),
        "bytes_done": n_ok,
        "sizes": sorted({f"{w}x{h}" for w, h in sizes}),
        "vessel_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
        "image_dir": os.path.relpath(out_img, ROOT),
        "label_dir": os.path.relpath(out_lab, ROOT),
    }
    print(f"[{key}] 转换 {n_ok} 张，已存在 {n_skip} 张，共 {len(ratios)} 张 | "
          f"尺寸 {info['sizes']} | 血管占比 {info['vessel_ratio']}")
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description="整理外部数据集")
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args(argv)

    keys = args.only or [k for k in ("stare", "chase", "hrf")
                         if os.path.isdir(os.path.join(RAW, k))]
    if not keys:
        raise SystemExit("data_external/_raw 为空，请先运行 scripts/download_external.py")

    meta_path = os.path.join(OUT, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    total = 0
    for k in keys:
        info = process(k)
        if info:
            meta[k] = info
            total += info["count"]

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print(f"外部数据合计 {total} 张 -> {OUT}")
    print("下一步：python -m src.pretrain --data_dirs "
          + ",".join(f"data_external/{k}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
