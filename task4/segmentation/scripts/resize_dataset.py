"""把 image/ + label/ 目录按长边缩放到指定尺寸，生成新的数据目录。

用于把 FIVES（下载时为 1024）缩放到与 DRIVE 相近的尺度（640），
减小预训练与目标域之间的"血管粗细/视野大小"域差。

用法：
    python scripts/resize_dataset.py --src data_external/fives --dst data_external/fives640 --max_side 640
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import ensure_dir  # noqa: E402


def resize_dir(src, dst, max_side, img_resample=Image.LANCZOS):
    ensure_dir(os.path.join(dst, "image"))
    ensure_dir(os.path.join(dst, "label"))
    files = sorted(glob.glob(os.path.join(src, "image", "*.png")))
    n = 0
    for ip in files:
        name = os.path.basename(ip)
        lp = os.path.join(src, "label", name)
        if not os.path.exists(lp):
            continue
        img = Image.open(ip).convert("RGB")
        msk = Image.open(lp).convert("L")
        w, h = img.size
        sc = min(1.0, max_side / max(w, h))
        if sc < 1.0:
            size = (int(round(w * sc)), int(round(h * sc)))
            img = img.resize(size, img_resample)
            msk = msk.resize(size, Image.NEAREST)
        img.save(os.path.join(dst, "image", name), optimize=True)
        msk.point(lambda v: 255 if v > 127 else 0).save(os.path.join(dst, "label", name), optimize=True)
        n += 1
    meta_src = os.path.join(src, "meta_fives.json")
    if os.path.exists(meta_src):
        shutil.copy(meta_src, os.path.join(dst, "meta_fives.json"))
    print(f"缩放完成：{n} 张 -> {dst}（长边 {max_side}）")
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description="按长边缩放 image/label 目录")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--max_side", type=int, default=640)
    a = p.parse_args(argv)
    resize_dir(a.src, a.dst, a.max_side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
