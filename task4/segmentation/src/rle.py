"""RLE 提交文件生成（与官方 ``segmentation_to_csv.py`` 完全等价）。

官方脚本的编码逻辑::

    image = Image.open(path).convert('L').resize((512,512), BILINEAR)
    image = ToTensor()(image)        # 归一化到 [0,1]，形状 (1,512,512)
    image[image > 0] = 1             # 任何非零像素都算前景 -> 即"血管=255"
    dots = np.where(image.flatten() == 1)[0]     # 行优先展开
    # 连续段 -> ("起点+1", "长度")

本模块用 numpy 复现同一逻辑，并提供两个附加能力：

1. ``--out submission.csv``：不依赖 torchvision 也能生成提交文件；
2. ``--verify 官方submission.csv``：逐行比对，确认两条链路结果一致。

用法::

    python -m src.rle                                  # ./image -> submission.csv
    python -m src.rle --image_dir image --out submission.csv
    python -m src.rle --verify submission.csv          # 校验一致性
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from .utils import list_ids

RLE_SHAPE = (512, 512)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# 核心编码
# --------------------------------------------------------------------------- #
def mask_to_binary_512(path: str) -> np.ndarray:
    """读取掩膜并按官方流程转为 512x512 的 bool 数组。"""
    img = Image.open(path).convert("L").resize(RLE_SHAPE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr > 0.0


def rle_encode(binary: np.ndarray) -> str:
    """行优先 RLE，返回 '起点 长度 起点 长度 ...'（起点从 1 开始）。"""
    flat = np.asarray(binary, dtype=bool).reshape(-1)
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return ""
    # 连续段切分：相邻索引差 > 1 处断开
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    start_pos = idx[starts] + 1                    # 1-based
    lengths = idx[ends] - idx[starts] + 1
    return " ".join(f"{s} {l}" for s, l in zip(start_pos.tolist(), lengths.tolist()))


def rle_decode(rle: str, shape=RLE_SHAPE) -> np.ndarray:
    """解码回 bool 数组（用于校验与可视化）。"""
    out = np.zeros(shape[0] * shape[1], dtype=bool)
    if not isinstance(rle, str) or not rle.strip():
        return out.reshape(shape)
    vals = np.array([int(v) for v in rle.split()], dtype=np.int64)
    starts, lengths = vals[0::2], vals[1::2]
    for s, l in zip(starts, lengths):
        out[s - 1: s - 1 + l] = True
    return out.reshape(shape)


def validate_rle(rle: str, shape=RLE_SHAPE) -> dict:
    """按竞赛要求检查：像素对升序、为正数、解码后无重复。"""
    if not isinstance(rle, str) or not rle.strip():
        return {"ok": True, "pairs": 0, "pixels": 0, "msg": "空掩膜"}
    vals = np.array([int(v) for v in rle.split()], dtype=np.int64)
    if vals.size % 2 != 0:
        return {"ok": False, "msg": "像素对个数不是偶数"}
    starts, lengths = vals[0::2], vals[1::2]
    if (starts < 1).any() or (lengths < 1).any():
        return {"ok": False, "msg": "存在非正数起点/长度"}
    if np.any(np.diff(starts) <= 0):
        return {"ok": False, "msg": "起点未严格递增"}
    # 段之间不能重叠（起点必须大于上一段末尾）
    ends = starts + lengths - 1
    if np.any(starts[1:] <= ends[:-1]):
        return {"ok": False, "msg": "存在重叠像素"}
    if ends.max() > shape[0] * shape[1]:
        return {"ok": False, "msg": "越界"}
    return {"ok": True, "pairs": int(starts.size), "pixels": int(lengths.sum())}


# --------------------------------------------------------------------------- #
# 批处理
# --------------------------------------------------------------------------- #
def build_submission(image_dir: str, out_csv: str) -> pd.DataFrame:
    ids = list_ids(image_dir) if os.path.isdir(image_dir) else []
    # 目录内可能还有别的扩展名，按实际文件再过滤一次
    ids = [i for i in ids
           if os.path.exists(os.path.join(image_dir, i + ".png"))
           or any(os.path.exists(os.path.join(image_dir, i + e)) for e in IMG_EXTS)]

    rows, problems = [], []
    for i in ids:
        path = _resolve(image_dir, i)
        rle = rle_encode(mask_to_binary_512(path))
        chk = validate_rle(rle)
        if not chk["ok"]:
            problems.append((i, chk["msg"]))
        rows.append({"Id": i, "Predicted": rle})

    df = pd.DataFrame(rows, columns=["Id", "Predicted"])
    df.to_csv(out_csv, index=False)
    if problems:
        print(f"[警告] {len(problems)} 张掩膜编码异常: {problems[:5]}")
    print(f"已写入 {out_csv}（{len(df)} 行）")
    return df


def _resolve(image_dir, i):
    for e in IMG_EXTS:
        p = os.path.join(image_dir, i + e)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{image_dir} 下找不到 {i} 的掩膜")


def verify(official_csv: str, mine_csv: str) -> bool:
    """逐行比对两份提交文件（按 Id 对齐）。"""
    a = pd.read_csv(official_csv, keep_default_na=False, dtype=str)
    b = pd.read_csv(mine_csv, keep_default_na=False, dtype=str)
    a = a.sort_values("Id", key=lambda s: s.astype(int)).reset_index(drop=True)
    b = b.sort_values("Id", key=lambda s: s.astype(int)).reset_index(drop=True)
    if list(a["Id"]) != list(b["Id"]):
        print(f"[不一致] Id 集合不同: {list(a['Id'])} vs {list(b['Id'])}")
        return False
    bad = [i for i, x, y in zip(a["Id"], a["Predicted"], b["Predicted"]) if x != y]
    if bad:
        print(f"[不一致] {len(bad)} 行不同: {bad[:5]}")
        return False
    print(f"[一致] {len(a)} 行 RLE 与官方脚本输出完全相同")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="生成 RLE 提交文件 / 校验与官方脚本一致性")
    p.add_argument("--image_dir", type=str, default="image", help="预测掩膜目录")
    p.add_argument("--out", type=str, default="submission.csv", help="输出 csv")
    p.add_argument("--verify", type=str, default=None,
                   help="与官方 segmentation_to_csv.py 生成的 csv 比对")
    args = p.parse_args(argv)

    if args.verify:
        mine = args.out if os.path.exists(args.out) else None
        if mine is None:
            build_submission(args.image_dir, args.out)
        ok = verify(args.verify, args.out)
        raise SystemExit(0 if ok else 1)

    build_submission(args.image_dir, args.out)


if __name__ == "__main__":
    main()
