"""评估已保存的预测掩膜（可与训练集的标签对比）。

用法::

    # 评估验证集 17-20 的预测结果
    python -m src.evaluate --pred_dir image --ids 17,18,19,20

    # 额外保存叠加可视化图
    python -m src.evaluate --pred_dir image --ids 17,18,19,20 --save_fig outputs/fig
"""

import argparse
import os

import numpy as np
import pandas as pd

from . import metrics as M
from .postprocess import fill_holes, remove_small_components
from .utils import Timer, ensure_dir, imread_gray, imread_rgb, imwrite, list_ids


def load_mask(path, shape_ref, threshold=127):
    import cv2
    m = imread_gray(path)
    if m.shape != shape_ref:
        m = cv2.resize(m, (shape_ref[1], shape_ref[0]), interpolation=cv2.INTER_NEAREST)
    return m > threshold


def main(argv=None):
    p = argparse.ArgumentParser(description="评估预测掩膜")
    p.add_argument("--pred_dir", type=str, default="image")
    p.add_argument("--label_dir", type=str, default="train/label")
    p.add_argument("--image_dir", type=str, default="train/image")
    p.add_argument("--ids", type=str, default="17,18,19,20")
    p.add_argument("--min_size", type=int, default=0, help="评估前再剔一次小连通域")
    p.add_argument("--save_fig", type=str, default=None, help="叠加可视化输出目录")
    args = p.parse_args(argv)

    ids = [i.strip() for i in args.ids.split(",") if i.strip()] or list_ids(args.pred_dir)
    if args.save_fig:
        ensure_dir(args.save_fig)

    timer = Timer()
    rows = []
    for i in ids:
        cands = [os.path.join(args.pred_dir, f"{i}{e}")
                 for e in (".png", ".jpg", ".jpeg", ".bmp", ".tif")]
        pred_path = next((c for c in cands if os.path.exists(c)), None)
        if pred_path is None:
            print(f"[跳过] 找不到 {i} 的预测")
            continue

        gt_path = os.path.join(args.label_dir, f"{i}.jpg")
        gt = imread_gray(gt_path) > 127
        pred = load_mask(pred_path, gt.shape)
        if args.min_size > 0:
            pred = remove_small_components(pred, args.min_size)

        m = M.evaluate_binary(pred, gt)
        m["id"] = i
        rows.append(m)

        if args.save_fig:
            rgb = imread_rgb(os.path.join(args.image_dir, f"{i}.jpg"))
            ov = rgb.copy()
            ov[pred & gt] = [255, 255, 255]   # 正确检测 -> 白
            ov[pred & ~gt] = [255, 0, 0]      # 误检 -> 红
            ov[~pred & gt] = [0, 255, 0]      # 漏检 -> 绿
            imwrite(os.path.join(args.save_fig, f"{i}_overlay.png"),
                    ov[:, :, ::-1].copy())

    if not rows:
        raise SystemExit("没有可评估的预测文件")

    df = pd.DataFrame(rows)[["id", "dice", "iou", "se", "sp", "acc"]]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("-" * 60)
    print(f"平均 Dice  = {df['dice'].mean():.4f}  ± {df['dice'].std():.4f}")
    print(f"平均 IoU   = {df['iou'].mean():.4f}")
    print(f"平均 Se/Sp = {df['se'].mean():.4f} / {df['sp'].mean():.4f}")
    print(f"平均 Acc   = {df['acc'].mean():.4f}")
    print(f"耗时 {timer}")

    out_csv = os.path.join(os.path.dirname(args.pred_dir) or ".",
                           "evaluate_report.csv")
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"明细已写入 {out_csv}")
    return df


if __name__ == "__main__":
    main()
