"""生成 README 里用到的可视化图片。

用法::

    python scripts/make_figs.py                 # 默认生成测试集预测图
    python scripts/make_figs.py --ids 13,6,9,15 --data_dir test --pred_dir image
    python scripts/make_figs.py --names figs/test_predictions_ensemble.png --rows 2

图含义：上排原图，下排红色叠加（预测血管）。内部一律按 RGB 处理，写盘前转 BGR。
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_dir, imread_gray, imread_rgb, imwrite, list_ids  # noqa: E402


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm")


def _find(dirpath, stem):
    for ext in IMG_EXTS:
        p = os.path.join(dirpath, stem + ext)
        if os.path.exists(p):
            return p
    return None


def read_mask(path):
    m = imread_gray(path)
    return m > 127


def overlay(rgb, mask, color=(255, 0, 0), alpha=0.85):
    out = rgb.copy()
    lay = out.copy()
    lay[mask] = color
    out = cv2.addWeighted(lay, alpha, out, 1 - alpha, 0)
    return out


def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 13 + 16, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成可视化图")
    ap.add_argument("--data_dir", default="test")
    ap.add_argument("--image_dir", default="", help="默认 <data_dir>/image")
    ap.add_argument("--label_dir", default="", help="默认 <data_dir>/label")
    ap.add_argument("--pred_dir", default="image")
    ap.add_argument("--ids", default="")
    ap.add_argument("--num", type=int, default=4, help="--ids 为空时自动挑前 n 张")
    ap.add_argument("--names", default="figs/test_predictions_ensemble.png")
    ap.add_argument("--scale", type=float, default=0.5, help="每张小图缩放比例")
    args = ap.parse_args(argv)

    image_dir = args.image_dir or os.path.join(args.data_dir, "image")
    ids_all = list_ids(image_dir)
    id_set = set(ids_all)
    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    else:
        ids = ids_all[: args.num]

    tops, bots = [], []
    for i in ids:
        if i not in id_set:
            print(f"[跳过] {image_dir}/{i}.* 不存在")
            continue
        image_path = _find(image_dir, i)
        pred_path = _find(args.pred_dir, i)
        if pred_path is None:
            print(f"[跳过] {args.pred_dir}/{i}.* 不存在")
            continue
        rgb = imread_rgb(image_path)
        mask = read_mask(pred_path)
        h, w = rgb.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        tops.append(put_label(rgb, f"{args.data_dir}/{i}"))
        bots.append(put_label(overlay(rgb, mask), "pred (vessel)"))

    if not tops:
        raise SystemExit("没有可用的图，检查 --data_dir / --pred_dir")

    s = args.scale
    rows = [np.hstack([cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                       for x in row]) for row in (tops, bots)]
    grid = np.vstack(rows)
    # 全流程按 RGB 处理，写盘前一次性转 BGR（cv2.imencode 期望 BGR）
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    ensure_dir(os.path.dirname(args.names) or ".")
    kwargs = {}
    if args.names.lower().endswith(".jpg"):
        kwargs["params"] = [cv2.IMWRITE_JPEG_QUALITY, 95]
    imwrite(args.names, grid, **kwargs)
    print(f"已写 {args.names}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
