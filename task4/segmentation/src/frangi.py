"""传统方法基线：Frangi 血管增强 + 自动阈值。

作用有两个：

1. 提供一个**不依赖深度学习**的参照分数，说明 U-Net 的增益来自哪里；
2. 在 GPU/训练不可用时，仍能产出可提交的掩膜（保底方案）。

方法：绿色通道 -> CLAHE -> Frangi(vesselness, black_ridges=True) -> 分位数阈值 -> FOV 裁剪
      -> 小连通域剔除。

注意：**阈值必须用分位数而不是 Otsu**。血管响应图的分布极度长尾（95% 以上像素
响应接近 0），Otsu 会切在很高的位置，只留下最强的一批像素，血管占比掉到 0.5%，
Dice 只有 0.01。改用"保留响应最强的 22%"后 Dice 恢复到 0.6 左右。

用法::

    python -m src.frangi --pred_dir image_frangi
"""

import argparse
import os

import cv2
import numpy as np
from skimage.filters import frangi

from . import metrics as M
from .postprocess import remove_small_components
from .utils import Timer, ensure_dir, fov_mask, imread_gray, imread_rgb, imwrite, list_ids


def frangi_vesselness(rgb: np.ndarray, sigmas=(1, 2, 3, 4, 5, 6),
                      clahe_clip=2.0, clahe_grid=8,
                      black_ridges=True, beta=0.5):
    """返回归一化到 [0,1] 的血管响应图。

    black_ridges=True 是关键：血管在绿色通道里是**暗色**脊线，
    skimage 需要 black_ridges=True 才会把暗脊线检出为高响应；
    设成 False 会去抓亮结构（视盘边缘、反光），实测 Dice 会从 0.65 掉到 0.12。
    """
    green = rgb[..., 1]
    op = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    g = op.apply(green).astype(np.float64) / 255.0
    resp = frangi(g, sigmas=sigmas, black_ridges=black_ridges, beta=beta)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
    mx = resp.max()
    if mx > 0:
        resp = resp / mx
    return resp.astype(np.float32)


def otsu_threshold(resp: np.ndarray, fov: np.ndarray) -> float:
    vals = resp[fov]
    if vals.size == 0:
        return 0.5
    u8 = np.clip(resp * 255, 0, 255).astype(np.uint8)
    sel = u8[fov]
    if len(np.unique(sel)) > 1:
        thr, _ = cv2.threshold(sel.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(thr) / 255.0
    return float(np.quantile(vals, 0.9))


def main(argv=None):
    p = argparse.ArgumentParser(description="Frangi 传统方法基线")
    p.add_argument("--image_dir", type=str, default="test/image")
    p.add_argument("--label_dir", type=str, default=None,
                   help="若提供标签目录，则同时报告 Dice")
    p.add_argument("--pred_dir", type=str, default="image_frangi")
    p.add_argument("--min_size", type=int, default=30)
    p.add_argument("--quantile", type=float, default=0.22,
                   help="0~1：保留 FOV 内响应最强的这一比例像素（设为 <=0 则退回 Otsu）")
    p.add_argument("--save_fig", action="store_true")
    args = p.parse_args(argv)

    ensure_dir(args.pred_dir)
    ids = list_ids(args.image_dir)
    timer = Timer()
    dices = []

    for i in ids:
        rgb = imread_rgb(os.path.join(args.image_dir, f"{i}.jpg"))
        fov = fov_mask(rgb)
        resp = frangi_vesselness(rgb)

        thr = (float(np.quantile(resp[fov], 1.0 - args.quantile))
               if args.quantile > 0 else otsu_threshold(resp, fov))
        mask = (resp > thr) & fov
        mask = remove_small_components(mask, args.min_size)

        imwrite(os.path.join(args.pred_dir, f"{i}.png"),
                (mask.astype(np.uint8) * 255))

        msg = f"  {i:>3s}  thr={thr:.3f}  血管占比 {mask.mean()*100:5.2f}%"
        if args.label_dir:
            gt_path = os.path.join(args.label_dir, f"{i}.jpg")
            if os.path.exists(gt_path):
                gt = imread_gray(gt_path) > 127
                d = M.dice_coef(mask, gt)
                dices.append(d)
                msg += f"  Dice={d:.4f}"
        print(msg)

    print("-" * 60)
    if dices:
        print(f"Frangi 基线 平均 Dice = {np.mean(dices):.4f}（{len(dices)} 张）")
    print(f"输出目录 {args.pred_dir}，耗时 {timer}")


if __name__ == "__main__":
    main()
