"""按官方预测转换评估，同时报告两种隐藏 GT 转换假设下的代理 Dice。

为什么需要这个脚本
------------------
本地一直在"原图分辨率、最近邻标签"上算 Dice（0.8156），而 Kaggle 上同一条提交能到
0.85676。差异来自官方脚本这段预处理::

    Image.open(p).convert("L").resize((512, 512), BILINEAR)   # 双线性缩放
    image = ToTensor()(img)                                   # /255
    image[image > 0] = 1                                      # 只要 >0 就算前景

双线性插值会在血管边缘产生中间值，`> 0` 于是把边缘"补"成前景 —— 相当于掩膜被膨胀了
约 1 个源像素（实测面积 9.18% → 12.90%）。评分方的 GT 大概率也走了同样的 pipeline，
本脚本不能确认隐藏 GT 的处理方式；双线性 GT 指标只是代理，不能视为已核实的线上评分。

用法
----
    # 单 run 五折诚实 OOF，扫阈值 + 小连通域
    python scripts/eval_official_metric.py --runs outputs_cv_sa

    # 比较多个 run（各自诚实 OOF）
    python scripts/eval_official_metric.py --runs outputs_cv,outputs_cv_sa,outputs_hr_s1

    # 只扫阈值，输出表格
    python scripts/eval_official_metric.py --runs outputs_cv_sa --thresholds 0.3,0.35,0.4,0.45,0.5
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_records  # noqa: E402
from src.evaluation import official_binarize, gt_bilinear, gt_nearest  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.utils import fov_mask, get_device, imread_gray, imread_rgb  # noqa: E402

OFFICIAL_SIZE = (512, 512)


# --------------------------------------------------------------------------- #
# 官方口径的两个变换
# --------------------------------------------------------------------------- #
def dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return float((2.0 * inter + eps) / (a.sum() + b.sum() + eps))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="官方口径（512 双线性 + >0）下评估 OOF 预测")
    p.add_argument("--runs", default="outputs_cv_sa", help="逗号分隔的 5 折输出目录")
    p.add_argument("--checkpoints", default="", help="逗号分隔的权重；给了就按 --ids 直接预测（用于单模型筛选）")
    p.add_argument("--ids", default="4,16,13,3", help="配合 --checkpoints 使用，默认第 0 折的留出图")
    p.add_argument("--train_image_dir", default="train/image")
    p.add_argument("--train_label_dir", default="train/label")
    p.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55")
    p.add_argument("--min_sizes", default="0,10,30")
    p.add_argument("--use_fov", default=True, type=lambda s: str(s).lower() in ("1", "true", "yes", "y"))
    p.add_argument("--tta", default=True, type=lambda s: str(s).lower() in ("1", "true", "yes", "y"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_csv", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from scripts.oof_ensemble import load_split, predict_ids  # 复用诚实 OOF 的取数逻辑
    from src.models import load_checkpoint

    device = get_device(args.device)
    records = build_records(args.train_image_dir, args.train_label_dir)
    by_id = {r["id"]: r for r in records}
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    min_sizes = [int(x) for x in args.min_sizes.split(",") if x.strip()]

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    if ckpts:
        runs = ckpts  # 单模型模式：每个"run"就是一个权重文件

    # 预先算好 GT 的两种 512 版本（与 run 无关）
    gt_cache = {}
    fov_cache = {}
    for r in records:
        gt = imread_gray(r["label"]) > 127
        gt_cache[r["id"]] = (gt, gt_bilinear(gt), gt_nearest(gt))
        img = imread_rgb(r["image"])
        fov_cache[r["id"]] = fov_mask(img) if args.use_fov else None

    print(f"{'run':>18} {'thr':>5} {'ms':>4} | {'512双线性':>9} {'512最近邻':>9} {'原图':>7} | {'面积%':>6}")
    print("-" * 78)

    rows_out = []
    for run in runs:
        if ckpts:
            # 单模型模式：直接对指定图预测
            model, cfg, _ = load_checkpoint(run, device=device)
            ids = [i.strip() for i in args.ids.split(",") if i.strip()]
            probs = predict_ids(model, cfg, ids, records, device, args.tta)
            tag = os.path.basename(os.path.dirname(run)) + "/" + os.path.basename(run)
        else:
            split = load_split(run)
            # 诚实 OOF：每个 fold 只预测它留出的图
            probs = {}
            for fold, val_ids in sorted(split.items()):
                ck = os.path.join(run, fold, "best.pt")
                if not os.path.exists(ck):
                    print(f"  [跳过] {ck} 不存在")
                    continue
                model, cfg, _ = load_checkpoint(ck, device=device)
                probs.update(predict_ids(model, cfg, list(val_ids), records, device, args.tta))
            tag = os.path.basename(run)

        for thr in thresholds:
            for ms in min_sizes:
                d_bi, d_nn, d_nat, areas = [], [], [], []
                for rid, p in probs.items():
                    gt, gt_bi, gt_nn = gt_cache[rid]
                    mask = postprocess(p, fov_cache[rid], threshold=thr,
                                       min_size=ms, do_fill_holes=False)
                    pred_bi = official_binarize(mask)
                    d_bi.append(dice(pred_bi, gt_bi))
                    d_nn.append(dice(pred_bi, gt_nn))
                    d_nat.append(dice(mask, gt))
                    areas.append(pred_bi.mean())
                row = dict(run=tag, threshold=thr, min_size=ms,
                           dice_512_bilinear=float(np.mean(d_bi)),
                           dice_512_nearest=float(np.mean(d_nn)),
                           dice_native=float(np.mean(d_nat)),
                           area_after_official=float(np.mean(areas)))
                rows_out.append(row)
                print(f"{tag:>18} {thr:>5.2f} {ms:>4d} | "
                      f"{row['dice_512_bilinear']:>9.4f} {row['dice_512_nearest']:>9.4f} "
                      f"{row['dice_native']:>7.4f} | {100*row['area_after_official']:>6.2f}")

    best = max(rows_out, key=lambda r: r["dice_512_bilinear"])
    print("-" * 78)
    print(f"最佳（按 512 双线性口径）: {os.path.basename(best['run'])} "
          f"thr={best['threshold']:.2f} min_size={best['min_size']} "
          f"→ {best['dice_512_bilinear']:.4f} (原图口径 {best['dice_native']:.4f})")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"已写 {args.out_csv}")
    return rows_out


if __name__ == "__main__":
    main()
