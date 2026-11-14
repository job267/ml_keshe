"""多尺度 TTA：在若干个放大倍率上各预测一次，概率平均。

思路很朴素：血管粗细在图上并不均匀，视盘附近的主血管粗、黄斑区末梢细。
单一尺度下总有一部分血管处在网络的"舒适区"之外，把 0.75 / 1.0 / 1.25 三个尺度的
概率图平均，等于让网络用三种"放大镜"各看一遍。

因为 :func:`src.engine.predict_prob` 支持 ``cfg.scale``（内部放大后前向、再把概率图
缩回原尺寸），这里只需要换 ``cfg.scale`` 多跑几次。

用法::

    # 在 5 折 OOF 上比较 单尺度 vs 多尺度，并按官方口径评分
    python scripts/ms_tta_eval.py --runs outputs_cv_sa --scales 1.0 --scales 0.75,1.0,1.25

    # 单模型
    python scripts/ms_tta_eval.py --checkpoints outputs_hr_s2/best.pt --ids 4,16,13,3 \
        --scales 0.8,1.0,1.25
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_records  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.utils import fov_mask, get_device, imread_gray, imread_rgb  # noqa: E402

from scripts.eval_official_metric import dice, gt_bilinear, gt_nearest, official_binarize  # noqa: E402
from scripts.oof_ensemble import load_split, predict_ids  # noqa: E402


def multi_scale_probs(model, cfg, ids, records, device, scales, tta=True):
    """对同一批图在多个尺度上预测并平均。"""
    acc = {}
    for s in scales:
        cfg.scale = float(s)
        p = predict_ids(model, cfg, ids, records, device, tta)
        for k, v in p.items():
            acc.setdefault(k, []).append(v)
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="多尺度 TTA 评估（官方 512 口径 + 原图口径）")
    ap.add_argument("--runs", default="", help="5 折目录，逗号分隔；按诚实 OOF 评估")
    ap.add_argument("--checkpoints", default="", help="权重列表，逗号分隔；配合 --ids")
    ap.add_argument("--ids", default="4,16,13,3")
    ap.add_argument("--train_image_dir", default="train/image")
    ap.add_argument("--train_label_dir", default="train/label")
    ap.add_argument("--scales", action="append", default=None,
                    help="可重复；每次给一组尺度，例如 --scales 1.0 --scales 0.75,1.0,1.25")
    ap.add_argument("--thresholds", default="0.40,0.45,0.50")
    ap.add_argument("--min_size", type=int, default=10)
    ap.add_argument("--tta", default=True, type=lambda s: str(s).lower() in ("1", "true", "yes", "y"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    records = build_records(args.train_image_dir, args.train_label_dir)
    scalesets = args.scales or ["1.0", "0.75,1.0,1.25"]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    gt_cache, fov_cache = {}, {}
    for r in records:
        gt = imread_gray(r["label"]) > 127
        gt_cache[r["id"]] = (gt, gt_bilinear(gt), gt_nearest(gt))
        fov_cache[r["id"]] = fov_mask(imread_rgb(r["image"]))

    print(f"{'模型/尺度':>34} {'thr':>5} | {'512双线性':>9} {'512最近邻':>9} {'原图':>7} | {'面积%':>6}")
    print("-" * 84)

    # 收集 (tag, probs) 列表
    todo = []
    if args.checkpoints:
        for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
            model, cfg, _ = load_checkpoint(ck, device=device)
            ids = [i.strip() for i in args.ids.split(",") if i.strip()]
            todo.append((os.path.basename(os.path.dirname(ck)) + "/" + os.path.basename(ck),
                         model, cfg, ids))
    if args.runs:
        for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
            split = load_split(run)
            for fold, val_ids in sorted(split.items()):
                ck = os.path.join(run, fold, "best.pt")
                if not os.path.exists(ck):
                    continue
                model, cfg, _ = load_checkpoint(ck, device=device)
                todo.append((f"{os.path.basename(run)}/{fold}", model, cfg, list(val_ids)))

    for tag, model, cfg, ids in todo:
        for sc in scalesets:
            scales = [float(x) for x in sc.split(",") if x.strip()]
            probs = multi_scale_probs(model, cfg, ids, records, device, scales, args.tta)
            for thr in thresholds:
                d_bi, d_nn, d_nat, areas = [], [], [], []
                for rid, p in probs.items():
                    gt, gt_bi, gt_nn = gt_cache[rid]
                    mask = postprocess(p, fov_cache[rid], threshold=thr,
                                       min_size=args.min_size, do_fill_holes=False)
                    pred = official_binarize(mask)
                    d_bi.append(dice(pred, gt_bi))
                    d_nn.append(dice(pred, gt_nn))
                    d_nat.append(dice(mask, gt))
                    areas.append(pred.mean())
                print(f"{tag + ' @' + sc:>34} {thr:>5.2f} | {np.mean(d_bi):>9.4f} "
                      f"{np.mean(d_nn):>9.4f} {np.mean(d_nat):>7.4f} | {100*np.mean(areas):>6.2f}")

    return None


if __name__ == "__main__":
    main()
