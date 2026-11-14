"""在验证集上网格搜索后处理超参（阈值 / 最小连通域面积）。

这类"零成本提分"步骤应该固定在流程里：模型权重不动，只调后处理，
就能把 Dice 拉回最优阈值附近（默认 0.5 往往不是 Dice 的最优点）。

用法::

    python scripts/tune_postprocess.py                       # 默认扫 validation 17-20
    python scripts/tune_postprocess.py --ids 17,18,19,20 --min_sizes 0,5,10,20,40
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import metrics as M  # noqa: E402
from src.config import Config  # noqa: E402
from src.dataset import ImageDataset, discover_ids, split_ids  # noqa: E402
from src.engine import predict_prob  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.utils import get_device  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="后处理超参搜索")
    p.add_argument("--checkpoint", type=str, default="outputs/best.pt")
    p.add_argument("--ids", type=str, default="")
    p.add_argument("--thresholds", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--min_sizes", type=str, default="0,5,10,20,40,80")
    p.add_argument("--fill_holes", type=str, default="0,1")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args(argv)

    device = get_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    if not ids:
        _, ids = split_ids(discover_ids(cfg), cfg.val_ids)
    if not ids:
        raise SystemExit("没有验证集图片，请用 --ids 指定")

    ds = ImageDataset(cfg, ids, with_label=True)
    print(f"验证图片: {ids}")

    cache = []
    for k in range(len(ds)):
        item = ds[k]
        prob = predict_prob(model, item["image"], cfg, device=device, tta=True,
                            rgb=item["rgb"].numpy())
        cache.append((prob, item["fov"].numpy().astype(bool),
                      item["label"].numpy().astype(bool), item["id"]))

    thresholds = [float(v) for v in args.thresholds.split(",")]
    min_sizes = [int(v) for v in args.min_sizes.split(",")]
    fills = [bool(int(v)) for v in args.fill_holes.split(",")]

    results = []
    for fh in fills:
        for ms in min_sizes:
            for th in thresholds:
                ds_ = []
                for prob, fov, gt, _ in cache:
                    pred = postprocess(prob, fov if cfg.use_fov else None,
                                       threshold=th, min_size=ms, do_fill_holes=fh)
                    ds_.append(M.dice_coef(pred, gt))
                results.append({"thr": th, "min_size": ms, "fill_holes": fh,
                                "dice": float(np.mean(ds_))})

    results.sort(key=lambda r: -r["dice"])
    print(f"\n{'thr':>6} {'min_size':>9} {'fill':>5} {'val Dice':>9}")
    print("-" * 34)
    for r in results[:12]:
        print(f"{r['thr']:>6.2f} {r['min_size']:>9d} {str(r['fill_holes']):>5} "
              f"{r['dice']:>9.4f}")
    best = results[0]
    print("-" * 34)
    print(f"最佳: threshold={best['thr']}, min_size={best['min_size']}, "
          f"fill_holes={best['fill_holes']} -> Dice {best['dice']:.4f}")
    print("\n把上述参数写进推理命令，例如：")
    print(f"  python -m src.infer --threshold {best['thr']} --min_size {best['min_size']} "
          f"--fill_holes {int(best['fill_holes'])}")


if __name__ == "__main__":
    main()
