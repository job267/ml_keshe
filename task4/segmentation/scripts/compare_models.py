"""在**同一套推理配方**下比较多个权重文件（避免拿不同后处理的分数做比较）。

默认在 DRIVE 验证集 17–20 上比较；给 ``--data_dir`` 可换到任意数据集
（例如外部数据，用来观察迁移能力）。

用法::

    python scripts/compare_models.py outputs/baseline_scratch.pt outputs/best.pt
    python scripts/compare_models.py outputs/baseline_scratch.pt outputs/best.pt \
        --data_dir data_external/stare --ids stare_im0001,stare_im0002
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import ImageDataset, build_records  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.train import evaluate  # noqa: E402
from src.utils import get_device  # noqa: E402


def short_name(path):
    """outputs_joint/best.pt -> joint/best.pt，便于在表里区分同名文件。"""
    parts = os.path.normpath(path).split(os.sep)
    tail = parts[-1].replace(".pt", "")
    if len(parts) >= 2:
        parent = parts[-2]
        if parent.lower() != "outputs":
            return f"{parent}/{tail}"
    return tail


def resolve_root(cfg, data_dir):
    """返回 (image_dir, label_dir)；默认用 DRIVE 训练目录（cfg.train_image_dir 的上一级）。"""
    root = data_dir or os.path.dirname(cfg.train_image_dir)
    return (os.path.join(ROOT, root, "image"), os.path.join(ROOT, root, "label"))


def main(argv=None):
    p = argparse.ArgumentParser(description="多权重统一配方对比")
    p.add_argument("checkpoints", nargs="+", help="一个或多个 .pt 权重")
    p.add_argument("--data_dir", default=None, help="含 image/ 与 label/ 的目录，默认用 DRIVE 验证集")
    p.add_argument("--ids", default=None, help="要评估的 id（逗号分隔），默认 DRIVE 17,18,19,20")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--min_size", type=int, default=10)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = get_device(args.device)
    results = {}
    ids_used = None

    for ck in args.checkpoints:
        if not os.path.exists(ck):
            print(f"[跳过] 不存在 {ck}")
            continue
        model, cfg, _ = load_checkpoint(ck, device=device)
        cfg.threshold, cfg.min_size, cfg.tta = args.threshold, args.min_size, args.tta
        img_dir, lab_dir = resolve_root(cfg, args.data_dir)
        records = build_records(img_dir, lab_dir)

        ids = [i.strip() for i in args.ids.split(",")] if args.ids else \
              [i.strip() for i in cfg.val_ids.split(",") if i.strip()]
        ds = ImageDataset(cfg, ids, with_label=True, records=records)
        if not len(ds):
            print(f"[跳过] {ck}: ids {ids} 在 {img_dir} 中不存在")
            continue
        ids_used = ds.ids
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        agg, rows = evaluate(model, loader, cfg, device, tta=args.tta)
        results[ck] = (agg, {r["id"]: r for r in rows})
        print(f"[完成] {os.path.basename(ck)}  Dice={agg['dice']:.4f}")

    if not results:
        print("没有可比较的权重")
        return 1

    print("\n" + "=" * 78)
    print(f"{'权重':<34}{'Dice':>8}{'IoU':>8}{'Se':>8}{'Sp':>8}{'Acc':>8}")
    print("-" * 78)
    best = max(results.items(), key=lambda kv: kv[1][0]["dice"])
    for ck, (agg, _) in results.items():
        flag = "  ← 最佳" if ck == best[0] else ""
        print(f"{short_name(ck):<34}{agg['dice']:>8.4f}{agg['iou']:>8.4f}"
              f"{agg['se']:>8.4f}{agg['sp']:>8.4f}{agg['acc']:>8.4f}{flag}")
    print("=" * 78)

    if len(results) > 1 and ids_used:
        print("\n逐图 Dice：")
        head = f"{'id':>12}" + "".join(f"{short_name(c):>26}" for c in results)
        print(head)
        for i in ids_used:
            line = f"{i:>12}"
            for ck, (_, rows) in results.items():
                line += f"{rows[i]['dice']:>26.4f}" if i in rows else f"{'-':>26}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
