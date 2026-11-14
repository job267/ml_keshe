"""跨实验的 OOF 对比 + **诚实**的集成收益估计。

为什么要单独写这个脚本（重要，涉及一个评测口径错误）
------------------------------------------------------
``src/cv.py`` 早期版本把"每张图用 4 个模型平均"的结果当成集成 OOF 公布。
但 5 折里第 k 折训练时用掉的是"除 val_ids[k] 之外的 16 张" —— 也就是说，
对某张图 i，**除它所在那一折以外的 4 个模型全都训练过它**。
那 4 个模型给出的预测是"在训练集上"的预测，分数自然虚高
（当时的数字 0.8308，比单模型 OOF 0.8124 高 0.018，这个差额主要是记忆效应）。

正确做法：一张图只能用**训练时没见过它**的模型来做集成。
单次 5 折里每张图恰好只有一个这样的模型，所以
"K 模型集成"的真实收益必须跨 **多个 run**（不同架构 / 不同划分种子）才能估计：
同一个划分下的 N 个 run，对每张图都能各贡献 1 个好模型，平均后就是
一个诚实的 N 模型集成估计。

用法::

    # 对比 3 个 run，并给出诚实集成估计
    python scripts/oof_ensemble.py --runs outputs_cv,outputs_cv_att,outputs_cv_full

    # 顺便量化"泄漏版"集成数字到底虚高多少（写文档用）
    python scripts/oof_ensemble.py --runs outputs_cv,outputs_cv_att --show_leak
"""

import argparse
import json
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ImageDataset, build_records  # noqa: E402
from src.engine import predict_prob  # noqa: E402
from src.metrics import evaluate_binary  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.utils import Timer, fov_mask, get_device, imread_gray, imread_rgb  # noqa: E402


def load_split(run_dir):
    path = os.path.join(run_dir, "cv_summary.json")
    if not os.path.exists(path):
        raise RuntimeError(f"缺少 {path}（先跑 src.cv）")
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    split = summary.get("split") or {}
    if not split:
        raise RuntimeError(f"{path} 里没有 split 字段")
    return split


def predict_ids(model, cfg, ids, records, device, tta):
    ds = ImageDataset(cfg, ids, with_label=False, records=records)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    out = {}
    for batch in loader:
        out[batch["id"][0]] = predict_prob(model, batch["image"][0], cfg,
                                           device=device, tta=tta, rgb=batch["rgb"][0].numpy())
    return out


def _load_fold(run_dir, fold, device, args):
    ck = os.path.join(run_dir, fold, "best.pt")
    if not os.path.exists(ck):
        return None, None
    model, cfg, _ = load_checkpoint(ck, device=device)
    cfg.threshold, cfg.min_size = args.threshold, args.min_size
    cfg.use_fov, cfg.fill_holes = args.use_fov, args.fill_holes
    return model, cfg


def oof_probs(run_dir, records, device, args):
    """诚实 OOF：每折只预测自己留出的那几张图。"""
    split = load_split(run_dir)
    probs = {}
    for fold, val_ids in sorted(split.items()):
        model, cfg = _load_fold(run_dir, fold, device, args)
        if model is None:
            print(f"  [跳过] {run_dir}/{fold}/best.pt 不存在")
            continue
        probs.update(predict_ids(model, cfg, list(val_ids), records, device, args.tta))
    return probs


def leaky_probs(run_dir, records, device, args):
    """**错误**口径（复现旧版 cv.py 的做法）：用训练过这张图的模型平均。"""
    split = load_split(run_dir)
    all_ids = [r["id"] for r in records]
    acc = {}
    for fold, val_ids in sorted(split.items()):
        model, cfg = _load_fold(run_dir, fold, device, args)
        if model is None:
            continue
        held = set(val_ids)
        ids = [i for i in all_ids if i not in held]
        for i, p in predict_ids(model, cfg, ids, records, device, args.tta).items():
            acc.setdefault(i, []).append(p)
    return {i: np.mean(v, axis=0) for i, v in acc.items()}


def score(probs, records, args):
    by_id = {r["id"]: r for r in records}
    rows = []
    for i, p in probs.items():
        r = by_id[i]
        img = imread_rgb(r["image"])
        gt = imread_gray(r["label"]) > 127
        fov = fov_mask(img) if args.use_fov else None
        mask = postprocess(p, fov, threshold=args.threshold,
                           min_size=args.min_size, do_fill_holes=args.fill_holes)
        m = evaluate_binary(mask, gt)
        m["id"] = i
        rows.append(m)
    rows.sort(key=lambda r: (not r["id"].isdigit(), int(r["id"]) if r["id"].isdigit() else 0, r["id"]))
    return rows


def brief(rows):
    d = [r["dice"] for r in rows]
    return (float(np.mean(d)), float(np.std(d)),
            float(np.mean([r["se"] for r in rows])),
            float(np.mean([r["sp"] for r in rows])),
            float(np.mean([r["iou"] for r in rows])))


def main(argv=None):
    ap = argparse.ArgumentParser(description="跨 run 的 OOF 对比 + 诚实集成估计")
    ap.add_argument("--runs", default="outputs_cv,outputs_cv_att,outputs_cv_full",
                    help="逗号分隔的 cv 输出目录（每个含 cv_summary.json 与 fold*/best.pt）")
    ap.add_argument("--train_image_dir", default="train/image")
    ap.add_argument("--train_label_dir", default="train/label")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min_size", type=int, default=10)
    ap.add_argument("--use_fov", default=True, type=lambda s: str(s).lower() in ("1", "true", "yes", "y"))
    ap.add_argument("--fill_holes", default=False, type=lambda s: str(s).lower() in ("1", "true", "yes", "y"))
    ap.add_argument("--tta", default=True, type=lambda s: str(s).lower() in ("1", "true", "yes", "y"))
    ap.add_argument("--show_leak", action="store_true", help="额外算一遍泄漏口径，量化偏差")
    ap.add_argument("--out_csv", default="", help="把逐图结果写到这个 csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    records = build_records(args.train_image_dir, args.train_label_dir)
    timer = Timer()

    print("=" * 82)
    print(f"OOF 对比：{len(runs)} 个 run，每个 run 的每张图只由「没见过它」的折模型预测")
    print("=" * 82)

    per_run, per_run_rows = {}, {}
    for run in runs:
        t0 = Timer()
        probs = oof_probs(run, records, device, args)
        rows = score(probs, records, args)
        per_run[run] = probs
        per_run_rows[run] = rows
        m, s, se, sp, iou = brief(rows)
        print(f"  {run:<22} Dice={m:.4f} ± {s:.4f}  IoU={iou:.4f}  Se={se:.4f}  Sp={sp:.4f}"
              f"  (n={len(rows)}, {t0})")

    # ---------------- 诚实集成：跨 run 平均（每张图都只用没见过它的模型） ---------------- #
    print("-" * 82)
    rows_by_run = per_run_rows
    print("诚实集成（对每张图，把各 run 中「没见过它」的那个模型做概率平均）")
    ens_levels = []
    for k in range(2, len(runs) + 1):
        subset = runs[:k]
        avg = {}
        for run in subset:
            for i, p in per_run[run].items():
                avg[i] = avg.get(i, 0.0) + p
        n = {i: len([r for r in subset if i in per_run[r]]) for i in avg}
        avg = {i: p / max(n[i], 1) for i, p in avg.items()}
        rows = score(avg, records, args)
        m, s, se, sp, iou = brief(rows)
        ens_levels.append((k, rows, (m, s, se, sp, iou)))
        print(f"  {'+'.join(os.path.basename(r) for r in subset):<46} "
              f"Dice={m:.4f} ± {s:.4f}  Se={se:.4f}  Sp={sp:.4f}")

    # ---------------- 配对比较（同一批图，比均值更灵敏） ---------------- #
    if len(runs) >= 2:
        print("-" * 82)
        print("配对比较（逐图差值，负值=后者更差）")
        base = {r["id"]: r["dice"] for r in rows_by_run[runs[0]]}
        for run in runs[1:]:
            cur = {r["id"]: r["dice"] for r in rows_by_run[run]}
            ids = sorted(base, key=lambda x: (not x.isdigit(), x))
            d = np.array([cur[i] - base[i] for i in ids])
            win = int((d > 0).sum())
            print(f"  {os.path.basename(run):<22} 均值差 {d.mean():+.4f}  "
                  f"中位数 {np.median(d):+.4f}  提升图数 {win}/{len(d)}  "
                  f"配对标准差 {d.std(ddof=1):.4f}  t≈{d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}")

    # ---------------- 泄漏口径（错误做法）的量化 ---------------- #
    if args.show_leak:
        print("-" * 82)
        print("[对照] 旧口径（用训练过该图的 4 个模型平均）—— 数字虚高，仅供参考")
        for run in runs:
            rows = score(leaky_probs(run, records, device, args), records, args)
            m, s, se, sp, iou = brief(rows)
            m0, *_ = brief(rows_by_run[run])
            print(f"  {os.path.basename(run):<22} 泄漏版 Dice={m:.4f}  诚实版={m0:.4f}  "
                  f"虚高 {m - m0:+.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8") as f:
            head = ",".join(["id"] + [os.path.basename(r) for r in runs] +
                            [f"ens{k}" for k, _, _ in ens_levels])
            f.write(head + "\n")
            ids = sorted(rows_by_run[runs[0]], key=lambda r: (not r["id"].isdigit(), r["id"]))
            id_list = [r["id"] for r in ids]
            table = {run: {r["id"]: r["dice"] for r in per_run_rows[run]} for run in runs}
            ens_tab = [({r["id"]: r["dice"] for r in rows}) for _, rows, _ in ens_levels]
            for i in id_list:
                vals = [f"{table[run][i]:.6f}" for run in runs] + \
                       [f"{t[i]:.6f}" for t in ens_tab]
                f.write(",".join([i] + vals) + "\n")
        print(f"\n逐图结果 -> {args.out_csv}")

    print(f"\n总耗时 {timer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
