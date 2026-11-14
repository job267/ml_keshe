"""把 ``scripts/oof_ensemble.py --out_csv`` 的结果画成逐图对比柱状图。

用法::

    python scripts/plot_oof.py --csv oof_compare.csv --out figs/cv_oof_dice.png
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description="OOF 逐图对比图")
    ap.add_argument("--csv", default="oof_compare.csv")
    ap.add_argument("--out", default="figs/cv_oof_dice.png")
    ap.add_argument("--title", default="5 折交叉验证：out-of-fold Dice（同划分、不同架构）")
    ap.add_argument("--ylim", default="0.6,0.9")
    ap.add_argument("--cols", default="",
                    help="逗号分隔，只画这些列（默认画 CSV 里全部非 id 列）")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c != "id"]
        rows = list(reader)

    if args.cols:
        want = [c.strip() for c in args.cols.split(",") if c.strip()]
        missing = [c for c in want if c not in cols]
        if missing:
            raise SystemExit(f"CSV 里没有这些列: {missing}；可用列: {cols}")
        cols = want

    rows.sort(key=lambda r: int(r["id"]))
    ids = [r["id"] for r in rows]
    data = {c: np.array([float(r[c]) for r in rows]) for c in cols}

    colors = ["#7aa6c2", "#d1495b", "#2e8b74", "#8a6bbe", "#e08b2f", "#555555"]
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=140)
    x = np.arange(len(ids))
    w = 0.8 / max(len(cols), 1)
    for k, c in enumerate(cols):
        v = data[c]
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, v, w * 0.92,
               label=f"{c}（均值 {v.mean():.4f}）", color=colors[k % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_xlabel("DRIVE 图像 id")
    ax.set_ylabel("Dice")
    lo, hi = (float(s) for s in args.ylim.split(","))
    ax.set_ylim(lo, hi)
    for c in cols:
        ax.axhline(float(data[c].mean()), color=colors[cols.index(c) % len(colors)],
                   lw=0.9, ls="--", alpha=0.55)
    ax.set_title(args.title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(len(cols), 3),
              fontsize=8.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out)
    print(f"已写 {args.out}")
    for c in cols:
        print(f"  {c:<24} {data[c].mean():.4f} ± {data[c].std():.4f}")


if __name__ == "__main__":
    main()
