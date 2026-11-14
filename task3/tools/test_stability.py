"""测试集预测稳定性自检：看 5 折模型的预测是否一致。

    python tools/test_stability.py --runs outputs/runs/s2_fold0 ... s2_fold4
    python tools/test_stability.py --runs outputs/runs/s2_fold0 ... --split train

用途：LB 只有 20 张图，RMSE 对单张离群极敏感。如果某张图在不同折之间预测差异很大，
说明模型对它没把握，LB 分数很可能被它左右。本工具输出逐图的分歧（px）并标出风险图。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.config import load_cfg
from src.data import load_submission_template, load_test_samples, load_train_samples, split_indices
from src.predict import Chain, ModelRunner
from src.utils import resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config", type=str, default="configs/stage2.yaml")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--out", type=str, default="outputs/test_stability.csv")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_test_samples(cfg) if args.split == "test" else load_train_samples(cfg)

    cache: dict[str, ModelRunner] = {}
    chains = [Chain(r, device, cache) for r in args.runs]
    print(f"{len(chains)} 个模型 | {len(samples)} 张图")

    per_model = []  # list[dict[id] -> (x_px, y_px)]
    for c in chains:
        r = c.predict(samples)["final"]
        dims = {s.id: (s.w, s.h) for s in samples}
        per_model.append({k: (v[0] * dims[k][0], v[1] * dims[k][1]) for k, v in r.items()})

    rows = []
    for s in samples:
        xs = np.array([m[s.id][0] for m in per_model])
        ys = np.array([m[s.id][1] for m in per_model])
        spread = float(np.hypot(xs - xs.mean(), ys - ys.mean()).mean())
        rows.append({
            "image": s.id,
            "x_mean": xs.mean(), "y_mean": ys.mean(),
            "x_std": xs.std(), "y_std": ys.std(),
            "x_range": xs.max() - xs.min(), "y_range": ys.max() - ys.min(),
            "spread_px": spread,
            "gt_x": s.x if s.has_gt else np.nan, "gt_y": s.y if s.has_gt else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("spread_px", ascending=False)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.2f")

    print(f"\n各模型间分歧（越小越稳）：median={df.spread_px.median():.2f}px  "
          f"mean={df.spread_px.mean():.2f}px  max={df.spread_px.max():.2f}px")
    print("\n最不稳定的 5 张：")
    print(df.head(5)[["image", "x_mean", "y_mean", "spread_px", "x_range", "y_range"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    if args.split == "train":
        err = np.hypot(df.gt_x - df.x_mean, df.gt_y - df.y_mean)
        print(f"\n（训练集）逐图误差: median={err.median():.1f}px  "
              f"p90={err.quantile(0.9):.1f}px  max={err.max():.1f}px")
        worst = df.assign(err=err).sort_values("err", ascending=False).head(5)
        print(worst[["image", "err", "spread_px"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
