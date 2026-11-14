"""测试时亮度/对比度归一化能否救回"困难图"。

动机：尾风险分析显示约 10~17 张低对比度图贡献了 50%~70% 的平方误差；训练期虽有
channel_gain/illumination/clahe 随机增强，但测试时从未做"把每张图拉到同一对比度"的归一化。
本脚本在现有 5 折检查点上做纯测试时 A/B（不改权重），评估：
  base      —— 现有流程
  clahe     —— LAB 的 L 通道 CLAHE
  pct       —— 按亮度分位数线性拉伸到固定目标（消除曝光差异）
  clahe+pct —— 两者串联
分别报告全体、8 张困难图、高分歧图、以及测试集上的表现。

用法: python tools/photo_ttt_probe.py
输出: outputs/analysis/photo_ttt/report.json
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import load_cfg
from src.data import FoveaDataset, load_train_samples, split_indices
from src.predict import Chain

OUT = Path("outputs/analysis/photo_ttt")
OUT.mkdir(parents=True, exist_ok=True)
HARD8 = {"0049", "0072", "0016", "0042", "0070", "0034", "0028", "0024"}
FOLDS = 5

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def t_base(bgr):
    return bgr


def t_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def t_pct(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    lo, hi = np.percentile(l, 2), np.percentile(l, 98)
    if hi - lo < 8:                      # 近乎全黑的图，拉伸会产生爆炸噪声
        return bgr
    lab[:, :, 0] = np.clip((l - lo) * (190.0 / (hi - lo)) + 18.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def t_both(bgr):
    return t_pct(t_clahe(bgr))


TRANSFORMS = {"base": t_base, "clahe": t_clahe, "pct": t_pct, "clahe+pct": t_both}


def comp_rmse(l2):
    l2 = np.asarray(l2, float)
    return float(np.sqrt((l2 ** 2).mean() / 2))


def main():
    cfg = load_cfg("configs/stage2.yaml")
    samples = load_train_samples(cfg)
    fold_of = {}
    for f in range(FOLDS):
        _, va = split_indices(len(samples), FOLDS, cfg["seed"], f)
        for i in va:
            fold_of[samples[i].id] = f
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 高分歧图（来自 tail_risk_probe 的输出）
    disp = {}
    p = Path("outputs/analysis/tail_risk/dispersion_vs_error.csv")
    if p.exists():
        import pandas as pd
        d = pd.read_csv(p, dtype={"image": str})
        disp = dict(zip(d.image.str.zfill(4), d.dispersion.astype(float)))
    high_disp = {k for k, v in disp.items() if v > 20}

    chains, cache = {}, {}
    for f in range(FOLDS):
        chains[f] = Chain(f"outputs/runs/s2_fold{f}", device, cache, prefer="stage2")

    results = {}
    per_image = {}
    for name, fn in TRANSFORMS.items():
        orig = FoveaDataset._load_full
        FoveaDataset._load_full = lambda self, s, _fn=fn: _fn(orig(self, s))
        errs = {}
        try:
            for f in range(FOLDS):
                sub = [s for s in samples if fold_of.get(s.id) == f]
                preds = chains[f].predict(sub)["final"]
                for s in sub:
                    x, y = preds[s.id]
                    errs[s.id] = float(np.hypot(x * s.w - s.x, y * s.h - s.y))
        finally:
            FoveaDataset._load_full = orig
        vals = list(errs.values())
        hard = [v for k, v in errs.items() if k in HARD8]
        hd = [v for k, v in errs.items() if k in high_disp]
        results[name] = dict(
            rmse=comp_rmse(vals), median=float(np.median(vals)),
            p90=float(np.quantile(vals, .9)), hard8_rmse=comp_rmse(hard),
            highdisp_rmse=comp_rmse(hd), n=len(vals))
        per_image[name] = errs
        print(f"{name:10s} RMSE={results[name]['rmse']:6.3f} median={results[name]['median']:6.2f} "
              f"p90={results[name]['p90']:7.2f} | hard8={results[name]['hard8_rmse']:7.2f} "
              f"| 高分歧({len(hd)})={results[name]['highdisp_rmse']:7.2f}")

    # 逐图对比：哪些图被救了、哪些被毁了
    if "base" in per_image:
        rows = []
        for k in sorted(per_image["base"]):
            r = dict(image=k, fold=fold_of.get(k), base=per_image["base"][k],
                     dispersion=disp.get(k, float("nan")))
            for name in TRANSFORMS:
                if name != "base":
                    r[name] = per_image[name][k]
                    r[f"d_{name}"] = per_image[name][k] - per_image["base"][k]
            rows.append(r)
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("base", ascending=False)
        df.to_csv(OUT / "per_image.csv", index=False)
        for name in TRANSFORMS:
            if name == "base":
                continue
            d = df[f"d_{name}"]
            print(f"  {name:10s} 逐图 Δ: 变好 {int((d < -2).sum()):2d} 张 / 变差 {int((d > 2).sum()):2d} 张 "
                  f"/ |Δ|<=2 {int((d.abs() <= 2).sum()):2d} 张")
        print("\n原本最差的 10 张在各变换下的误差:")
        print(df.head(10)[["image", "dispersion", "base", "clahe", "pct", "clahe+pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    json.dump(results, open(OUT / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
