"""尾风险诊断：模型间分歧(dispersion)能否预测单图误差，以及门控聚合能否降 RMSE。

三个成员都是同一划分(seed 42)下的袋外预测：
  v5  baseline_s42      (旧阶段二最优)
  v12 baseline_s43      (同配置换训练种子)
  v12 mass_kl_s43       (换损失 + 换种子)
因此每张图的三个预测都来自"没见过该图的模型"，可以当作 3 成员 OOF 集成来评估。

注意口径：oof.l2 是欧氏像素距离，竞赛 RMSE = sqrt(mean(l2^2) / 2)。
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERS = {
    "v5_base42": "outputs/experiments/v5_20261024/baseline_s42/oof.csv",
    "v12_base43": "outputs/experiments/v12_seed_replication/baseline_s43/oof.csv",
    "v12_kl43": "outputs/experiments/v12_seed_replication/mass_kl_s43/oof.csv",
}
HARD8 = {"0049", "0072", "0016", "0042", "0070", "0034", "0028", "0024"}
OUT = Path("outputs/analysis/tail_risk")
OUT.mkdir(parents=True, exist_ok=True)


def comp_rmse(l2):
    l2 = np.asarray(l2, dtype=float)
    return float(np.sqrt((l2 ** 2).mean() / 2))


def load():
    frames = {}
    for k, p in MEMBERS.items():
        d = pd.read_csv(p, dtype={"image": str})
        d["image"] = d.image.str.zfill(4)
        frames[k] = d.set_index("image")[["fold", "x", "y", "gt_x", "gt_y", "l2"]].astype(float)
    idx = frames[MEMBERS and "v5_base42"].index
    for k, d in frames.items():
        idx = idx.intersection(d.index)
    idx = sorted(idx)
    return frames, idx


def main():
    frames, idx = load()
    names = list(MEMBERS)
    P = np.stack([[frames[n].loc[i, "x"], frames[n].loc[i, "y"]] for i in idx for n in names]).reshape(
        len(idx), len(names), 2)
    gt = np.stack([[frames[names[0]].loc[i, "gt_x"], frames[names[0]].loc[i, "gt_y"]] for i in idx])
    l2 = {n: frames[n].loc[idx, "l2"].values for n in names}

    print(f"成员单模性能（竞赛口径 RMSE）: " + "  ".join(f"{n}={comp_rmse(l2[n]):.2f}" for n in names))
    for n in names:
        e = l2[n]
        print(f"  {n:11s} median={np.median(e):6.2f}  p90={np.quantile(e, .9):7.2f}  max={e.max():7.2f}")

    # ---- 分歧度 ----
    disp = np.zeros(len(idx))
    pairw = []
    for a, b in combinations(range(len(names)), 2):
        d = np.hypot(*(P[:, a] - P[:, b]).T)
        pairw.append(d)
        disp = np.maximum(disp, d)
    pairw = np.stack(pairw).max(0)

    mean_l2 = np.stack([l2[n] for n in names]).mean(0)
    both_hard = np.array([i in HARD8 for i in idx])
    r_all = np.corrcoef(disp, mean_l2)[0, 1]
    r_easy = np.corrcoef(disp[~both_hard], mean_l2[~both_hard])[0, 1]
    sp = pd.Series(disp).corr(pd.Series(mean_l2), method="spearman")
    print(f"\n分歧度(max pairwise) vs 单图误差: pearson(全体 n={len(idx)})={r_all:.3f}  "
          f"pearson(非8张困难图 n={(~both_hard).sum()})={r_easy:.3f}  spearman={sp:.3f}")

    df = pd.DataFrame(dict(image=idx, fold=frames[names[0]].loc[idx, "fold"].values,
                           gt_x=gt[:, 0], gt_y=gt[:, 1], dispersion=disp,
                           l2_v5=l2["v5_base42"], l2_b43=l2["v12_base43"], l2_kl=l2["v12_kl43"],
                           l2_mean=mean_l2))
    df["is_hard8"] = both_hard
    df = df.sort_values("dispersion", ascending=False)
    df.to_csv(OUT / "dispersion_vs_error.csv", index=False)

    print("\n分歧最大的 12 张（H=困难图）:")
    print(f"{'img':6s}{'H':>3s}{'disp':>8s}{'v5':>8s}{'b43':>8s}{'kl':>8s}")
    for _, r in df.head(12).iterrows():
        print(f"{r.image:6s}{'H' if r.is_hard8 else '':>3s}{r.dispersion:8.1f}"
              f"{r.l2_v5:8.1f}{r.l2_b43:8.1f}{r.l2_kl:8.1f}")

    # 分桶：分歧度分位与误差
    print("\n按分歧度分桶(全体 80 张):")
    qs = df.dispersion.quantile([0, .25, .5, .75, .9, 1.0]).values
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (df.dispersion >= lo) & (df.dispersion <= hi)
        if m.sum():
            print(f"  [{lo:7.1f},{hi:7.1f}] n={m.sum():2d} 平均单图误差={df.l2_mean[m].mean():7.1f}"
                  f"  RMSE贡献前占位")

    # ---- 聚合策略 ----
    print("\n=== 聚合策略（同一 OOF 划分，每张图 3 个候选） ===")
    res = {}
    for n in names:
        res[n] = comp_rmse(l2[n])
    res["mean3"] = comp_rmse(np.hypot(*(P.mean(1) - gt).T))
    med = np.median(P, axis=1)
    res["median3"] = comp_rmse(np.hypot(*(med - gt).T))
    # 去掉离中位数最远的 1 个，剩 2 个取平均
    d_to_med = np.stack([np.hypot(*(P[:, k] - med).T) for k in range(len(names))], 1)
    keep = np.argsort(d_to_med, 1)[:, :2]
    trimmed = np.stack([P[i, keep[i]].mean(0) for i in range(len(idx))])
    res["trim2_of_3"] = comp_rmse(np.hypot(*(trimmed - gt).T))

    # 门控：分歧小用 mean3，分歧大用 median3
    for tau in [5, 10, 15, 20, 25, 30, 40]:
        sel = np.where(disp[:, None] > tau, med, P.mean(1))
        e = np.hypot(*(sel - gt).T)
        res[f"gate(tau={tau})"] = comp_rmse(e)
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"  {k:14s} RMSE={v:6.3f}")

    # ---- 关键：若把"分歧最大的一小部分图"判为不可信，收益多少 ----
    print("\n=== 分歧度作为\"不可信标记\"的筛查能力 ===")
    for th in [10, 15, 20, 25, 30]:
        m = disp > th
        if m.sum():
            print(f"  分歧>{th:2d}px: n={m.sum():2d}  这些图的平均误差={mean_l2[m].mean():6.1f} "
                  f"(其余={mean_l2[~m].mean():5.1f})  占全部平方误差="
                  f"{np.sum(mean_l2[m]**2)/np.sum(mean_l2**2):.1%}")

    json.dump({"members": {n: comp_rmse(l2[n]) for n in names},
               "strategies": res,
               "corr_all": float(r_all), "corr_easy": float(r_easy), "spearman": float(sp)},
              open(OUT / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
