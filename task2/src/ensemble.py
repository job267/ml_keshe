"""融合权重搜索：非负、和为 1，先在 0.05 网格上粗搜，再在邻域内 0.01 细化。

输入是 src/predict.py --save-probs 导出的 npz（probs/labels/groups），
因此每个成员的推理只做一次，权重搜索只是矩阵加权。

示例：
    python -m src.ensemble --probs outputs/probs/r2_r18.npz outputs/probs/r2_convnext.npz \
        outputs/probs/r2_b0.npz --baseline 0 --out reports/round2/ensemble.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .metrics import paired_bootstrap_diff, summarize


def simplex_grid(k: int, step: float = 0.05) -> List[tuple]:
    n = int(round(1.0 / step))
    out = []
    for cuts in itertools.combinations(range(n + k - 1), k - 1):
        prev, counts = -1, []
        for c in cuts:
            counts.append(c - prev - 1)
            prev = c
        counts.append(n + k - 1 - prev - 1)
        if all(c >= 0 for c in counts):
            out.append(tuple(c * step for c in counts))
    return out


def search(probs: Sequence[np.ndarray], labels: np.ndarray, step: float = 0.05,
           min_w: float = 0.0) -> dict:
    k = len(probs)
    best = (-1.0, None)
    for w in simplex_grid(k, step):
        if min_w > 0 and any(x < min_w for x in w):
            continue
        p = sum(wi * pi for wi, pi in zip(w, probs))
        acc = float((p.argmax(1) == labels).mean())
        if acc > best[0]:
            best = (acc, w)
    return {"accuracy": best[0], "weights": list(best[1])}


def refine(probs: Sequence[np.ndarray], labels: np.ndarray, w0: Sequence[float],
           delta: float = 0.10, step: float = 0.01) -> dict:
    k = len(probs)
    best = (-1.0, list(w0))
    grid = np.arange(-delta, delta + 1e-9, step)
    for shift in itertools.product(grid, repeat=k - 1):
        cand = np.array(list(shift) + [-sum(shift)]) + np.asarray(w0)
        if (cand < -1e-9).any():
            continue
        cand = np.clip(cand, 0, None)
        cand = cand / cand.sum()
        p = sum(wi * pi for wi, pi in zip(cand, probs))
        acc = float((p.argmax(1) == labels).mean())
        if acc > best[0]:
            best = (acc, cand.tolist())
    return {"accuracy": best[0], "weights": [round(x, 4) for x in best[1]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="融合权重搜索")
    ap.add_argument("--probs", nargs="+", required=True, help="各成员 npz（predict --save-probs）")
    ap.add_argument("--baseline", type=int, default=None, help="作为对照的成员下标，用于配对 bootstrap")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--min-w", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    zs = [np.load(p) for p in args.probs]
    labels = zs[0]["labels"]
    for z in zs[1:]:
        if not np.array_equal(z["labels"], labels):
            raise ValueError("成员的标签数组不一致，请确认使用同一验证划分")
    probs = [z["probs"].astype(np.float64) for z in zs]
    groups = zs[0]["groups"] if "groups" in zs[0] else None

    coarse = search(probs, labels, args.step, args.min_w)
    fine = refine(probs, labels, coarse["weights"])
    w = fine["weights"]
    fused = sum(wi * pi for wi, pi in zip(w, probs))
    rep = summarize(labels, fused.argmax(1))
    rep.pop("confusion", None)
    rep.update({"members": args.probs, "weights": w, "coarse": coarse, "fine": fine})

    singles = []
    for i, p in enumerate(probs):
        s = summarize(labels, p.argmax(1))
        singles.append({"member": args.probs[i], "accuracy": s["accuracy"], "macro_f1": s["macro_f1"]})
    rep["singles"] = singles

    if args.baseline is not None:
        base = probs[args.baseline].argmax(1)
        rep["bootstrap_vs_baseline"] = paired_bootstrap_diff(labels, base, fused.argmax(1), groups)

    print(json.dumps({k: v for k, v in rep.items() if k != "singles"}, ensure_ascii=False, indent=2))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
