"""数据体检 + 模型冒烟测试（不需要训练即可运行）。

    python tools/quick_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_cfg
from src.data import (
    FoveaDataset,
    load_submission_template,
    load_test_samples,
    load_train_samples,
    patch_to_global,
    split_indices,
)
from src.losses import FoveaLoss
from src.metrics import compute_metrics, format_metrics
from src.models import build_model, predict_coords, soft_argmax


def main() -> None:
    cfg = load_cfg("configs/stage1.yaml")
    train = load_train_samples(cfg)
    test = load_test_samples(cfg)
    tmpl = load_submission_template(cfg)
    print("=" * 78)
    print(f"训练集 {len(train)} 张 | 测试集 {len(test)} 张 | 提交模板 {len(tmpl['keys'])} 行")
    whs = [(s.w, s.h) for s in train]
    print(f"图像尺寸: {sorted(set(whs))}")
    xs = np.array([s.x for s in train]); ys = np.array([s.y for s in train])
    ws = np.array([s.w for s in train], dtype=float); hs = np.array([s.h for s in train], dtype=float)
    print(f"中心凹像素范围: x [{xs.min():.0f}, {xs.max():.0f}]  y [{ys.min():.0f}, {ys.max():.0f}]")
    print(f"归一化均值: x={(xs/ws).mean():.4f} y={(ys/hs).mean():.4f} | "
          f"标准差: x={(xs/ws).std():.4f} y={(ys/hs).std():.4f}")
    print(f"不可见样本(坐标为 0,0): {sum(s.vis == 0 for s in train)} 张")
    print(f"测试集图像: {[s.id for s in test]}")
    assert [s.id for s in test] == sorted({sid for sid, _ in tmpl["rows"]}), "测试集顺序与提交模板不一致"

    # ---- 常数基线（全部预测训练集均值），作为 RMSE 的参照 ----
    mx, my = (xs / ws).mean(), (ys / hs).mean()
    P = np.tile([mx, my], (len(train), 1))
    G = np.stack([xs / ws, ys / hs], 1)
    WH = np.stack([ws, hs], 1)
    base = compute_metrics(P, G, WH, vis=np.array([s.vis for s in train]))
    print("-" * 78)
    print("常数基线（预测均值）:", format_metrics(base))

    # ---- stage1 前向 / 损失 / 反传 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = split_indices(len(train), cfg["data"]["n_folds"], cfg["seed"], 0)
    ds = FoveaDataset([train[i] for i in va], cfg, train=False, task="stage1")
    b = next(iter(DataLoader(ds, batch_size=4)))
    model = build_model(cfg, device)
    x = b["image"].to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    print("-" * 78)
    print(f"stage1 前向: 输入 {tuple(x.shape)} -> 热力图 {tuple(out['heatmap'].shape)} 用时 {time.time()-t0:.2f}s")
    print(f"  预测(归一化) {np.round(out['xy_soft'][0].tolist(), 4)} vs GT {b['xy'][0].tolist()}")
    assert out["heatmap"].shape[0] == x.shape[0] == 4

    model.train()
    loss_fn = FoveaLoss(cfg)
    parts = loss_fn(model(x), b["xy"].to(device), b["vis"].to(device))
    parts["total"].backward()
    print(f"  损失 {parts['total'].item():.4f} (hm {parts['hm'].item():.4f}, soft {parts['soft'].item():.4f}) 反传 OK")

    # ---- 热力图标签 / soft-argmax 一致性 ----
    from src.losses import make_heatmap_target

    xy = b["xy"][:1].to(device)
    tgt = make_heatmap_target(xy, 64, 96, sigma=1.5)
    dec = soft_argmax(tgt)
    err = (dec - xy).abs().max().item()
    print(f"  高斯标签 -> soft-argmax 还原误差 = {err:.2e}（应接近 0）")
    assert err < 2e-3, "soft-argmax 与热力图标签的坐标约定不一致"

    # ---- stage2 裁切 / 坐标换算一致性（以 GT 为中心，train=False 保证无增强干扰）----
    cfg2 = load_cfg("configs/stage2.yaml")
    centers = {train[i].id: (train[i].x / train[i].w, train[i].y / train[i].h) for i in va}
    ds2 = FoveaDataset([train[i] for i in va], cfg2, train=False, task="stage2", centers=centers)
    b2 = next(iter(DataLoader(ds2, batch_size=4)))
    glob = patch_to_global(b2["xy"], b2["crop"])
    err2 = (glob - b2["xy_global"]).abs().max().item()
    print("-" * 78)
    print(f"stage2: patch {tuple(b2['image'].shape)}  crop(cx,cy,size) 示例 {np.round(b2['crop'][0].tolist(), 4)}")
    print(f"  patch 内坐标 -> 全图坐标 与 GT 最大偏差 = {err2:.2e}（应为 0）")
    assert err2 < 1e-5, "stage2 坐标换算不一致"

    print("=" * 78)
    print("全部检查通过")


if __name__ == "__main__":
    main()
