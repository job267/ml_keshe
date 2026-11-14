"""自检：模型前向、损失反传、MixUp/遮挡、划分确定性、提交写出与校验。

    python tools\\quick_check.py            # 全部检查（CPU 即可）
    python tools\\quick_check.py --data     # 额外用真实数据确认划分张数与提交行数
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import load_cfg  # noqa: E402
from src.data import RandomErasing, build_transforms, mixup_batch, soft_targets  # noqa: E402
from src.losses import Criterion, gce, soft_ce  # noqa: E402
from src.models import build_model, predict_proba  # noqa: E402
from src.predict import validate_submission, write_submission  # noqa: E402

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="用真实数据检查划分与提交")
    args = ap.parse_args(argv)

    print("[1] 模型前向（48px 单通道 / 160px 三通道）")
    for name, img, ch, size in [("ferresnet", "ferresnet", 1, 48),
                                ("ferresnet+se+cbam", "ferresnet", 1, 48),
                                ("resnet18", "resnet18", 3, 160),
                                ("convnext_tiny", "convnext_tiny", 3, 160),
                                ("efficientnet_b0", "efficientnet_b0", 3, 160)]:
        cfg = load_cfg(None, [f"model.name={img}", f"data.img_size={size}", f"model.in_ch={ch}",
                              f"model.pretrained=false"])
        if name.endswith("se+cbam"):
            cfg["model"].update({"se": True, "cbam": True})
        try:
            model, info = build_model(cfg)
            x = torch.randn(2, ch, size, size)
            y = model(x)
            check(f"{name} 输出 {tuple(y.shape)}", tuple(y.shape) == (2, 6),
                  f"{info['params'] / 1e6:.2f}M 参数")
        except Exception as e:  # timm 缺失时跳过而不是整体失败
            check(f"{name} 构建", False, f"{type(e).__name__}: {e}")

    print("[2] 损失与梯度")
    logits = torch.randn(8, 6, requires_grad=True)
    target = soft_targets(torch.randint(0, 6, (8,)), 0.05)
    loss = soft_ce(logits, target)
    loss.backward()
    check("soft_ce 反传得到有限梯度", torch.isfinite(logits.grad).all().item())
    l2 = gce(logits, target, q=0.7)
    check("gce(q=0.7) 数值有限", bool(torch.isfinite(l2)))
    # 均匀预测处 GCE 与 CE 的**梯度**尺度应一致（6^q 缩放的目的；损失值本身不同）
    unif = torch.zeros(4, 6, requires_grad=True)
    onehot = torch.full((4, 6), 1 / 6)
    ce_v = soft_ce(unif, onehot)
    ce_v.backward()
    g_ce = unif.grad.clone()
    unif.grad = None
    gce_v = gce(unif, onehot, q=0.7)
    gce_v.backward()
    g_gce = unif.grad.clone()
    check("均匀预测处 CE 与 GCE 梯度一致", torch.allclose(g_ce, g_gce, atol=1e-6),
          f"|Δ|={float((g_ce - g_gce).abs().max()):.2e}")

    print("[3] MixUp / 随机遮挡 / TTA")
    x = torch.rand(4, 1, 48, 48)
    y = torch.randint(0, 6, (4,))
    xm, ya, yb, lam = mixup_batch(x, y, 0.4)
    check("mixup 保持形状与 lambda 范围", xm.shape == x.shape and 0.5 <= lam <= 1.0)
    er = RandomErasing(prob=1.0, area=(0.02, 0.12), ratio=(0.5, 2.0))
    xe = er(torch.ones(1, 48, 48))
    erased = float((xe == 0).float().mean())
    check("随机遮挡面积在 2%–12%", 0.015 <= erased <= 0.13, f"{erased:.3%}")
    cfg = load_cfg(None, ["model.name=ferresnet", "model.pretrained=false"])
    model, _ = build_model(cfg)
    p = predict_proba(model, torch.randn(3, 1, 48, 48), flip_tta=True)
    check("TTA 概率和为 1", torch.allclose(p.sum(1), torch.ones(3), atol=1e-5))

    print("[4] 提交写出与校验")
    ids = [f"{i:05d}" for i in range(7065)]
    prob = np.random.rand(7065, 6)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub.csv"
        rep = write_submission(ids, prob, out)
        check("7065 行提交通过校验", rep["ok"], f"rows={rep['n_rows']}")

    if args.data:
        print("[5] 真实数据：划分确定性与张数")
        from src.data import prepare_data
        s1, g1, sp1, _w, _m = prepare_data(load_cfg(None))
        s2, g2, sp2, _w2, _m2 = prepare_data(load_cfg(None))
        check("两次划分完全一致", sp1 == sp2)
        check("总张数 28275", len(s1) == 28275, str(len(s1)))
        check("验证集 2848 / 留出 2836",
              len(sp1["val"]) == 2848 and len(sp1["holdout"]) == 2836,
              f"{len(sp1['val'])}/{len(sp1['holdout'])}")
        check("组不跨划分", not (set(g1[i] for i in sp1["train"]) & set(g1[i] for i in sp1["val"])))

    print()
    if FAIL:
        print(f"失败 {len(FAIL)} 项：" + ", ".join(FAIL))
        return 1
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
