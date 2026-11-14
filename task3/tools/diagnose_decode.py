"""诊断：对比不同坐标解码方式的效果（soft-argmax / argmax / 回归头 / 平均）。

    python tools/diagnose_decode.py outputs/runs/s1_fold0/best.pt
    python tools/diagnose_decode.py outputs/runs/s2_fold0/best.pt --tta
    python tools/diagnose_decode.py outputs/runs/s1_fold0/best.pt --split train

会给出每种解码在验证集（或训练集）上的赛题 RMSE，并导出两张热力图可视化到
outputs/decode_vis/。常用于判断：热力图峰值是否比"积分期望"更准。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import FoveaDataset, patch_to_global, load_submission_template, load_test_samples, load_train_samples, split_indices
from src.metrics import compute_metrics, format_metrics
from src.models import argmax_coords, build_model, soft_argmax
from src.utils import load_checkpoint, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", choices=["val", "train", "test"], default="val")
    ap.add_argument("--tta", dest="tta", action="store_true", default=None,
                    help="强制开 TTA；默认沿用检查点里的 train.eval_tta（与训练时模型选择口径一致）")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    stage = int(ckpt["stage"])
    fold = int(ckpt["fold"])
    print(f"检查点: {args.ckpt}  stage={stage} fold={fold}  epoch={ckpt.get('epoch')}")
    print(f"训练时验证指标: {ckpt.get('val')}")

    samples = load_train_samples(cfg)
    if args.split == "test":
        samples = load_test_samples(cfg)
    elif args.split == "val":
        _, va = split_indices(len(samples), int(cfg["data"]["n_folds"]), int(cfg["seed"]), fold)
        samples = [samples[i] for i in va]

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tta = bool(cfg["train"].get("eval_tta", False)) if args.tta is None else bool(args.tta)
    print(f"TTA: {tta}")

    centers = None
    if stage == 2:
        # 必须走和 predict.py 完全相同的路径（含 TTA / 相同的中心点来源），
        # 否则得到的 RMSE 会和训练日志对不上，容易误判。
        from src.predict import ModelRunner

        centers = ModelRunner(ckpt["s1_ckpt"], device, tta=tta).predict(samples)

    ds = FoveaDataset(samples, cfg, train=False, task="stage1" if stage == 1 else "stage2", centers=centers)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    names = ["soft", "argmax", "head", "mean"]
    acc = {k: {"p": [], "g": [], "wh": [], "vis": []} for k in names}
    heatmaps = {}

    with torch.no_grad():
        for b in dl:
            x = b["image"].to(device)
            out = model(x)
            prob = torch.sigmoid(out["heatmap"])
            soft = soft_argmax(prob)
            hard = argmax_coords(prob)
            head = out.get("xy_head")
            raw = {"soft": soft, "argmax": hard}
            if head is not None:
                raw["head"] = head
                raw["mean"] = (soft + head) / 2.0
            if tta:
                out2 = model(torch.flip(x, dims=[3]))
                p2 = torch.sigmoid(out2["heatmap"])
                s2 = soft_argmax(p2); s2[:, 0] = 1 - s2[:, 0]
                h2 = argmax_coords(p2); h2[:, 0] = 1 - h2[:, 0]
                raw["soft"] = (raw["soft"] + s2) / 2
                raw["argmax"] = (raw["argmax"] + h2) / 2
                if head is not None:
                    hd2 = out2["xy_head"]; hd2 = torch.stack([1 - hd2[:, 0], hd2[:, 1]], 1)
                    raw["head"] = (raw["head"] + hd2) / 2
                    raw["mean"] = (raw["soft"] + raw["head"]) / 2

            gt = b["xy_global"] if stage == 2 else b["xy"]
            crop = b["crop"].to(device) if stage == 2 else None
            for k, v in raw.items():
                if stage == 2:
                    v = patch_to_global(v, crop)
                acc[k]["p"].append(v.float().cpu().numpy())
                acc[k]["g"].append(gt.numpy())
                acc[k]["wh"].append(b["wh"].numpy())
                acc[k]["vis"].append(b["vis"].numpy())

            # 记录前两张热力图用于可视化
            for i, sid in enumerate(b["id"]):
                if len(heatmaps) < 2 and sid not in heatmaps:
                    heatmaps[sid] = (prob[i, 0].cpu().numpy(), samples[[s.id for s in samples].index(sid)])

    print("-" * 78)
    best_k, best_v = None, 1e9
    for k in names:
        if not acc[k]["p"]:
            continue
        m = compute_metrics(np.concatenate(acc[k]["p"]), np.concatenate(acc[k]["g"]),
                            np.concatenate(acc[k]["wh"]), vis=np.concatenate(acc[k]["vis"]))
        print(f"{k:>7}: {format_metrics(m)}")
        if m["rmse"] < best_v:
            best_k, best_v = k, m["rmse"]
    print(f"最佳解码: {best_k} ({best_v:.2f}px)")

    # 热力图可视化
    if heatmaps:
        out_dir = resolve_path("outputs/decode_vis")
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid, (hm, s) in heatmaps.items():
            im = cv2.imread(str(s.path))
            sc = 1000 / im.shape[1]
            im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)))
            hm_n = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            hm_big = cv2.resize(hm_n, (im.shape[1], im.shape[0]))
            heat = cv2.applyColorMap((hm_big * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis = cv2.addWeighted(im, 0.6, heat, 0.4, 0)
            if s.has_gt:
                gx, gy = int(s.x / s.w * im.shape[1]), int(s.y / s.h * im.shape[0])
                cv2.drawMarker(vis, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 36, 2)
            cv2.imwrite(str(out_dir / f"{sid}_heatmap.jpg"), vis)
        print(f"热力图已保存到 {out_dir}")


if __name__ == "__main__":
    main()
