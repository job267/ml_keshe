"""多模型概率平均（集成）推理。

集成收益必须通过未见过目标图的 OOF 评估验证；支持预先确定的粗粒度权重。

用法::

    # 1) 在验证集上看集成是否更好（有标签时自动算 Dice）
    python scripts/ensemble.py --checkpoints outputs/baseline_scratch.pt,outputs/best.pt,outputs_joint/best.pt \
        --eval --ids 17,18,19,20

    # 2) 生成集成掩膜用于提交
    python scripts/ensemble.py --checkpoints a.pt,b.pt,c.pt --data_dir test --pred_dir image
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import metrics as M  # noqa: E402
from src.dataset import ImageDataset, build_records  # noqa: E402
from src.engine import predict_prob  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.utils import Timer, ensure_dir, get_device, imwrite  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="多模型概率平均集成")
    p.add_argument("--checkpoints", required=True, help="逗号分隔的权重列表")
    p.add_argument("--weights", default="", help="与 checkpoints 一一对应的非负权重；默认等权")
    p.add_argument("--preprocess_mode", choices=["checkpoint", "legacy", "rgb_first"], default="checkpoint")
    p.add_argument("--data_dir", default="test", help="含 image/（可选 label/）的目录，默认 test")
    p.add_argument("--ids", default="", help="只处理指定 id，逗号分隔，留空=全部")
    p.add_argument("--pred_dir", default="", help="掩膜输出目录，留空则只算分不写盘")
    p.add_argument("--eval", action="store_true", help="有 label 时计算 Dice 等指标")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--min_size", type=int, default=10)
    p.add_argument("--tta", action="store_true", default=True)
    p.add_argument("--no-tta", dest="tta", action="store_false")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = get_device(args.device)
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    weights = np.array([float(x) for x in args.weights.split(',')] if args.weights else [1.] * len(ckpts))
    if len(weights) != len(ckpts) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights 必须与权重文件数一致、有限非负且总和 >0")
    weights /= weights.sum()
    models, model_cfgs, cfg = [], [], None
    for ck in ckpts:
        if not os.path.exists(ck):
            raise SystemExit(f"找不到权重: {ck}")
        m, c, _ = load_checkpoint(ck, device=device)
        if args.preprocess_mode != "checkpoint":
            c.preprocess_mode = args.preprocess_mode
        models.append(m)
        model_cfgs.append(c)
        cfg = cfg or c
    cfg.threshold, cfg.min_size, cfg.tta = args.threshold, args.min_size, args.tta

    def _tag(path: str) -> str:
        """outputs_cv_sa/fold2/best.pt -> outputs_cv_sa/fold2；outputs_all/best.pt -> outputs_all"""
        parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
        if len(parts) >= 2 and parts[-1].startswith("best"):
            parent = parts[-2]
            if parent.startswith("fold"):
                return f"{parts[-3]}/{parent}"
            return parent
        return os.path.basename(path)

    print(f"集成 {len(models)} 个模型: " + ", ".join(_tag(c) for c in ckpts))

    img_dir = os.path.join(ROOT, args.data_dir, "image")
    lab_dir = os.path.join(ROOT, args.data_dir, "label")
    has_label = args.eval and os.path.isdir(lab_dir)
    records = build_records(img_dir, lab_dir if has_label else img_dir)
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] or [r["id"] for r in records]
    ds = ImageDataset(cfg, ids, with_label=has_label, records=records)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    if args.pred_dir:
        ensure_dir(os.path.join(ROOT, args.pred_dir))

    timer = Timer()
    rows = []
    # 每个权重自带训练配置（模型结构、放大倍率 cfg.scale 可能不同），
    # 所以前向时必须用**各自的 cfg**，只有后处理参数统一用第一个模型的。
    for c in model_cfgs:
        c.threshold, c.min_size, c.tta = args.threshold, args.min_size, args.tta
        c.use_fov, c.fill_holes = cfg.use_fov, cfg.fill_holes

    for batch in loader:
        image = batch["image"][0]
        fov = batch["fov"][0].numpy().astype(bool)
        prob = None
        with torch.no_grad():
            for m, mc, weight in zip(models, model_cfgs, weights):
                q = weight * np.asarray(predict_prob(m, image, mc, device=device, tta=mc.tta,
                                                     rgb=batch["rgb"][0].numpy()))
                prob = q if prob is None else prob + q
        mask = postprocess(prob, fov if cfg.use_fov else None,
                           threshold=cfg.threshold, min_size=cfg.min_size,
                           do_fill_holes=cfg.fill_holes)
        i = batch["id"][0]
        if args.pred_dir:
            out_mask = (~mask) if cfg.invert_output else mask
            imwrite(os.path.join(ROOT, args.pred_dir, f"{i}.png"),
                    (out_mask.astype(np.uint8) * 255))
        if has_label:
            gt = batch["label"][0].numpy().astype(bool)
            m = M.evaluate_binary(mask, gt)
            m["id"] = i
            rows.append(m)
            print(f"  {i:>16s}  Dice={m['dice']:.4f}")

    if rows:
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ("dice", "iou", "se", "sp", "acc")}
        print(f"\n集成结果 ({len(models)} 模型, {len(rows)} 张): "
              f"Dice={agg['dice']:.4f}  IoU={agg['iou']:.4f}  Se={agg['se']:.4f}  "
              f"Sp={agg['sp']:.4f}  Acc={agg['acc']:.4f}")
    print(f"耗时 {timer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
