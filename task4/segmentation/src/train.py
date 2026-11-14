"""训练入口。

用法::

    python -m src.train                     # 使用 config.py 中的默认参数
    python -m src.train --epochs 300 --lr 5e-4
    python -m src.train --val_ids ""        # 不留验证集（全部 20 张训练）

产物（默认在 ``outputs/``）::

    best.pt        验证 Dice 最高的权重（推理默认加载它）
    last.pt        最后一轮权重
    train_log.csv  每轮的 loss / lr / 验证指标
    history.json   完整训练曲线
"""

import json
import csv
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .config import parse_config
from .dataset import (ImageDataset, PatchDataset, build_external_records,
                      build_records, discover_ids, filter_records, split_ids)
from .engine import predict_prob
from .losses import build_loss, training_loss
from .evaluation import official_binarize, gt_bilinear
from .models import build_model, load_weights
from .checkpointing import save_resume, load_resume
from .postprocess import postprocess
from .utils import AverageMeter, Timer, count_params, ensure_dir, get_device, set_seed


# --------------------------------------------------------------------------- #
# 学习率调度：线性 warmup + 余弦退火
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, cfg):
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        t = min(1.0, max(0.0, t))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_scaler(enabled: bool):
    try:  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # 兼容旧版本
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- #
# 整图验证
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, cfg, device, tta=False):
    """在整图上评估，返回 (平均指标 dict, 每图指标列表)。

    ``tta=False`` 用于训练期快速验证；离线评测传 ``tta=True`` 与提交配方对齐。
    """
    model.eval()
    rows = []
    for batch in loader:
        image = batch["image"][0]
        fov = batch["fov"][0].numpy().astype(bool)
        gt = batch["label"][0].numpy().astype(bool)
        prob = predict_prob(model, image, cfg, device=device, tta=tta,
                            rgb=batch["rgb"][0].numpy())
        pred = postprocess(prob, fov if cfg.use_fov else None,
                           threshold=cfg.threshold, min_size=cfg.min_size,
                           do_fill_holes=cfg.fill_holes)
        m = M.evaluate_binary(pred, gt)
        m["dice_512_proxy"] = M.dice_coef(official_binarize(pred), gt_bilinear(gt))
        m["id"] = batch["id"][0]
        rows.append(m)
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("dice", "iou", "se", "sp", "acc", "dice_512_proxy")} if rows else {}
    return agg, rows


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv=None):
    cfg = parse_config(argv, description="训练眼底血管分割模型")
    if cfg.selection_metric not in ("dice", "dice_512_proxy") or cfg.checkpoint_policy not in ("best", "last"):
        raise ValueError("不支持的 checkpoint 选择策略")
    if cfg.epochs < 1 or cfg.patches_per_epoch < cfg.batch_size or cfg.val_every < 1 or cfg.checkpoint_every < 1:
        raise ValueError("epochs/val_every 必须正数，patches_per_epoch 必须 >= batch_size")
    set_seed(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)
    ensure_dir(cfg.out_dir)

    # ---------------- 数据 ---------------- #
    all_ids = discover_ids(cfg)
    train_ids, val_ids = split_ids(all_ids, cfg.val_ids)
    requested_val = [i.strip() for i in cfg.val_ids.split(',') if i.strip()]
    if len(val_ids) != len(requested_val) or len(set(requested_val)) != len(requested_val):
        raise ValueError("val_ids 存在重复或未知图像，拒绝静默忽略")
    if not train_ids:
        raise RuntimeError("训练集为空，请检查 --val_ids 设置")

    cache = {}
    all_records = build_records(cfg.train_image_dir, cfg.train_label_dir)
    train_records = filter_records(all_records, train_ids)
    if cfg.extra_train_dirs:
        extra_roots = [d.strip() for d in cfg.extra_train_dirs.split(",") if d.strip()]
        extra = build_external_records(extra_roots)
        rep = max(1, int(cfg.extra_repeat))
        train_records = train_records * rep + extra
    train_set = PatchDataset(cfg, train_records, cache=cache)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    provenance = {"cfg": cfg.to_dict(), "train_ids": train_ids, "val_ids": val_ids,
                  "sampled_record_ids": [r["id"] for r in train_records]}
    with open(os.path.join(cfg.out_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    val_loader = None
    if val_ids:
        val_set = ImageDataset(cfg, val_ids, with_label=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # ---------------- 模型 ---------------- #
    model = build_model(cfg).to(device)
    init_info = None
    if cfg.init_from:
        if not os.path.exists(cfg.init_from):
            raise SystemExit(f"找不到预训练权重: {cfg.init_from}")
        n_loaded, skipped, missing = load_weights(model, cfg.init_from, device)
        init_info = (n_loaded, len(skipped), len(missing))
        if not n_loaded:
            raise SystemExit(f"{cfg.init_from} 与当前模型结构完全不匹配，无法热启动")
    criterion = build_loss(cfg.loss, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = make_scaler(cfg.amp and device.type == "cuda")

    print("=" * 78)
    print(f"设备        : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"模型        : {cfg.model}  参数 {count_params(model)/1e6:.2f} M")
    print(f"输入通道    : {cfg.channel_list}")
    if init_info:
        print(f"热启动      : {cfg.init_from}  载入 {init_info[0]} 个张量"
              f"（跳过 {init_info[1]}，未初始化 {init_info[2]}）")
    print(f"训练/验证   : {len(train_ids)} 张 / {len(val_ids)} 张   {train_ids}")
    if val_ids:
        print(f"验证集      : {val_ids}")
    print(f"Patch       : {cfg.patch_size}  每个 epoch {cfg.patches_per_epoch} 个 patch"
          f"（{cfg.patches_per_epoch // cfg.batch_size} 步）")
    print(f"损失/优化器 : {cfg.loss} / AdamW(lr={cfg.lr}, wd={cfg.weight_decay})")
    print(f"训练轮数    : {cfg.epochs}   AMP: {cfg.amp and device.type == 'cuda'}")
    from .dataset import _resolve_aug
    _aug = _resolve_aug(cfg)
    _on = [f"{k}={v}" for k, v in _aug.items() if v]
    print(f"增强        : " + (" ".join(_on) if _on else "基础（旋转/翻转/缩放/光度）"))
    print("=" * 78)

    # ---------------- 训练循环 ---------------- #
    history = []
    best_dice = -1.0
    best_score = -1.0
    best_epoch = -1
    timer = Timer()

    start_epoch = 0
    if cfg.resume_from:
        resumed = load_resume(cfg.resume_from, 'train', cfg, model, optimizer, scheduler,
                              scaler, train_set, device)
        history, start_epoch = resumed['history'], resumed['epoch']
        best_dice = resumed['stats']['best_dice']
        best_score = resumed['stats']['best_score']
        best_epoch = resumed['stats']['best_epoch']
        print(f'Resuming after epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()

        for batch in train_loader:
            img = batch[0].to(device, non_blocking=True)
            mask = batch[1].to(device, non_blocking=True)
            auxiliary = {k:v.to(device, non_blocking=True) for k,v in batch[2].items()} if len(batch)>2 else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logit = model(img)
                loss = training_loss(logit, mask, criterion, cfg, auxiliary, epoch+1)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            loss_meter.update(loss.item(), img.size(0))

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        rec = {"epoch": epoch + 1, "loss": loss_meter.avg, "lr": lr_now,
               "time": round(time.time() - t0, 2)}

        # ---- 定期整图验证 ----
        do_val = val_loader is not None and (
            (epoch + 1) % cfg.val_every == 0 or epoch == 0 or epoch + 1 == cfg.epochs)
        if do_val:
            agg, rows = evaluate(model, val_loader, cfg, device, tta=cfg.selection_tta)
            rec.update({f"val_{k}": float(v) for k, v in agg.items()})
            score = agg[cfg.selection_metric]
            if score > best_score:
                best_score = score
                best_dice = agg["dice"]
                best_epoch = epoch + 1
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                            "epoch": epoch + 1, "val_dice": best_dice,
                            "selection_score": score, "train_ids": train_ids, "val_ids": val_ids},
                           os.path.join(cfg.out_dir, "best.pt"))
            print(f"[{epoch+1:4d}/{cfg.epochs}] loss={loss_meter.avg:.4f} lr={lr_now:.2e} "
                  f"val dice={agg.get('dice', 0):.4f} iou={agg.get('iou', 0):.4f} "
                  f"se={agg.get('se', 0):.4f} sp={agg.get('sp', 0):.4f} "
                  f"proxy={agg['dice_512_proxy']:.4f} | best {cfg.selection_metric}={best_score:.4f}@{best_epoch} "
                  f"| {time.time()-t0:.1f}s "
                  f"| total {timer}")
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{epoch+1:4d}/{cfg.epochs}] loss={loss_meter.avg:.4f} lr={lr_now:.2e} "
                  f"| {time.time()-t0:.1f}s | total {timer}")

        history.append(rec)
        if (epoch+1) % cfg.checkpoint_every == 0 or epoch+1 == cfg.epochs:
            save_resume(os.path.join(cfg.out_dir, 'resume.pt'), 'train', cfg, model,
                        optimizer, scheduler, scaler, train_set, history,
                        best_dice=best_dice, best_score=best_score, best_epoch=best_epoch)
            with open(os.path.join(cfg.out_dir, 'history.partial.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    # ---------------- 收尾 ---------------- #
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                "epoch": cfg.epochs, "val_dice": history[-1].get("val_dice"),
                "train_ids": train_ids, "val_ids": val_ids},
               os.path.join(cfg.out_dir, "last.pt"))

    # 若从未验证过（--val_ids ""），把最后一轮保存为 best
    if cfg.checkpoint_policy == "last" or not os.path.exists(os.path.join(cfg.out_dir, "best.pt")):
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                    "epoch": cfg.epochs, "train_ids": train_ids, "val_ids": val_ids,
                    "checkpoint_policy": cfg.checkpoint_policy}, os.path.join(cfg.out_dir, "best.pt"))

    with open(os.path.join(cfg.out_dir, "train_log.csv"), "w", encoding="utf-8") as f:
        cols = list(dict.fromkeys(k for r in history for k in r))
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(history)
    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    if cfg.checkpoint_policy == "last":
        print(f"训练完成：按预先固定的训练日程使用最后一轮 epoch={cfg.epochs}")
    elif best_epoch > 0:
        print(f"训练完成：最佳验证 Dice = {best_dice:.4f} (epoch {best_epoch})"
              f" -> {os.path.join(cfg.out_dir, 'best.pt')}")
    else:
        print(f"训练完成：未做验证，权重 -> {os.path.join(cfg.out_dir, 'best.pt')}")
    print(f"总耗时 {timer}")
    print("=" * 78)
    return history


if __name__ == "__main__":
    main()
