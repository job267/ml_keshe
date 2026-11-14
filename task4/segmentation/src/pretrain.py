"""外部数据预训练：在 STARE / CHASE_DB1 / HRF 上先学"血管长什么样"，再到 DRIVE 上微调。

为什么这样做：DRIVE 只有 20 张（16 训练 / 4 验证），从头训很容易过拟合；
外部数据把训练图像从 16 张扩到 109 张（16 + 93），且覆盖不同相机、分辨率、
病理（HRF 含糖网与青光眼），能显著改善细血管的召回。

流程::

    python scripts/download_external.py      # 下载到 data_external/_raw
    python scripts/prepare_external.py       # 整理成 data_external/<name>/{image,label}
    python -m src.pretrain                   # 预训练 -> outputs/pretrained.pt
    python -m src.train --init_from outputs/pretrained.pt --epochs 120 --lr 5e-4

预训练结束后会顺带在 DRIVE 验证集上算一次 **零样本 Dice**（不微调直接预测），
用来确认预训练真的学到了可迁移的血管特征。
"""

import os
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import parse_config
from .dataset import (ImageDataset, PatchDataset, build_external_records,
                      build_records)
from .losses import build_loss, training_loss, region_logits
from .models import build_model
from .checkpointing import save_resume, load_resume
from .train import build_scheduler, evaluate, make_scaler
from .utils import AverageMeter, Timer, count_params, ensure_dir, get_device, set_seed


def main(argv=None):
    cfg = parse_config(argv, description="外部数据预训练（STARE/CHASE/HRF -> 血管分割）")
    set_seed(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.dirname(os.path.abspath(cfg.pretrain_out)))
    if os.path.exists(cfg.pretrain_out):
        raise FileExistsError(f'预训练输出已存在，拒绝覆盖：{cfg.pretrain_out}')

    roots = [d.strip() for d in cfg.external_dirs.split(",") if d.strip()]
    missing = [r for r in roots if not os.path.isdir(os.path.join(r, "image"))]
    if missing:
        raise SystemExit(
            "缺少外部数据目录: " + ", ".join(missing) +
            "\n请先运行: python scripts/download_external.py && python scripts/prepare_external.py")

    records = build_external_records(roots)
    train_set = PatchDataset(cfg, records)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg.loss, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = make_scaler(cfg.amp and device.type == "cuda")
    steps = max(1, cfg.patches_per_epoch // cfg.batch_size)

    per_ds = {}
    for r in records:
        per_ds[r["dataset"]] = per_ds.get(r["dataset"], 0) + 1

    print("=" * 78)
    print(f"设备        : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"模型        : {cfg.model}  参数 {count_params(model)/1e6:.2f} M")
    print(f"外部数据    : {len(records)} 张  "
          + "  ".join(f"{k}={v}" for k, v in per_ds.items()))
    print(f"Patch       : {cfg.patch_size}  每个 epoch {cfg.patches_per_epoch} 个"
          f"（{steps} 步）")
    print(f"损失/优化器 : {cfg.loss} / AdamW(lr={cfg.lr}, wd={cfg.weight_decay})")
    print(f"预训练轮数  : {cfg.epochs}   AMP: {cfg.amp and device.type == 'cuda'}")
    print("=" * 78)

    history = []
    timer = Timer()
    start_epoch = 0
    if cfg.resume_from:
        resumed = load_resume(cfg.resume_from, 'pretrain', cfg, model, optimizer, scheduler,
                              scaler, train_set, device)
        history, start_epoch = resumed['history'], resumed['epoch']
        print(f'Resuming pretraining after epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        t0 = time.time()

        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            auxiliary = {k:v.to(device, non_blocking=True) for k,v in batch[2].items()} if len(batch)>2 else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                enabled=scaler.is_enabled()):
                logits = model(x)
                loss = training_loss(logits, y, criterion, cfg, auxiliary, epoch+1)
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

            with torch.no_grad():
                prob = torch.sigmoid(region_logits(logits).float())
                pred = (prob > 0.5).float()
                inter = (pred * y).sum(dim=(1, 2, 3))
                denom = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
                batch_dice = ((2 * inter + 1e-7) / (denom + 1e-7)).mean().item()
            loss_meter.update(loss.item(), x.size(0))
            dice_meter.update(batch_dice, x.size(0))

        scheduler.step()   # 注意：调度器按 epoch 步进（与 train.py 一致）

        rec = {"epoch": epoch + 1, "loss": loss_meter.avg, "patch_dice": dice_meter.avg,
               "lr": optimizer.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(rec)
        print(f"[pretrain {epoch+1:>3d}/{cfg.epochs}] loss={rec['loss']:.4f}  "
              f"patch_dice={rec['patch_dice']:.4f}  lr={rec['lr']:.2e}  {rec['sec']:.1f}s",
              flush=True)
        if (epoch+1) % cfg.checkpoint_every == 0 or epoch+1 == cfg.epochs:
            save_resume(os.path.join(cfg.out_dir, 'resume.pt'), 'pretrain', cfg, model,
                        optimizer, scheduler, scaler, train_set, history)
            with open(os.path.join(cfg.out_dir, 'history.partial.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                "meta": {"external_dirs": roots, "records": len(records),
                         "per_dataset": per_ds, "history": history}},
               cfg.pretrain_out)
    print("-" * 78)
    print(f"预训练权重 -> {cfg.pretrain_out}   耗时 {timer}")

    # ---------------- 零样本迁移检查（DRIVE 验证集，不微调） ---------------- #
    eval_ids = [i.strip() for i in cfg.val_ids.split(",") if i.strip()]
    if eval_ids:
        drive_records = build_records(cfg.train_image_dir, cfg.train_label_dir)
        val_set = ImageDataset(cfg, eval_ids, with_label=True, records=drive_records)
        if len(val_set):
            loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
            agg, rows = evaluate(model, loader, cfg, device)
            print(f"零样本迁移（外部数据直接预测 DRIVE {eval_ids}）: "
                  f"Dice={agg['dice']:.4f}  Se={agg['se']:.4f}  Sp={agg['sp']:.4f}")
            for r in rows:
                print(f"   {r['id']:>3s}  Dice={r['dice']:.4f}")
    print("=" * 78)
    print(f"下一步：python -m src.train --init_from {cfg.pretrain_out} "
          f"--epochs 120 --lr 5e-4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
