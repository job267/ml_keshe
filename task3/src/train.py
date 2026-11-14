"""训练入口（stage1 粗定位 / stage2 精定位）。

示例
----
# 阶段一：整图 -> 黄斑区域热力图 -> 粗坐标
python -m src.train --config configs/stage1.yaml --stage 1 --fold 0

# 阶段二：以 stage1 结果为中心裁切高清 patch -> 精坐标
python -m src.train --config configs/stage2.yaml --stage 2 --fold 0 \
       --s1-ckpt outputs/runs/s1_fold0/best.pt

# 5 折全跑
python -m src.train --config configs/stage1.yaml --stage 1 --fold all
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_cfg
from .data import FoveaDataset, Sample, dataloader_workers_ok, load_train_samples, split_indices
from .infer import run_inference
from .losses import FoveaLoss
from .metrics import format_metrics
from .models import build_model
from .utils import (
    AverageMeter,
    Logger,
    ModelEma,
    count_params,
    format_seconds,
    load_checkpoint,
    resolve_path,
    save_checkpoint,
    save_json,
    set_seed,
    worker_init_fn,
)


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #
def _loader(ds: FoveaDataset, cfg: Dict, train: bool) -> DataLoader:
    nw = int(cfg["data"]["num_workers"])
    if nw > 0 and not dataloader_workers_ok():
        nw = 0
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=nw > 0,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4 if nw > 0 else None,
    )


def build_stage1_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, List[Sample], np.ndarray]:
    samples = load_train_samples(cfg)
    tr, va = split_indices(len(samples), int(cfg["data"]["n_folds"]), int(cfg["seed"]), int(cfg["data"]["fold"]))
    ds_tr = FoveaDataset([samples[i] for i in tr], cfg, train=True, task="stage1")
    ds_va = FoveaDataset([samples[i] for i in va], cfg, train=False, task="stage1")
    return _loader(ds_tr, cfg, True), _loader(ds_va, cfg, False), samples, va


def build_stage2_loaders(
    cfg: Dict,
    s1_ckpt_path: str,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, List[Sample], Dict]:
    """stage2 的验证集必须用 stage1 的**真实预测**作为裁切中心，否则指标会虚高。"""
    samples = load_train_samples(cfg)
    tr, va = split_indices(len(samples), int(cfg["data"]["n_folds"]), int(cfg["seed"]), int(cfg["data"]["fold"]))
    val_samples = [samples[i] for i in va]

    ckpt = load_checkpoint(s1_ckpt_path, map_location="cpu")
    cfg1 = ckpt["cfg"]
    model1 = build_model(cfg1, device)
    model1.load_state_dict(ckpt["state_dict"])
    model1.eval()
    tta1 = bool(cfg1["train"].get("eval_tta", False))
    centers, s1_metrics = run_inference(
        model1, FoveaDataset(val_samples, cfg1, train=False, task="stage1"), cfg1, device, tta1
    )
    del model1
    torch.cuda.empty_cache()

    ds_tr = FoveaDataset([samples[i] for i in tr], cfg, train=True, task="stage2")
    ds_va = FoveaDataset(val_samples, cfg, train=False, task="stage2", centers=centers)
    meta = {
        "s1_ckpt": str(resolve_path(s1_ckpt_path)),
        "s1_val_metrics": s1_metrics,
        "val_centers": centers,
    }
    return _loader(ds_tr, cfg, True), _loader(ds_va, cfg, False), samples, meta


# --------------------------------------------------------------------------- #
# 优化器与调度
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    bb, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if n.startswith("backbone.") else other).append(p)
    groups = [{"params": other, "lr": float(t["lr"])}]
    if bb:
        groups.append({"params": bb, "lr": float(t["lr"]) * float(t["backbone_lr_scale"])})
    return torch.optim.AdamW(groups, weight_decay=float(t["weight_decay"]))


def build_scheduler(opt: torch.optim.Optimizer, cfg: Dict, steps_per_epoch: int):
    t = cfg["train"]
    total = max(1, int(t["epochs"]) * steps_per_epoch)
    warmup = max(1, int(t["warmup_epochs"]) * steps_per_epoch)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# --------------------------------------------------------------------------- #
# 单个 epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: FoveaLoss,
    opt: torch.optim.Optimizer,
    sched,
    scaler,
    ema: Optional[ModelEma],
    device: torch.device,
    cfg: Dict,
    epoch: int,
    log: Logger,
) -> Dict[str, float]:
    model.train()
    meters = {k: AverageMeter() for k in ("total", "hm", "soft", "coord", "vis")}
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    clip = float(cfg["train"]["grad_clip"])
    limit = int(cfg["train"]["limit_batches"])
    t0 = time.time()

    for it, b in enumerate(loader):
        if limit and it >= limit:
            break
        x = b["image"].to(device, non_blocking=True)
        xy = b["xy"].to(device, non_blocking=True)
        vis = b["vis"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            parts = loss_fn(pred, xy, vis)
        scaler.scale(parts["total"]).backward()
        if clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update(model)
        if sched is not None:
            sched.step()

        n = x.shape[0]
        for k in meters:
            if k in parts:
                meters[k].update(float(parts[k].detach()), n)

        if (it + 1) % 20 == 0:
            log(f"    ep{epoch:03d} it{it+1:04d} loss={meters['total'].avg:.5f}")

    dt = time.time() - t0
    lr = opt.param_groups[0]["lr"]
    return {
        "loss": meters["total"].avg,
        "hm": meters["hm"].avg,
        "soft": meters["soft"].avg,
        "coord": meters["coord"].avg,
        "lr": lr,
        "sec": dt,
    }


# --------------------------------------------------------------------------- #
# 主训练流程
# --------------------------------------------------------------------------- #
def fit(
    cfg: Dict,
    stage: int,
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    extra_meta: Dict,
    out_dir: Path,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据划分固定用顶层 seed；训练随机性用 run_seed，方便在同一划分上做多种子实验
    run_seed = cfg["train"].get("run_seed")
    run_seed = int(cfg["seed"]) if run_seed is None else int(run_seed)
    set_seed(run_seed + fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    log(f"===== {cfg['task']} | fold {fold} | device {device} | out {out_dir}")
    log(f"seeds: split={cfg['seed']}  run={run_seed}（fold 内实际 = {run_seed + fold}）")

    model = build_model(cfg, device)
    log(f"model: {cfg['model']['backbone']}  params={count_params(model)/1e6:.2f}M  "
        f"input={cfg['data']['img_size']}  out_stride={cfg['model']['out_stride']}")
    loss_fn = FoveaLoss(cfg).to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, max(1, len(train_loader)))
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEma(model, decay=float(cfg["train"]["ema_decay"])) if cfg["train"]["ema_decay"] > 0 else None
    eval_model = copy.deepcopy(model).eval()
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_loader.dataset)      # type: ignore[arg-type]
    log(f"train={n_train} val={n_val} batch={cfg['train']['batch_size']} epochs={cfg['train']['epochs']}")

    hist_path = out_dir / "history.csv"
    hist_keys = ["epoch", "lr", "loss", "hm", "soft", "coord", "val_rmse", "val_mean_l2", "best_rmse", "sec"]
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(hist_keys)

    best = float("inf")
    best_epoch = -1
    bad = 0
    tta = bool(cfg["train"].get("eval_tta", False))
    source = cfg["model"].get("coord_source", "soft")
    t_start = time.time()

    for epoch in range(int(cfg["train"]["epochs"])):
        tr = train_one_epoch(model, train_loader, loss_fn, opt, sched, scaler, ema,
                             device, cfg, epoch, log)
        if ema is not None:
            ema.load_into(eval_model)
        else:
            eval_model.load_state_dict(model.state_dict())
        _, vm = run_inference(eval_model, val_loader.dataset, cfg, device, tta=tta, source=source)  # type: ignore[arg-type]

        is_best = vm["rmse"] < best - 1e-9
        if is_best:
            best, best_epoch, bad = vm["rmse"], epoch, 0
        else:
            bad += 1

        log(
            f"ep {epoch:03d}/{cfg['train']['epochs']-1:03d} | lr {tr['lr']:.2e} | "
            f"loss {tr['loss']:.5f} (hm {tr['hm']:.5f} xy {tr['soft']:.5f}) | "
            f"val {vm['rmse']:.2f}px meanL2 {vm['mean_l2']:.2f} | best {best:.2f}@{best_epoch} | "
            f"{tr['sec']:.1f}s"
        )
        with hist_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{tr['lr']:.6e}", f"{tr['loss']:.6f}", f"{tr['hm']:.6f}",
                f"{tr['soft']:.6f}", f"{tr['coord']:.6f}", f"{vm['rmse']:.4f}",
                f"{vm['mean_l2']:.4f}", f"{best:.4f}", f"{tr['sec']:.2f}",
            ])

        every = max(1, int(cfg["train"].get("save_last_every", 1)))
        if is_best or (epoch + 1) % every == 0:
            t_save = time.time()
            # 只在真正保存时做 CPU 拷贝，避免每个 epoch 白白同步几十 MB
            payload = {
                "state_dict": {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()},
                "cfg": cfg,
                "stage": stage,
                "fold": fold,
                "epoch": epoch,
                "val": vm,
                "best_val_rmse": best,
                **extra_meta,
            }
            if bool(cfg["train"].get("save_raw", False)):
                payload["raw_state_dict"] = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if is_best:
                save_checkpoint(out_dir / "best.pt", payload)
            if (epoch + 1) % every == 0:
                save_checkpoint(out_dir / "last.pt", payload)
            dt_save = time.time() - t_save
            if dt_save > 2.0:
                log(f"    [ckpt] 保存耗时 {dt_save:.1f}s（磁盘/杀毒软件较慢时可调大 save_last_every）")

        if int(cfg["train"]["patience"]) > 0 and bad >= int(cfg["train"]["patience"]):
            log(f"early stop: {bad} epochs 无提升")
            break

    total = time.time() - t_start
    log(f"done in {format_seconds(total)} | best val RMSE = {best:.2f}px @ epoch {best_epoch}")
    save_json(out_dir / "summary.json", {"stage": stage, "fold": fold, "best_val_rmse": best,
                                        "best_epoch": best_epoch, "seconds": total})
    return {"best_rmse": best, "best_epoch": best_epoch}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="中心凹定位训练")
    ap.add_argument("--config", nargs="+", default=None,
                    help="一个或多个 yaml，后者覆盖前者（便于做消融）")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--fold", type=str, default="0", help="0..k-1 或 all")
    ap.add_argument("--s1-ckpt", type=str, default=None, help="stage2 需要的 stage1 检查点")
    ap.add_argument("--tag", type=str, default="", help="输出目录后缀")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--run-seed", type=int, default=None,
                    help="训练随机种子（数据划分仍由 cfg.seed 决定，用于同划分下的多种子实验）")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置，如 --set train.epochs=60")
    args = ap.parse_args(argv)

    if os.environ.get("DSH_STACK_DUMP"):  # 排查卡死用：定期打印所有线程栈
        import faulthandler

        faulthandler.dump_traceback_later(int(os.environ["DSH_STACK_DUMP"]), repeat=True)

    cfg = load_cfg(args.config, args.set)
    cfg["task"] = f"stage{args.stage}"
    if args.run_seed is not None:
        cfg["train"]["run_seed"] = int(args.run_seed)
    # stage2 用 6 通道（patch + 整图缩略图）时，模型的第一个卷积也要相应扩展
    if args.stage == 2:
        cfg["model"]["in_ch"] = 6 if cfg["stage2"].get("use_context") else 3
    folds: List[int] = (
        list(range(int(cfg["data"]["n_folds"]))) if args.fold == "all" else [int(args.fold)]
    )

    for fold in folds:
        cfg["data"]["fold"] = fold
        out_dir = (
            resolve_path(args.out_dir) if args.out_dir
            else resolve_path(cfg["output"]["dir"]) / "runs" / f"s{args.stage}_fold{fold}{args.tag}"
        )
        if args.stage == 1:
            tr_dl, va_dl, _, _ = build_stage1_loaders(cfg)
            extra = {"s1_ckpt": None}
        else:
            # 支持模板：--s1-ckpt "outputs/runs/s1_fold{fold}/best.pt"，配合 --fold all 使用
            s1 = args.s1_ckpt or cfg["stage2"].get("s1_ckpt") or "outputs/runs/s1_fold{fold}/best.pt"
            s1 = str(s1).format(fold=fold)
            if not resolve_path(s1).exists():
                raise SystemExit(f"找不到 stage1 检查点: {resolve_path(s1)}")
            cfg["stage2"]["s1_ckpt"] = s1
            tr_dl, va_dl, _, extra = build_stage2_loaders(
                cfg, s1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            print(f"stage1(fold{fold}) 在验证集上的指标: {format_metrics(extra['s1_val_metrics'])}")
        fit(cfg, args.stage, fold, tr_dl, va_dl, extra, out_dir)


if __name__ == "__main__":
    main()
