"""训练入口：AdamW + 预热余弦 + 标签平滑 + MixUp + 随机遮挡 + EMA + AMP。

命令行示例：
    # 第一轮小基线（48px，35 轮）
    python -m src.train --config configs/round1_baseline.yaml --out-dir outputs/runs/r1_base

    # 第十二轮遮挡配方（160px ConvNeXt-Tiny）
    python -m src.train --config configs/round12_erasing.yaml --out-dir outputs/runs/r12 --refit

    # 折外评估（3 折按重复组划分，输出 oof.npz 供融合/质量权重研究）
    python -m src.train --config configs/round7_gce.yaml --oof 3 --out-dir outputs/runs/oof7
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import ARTIFACTS, CLASSES, NUM_CLASSES, RUNS
from .config import load_cfg, save_cfg
from .data import (FERDataset, build_transforms, class_weights_from_counts, group_kfold,
                   make_loader, mixup_batch, mixup_targets, prepare_data, soft_targets)
from .losses import Criterion
from .metrics import summarize
from .models import build_model, param_groups, predict_proba


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)


def warmup_cosine(optimizer, total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, flip_tta: bool = True, fp32: bool = True) -> dict:
    model.eval()
    probs, ys = [], []
    for x, y, _idx, _w in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=False if fp32 else device.type == "cuda"):
            p = predict_proba(model, x.float(), flip_tta=flip_tta)
        probs.append(p.float().cpu())
        ys.append(y)
    prob = torch.cat(probs).numpy()
    y = torch.cat(ys).numpy()
    rep = summarize(y, prob.argmax(1))
    rep["probs"] = prob
    rep["labels"] = y
    return rep


def _train_one_epoch(model, loader, crit, opt, sched, device, cfg, epoch: int, limit_batches: int | None,
                     amp: bool, scaler, ema: EMA | None) -> dict:
    model.train()
    tl = cfg["train"]
    mix_prob, alpha = float(tl.get("mixup_prob", 0.0)), float(tl.get("mixup_alpha", 0.2))
    n_seen, loss_sum, correct = 0, 0.0, 0
    t0 = time.time()
    for step, (x, y, _idx, w) in enumerate(loader):
        if limit_batches and step >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True).float()
        target = soft_targets(y, float(tl.get("label_smoothing", 0.0)))
        lam = 1.0
        if mix_prob > 0 and random.random() < mix_prob:
            x, _ya, yb, lam = mixup_batch(x, y, alpha)
            tb = soft_targets(yb, float(tl.get("label_smoothing", 0.0)))
            target = mixup_targets(target, tb, lam)
            # 质量权重同样混合：避免"混合后整批统一权重"
            perm = torch.randperm(w.size(0), device=w.device)
            w = lam * w + (1 - lam) * w[perm]

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = crit(logits, target, sample_weight=w)
        if scaler is not None:
            scaler.scale(loss).backward()
            if tl.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tl["grad_clip"]))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if tl.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tl["grad_clip"]))
            opt.step()
        sched.step()
        if ema is not None:
            ema.update(model)

        bs = y.size(0)
        n_seen += bs
        loss_sum += float(loss.detach()) * bs
        correct += int((logits.argmax(1) == y).sum())
        if tl.get("log_every") and step % int(tl["log_every"]) == 0:
            print(f"  ep{epoch} step{step}/{len(loader)} loss={float(loss):.4f} "
                  f"acc={correct / max(1, n_seen):.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
    return {"loss": loss_sum / max(1, n_seen), "train_acc": correct / max(1, n_seen),
            "sec": time.time() - t0}


def run_training(cfg: dict, out_dir: Path, seed: int | None = None, limit_batches: int | None = None,
                 train_idx=None, val_idx=None, samples=None, weights=None, refit_epochs: int = 0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(seed if seed is not None else cfg["exp"]["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_cfg(cfg, out_dir / "config.yaml")

    if samples is None:
        samples, _groups, splits, weights, meta = prepare_data(cfg, seed)
        train_idx, val_idx = splits["train"], splits["val"]
        (out_dir / "data_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tf_tr = build_transforms(cfg, train=True)
    tf_ev = build_transforms(cfg, train=False)
    w_tr = weights[train_idx] if weights is not None else None
    w_ev = weights[val_idx] if weights is not None else None
    ds_tr = FERDataset([samples[i] for i in train_idx], tf_tr, w_tr)
    ds_ev = FERDataset([samples[i] for i in val_idx], tf_ev, w_ev)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["data"].get("num_workers", 4))
    dl_tr = make_loader(ds_tr, bs, True, nw, drop_last=len(ds_tr) > bs)
    dl_ev = make_loader(ds_ev, int(cfg["eval"].get("batch_size", 512)), False, nw)

    model, info = build_model(cfg)
    model.to(device)
    print(f"[model] {info}", flush=True)

    cw = None
    if cfg["train"].get("class_weights"):
        cw = class_weights_from_counts([samples[i] for i in train_idx])
    crit = Criterion(float(cfg["train"].get("gce_q", 0.0)), float(cfg["train"]["label_smoothing"]), cw).to(device)

    opt = torch.optim.AdamW(param_groups(model, float(cfg["train"]["lr"]),
                                         float(cfg["model"].get("head_lr_mult", 1.0)),
                                         float(cfg["train"]["weight_decay"])))
    total_steps = max(1, int(cfg["train"]["epochs"]) * len(dl_tr))
    warm = int(float(cfg["train"].get("warmup_epochs", 1)) * len(dl_tr))
    sched = warmup_cosine(opt, total_steps, warm)
    amp = bool(cfg["train"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda") if hasattr(torch, "amp") else None
    ema = EMA(model, float(cfg["train"]["ema_decay"])) if float(cfg["train"].get("ema_decay", 0)) > 0 else None

    best_acc, best_ep, history = -1.0, -1, []
    for ep in range(1, int(cfg["train"]["epochs"]) + 1):
        tr = _train_one_epoch(model, dl_tr, crit, opt, sched, device, cfg, ep, limit_batches, amp, scaler, ema)
        ev_model = ema.shadow if ema is not None else model
        ev = evaluate(ev_model, dl_ev, device, bool(cfg["train"].get("eval_tta_flip", True)))
        row = {"epoch": ep, **tr, "val_acc": ev["accuracy"], "val_macro_f1": ev["macro_f1"],
               "val_ci": ev["wilson_ci"]}
        history.append(row)
        print(f"[ep{ep}] loss={tr['loss']:.4f} train={tr['train_acc']:.4f} "
              f"val={ev['accuracy']:.4f} f1={ev['macro_f1']:.4f} ({tr['sec']:.1f}s)", flush=True)
        if ev["accuracy"] > best_acc:
            best_acc, best_ep = ev["accuracy"], ep
            torch.save({"model": ev_model.state_dict(), "cfg": cfg, "info": info, "epoch": ep,
                        "val": {k: v for k, v in ev.items() if k not in ("probs", "labels")},
                        "classes": CLASSES}, out_dir / "best.pt")
    torch.save({"model": (ema.shadow if ema is not None else model).state_dict(), "cfg": cfg,
                "info": info, "epoch": int(cfg["train"]["epochs"]), "classes": CLASSES},
               out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {"best_acc": best_acc, "best_epoch": best_ep, "out_dir": str(out_dir), "model": info}

    if refit_epochs and refit_epochs > 0:
        print(f"[refit] 合并 train+val，追加 {refit_epochs} 轮", flush=True)
        merged = sorted(set(train_idx) | set(val_idx))
        ds_all = FERDataset([samples[i] for i in merged], tf_tr, weights[merged] if weights is not None else None)
        dl_all = make_loader(ds_all, bs, True, nw, drop_last=len(ds_all) > bs)
        opt2 = torch.optim.AdamW(param_groups(model, float(cfg["train"]["lr"]) * 0.1,
                                              float(cfg["model"].get("head_lr_mult", 1.0)),
                                              float(cfg["train"]["weight_decay"])))
        # 按报告第十四轮：8 轮计划的末尾余弦段，而不是重新整段调度
        steps2 = refit_epochs * len(dl_all)
        sched2 = torch.optim.lr_scheduler.LambdaLR(
            opt2, lambda s: 0.5 * (1 + math.cos(math.pi * (s / max(1, steps2)))))
        for ep in range(1, refit_epochs + 1):
            tr = _train_one_epoch(model, dl_all, crit, opt2, sched2, device, cfg, ep, limit_batches, amp, scaler, ema)
            print(f"[refit ep{ep}] loss={tr['loss']:.4f} train={tr['train_acc']:.4f}", flush=True)
        torch.save({"model": (ema.shadow if ema is not None else model).state_dict(), "cfg": cfg,
                    "info": info, "epoch": int(cfg["train"]["epochs"]) + refit_epochs,
                    "classes": CLASSES, "refit_epochs": refit_epochs}, out_dir / "deploy.pt")
        result["deploy"] = str(out_dir / "deploy.pt")
    return result


def run_oof(cfg: dict, out_dir: Path, k: int = 3, seed: int | None = None,
            limit_batches: int | None = None) -> dict:
    """按重复组做 k 折，输出 oof.npz（索引/概率/标签/组号）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(seed if seed is not None else cfg["exp"]["seed"])
    samples, groups, splits, weights, meta = prepare_data(cfg, seed)
    pool = sorted(set(splits["train"]) | set(splits["val"]))  # 25,427 开发池
    pool_samples = [samples[i] for i in pool]
    pool_groups = [groups[i] for i in pool]
    folds = group_kfold(pool_samples, pool_groups, k, seed)

    oof = np.full((len(pool), NUM_CLASSES), np.nan, dtype=np.float32)
    for f, (tr_local, te_local) in enumerate(folds):
        tr = [pool[i] for i in tr_local]
        te = [pool[i] for i in te_local]
        print(f"[oof fold{f}] train={len(tr)} test={len(te)}", flush=True)
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["exp"]["name"] = f"{cfg['exp']['name']}_f{f}"
        res = run_training(sub_cfg, out_dir / f"fold{f}", seed=seed + f, limit_batches=limit_batches,
                           train_idx=tr, val_idx=te, samples=samples, weights=weights)
        ck = torch.load(res["out_dir"] + "/best.pt", map_location=device, weights_only=False)
        model, _ = build_model(ck["cfg"])
        model.load_state_dict(ck["model"])
        model.to(device)
        ds = FERDataset([samples[i] for i in te], build_transforms(cfg, train=False))
        dl = make_loader(ds, int(cfg["eval"]["batch_size"]), False, int(cfg["data"]["num_workers"]))
        ev = evaluate(model, dl, device, bool(cfg["train"].get("eval_tta_flip", True)))
        oof[te_local] = ev["probs"]
        print(f"[oof fold{f}] acc={ev['accuracy']:.4f}", flush=True)

    y = np.array([pool_samples[i][1] for i in range(len(pool_samples))])
    g = np.array(pool_groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "oof.npz", probs=oof, labels=y, groups=g,
                        pool_idx=np.array(pool))
    acc = float((np.nan_to_num(oof).argmax(1) == y).mean())
    print(f"[oof] merged accuracy={acc:.4f}  文件：{out_dir / 'oof.npz'}", flush=True)
    return {"oof_acc": acc, "path": str(out_dir / "oof.npz")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2026task2 FER 训练")
    ap.add_argument("--config", default="configs/round1_baseline.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="点路径覆盖，如 train.epochs=2 model.dropout=0.3")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--refit", action="store_true", help="训练后合并 train+val 追加重训（部署权重）")
    ap.add_argument("--oof", type=int, default=0, help=">1 时执行按组 k 折折外评估")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config, args.set)
    name = cfg["exp"]["name"]
    out_dir = Path(args.out_dir) if args.out_dir else RUNS / name
    limit = args.limit_batches if args.limit_batches is not None else cfg["train"].get("limit_batches")
    if args.oof and args.oof > 1:
        res = run_oof(cfg, out_dir, args.oof, args.seed, limit)
    else:
        res = run_training(cfg, out_dir, args.seed, limit,
                           refit_epochs=int(cfg["train"].get("refit_epochs", 0)) if args.refit else 0)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
