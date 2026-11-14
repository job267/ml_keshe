"""5 折交叉验证 + 集成。

为什么需要它：只有一个 4 张图的验证集时，模型之间 0.003 的差距完全被噪声淹没
（前两轮已经踩过这个坑）。5 折交叉验证把 20 张图全部轮流当一次验证集，
得到覆盖全体的 **20 个 out-of-fold 预测**。逐图标准差、均值不确定性与训练种子
波动必须分别报告，不能用一个 ± 数字替代。

流程::

    # 训练 5 折（每折 16 训 / 4 验），并给出 OOF（out-of-fold）指标
    python -m src.cv --folds 5 --epochs 150

    # 只重新评估已有的 5 折权重
    python -m src.cv --skip_train

    # 用 5 折模型做概率平均集成，生成提交掩膜
    python scripts/ensemble.py \
        --checkpoints outputs_cv/fold0/best.pt,outputs_cv/fold1/best.pt,outputs_cv/fold2/best.pt,\
outputs_cv/fold3/best.pt,outputs_cv/fold4/best.pt --data_dir test --pred_dir image

产物：
    outputs_cv/fold{k}/best.pt        每折最优权重
    outputs_cv/cv_report.csv          逐图 OOF 指标（含 fold 编号）
    outputs_cv/cv_summary.json        每折均值 / 总体均值 / 标准差
"""

import json
import os
import time
from dataclasses import fields

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ImageDataset, build_records, split_ids
from .models import load_checkpoint
from .train import evaluate, main as train_main
from .utils import Timer, ensure_dir, get_device, imread_gray


def stratified_folds(records, k, seed=0, strategy="stratified"):
    """按血管像素占比排序后交错分配，使每折的血管密度尽量接近。

    交叉验证里如果某一折恰好都是"血管多"的图，指标会被系统性拉高/拉低，
    所以这里用 cheap stratification：按 target 比例排序 -> 轮流发牌。
    """
    if not 2 <= k <= len(records):
        raise ValueError("folds 必须在 2 与样本数之间")
    items = []
    for r in records:
        lab = imread_gray(r["label"])
        items.append((float((lab > 127).mean()), r["id"]))
    items.sort()
    folds = [[] for _ in range(k)]
    rng = np.random.RandomState(seed)
    if strategy == "legacy":
        order = rng.permutation(k)
        for idx, (_, i) in enumerate(items):
            folds[order[idx % k]].append(i)
    elif strategy == "stratified":
        # 每个相邻密度区间独立发牌，seed 改变成员而非只改变折编号。
        for start in range(0, len(items), k):
            for fold, (_, rid) in zip(rng.permutation(k), items[start:start+k]):
                folds[int(fold)].append(rid)
    else:
        raise ValueError(f"未知 split_strategy: {strategy}")
    return folds


def read_folds(path, ids):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    split = raw.get("split", raw)
    names = sorted(split, key=lambda s: int(s.removeprefix("fold")))
    if names != [f"fold{i}" for i in range(len(names))]:
        raise ValueError("split 必须连续命名为 fold0..foldK")
    folds = [[str(i) for i in split[name]] for name in names]
    flat = [i for fold in folds for i in fold]
    if any(not f for f in folds) or len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError("划分必须无重复、无空折、完整覆盖全部图像")
    return folds


def parse_argv(argv=None):
    """复用 Config 的字段生成命令行（与 train.py 完全一致的参数集），再加 CV 专用参数。"""
    from .config import build_parser

    p = build_parser("K 折交叉验证 + OOF 评估")
    p.add_argument("--folds", type=int, default=5, help="折数")
    p.add_argument("--skip_train", action="store_true", help="跳过训练，只重算 OOF 指标")
    p.add_argument("--split_file", default="", help="使用保存的 JSON 划分，禁止重新推断")
    # 覆盖 Config 的默认值：CV 的输出目录与每折轮数
    p.set_defaults(out_dir="outputs_cv", epochs=150, val_every=5)
    return p.parse_args(argv)


def fold_argv(args, fold, val_ids):
    """把本次实验的公共参数翻译成 train.py 的命令行。"""
    overrides = {"val_ids": ",".join(val_ids), "out_dir": os.path.join(args.out_dir, f"fold{fold}"),
                 "seed": args.seed + fold}
    argv = []
    for field in fields(Config):
        value = overrides.get(field.name, getattr(args, field.name))
        # 等号形式保留空字符串；不能遗漏任何 Config 参数。
        argv.append(f"--{field.name}={value}")
    return argv


def oof_evaluate(args, folds, records, device):
    """每张图只用"没见过它"的那一折模型预测，得到 out-of-fold 指标。"""
    rows = []
    for fold, val_ids in enumerate(folds):
        ck = os.path.join(args.out_dir, f"fold{fold}", "best.pt")
        if not os.path.exists(ck):
            raise FileNotFoundError(f"缺少 {ck}，拒绝把部分折当作完整 OOF")
        model, cfg, _ = load_checkpoint(ck, device=device)
        if set(cfg.val_ids.split(',')) != set(val_ids):
            raise ValueError(f"{ck} 的验证 ID 与划分不符，拒绝可能泄漏的 OOF")
        cfg.threshold = args.threshold
        cfg.min_size = args.min_size
        cfg.use_fov = args.use_fov
        cfg.fill_holes = args.fill_holes
        ds = ImageDataset(cfg, val_ids, with_label=True, records=records)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        agg, fold_rows = evaluate(model, loader, cfg, device, tta=args.tta)
        for r in fold_rows:
            r["fold"] = fold
            rows.append(r)
        print(f"  fold{fold}  ({len(val_ids)} 张): Dice={agg['dice']:.4f}  "
              f"Se={agg['se']:.4f}  Sp={agg['sp']:.4f}")
    return rows


def ensemble_note(runs_hint="outputs_cv,outputs_cv_att"):
    """为什么这里不再输出"集成 OOF"。

    单次 K 折里，对某张图 i，**除它所在那一折以外的 K-1 个模型全都训练过它**：
    用它们平均得到的是"在训练集上的预测"，分数天然虚高（本项目实测虚高约 0.018）。
    想诚实地估计 N 模型集成的收益，必须让 N 个 run（不同架构/不同划分种子）
    各自提供"没见过这张图"的那一个模型，再平均 —— 见 ``scripts/oof_ensemble.py``。
    """
    print("-" * 78)
    print("关于集成：单次交叉验证**无法**估计集成的收益（每张图只有一个没见过它的模型）。")
    print("           跨 run 的诚实集成估计请用：")
    print(f"             python scripts/oof_ensemble.py --runs {runs_hint}")
    print("-" * 78)


def main(argv=None):
    args = parse_argv(argv)
    device = get_device(args.device)
    ensure_dir(args.out_dir)

    records = build_records(args.train_image_dir, args.train_label_dir)
    ids = [r["id"] for r in records]
    saved = os.path.join(args.out_dir, "cv_summary.json")
    if args.split_file:
        folds = read_folds(args.split_file, ids)
    elif args.skip_train:
        if not os.path.exists(saved):
            raise ValueError("skip_train 必须提供 split_file 或已有 cv_summary.json，不能按 seed 猜测历史划分")
        folds = read_folds(saved, ids)
    else:
        folds = stratified_folds(records, args.folds, seed=args.split_seed, strategy=args.split_strategy)
    if len(folds) != args.folds:
        raise ValueError("划分折数与 --folds 不符")
    print("=" * 78)
    print(f"K 折交叉验证: K={args.folds}  样本 {len(ids)} 张  |  每折 {[len(f) for f in folds]}")
    for k, f in enumerate(folds):
        print(f"  fold{k}: {f}")
    print("=" * 78)

    timer = Timer()
    if not args.skip_train:
        if any(os.path.exists(os.path.join(args.out_dir, f"fold{fold}", "best.pt")) for fold in range(args.folds)):
            raise ValueError("输出目录已有权重，请使用新 out_dir，避免覆盖历史结果")
        with open(os.path.join(args.out_dir, "split.json"), "w", encoding="utf-8") as f:
            json.dump({"split": {f"fold{k}": v for k, v in enumerate(folds)},
                       "split_seed": args.split_seed, "split_strategy": args.split_strategy}, f, indent=2)
        for fold, val_ids in enumerate(folds):
            ck = os.path.join(args.out_dir, f"fold{fold}", "best.pt")
            if os.path.exists(ck) and args.skip_train:
                continue
            print(f"\n{'#' * 78}\n# fold{fold}/{args.folds - 1}  val={val_ids}\n{'#' * 78}")
            t0 = time.time()
            train_main(fold_argv(args, fold, val_ids))
            print(f"# fold{fold} 完成，用时 {(time.time() - t0) / 60:.1f} min")

    # ---------------- OOF 评估 ---------------- #
    print("\n" + "=" * 78)
    print("Out-of-Fold 评估（每张图只用没见过它的模型）")
    rows = oof_evaluate(args, folds, records, device)
    if not rows:
        print("没有可用权重，退出")
        return 1

    rows.sort(key=lambda r: int(r["id"]))
    dice = [r["dice"] for r in rows]
    per_fold = {}
    for r in rows:
        per_fold.setdefault(r["fold"], []).append(r["dice"])
    print("-" * 78)
    print(f"{'id':>4}{'fold':>6}{'Dice':>9}{'IoU':>9}{'Se':>9}{'Sp':>9}{'Acc':>9}")
    for r in rows:
        print(f"{r['id']:>4}{r['fold']:>6}{r['dice']:>9.4f}{r['iou']:>9.4f}"
              f"{r['se']:>9.4f}{r['sp']:>9.4f}{r['acc']:>9.4f}")
    print("-" * 78)
    for k in sorted(per_fold):
        d = np.array(per_fold[k])
        print(f"  fold{k}: Dice={d.mean():.4f} ± {d.std():.4f}  (n={len(d)})")
    print(f"\n**{len(rows)} 折 OOF 平均 Dice = {np.mean(dice):.4f} ± {np.std(dice):.4f}**")
    print("   （± 为逐图标准差，不是均值置信区间，也不代表训练种子波动）")

    # ---------------- 落盘 ---------------- #
    report = os.path.join(args.out_dir, "cv_report.csv")
    with open(report, "w", encoding="utf-8") as f:
        f.write("id,fold,dice,iou,se,sp,acc\n")
        for r in rows:
            f.write(f"{r['id']},{r['fold']},{r['dice']:.6f},{r['iou']:.6f},"
                    f"{r['se']:.6f},{r['sp']:.6f},{r['acc']:.6f}\n")
    summary = {
        "folds": args.folds,
        "config": {f.name: getattr(args, f.name) for f in fields(Config)},
        "oof_dice_512_proxy": float(np.mean([r["dice_512_proxy"] for r in rows])),
        "epochs": args.epochs,
        "init_from": args.init_from,
        "extra_train_dirs": args.extra_train_dirs,
        "extra_repeat": args.extra_repeat,
        "oof_dice_mean": float(np.mean(dice)),
        "oof_dice_std": float(np.std(dice)),
        "oof_se_mean": float(np.mean([r["se"] for r in rows])),
        "oof_sp_mean": float(np.mean([r["sp"] for r in rows])),
        "per_fold": {k: [float(np.mean(v)), float(np.std(v))] for k, v in per_fold.items()},
        "split": {f"fold{k}": v for k, v in enumerate(folds)},
        "seconds": timer.elapsed(),
    }
    with open(os.path.join(args.out_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"报告 -> {report}")
    print(f"汇总 -> {os.path.join(args.out_dir, 'cv_summary.json')}  （总耗时 {timer}）")

    # ---------------- 关于"集成 OOF" ---------------- #
    ensemble_note(",".join(["outputs_cv", "outputs_cv_att", "outputs_cv_full"]))

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
