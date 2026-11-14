"""推理 / 生成提交文件。

三种用法
--------
# 1) 对测试集预测，生成 outputs/submission.csv（可直接上传）
python -m src.predict --config configs/stage2.yaml \
    --runs outputs/runs/s2_fold0 outputs/runs/s2_fold1 outputs/runs/s2_fold2 \
           outputs/runs/s2_fold3 outputs/runs/s2_fold4 --split test

# 2) 交叉验证报告（每个 run 在它自己的验证折上评估，汇总成总体 RMSE）
python -m src.predict --runs outputs/runs/s2_fold0 ... --split cv

# 3) 训练集拟合情况（含 stage1 对比）
python -m src.predict --runs outputs/runs/s1_fold0 --split train

`--runs` 可以给目录（自动找 best.pt）或 .pt 文件；如果给的是 stage2 检查点，
会自动根据 ckpt 里记录的 `s1_ckpt` 串起 stage1 -> stage2 的两阶段流程。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from .config import load_cfg
from .data import (
    FoveaDataset,
    Sample,
    load_submission_template,
    load_test_samples,
    load_train_samples,
    split_indices,
)
from .infer import run_inference
from .metrics import compute_metrics, format_metrics
from .models import build_model
from .utils import load_checkpoint, resolve_path, save_json

Coord = Tuple[float, float]


# --------------------------------------------------------------------------- #
# 单个模型 / 完整链条的预测器
# --------------------------------------------------------------------------- #
def resolve_ckpt(p: str | Path) -> Path:
    p = resolve_path(p)
    if p.is_dir():
        for name in ("best.pt", "last.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"{p} 下没有 best.pt / last.pt")
    if not p.exists():
        raise FileNotFoundError(f"检查点不存在: {p}")
    return p


class ModelRunner:
    """一个检查点 = 一个模型。"""

    def __init__(self, path: str | Path, device: torch.device, source: Optional[str] = None,
                 tta: Optional[bool] = None, tta_spec: Optional[Dict] = None):
        self.path = resolve_ckpt(path)
        ckpt = load_checkpoint(self.path, map_location="cpu")
        self.cfg: Dict = ckpt["cfg"]
        self.stage: int = int(ckpt["stage"])
        self.fold: int = int(ckpt.get("fold", 0))
        self.s1_ckpt: Optional[str] = ckpt.get("s1_ckpt")
        self.val_rmse: Optional[float] = (ckpt.get("val") or {}).get("rmse")
        self.device = device
        self.model = build_model(self.cfg, device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tta = bool(self.cfg["train"].get("eval_tta", False)) if tta is None else bool(tta)
        self.tta_spec = tta_spec
        self.source = source or self.cfg["model"].get("coord_source", "soft")

    def predict(self, samples: Sequence[Sample], centers: Optional[Dict[str, Coord]] = None) -> Dict[str, Coord]:
        ds = FoveaDataset(
            samples, self.cfg, train=False,
            task="stage1" if self.stage == 1 else "stage2",
            centers=centers,
        )
        preds, _ = run_inference(self.model, ds, self.cfg, self.device,
                                 tta=None if self.tta_spec else self.tta,
                                 source=self.source, tta_spec=self.tta_spec)
        return preds


class Chain:
    """stage1（可选）+ stage2 组成的完整预测链。

    `prefer` 控制最终用哪一级的结果：
      stage2 —— 总是用精定位结果；
      stage1 —— 总是用粗定位结果；
      auto   —— 比较两个检查点在各自验证集上的 RMSE，取更好的那个（默认）。
    """

    def __init__(self, run: str | Path, device: torch.device, cache: Dict[str, ModelRunner],
                 source: Optional[str] = None, tta: Optional[bool] = None,
                 prefer: str = "auto", tta_spec: Optional[Dict] = None):
        self.runner = ModelRunner(run, device, source, tta, tta_spec)
        self.s1: Optional[ModelRunner] = None
        self.prefer = prefer
        if self.runner.stage == 2:
            if not self.runner.s1_ckpt:
                raise ValueError(f"{self.runner.path} 是 stage2 模型但缺少 s1_ckpt 记录")
            key = str(resolve_ckpt(self.runner.s1_ckpt))
            if key not in cache:
                cache[key] = ModelRunner(key, device, source, tta, tta_spec)
            self.s1 = cache[key]
        self.fold = self.runner.fold

    @property
    def cfg(self) -> Dict:
        return self.runner.cfg

    @property
    def stage(self) -> int:
        return self.runner.stage

    def _use_stage2(self) -> bool:
        if self.s1 is None:
            return False
        if self.prefer == "stage2":
            return True
        if self.prefer == "stage1":
            return False
        v2 = self.runner.val_rmse if self.runner.val_rmse is not None else 1e9
        v1 = self.s1.val_rmse if self.s1.val_rmse is not None else 1e9
        return v2 <= v1

    def predict(self, samples: Sequence[Sample]) -> Dict[str, Dict[str, Coord]]:
        if self.s1 is None:
            p1 = self.runner.predict(samples)
            return {"stage1": p1, "final": p1}
        p1 = self.s1.predict(samples)
        if not self._use_stage2():
            return {"stage1": p1, "final": p1}
        p2 = self.runner.predict(samples, centers=p1)
        return {"stage1": p1, "final": p2}


# --------------------------------------------------------------------------- #
# 指标汇总
# --------------------------------------------------------------------------- #
def metrics_vs_samples(samples: Sequence[Sample], preds: Dict[str, Coord]) -> Dict[str, float]:
    ids = [s.id for s in samples]
    P = np.array([preds[i] for i in ids], dtype=np.float64)
    G = np.array([[s.x / s.w, s.y / s.h] for s in samples], dtype=np.float64)
    WH = np.array([[s.w, s.h] for s in samples], dtype=np.float64)
    VIS = np.array([s.vis for s in samples], dtype=np.float64)
    return compute_metrics(P, G, WH, vis=VIS)


def pool_metrics(per_run: List[Tuple[Sequence[Sample], Dict[str, Coord]]]) -> Dict[str, float]:
    """把多折的逐样本预测拼起来统一算指标（比"各折 RMSE 求平均"更严谨）。"""
    samples: List[Sample] = []
    preds: Dict[str, Coord] = {}
    for s_list, p in per_run:
        for s in s_list:
            samples.append(s)
            preds[s.id] = p[s.id]
    return metrics_vs_samples(samples, preds)


# --------------------------------------------------------------------------- #
# 提交文件
# --------------------------------------------------------------------------- #
def write_submission(tmpl: Dict, preds: Dict[str, Coord], samples: Sequence[Sample],
                     out_path: Path) -> pd.DataFrame:
    """preds 是**归一化**坐标；这里乘回原图宽高，按模板行序写成像素坐标。"""
    dims = {s.id: (s.w, s.h) for s in samples}
    rows = []
    missing = set()
    for key, (sid, coord) in zip(tmpl["keys"], tmpl["rows"]):
        if sid not in preds:
            missing.add(sid)
            v = 0.0
        else:
            w, h = dims[sid]
            v = preds[sid][0] * w if coord == "X" else preds[sid][1] * h
        rows.append({tmpl["image_col"]: key, tmpl["value_col"]: f"{float(v):.6f}"})
    if missing:
        raise ValueError(f"缺少这些图像的预测: {sorted(missing)}")
    df = pd.DataFrame(rows, columns=tmpl["columns"])
    vmax = pd.to_numeric(df[tmpl["value_col"]]).abs().max()
    if vmax < 10:  # 全都 < 10 基本说明忘了乘回像素尺寸
        print(f"[warn] 提交值最大值仅 {vmax:.3f}，疑似仍是归一化坐标，请检查坐标换算！")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    return df


def write_preds_csv(samples: Sequence[Sample], final: Dict[str, Coord], stage1: Dict[str, Coord],
                    out_path: Path) -> None:
    rows = []
    for s in samples:
        fx, fy = final[s.id]
        x1, y1 = stage1.get(s.id, (float("nan"), float("nan")))
        rows.append({
            "image": s.id,
            "fovea_x": fx * s.w, "fovea_y": fy * s.h,
            "stage1_x": x1 * s.w, "stage1_y": y1 * s.h,
            "width": s.w, "height": s.h,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.3f")


def save_visualization(samples: Sequence[Sample], final: Dict[str, Coord], stage1: Dict[str, Coord],
                       out_dir: Path, max_width: int = 1000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        im = cv2.imread(str(s.path))
        if im is None:
            continue
        h, w = im.shape[:2]
        sc = max_width / w
        im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        if s.id in stage1:
            x1, y1 = int(stage1[s.id][0] * im.shape[1]), int(stage1[s.id][1] * im.shape[0])
            cv2.circle(im, (x1, y1), 18, (0, 255, 0), 2)
        if s.id in final:
            x, y = int(final[s.id][0] * im.shape[1]), int(final[s.id][1] * im.shape[0])
            cv2.drawMarker(im, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
            cv2.putText(im, f"({final[s.id][0] * s.w:.1f}, {final[s.id][1] * s.h:.1f})",
                        (max(0, x - 90), max(20, y - 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if s.has_gt and s.vis:
            gx, gy = int(s.x / w * im.shape[1]), int(s.y / h * im.shape[0])
            cv2.circle(im, (gx, gy), 8, (255, 0, 0), 2)
        cv2.imwrite(str(out_dir / f"{s.id}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="中心凹定位推理 / 提交生成")
    ap.add_argument("--config", nargs="+", default=None, help="一个或多个 yaml，后者覆盖前者")
    ap.add_argument("--runs", nargs="+", required=True, help="检查点目录或 .pt 文件，可多个（集成）")
    ap.add_argument("--split", choices=["test", "cv", "train"], default="test")
    ap.add_argument("--source", choices=["soft", "head", "mean"], default=None,
                    help="覆盖 coord_source（用于对比不同解码方式）")
    ap.add_argument("--tta", dest="tta", action="store_true", default=None,
                    help="强制开启水平翻转 TTA（默认沿用检查点里的 train.eval_tta）")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--tta-scales", type=str, default=None,
                    help="多尺度 TTA，逗号分隔，如 1.0,1.15,0.85（默认 1.0）")
    ap.add_argument("--tta-shifts", type=str, default=None,
                    help="平移 TTA，分号分隔的 dx,dy，如 0,0;0.02,0;-0.02,0")
    ap.add_argument("--prefer", choices=["auto", "stage2", "stage1"], default="auto",
                    help="两阶段结果取舍：auto=按各自验证集 RMSE 选更好的")
    ap.add_argument("--out", type=str, default=None, help="提交文件路径")
    ap.add_argument("--vis", action="store_true", help="输出预测可视化")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config, args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(cfg["output"]["dir"])

    # TTA 规格：命令行优先，其次配置里的 tta 段
    spec = dict(cfg.get("tta") or {})
    if args.tta is not None:
        spec["flip"] = bool(args.tta)
    if args.tta_scales:
        spec["scales"] = [float(v) for v in args.tta_scales.replace(" ", "").split(",") if v]
    if args.tta_shifts:
        spec["shifts"] = [[float(v) for v in p.split(",")] for p in args.tta_shifts.split(";") if p]
    use_spec = bool(args.tta_scales or args.tta_shifts or args.tta is not None)
    spec = spec if use_spec else None

    cache: Dict[str, ModelRunner] = {}
    chains = [Chain(r, device, cache, args.source, args.tta, args.prefer, spec) for r in args.runs]
    if spec:
        print(f"TTA: flip={spec.get('flip')} scales={spec.get('scales')} shifts={len(spec.get('shifts') or [1])} 组")
    print(f"加载 {len(chains)} 个预测链: " + ", ".join(f"{c.runner.path.name}(fold{c.fold},stage{c.stage})" for c in chains))

    # ---------------- 测试集：生成提交文件 ---------------- #
    if args.split == "test":
        samples = load_test_samples(cfg)
        tmpl = load_submission_template(cfg)
        finals, stage1s = [], []
        for c in chains:
            r = c.predict(samples)
            finals.append(r["final"])
            stage1s.append(r["stage1"])

        final = {s.id: (float(np.mean([f[s.id][0] for f in finals])),
                        float(np.mean([f[s.id][1] for f in finals]))) for s in samples}
        stage1 = {s.id: (float(np.mean([f[s.id][0] for f in stage1s])),
                         float(np.mean([f[s.id][1] for f in stage1s]))) for s in samples}

        out_path = resolve_path(args.out) if args.out else out_dir / "submission.csv"
        write_submission(tmpl, final, samples, out_path)
        write_preds_csv(samples, final, stage1, out_dir / "preds_test.csv")
        save_json(out_dir / "preds_test.json", {
            s.id: {"x": final[s.id][0] * s.w, "y": final[s.id][1] * s.h,
                   "stage1_x": stage1[s.id][0] * s.w, "stage1_y": stage1[s.id][1] * s.h}
            for s in samples
        })
        if args.vis:
            save_visualization(samples, final, stage1, out_dir / "pred_vis")
        print(f"已写出提交文件: {out_path}  ({len(tmpl['keys'])} 行)")
        print(f"逐图预测: {out_dir / 'preds_test.csv'}")
        return

    # ---------------- 验证 / 训练集：只看指标 ---------------- #
    all_samples = load_train_samples(cfg)
    per_run_stage1, per_run_final = [], []
    for c in chains:
        if args.split == "train":
            samples = all_samples
        else:
            cf = c.cfg
            fold = c.fold % int(cf["data"]["n_folds"])
            _, va = split_indices(len(all_samples), int(cf["data"]["n_folds"]), int(cf["seed"]), fold)
            samples = [all_samples[i] for i in va]
        r = c.predict(samples)
        m1 = metrics_vs_samples(samples, r["stage1"])
        m2 = metrics_vs_samples(samples, r["final"])
        print(f"  fold{c.fold} stage{c.stage} | stage1: {format_metrics(m1)}")
        if c.stage == 2:
            used = "stage2" if c._use_stage2() else "stage1"
            print(f"  fold{c.fold} stage{c.stage} | final({used}): {format_metrics(m2)}")
        per_run_stage1.append((samples, r["stage1"]))
        per_run_final.append((samples, r["final"]))

    if args.split == "cv" and len(chains) > 1:
        print("-" * 70)
        print(f"逐模型拼接（含重复样本）stage1: {format_metrics(pool_metrics(per_run_stage1))}")
        print(f"逐模型拼接（含重复样本）final : {format_metrics(pool_metrics(per_run_final))}")

        # 按折分组集成：同一折的多个模型（例如多个 seed）先平均，再汇总。
        # 这是评估"多种子集成"的正确口径——每折只算一次，不会重复计样本。
        groups: Dict[int, list] = {}
        for c in chains:
            groups.setdefault(c.fold, []).append(c)
        if len(groups) == int(cfg["data"]["n_folds"]) and any(len(v) > 1 for v in groups.values()):
            print("-" * 70)
            print(f"折内集成（每折 {len(chains) // len(groups)} 个模型平均）：")
            pairs = []
            for fold in sorted(groups):
                cs = groups[fold]
                cf = cs[0].cfg
                _, va = split_indices(len(all_samples), int(cf["data"]["n_folds"]),
                                      int(cf["seed"]), fold)
                samples = [all_samples[i] for i in va]
                preds = [c.predict(samples)["final"] for c in cs]
                ens = {s.id: (float(np.mean([p[s.id][0] for p in preds])),
                              float(np.mean([p[s.id][1] for p in preds]))) for s in samples}
                m = metrics_vs_samples(samples, ens)
                print(f"  fold{fold} × {len(cs)} 模型: {format_metrics(m)}")
                pairs.append((samples, ens))
            print(f"汇总 折内集成: {format_metrics(pool_metrics(pairs))}")


if __name__ == "__main__":
    main()
