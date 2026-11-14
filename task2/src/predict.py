"""推理与提交：多成员加权融合、翻转 TTA、竞赛编号映射、提交校验与 SHA256 绑定。

示例：
    python -m src.predict --runs outputs/runs/r2_r18 outputs/runs/r2_convnext outputs/runs/r2_b0 \
        --weights 0.25 0.50 0.25 --out submission_round2.csv

    # 在开发验证集上评估同一融合（不写提交）
    python -m src.predict --runs ... --weights ... --split val --save-probs outputs/ens_val.npz

    # 部署权重（全数据微调后的 deploy.pt）
    python -m src.predict --runs outputs/runs/r12 --ckpt deploy.pt --out submission_round12_full.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from . import CLASSES, IDX_TO_COMP, NUM_CLASSES, REPORTS, ROOT
from .config import load_cfg
from .data import build_transforms, list_test, prepare_data
from .metrics import summarize
from .models import build_model, predict_proba


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_member(run: str | Path, ckpt: str = "best.pt") -> Tuple[torch.nn.Module, dict]:
    path = Path(run)
    if path.is_file():
        ck_path = path
    else:
        ck_path = path / ckpt
    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or load_cfg(None)
    model, info = build_model(cfg)
    sd = ck["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ck_path} 与模型不匹配：missing={missing[:4]} unexpected={unexpected[:4]}")
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, cfg


@torch.no_grad()
def predict_paths(model: torch.nn.Module, paths: Sequence[Path], cfg: dict,
                  batch_size: int = 512, flip_tta: bool = True) -> np.ndarray:
    device = next(model.parameters()).device
    tf = build_transforms(cfg, train=False)
    out: List[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        xs = []
        for p in chunk:
            with Image.open(p) as im:
                xs.append(tf(im.convert("L")))
        x = torch.stack(xs).to(device)
        out.append(predict_proba(model, x, flip_tta=flip_tta).float().cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"  predict {min(len(paths), i + batch_size)}/{len(paths)}", flush=True)
    return np.concatenate(out, axis=0)


def fuse(members: List[Tuple[torch.nn.Module, dict]], paths: Sequence[Path], weights: Sequence[float],
         flip_tta: bool = True) -> np.ndarray:
    if len(weights) != len(members):
        raise ValueError(f"权重数 {len(weights)} 与成员数 {len(members)} 不一致")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    prob = None
    for (model, cfg), wi in zip(members, w):
        p = predict_paths(model, paths, cfg, flip_tta=flip_tta)
        prob = p * wi if prob is None else prob + p * wi
    return prob


def write_submission(ids: Sequence[str], prob: np.ndarray, out_path: str | Path,
                     template: str | Path | None = None) -> Dict:
    """按 sample_submission 的 ID 顺序写出竞赛编号；写后立即校验。

    提交 ID 是"文件名.jpg"这类无规律字符串，不能假定与测试目录的排序一致，
    因此只要模板存在，就按模板的 ID 顺序重排后再写。
    """
    label_by_id = {i: int(IDX_TO_COMP[k]) for i, k in zip(ids, prob.argmax(1).tolist())}
    order = list(ids)
    if template and Path(template).exists():
        with open(template, "r", encoding="utf-8", newline="") as f:
            order = [r[0] for r in list(csv.reader(f))[1:]]
        missing = [i for i in order if i not in label_by_id]
        if missing:
            raise ValueError(f"{len(missing)} 个模板 ID 没有对应预测，例如 {missing[:3]}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["ID", "Emotion"])
        for i in order:
            wr.writerow([i, label_by_id[i]])
    rep = validate_submission(out_path, template)
    rep["sha256"] = sha256_file(out_path)
    rep["file"] = str(out_path)
    vals = np.array([label_by_id[i] for i in order])
    rep["class_distribution"] = {CLASSES[k]: int((vals == IDX_TO_COMP[k]).sum()) for k in range(NUM_CLASSES)}
    return rep


def validate_submission(path: str | Path, template: str | Path | None = None) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    errors: List[str] = []
    if header != ["ID", "Emotion"]:
        errors.append(f"表头应为 ['ID','Emotion']，实际 {header}")
    ids = [r[0] for r in body]
    if len(set(ids)) != len(ids):
        errors.append("存在重复 ID")
    labels = []
    for r in body:
        try:
            v = int(r[1])
        except (ValueError, IndexError):
            errors.append(f"非法标签行：{r}")
            continue
        labels.append(v)
        if not 0 <= v < NUM_CLASSES:
            errors.append(f"标签越界：{v}")
    expected = 7065
    if template:
        with open(template, "r", encoding="utf-8", newline="") as f:
            trows = list(csv.reader(f))[1:]
        expected = len(trows)
        tid = [r[0] for r in trows]
        if ids != tid:
            errors.append("ID 顺序与 sample_submission 不一致")
    if len(body) != expected:
        errors.append(f"行数 {len(body)} 与期望 {expected} 不一致")
    return {"ok": not errors, "errors": errors[:20], "n_rows": len(body),
            "n_classes_used": len(set(labels))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="推理/融合/提交")
    ap.add_argument("--runs", nargs="+", required=True, help="run 目录或 .pt 文件，可多个")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--ckpt", default="best.pt", help="run 目录内读取的检查点文件名")
    ap.add_argument("--split", default="test", choices=["test", "val", "holdout", "train"])
    ap.add_argument("--out", default=None, help="提交 CSV 路径")
    ap.add_argument("--template", default="submission.csv", help="sample_submission 路径（用于校验 ID 顺序）")
    ap.add_argument("--save-probs", default=None, help="把概率/标签存成 npz，供融合搜索")
    ap.add_argument("--no-tta-flip", action="store_true")
    args = ap.parse_args(argv)

    members = [load_member(r, args.ckpt) for r in args.runs]
    weights = args.weights or [1.0] * len(members)
    flip = not args.no_tta_flip

    if args.split == "test":
        test_paths = list_test(ROOT / "fer_data" / "fer_data")
        ids = [p.name for p in test_paths]  # 提交 ID 带 .jpg 后缀，与 sample_submission 一致
        prob = fuse(members, test_paths, weights, flip)
        out = args.out or "submission_fusion.csv"
        rep = write_submission(ids, prob, out, args.template if Path(args.template).exists() else None)
    else:
        cfg = members[0][1]
        samples, groups, splits, _w, _meta = prepare_data(cfg)
        idx = splits[args.split]
        paths = [samples[i][0] for i in idx]
        y = np.array([samples[i][1] for i in idx])
        prob = fuse(members, paths, weights, flip)
        rep = summarize(y, prob.argmax(1))
        rep.pop("confusion", None)
        rep["weights"] = list(weights)
        if args.save_probs:
            np.savez_compressed(args.save_probs, probs=prob, labels=y,
                                groups=np.array([groups[i] for i in idx]))
        print(f"[{args.split}] acc={rep['accuracy']:.4f} macro_f1={rep['macro_f1']:.4f} n={rep['n']}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / (Path(args.out).stem + "_check.json" if args.out else "predict_check.json")).write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_class"}, ensure_ascii=False, indent=2))
    if not rep.get("ok", True):
        print("提交校验失败：", rep["errors"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
