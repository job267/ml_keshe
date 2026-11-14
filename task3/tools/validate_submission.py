"""校验提交文件格式（与 sample_submission.csv 逐行比对）。

    python tools/validate_submission.py outputs/submission.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

from src.config import load_cfg
from src.data import load_submission_template
from src.utils import resolve_path


def main() -> None:
    sub_path = resolve_path(sys.argv[1] if len(sys.argv) > 1 else "outputs/submission.csv")
    cfg = load_cfg("configs/stage1.yaml")
    tmpl = load_submission_template(cfg)

    sub = pd.read_csv(sub_path)
    print(f"文件: {sub_path}")
    print(f"形状: {sub.shape} | 列: {list(sub.columns)}")

    ok = True
    if list(sub.columns) != tmpl["columns"]:
        print(f"[x] 列名不匹配，期望 {tmpl['columns']}")
        ok = False
    if len(sub) != len(tmpl["keys"]):
        print(f"[x] 行数不匹配，期望 {len(tmpl['keys'])} 实际 {len(sub)}")
        ok = False

    keys = sub[tmpl["image_col"]].astype(str).tolist() if len(sub) else []
    if keys != tmpl["keys"]:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(keys, tmpl["keys"])) if a != b][:5]
        print(f"[x] ImageID 顺序与模板不一致（可能已按字典序排序），前几处差异: {diff}")
        ok = False

    vals = pd.to_numeric(sub[tmpl["value_col"]], errors="coerce")
    if vals.isna().any():
        print(f"[x] 存在非数值/缺失值，共 {int(vals.isna().sum())} 个")
        ok = False

    # 与测试图像尺寸比对，检查是否越界
    root = resolve_path(cfg["data"]["root"])
    dims = {}
    for sid, _ in tmpl["rows"]:
        if sid not in dims:
            with Image.open(root / "test" / f"{sid}.jpg") as im:
                dims[sid] = im.size

    xs, ys, bad = [], [], 0
    for v, (sid, coord) in zip(vals.tolist(), tmpl["rows"]):
        w, h = dims[sid]
        limit = w if coord == "X" else h
        if not (-1e-6 <= float(v) <= limit + 1e-6):
            bad += 1
            print(f"[!] {sid}_Fovea_{coord} = {float(v):.2f} 超出图像范围 [0, {limit}]")
        (xs if coord == "X" else ys).append(float(v))

    x, y = np.array(xs), np.array(ys)
    print(f"X: mean={x.mean():.2f} min={x.min():.2f} max={x.max():.2f}")
    print(f"Y: mean={y.mean():.2f} min={y.min():.2f} max={y.max():.2f}")
    print(f"零值对 (0,0) 的数量: {int(((x == 0) & (y == 0)).sum())}")
    if bad:
        print(f"[!] 共 {bad} 个值越界（(0,0) 预测也可能触发越界提示，属正常）")
    print("-" * 60)
    print("格式检查通过" if ok else "格式检查未通过")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
