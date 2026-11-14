"""视盘—黄斑解剖先验诊断。

目的：检验"视盘相对位置"能否作为一个独立的裁决信号。
  1) 在 80 张训练图上用简易视盘检测（亮斑质心）估计 盘->中心凹 的偏移分布；
  2) 看 8 张困难图的跨模型一致性：模型互相接近但都偏离 GT => 标注可疑；
  3) 对测试集不稳定图（0081/0086）用该先验裁决 v5 与 v12 两个候选。

输出: outputs/analysis/disc_prior/*.csv + *_vis.jpg
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "outputs" / "analysis" / "disc_prior"
OUT.mkdir(parents=True, exist_ok=True)


def imread(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _disc_kernel(r_in=6, r_out=15):
    """正中心圆盘 + 负外环 = 与视盘尺寸匹配的带通模板。"""
    size = r_out * 2 + 1
    yy, xx = np.mgrid[0:size, 0:size] - r_out
    rr = np.hypot(xx, yy)
    k = (rr <= r_in).astype(np.float32) - ((rr > r_in) & (rr <= r_out)).astype(np.float32) * (
        (r_in ** 2) / (r_out ** 2 - r_in ** 2))
    return k - k.mean()


_DK = _disc_kernel()


def detect_disc(bgr):
    """视盘粗定位：绿色通道模糊后与圆盘模板做相关。返回归一化 (x, y) 与响应比。"""
    h, w = bgr.shape[:2]
    s = 16
    small = cv2.resize(bgr, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    g = small[:, :, 1].astype(np.float32)
    gb = cv2.GaussianBlur(g, (0, 0), 1.5)
    resp = cv2.filter2D(gb, -1, _DK)
    iy, ix = np.unravel_index(int(np.argmax(resp)), resp.shape)
    peak = float(resp[iy, ix])
    # 峰值附近取响应加权的质心，抵消网格量化
    r = 6
    y0, y1 = max(0, iy - r), min(resp.shape[0], iy + r + 1)
    x0, x1 = max(0, ix - r), min(resp.shape[1], ix + r + 1)
    win = np.clip(resp[y0:y1, x0:x1], 0, None)
    tot = win.sum()
    if tot <= 1e-6:
        return None, 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx = float((win * xs).sum() / tot) / resp.shape[1]
    cy = float((win * ys).sum() / tot) / resp.shape[0]
    return (cx, cy), peak


def load_gt():
    df = pd.read_csv("detection/fovea_localization_train_GT.csv")
    df.columns = [c.strip() for c in df.columns]
    def _id(v):
        s = str(v).strip()
        return f"{int(float(s)):04d}" if s.replace(".", "").isdigit() else s.split(".")[0]
    return {_id(r["data"]): (float(r["Fovea_X"]), float(r["Fovea_Y"])) for _, r in df.iterrows()}


def main():
    gt = load_gt()
    rows = []
    for img_id, (gx, gy) in sorted(gt.items()):
        p = ROOT / "detection" / "train" / f"{img_id}.jpg"
        bgr = imread(p)
        h, w = bgr.shape[:2]
        disc, conf = detect_disc(bgr)
        if disc is None:
            continue
        dx, dy = disc[0] * w, disc[1] * h
        rows.append(dict(image=img_id, w=w, h=h,
                         disc_x=dx, disc_y=dy, disc_conf=conf,
                         gt_x=gx, gt_y=gy,
                         off_x=(gx - dx) / w, off_y=(gy - dy) / h,
                         dist_px=float(np.hypot(gx - dx, gy - dy)),
                         angle_deg=float(np.degrees(np.arctan2(gy - dy, gx - dx)))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "train_disc_prior.csv", index=False)

    print(f"训练集 {len(df)} 张，视盘响应峰值 conf: "
          f"median={df.disc_conf.median():.1f} p10={df.disc_conf.quantile(.1):.1f}")
    print(f"盘->凹 归一化偏移: x {df.off_x.mean():+.4f}±{df.off_x.std():.4f}  "
          f"y {df.off_y.mean():+.4f}±{df.off_y.std():.4f}")
    print(f"欧氏距离(原图像素): median={df.dist_px.median():.0f} "
          f"p10={df.dist_px.quantile(.1):.0f} p90={df.dist_px.quantile(.9):.0f}")
    print(f"方向角: median={df.angle_deg.median():.1f}°  "
          f"落在 [150°,210°] 的比例={((df.angle_deg.abs() >= 150) & (df.angle_deg.abs() <= 210)).mean():.1%}")

    # 用 (均值, 协方差) 计算马氏距离，量化"该先验能多准地预测中心凹"
    mu = df[["off_x", "off_y"]].values.mean(0)
    cov = np.cov(df[["off_x", "off_y"]].values.T)
    err = np.hypot(df.off_x - mu[0], df.off_y - mu[1]) * df.w  # 用均值偏移预测时的误差(px)
    print(f"\n仅用\"均值偏移\"预测中心凹的误差: median={err.median():.0f}px "
          f"p75={err.quantile(.75):.0f}px p90={err.quantile(.9):.0f}px")
    print(f"（对比：常数基线 110px，当前模型 CV 中位数 9px —— 先验单独用不够，但作为裁决器足够）")

    # 困难图与最佳模型的一致性
    hard = ["0049", "0072", "0016", "0042", "0070", "0034", "0028", "0024"]
    oof = {}
    for tag, path in [("v5_s42", "outputs/experiments/v5_20261024/baseline_s42/oof.csv"),
                      ("v12_s43", "outputs/experiments/v12_seed_replication/mass_kl_s43/oof.csv"),
                      ("base_s43", "outputs/experiments/v12_seed_replication/baseline_s43/oof.csv")]:
        if Path(path).exists():
            oof[tag] = pd.read_csv(path, dtype={"image": str}).assign(
                image=lambda d: d.image.str.zfill(4)).set_index("image")
    print("\n=== 8 张困难图：模型间一致性 vs 与 GT 的距离 ===")
    rec = []
    for i in hard:
        if i not in df.image.values:
            continue
        r = df[df.image == i].iloc[0]
        preds = {t: (o.loc[i, "x"], o.loc[i, "y"]) for t, o in oof.items() if i in o.index}
        pts = np.array(list(preds.values()))
        spread = float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean())) if len(pts) > 1 else 0.0
        gtp = np.array([r.gt_x, r.gt_y])
        d_gt = float(np.hypot(*(pts.mean(0) - gtp)))
        prior = np.array([r.disc_x + mu[0] * r.w, r.disc_y + mu[1] * r.h])
        d_prior = float(np.hypot(*(pts.mean(0) - prior)))
        rec.append(dict(image=i, mean_pred_vs_gt=d_gt, model_spread=spread,
                        mean_pred_vs_discprior=d_prior,
                        gt_vs_discprior=float(np.hypot(*(gtp - prior))),
                        **{f"pred_{k}": f"{v[0]:.0f},{v[1]:.0f}" for k, v in preds.items()}))
    hdf = pd.DataFrame(rec)
    hdf.to_csv(OUT / "hard8_agreement.csv", index=False)
    print(hdf.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n判读: model_spread 远小于 mean_pred_vs_gt  => 模型彼此一致但与标注不符（标注可疑或系统性偏移）")


if __name__ == "__main__":
    main()
