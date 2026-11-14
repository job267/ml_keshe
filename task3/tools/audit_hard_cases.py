"""逐图目视复核：把高分歧训练图的 GT 与三个模型预测画在一起。

仅用于判断"分歧大"的图到底是模型错还是标注错——这是 CV 与 LB 差距归因的关键证据。
输出 outputs/analysis/tail_risk/audit_hard8.jpg
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HARD = ["0049", "0072", "0016", "0042", "0070", "0034", "0028", "0024"]
MEMBERS = {
    "v5": ("outputs/experiments/v5_20261024/baseline_s42/oof.csv", (0, 200, 255)),      # 橙
    "b43": ("outputs/experiments/v12_seed_replication/baseline_s43/oof.csv", (255, 0, 255)),  # 品红
    "kl43": ("outputs/experiments/v12_seed_replication/mass_kl_s43/oof.csv", (255, 255, 0)),  # 青
}
OUT = Path("outputs/analysis/tail_risk")
OUT.mkdir(parents=True, exist_ok=True)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    preds = {}
    for k, (p, _) in MEMBERS.items():
        d = pd.read_csv(p, dtype={"image": str})
        d["image"] = d.image.str.zfill(4)
        preds[k] = d.set_index("image")

    tiles = []
    for img_id in HARD:
        bgr = imread(f"detection/train/{img_id}.jpg")
        h, w = bgr.shape[:2]
        ref = preds["v5"].loc[img_id]
        gt = np.array([float(ref.gt_x), float(ref.gt_y)])
        pts = {k: np.array([float(p.loc[img_id, "x"]), float(p.loc[img_id, "y"])]) for k, p in preds.items()}
        center = np.mean([gt] + list(pts.values()), axis=0)
        half = 260
        cx, cy = int(center[0]), int(center[1])
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, x0 + 2 * half), min(h, y0 + 2 * half)
        tile = bgr[y0:y1, x0:x1].copy()
        sc = 560 / tile.shape[1]
        tile = cv2.resize(tile, (560, int(tile.shape[0] * sc)), interpolation=cv2.INTER_AREA)

        def to_tile(pt):
            return (int((pt[0] - x0) * sc), int((pt[1] - y0) * sc))

        # GT：红色十字 + 圆环
        g = to_tile(gt)
        cv2.drawMarker(tile, g, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.circle(tile, g, 16, (0, 0, 255), 1)
        for k, (_, color) in MEMBERS.items():
            cv2.drawMarker(tile, to_tile(pts[k]), color, cv2.MARKER_TILTED_CROSS, 14, 1)
            cv2.line(tile, g, to_tile(pts[k]), color, 1, cv2.LINE_AA)
        err = float(np.hypot(*(np.mean(list(pts.values()), 0) - gt)))
        cv2.putText(tile, f"{img_id}  err={err:.0f}px", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, f"{img_id}  err={err:.0f}px", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(tile, "GT=red  v5=orange  b43=magenta  kl43=cyan", (8, tile.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, "GT=red  v5=orange  b43=magenta  kl43=cyan", (8, tile.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(tile)

    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    canvas = np.vstack(rows)
    cv2.imwrite(str(OUT / "audit_hard8.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("saved", OUT / "audit_hard8.jpg", canvas.shape)


if __name__ == "__main__":
    main()
