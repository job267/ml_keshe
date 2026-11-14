"""环境与数据自检：跑训练前先确认依赖、数据、标签约定都没问题。

    python scripts/check_env.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def main():
    print("=" * 70)
    print("Python :", sys.version.split()[0], sys.executable)

    ok = True
    for mod in ("numpy", "pandas", "PIL", "cv2", "skimage", "torch", "torchvision"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12} {getattr(m, '__version__', '?'):<12} OK")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {mod:<12} 缺失 -> {e}")

    try:
        import torch
        print(f"  CUDA        {torch.cuda.is_available()}"
              + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    except Exception:
        pass
    print("=" * 70)

    # ---------------- 数据检查 ---------------- #
    from src.utils import fov_mask, imread_gray, imread_rgb, list_ids

    tr_img = os.path.join(ROOT, "train", "image")
    tr_lab = os.path.join(ROOT, "train", "label")
    te_img = os.path.join(ROOT, "test", "image")

    ids = list_ids(tr_img)
    print(f"训练图 {len(ids)} 张: {ids}")
    print(f"测试图 {len(list_ids(te_img))} 张")

    ratios, fovs, shapes = [], [], set()
    values = set()
    for i in ids:
        rgb = imread_rgb(os.path.join(tr_img, f"{i}.jpg"))
        lab = imread_gray(os.path.join(tr_lab, f"{i}.jpg"))
        shapes.add(rgb.shape[:2])
        values.update(np.unique(lab).tolist())
        ratios.append(float((lab > 127).mean()))
        fovs.append(float(fov_mask(rgb).mean()))

    print(f"图像尺寸      : {sorted(shapes)}")
    print(f"标签取值      : {sorted(values)}  -> "
          f"{'血管=255（与官方脚本一致）' if 255 in values and 0 in values else '无法判断'}")
    print(f"血管像素占比  : {np.mean(ratios)*100:.2f}%  (min {min(ratios)*100:.2f}% / "
          f"max {max(ratios)*100:.2f}%)")
    print(f"FOV 占比      : {np.mean(fovs)*100:.2f}%")

    need = {"train/image", "train/label", "test/image", "segmentation_to_csv.py"}
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    print("-" * 70)
    if missing:
        ok = False
        print("缺少必需路径:", missing)
    else:
        print("数据目录完整：train/image, train/label, test/image, segmentation_to_csv.py")

    # ---------------- 外部数据（可选，第二轮迭代用） ---------------- #
    ext = os.path.join(ROOT, "data_external")
    ready = []
    for name in ("stare", "chase", "hrf", "fives640", "fives"):
        d = os.path.join(ext, name, "image")
        if os.path.isdir(d):
            ready.append(f"{name}={len(list_ids(d))}")
    if ready:
        print("外部数据      : " + "  ".join(ready) + "  （可用于 src.pretrain）")
    else:
        print("外部数据      : 未准备（可选）-> python scripts/download_external.py "
              "&& python scripts/prepare_external.py")

    print("自检", "通过" if ok else "未通过", "—— 可以运行 python -m src.train")
    print("=" * 70)


if __name__ == "__main__":
    main()
