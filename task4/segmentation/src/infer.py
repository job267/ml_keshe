"""测试集推理入口：生成血管掩膜 PNG。

输出格式（与官方 ``segmentation_to_csv.py`` 的约定一致）：

* 文件名与测试图同名（``1.png`` ... ``20.png``），Id 取文件名去掉扩展名；
* 尺寸为 **原图尺寸** 565x584，像素值 0（背景）/ 255（血管）；
* 保存为 PNG（无损），避免 JPG 压缩噪点被 ``>0`` 阈值误判为血管。

为什么保存原图尺寸？官方脚本会把掩膜 ``resize`` 到 512x512 后取 ``>0``，
这与生成标签 RLE 的处理完全一致，因此交给它做统一缩放最不容易出现偏差。

用法::

    python -m src.infer                          # 默认加载 outputs/best.pt
    python -m src.infer --checkpoint outputs/last.pt --no-tta
    python -m src.infer --threshold 0.45 --min_size 12 --save_prob
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ImageDataset
from .engine import predict_prob
from .models import load_checkpoint
from .postprocess import postprocess
from .utils import Timer, ensure_dir, get_device, imwrite, list_ids, set_seed


def build_infer_parser():
    p = argparse.ArgumentParser(description="测试集推理：生成掩膜 PNG")
    p.add_argument("--checkpoint", type=str, default="outputs/best.pt",
                   help="权重路径")
    p.add_argument("--test_dir", type=str, default=None, help="测试图像目录，默认取权重中的配置")
    p.add_argument("--ids", type=str, default="", help="只推理指定 id（逗号分隔），留空=全部")
    p.add_argument("--pred_dir", type=str, default=None, help="掩膜输出目录（默认 image）")
    # 后处理超参属于"推理配方"，默认取当前 Config（已在验证集上调好），而不是训练时存下的旧值
    _d = Config()
    p.add_argument("--threshold", type=float, default=_d.threshold, help="二值化阈值")
    p.add_argument("--min_size", type=int, default=_d.min_size, help="小连通域剔除阈值")
    p.add_argument("--use_fov", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                   nargs="?", const=True, default=_d.use_fov, help="是否使用视场掩膜")
    p.add_argument("--fill_holes", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                   nargs="?", const=True, default=_d.fill_holes, help="是否填洞")
    p.add_argument("--tta", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                   nargs="?", const=True, default=_d.tta, help="是否使用翻转 TTA")
    p.add_argument("--save_prob", action="store_true", help="同时保存概率图 npy")
    p.add_argument("--device", type=str, default="auto")
    return p


def main(argv=None):
    args = build_infer_parser().parse_args(argv)
    device = get_device(args.device)

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"找不到权重文件: {args.checkpoint}，请先运行 python -m src.train")

    model, cfg, extra = load_checkpoint(args.checkpoint, device=device)
    # 用命令行覆盖推理相关配置（网络结构相关参数必须沿用训练时的配置）
    if args.test_dir is not None:
        cfg.test_image_dir = args.test_dir
    if args.pred_dir is not None:
        cfg.pred_dir = args.pred_dir
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.min_size is not None:
        cfg.min_size = args.min_size
    if args.use_fov is not None:
        cfg.use_fov = args.use_fov
    if args.fill_holes is not None:
        cfg.fill_holes = args.fill_holes
    if args.tta is not None:
        cfg.tta = args.tta
    # 后处理参数始终以本次推理的取值为准（不沿用训练时记录的值）
    cfg.threshold = args.threshold
    cfg.min_size = args.min_size
    cfg.use_fov = bool(args.use_fov)
    cfg.fill_holes = bool(args.fill_holes)
    cfg.tta = bool(args.tta)

    set_seed(cfg.seed)
    ensure_dir(cfg.pred_dir)
    if args.save_prob:
        ensure_dir(os.path.join(cfg.pred_dir, "prob"))

    ids = list_ids(cfg.test_image_dir)
    if args.ids:
        keep = [i.strip() for i in args.ids.split(",") if i.strip()]
        ids = [i for i in ids if i in keep]
    if not ids:
        raise SystemExit(f"测试目录为空: {cfg.test_image_dir}")

    ds = ImageDataset(cfg, ids, with_label=False,
                      image_dir=cfg.test_image_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    print("=" * 78)
    print(f"权重      : {args.checkpoint}")
    print(f"测试图像  : {cfg.test_image_dir}  ({len(ids)} 张)")
    print(f"输出目录  : {cfg.pred_dir}")
    print(f"阈值/最小面积/FOV/TTA : {cfg.threshold} / {cfg.min_size} / "
          f"{cfg.use_fov} / {cfg.tta}")
    print("=" * 78)

    timer = Timer()
    ratios = []
    for batch in loader:
        i = batch["id"][0]
        image = batch["image"][0]
        fov = batch["fov"][0].numpy().astype(bool)

        prob = predict_prob(model, image, cfg, device=device, tta=cfg.tta,
                            rgb=batch["rgb"][0].numpy())
        mask = postprocess(prob, fov if cfg.use_fov else None,
                           threshold=cfg.threshold, min_size=cfg.min_size,
                           do_fill_holes=cfg.fill_holes)

        out_png = os.path.join(cfg.pred_dir, f"{i}.png")
        out_mask = (~mask) if getattr(cfg, "invert_output", False) else mask
        imwrite(out_png, (out_mask.astype(np.uint8) * 255))
        if args.save_prob:
            np.save(os.path.join(cfg.pred_dir, "prob", f"{i}.npy"), prob.astype(np.float16))

        ratios.append(mask.mean())
        print(f"  {i:>3s}  mask 血管占比 {mask.mean()*100:5.2f}%  -> {out_png}")

    print("-" * 78)
    print(f"完成 {len(ids)} 张，平均血管占比 {np.mean(ratios)*100:.2f}%，耗时 {timer}")
    print("下一步：python segmentation_to_csv.py   (或 python -m src.rle)")
    print("=" * 78)


if __name__ == "__main__":
    main()
