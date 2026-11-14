"""推理引擎：整图前向 + 翻转 TTA + 权重加载。

训练阶段的整图验证与最终测试集推理都调用 :func:`predict_prob`，
避免"验证用一套后处理、提交用另一套"的偏差。
"""

import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model


def multiple_of(cfg) -> int:
    """网络下采样倍数，padding 必须是它的整数倍。"""
    return int(2 ** max(1, cfg.depth))


def pad_to_multiple(x: torch.Tensor, m: int):
    """(B,C,H,W) -> 常量 0 padding 到 H,W 均为 m 的倍数；返回 (padded, (H,W))。"""
    H, W = x.shape[-2:]
    ph = (m - H % m) % m
    pw = (m - W % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
    return x, (H, W)


@torch.no_grad()
def predict_prob(model, image: torch.Tensor, cfg, device=None, tta: bool = None,
                 rgb: np.ndarray = None) -> np.ndarray:
    """对单张整图 (C,H,W) 预测血管概率，返回 (H,W) float32 numpy 数组。

    ``tta=True`` 时对 4 种翻转（原图 / 水平 / 垂直 / 二者）取平均。
    """
    if tta is None:
        tta = getattr(cfg, "tta", False)
    device = device or next(model.parameters()).device
    model.eval()

    H0, W0 = int(image.shape[-2]), int(image.shape[-1])
    sf = float(getattr(cfg, "scale", 1.0) or 1.0)
    mode = getattr(cfg, "preprocess_mode", "legacy")
    if mode not in ("legacy", "rgb_first"):
        raise ValueError(f"未知 preprocess_mode: {mode}")
    if rgb is not None:
        from .preprocess import prepare_rgb
        prepared = prepare_rgb(np.asarray(rgb), cfg, sf if mode == "rgb_first" else 1.0)
        x = torch.from_numpy(prepared).unsqueeze(0).to(device)
    else:
        if mode == "rgb_first" and sf != 1.0:
            raise ValueError("rgb_first 缩放推理必须传入原始 rgb")
        x = image.unsqueeze(0).to(device)

    # "提高分辨率"：训练时按 cfg.scale 放大后切 patch，推理时同样放大再前向，
    # 最后把概率图缩回原尺寸，保证下游（后处理 / 官方转换 / 与标签比对）全部在原尺度。
    if sf != 1.0 and mode == "legacy":
        x = F.interpolate(x, scale_factor=sf, mode="bilinear", align_corners=False)
    x, (H, W) = pad_to_multiple(x, multiple_of(cfg))

    transforms = [(False, False)]
    if tta:
        transforms += [(True, False), (False, True), (True, True)]

    prob = None
    for flip_h, flip_v in transforms:
        xi = x
        if flip_h:
            xi = torch.flip(xi, dims=[3])
        if flip_v:
            xi = torch.flip(xi, dims=[2])
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logit = model(xi)
        p = torch.sigmoid(logit.float())[0, 0]          # (Hp, Wp) 含 padding
        # 注意：必须先在 padded 尺度上翻转回来，再裁剪。
        # 若先裁剪再翻转，padding 造成的内容位移会让结果完全错位。
        if flip_h:
            p = torch.flip(p, dims=[1])
        if flip_v:
            p = torch.flip(p, dims=[0])
        p = p[:H, :W]
        prob = p if prob is None else prob + p

    prob = (prob / len(transforms)).float()
    if sf != 1.0 and (H, W) != (H0, W0):
        prob = F.interpolate(prob[None, None], size=(H0, W0), mode="area")[0, 0]
    return prob.cpu().numpy()


def build_and_load(cfg, device, path=None):
    """按 cfg 构建模型并加载权重（若存在）。"""
    model = build_model(cfg)
    path = path or cfg.checkpoint
    import os
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        extra = {k: v for k, v in ckpt.items() if k not in ("model", "cfg")}
        model.to(device).eval()
        return model, extra
    model.to(device).eval()
    return model, {}
