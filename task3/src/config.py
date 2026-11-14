"""配置系统：默认配置 + YAML 覆盖 + 命令行 `--set a.b=value` 覆盖。

用法::

    python -m src.train --config configs/stage1.yaml --set train.epochs=60
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import yaml

from .utils import resolve_path

# fmt: off
DEFAULT_CFG: Dict[str, Any] = {
    "seed": 42,
    "task": "stage1",                       # stage1 | stage2

    "data": {
        "root": "detection",                # 数据根目录
        "gt_csv": "detection/fovea_localization_train_GT.csv",
        "sample_submission": "detection/sample_submission.csv",
        "img_size": [768, 512],             # stage1 整图输入尺寸 (W, H)
        "n_folds": 5,                       # K 折交叉验证
        "fold": 0,
        "num_workers": 4,
        "cache_mb": 300,                    # 每进程图像缓存上限
        "decode_scale": 2,                  # 1=原始分辨率, 2=1/2, 4=1/4（更快）
    },

    "aug": {
        "hflip": 0.5,                       # 水平翻转概率（眼底图左右眼互为镜像，安全可用）
        "rotate": 8.0,                      # 旋转角度 ±deg
        "scale": [0.92, 1.08],              # 缩放范围
        "translate": 0.03,                  # 平移幅度（占图像比例）
        "brightness": 0.15,
        "contrast": 0.15,
        "gamma": [0.8, 1.25],
        "hue": 0.03,                        # HSV 色相 ±(占整圈比例)；0 关闭
        "saturation": [0.7, 1.3],           # HSV 饱和度缩放范围
        "channel_gain": 0.10,               # RGB 每通道独立对数增益 σ（模拟白平衡差异）
        "illumination_p": 0.25,             # 低频光照梯度增强概率
        "clahe_p": 0.20,                    # CLAHE（局部对比度）概率
        "lab_shift": 0.0,                   # Lab 各通道均值平移强度（0 关闭；强增强建议 3.0）
        "lab_scale": [1.0, 1.0],            # Lab 各通道标准差缩放（0 关闭；强增强建议 [0.85, 1.15]）
        "gray_p": 0.0,                      # 随机转灰度（中心凹是亮度特征，安全；强增强建议 0.1）
        "channel_dropout_p": 0.0,           # 随机丢弃一个 RGB 通道
        "channel_shuffle_p": 0.0,           # 随机打乱 RGB 通道顺序
        "blur_p": 0.15,
        "noise_p": 0.15,
    },

    "model": {
        "arch": "fpn",                      # fpn（多尺度融合）| dilated（无 FPN，空洞卷积保持高分辨率）
        "backbone": "resnet18",             # resnet18 | resnet34 | resnet50 | resnet101
        "pretrained": True,
        "in_ch": 3,                         # 输入通道；stage2.use_context=True 时自动置 6
        "out_stride": 4,                    # 热力图相对输入的下采样倍率（只支持 2 / 4）
        "hidden": 128,                      # FPN 特征通道
        "use_coord_head": True,             # 额外全局回归头（与 soft-argmax 联合监督）
        "use_vis_head": False,              # 可见性分类头（训练集中没有 (0,0) 样本时关掉）
        "coord_source": "soft",             # 最终坐标来源: soft | head | mean
        "dropout": 0.0,
    },

    "train": {
        "run_seed": None,                   # 训练随机种子；None = 用顶层 seed。多 seed 时用它区分
        "epochs": 120,
        "batch_size": 8,
        "lr": 3.0e-4,
        "backbone_lr_scale": 0.3,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 5,
        "amp": True,
        "grad_clip": 1.0,
        "ema_decay": 0.99,
        "patience": 40,
        "eval_tta": False,                  # 验证时是否使用水平翻转 TTA
        "save_raw": False,                  # 是否把未做 EMA 的权重也存进检查点（文件会翻倍）
        "save_last_every": 10,              # 每 N 个 epoch 存一次 last.pt
        "limit_batches": 0,                 # >0 时只跑少量 batch（冒烟测试）
    },

    "loss": {
        "heatmap": "mse",                   # mse | bce
        "w_heatmap": 1.0,
        "w_soft": 1.0,                      # 对 soft-argmax 坐标的监督权重
        "w_coord": 0.5,                     # 对 coord_head 的监督权重
        "w_vis": 1.0,
        "coord": "smoothl1",                # smoothl1 | wing | l1
        "smoothl1_beta": 0.01,              # 归一化坐标单位下 ~= 30px
        "wing_w": 0.05,
        "wing_e": 0.1,
    },

    "heatmap": {
        "sigma": 2.0,                       # 高斯标签标准差（热力图网格单位）
        "beta": 1.0,                        # soft-argmax 温度
    },

    "tta": {
        "flip": True,                       # 水平翻转 TTA
        "scales": [1.0],                    # 多尺度 TTA，如 [1.0, 1.15, 0.87]
        "shifts": [[0.0, 0.0]],             # 平移 TTA（占图像比例），如 [[0,0],[0.02,0],[-0.02,0]]
    },

    "stage2": {
        "patch_size": 0.32,                 # 裁切框边长（占图像宽/高的比例）
        "patch_resize": 384,                # patch 输入网络的边长
        "jitter_sigma": 0.05,               # 训练时裁切中心相对 GT 的抖动
        "big_jitter_prob": 0.2,             # 小概率大体抖动，模拟 stage1 大误差
        "big_jitter_sigma": 0.12,
        "use_context": False,               # True 时 patch 额外拼一张整图缩略图（6 通道输入）
        "s1_ckpt": None,                    # stage1 检查点路径（由 --s1-ckpt 传入）
    },

    "output": {"dir": "outputs"},
}
# fmt: on


def _deep_update(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_value(s: str) -> Any:
    for parser in (json.loads, lambda x: json.loads(x.replace("'", '"'))):
        try:
            return parser(s)
        except Exception:
            pass
    return s


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式，收到: {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = _parse_value(raw.strip())
    return cfg


def load_cfg(
    path: Optional[str | Path | Iterable[str | Path]] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """读取配置：默认值 <- yaml(可多个，按顺序叠加) <- 命令行覆盖。

    传入多个 yaml 时后者覆盖前者，方便做实验：
        --config configs/stage1.yaml configs/v4/stage1_strongaug.yaml
    """
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path is not None:
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        for p in paths:
            fp = resolve_path(p)
            if not fp.exists():
                raise FileNotFoundError(f"找不到配置文件: {fp}")
            with fp.open("r", encoding="utf-8") as f:
                _deep_update(cfg, yaml.safe_load(f) or {})
    return apply_overrides(cfg, overrides)
