"""配置系统：默认值 + YAML 覆盖 + 命令行 --set 覆盖。

用法：
    from src.config import load_cfg
    cfg = load_cfg("configs/round12_erasing.yaml", ["train.epochs=12", "model.pretrained=false"])

--set 的值按 YAML 解析（0.25 / true / 12 / [1, 2] / null），因此 `--set model.dropout=0.3`
得到的仍是 float，而不是字符串。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

DEFAULT_CFG: dict[str, Any] = {
    "exp": {"name": "round1", "seed": 42, "notes": ""},
    "data": {
        "root": "fer_data/fer_data",
        "img_size": 48,
        "val_size": 2848,
        "holdout_size": 2836,
        "num_workers": 4,
        "cache_dup_index": True,
        "grayscale_to_rgb": False,
    },
    "aug": {
        # 几何增强强度：1.0 表示报告里"标准"配置，0.35 对应第六轮的弱增强
        "strength": 1.0,
        "rotation": 12.0,
        "scale": [0.90, 1.00],
        "hflip": 0.5,
        "brightness": 0.20,
        "contrast": 0.20,
        "erasing": {
            "enabled": False,   # 第十二轮启用
            "prob": 0.25,
            "area": [0.02, 0.12],
            "ratio": [0.5, 2.0],
        },
    },
    "model": {
        "name": "ferresnet",        # ferresnet | resnet18 | convnext_tiny | efficientnet_b0 | rexnet_150
        "pretrained": False,
        "pretrained_path": None,    # 人脸领域权重（VGGFace2）等本地检查点
        "dropout": 0.25,
        "head_lr_mult": 1.0,
        "se": False,                # 通道重标定
        "cbam": False,              # 池化前的残差 CBAM（第五轮 C）
        "stn": False,               # 有界空间变换（第十三轮）
        "width_mult": 1.0,
        "in_ch": 1,
    },
    "train": {
        "epochs": 35,
        "batch_size": 256,
        "lr": 0.003,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
        "label_smoothing": 0.0,
        "mixup_prob": 0.0,
        "mixup_alpha": 0.2,
        "ema_decay": 0.0,           # 0 表示关闭
        "amp": True,
        "grad_clip": 5.0,
        "class_weights": False,     # 类别加权（第八轮 B 取消）
        "quality_weights": False,   # 数据质量权重（第五轮 B / 第八轮 C）
        "quality_mode": "audit",    # audit | conservative_oof
        "gce_q": 0.0,               # >0 时用广义交叉熵替代交叉熵（第七轮 C）
        "refit_epochs": 8,          # 全数据微调轮次（第十四轮结论：8 轮）
        "eval_tta_flip": True,
        "log_every": 50,
    },
    "eval": {"fp32": True, "batch_size": 512},
}


def deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(dict(out[k]), v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str):
    try:
        val = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    return text if val is None and text.strip() not in ("null", "~") else val


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式，收到：{item!r}")
        key, raw = item.split("=", 1)
        val = _parse_scalar(raw)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = val
    return out


def load_cfg(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def save_cfg(cfg: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
