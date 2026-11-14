"""模型：小型残差基线、注意力变体、迁移学习骨干、有界空间变换与 CBAM 残差。

对应报告第 3 节：
- FERResNet：3×3 输入卷积 + 四个残差阶段（每阶段两个块）+ GAP + Dropout + 六类线性头；
- 注意力版本在残差分支中加入通道重标定（se=True），同时加宽、提高 Dropout；
- 迁移骨干：resnet18 / convnext_tiny / efficientnet_b0（torchvision）、rexnet_150（timm）；
- 人脸领域权重可由 model.pretrained_path 指定本地检查点；
- 第十三轮的有界空间变换：旋转 |θ|≤0.17 rad、各向同性缩放 ±10%、平移 ±0.1，
  单位变换初始化 + 反射填充。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NUM_CLASSES

__all__ = ["FERResNet", "BoundedSTN", "CBAM", "build_model", "param_groups", "predict_proba",
           "load_state_dict_file"]


# --------------------------------------------------------------------------------------
# 注意力模块
# --------------------------------------------------------------------------------------
class SEBlock(nn.Module):
    """通道重标定：全局平均池化 → 降维 → 升维 → sigmoid。"""

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class CBAM(nn.Module):
    """通道 + 空间注意力；外层以零初始化残差增益挂接，保证初始映射不变。"""

    def __init__(self, ch: int, reduction: int = 16, kernel: int = 7):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(ch, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, ch, 1))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg, mx = F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ca
        sp = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        x = x * torch.sigmoid(self.spatial(sp))
        return x


class BoundedSTN(nn.Module):
    """有界空间变换：只允许小角度旋转、等比缩放与小平移，避免人脸被剪切。"""

    def __init__(self, in_ch: int = 1, max_rot: float = 0.17, max_scale: float = 0.10,
                 max_trans: float = 0.10, pad: int = 8):
        super().__init__()
        self.max_rot, self.max_scale, self.max_trans, self.pad = max_rot, max_scale, max_trans, pad
        self.loc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, 4))
        nn.init.zeros_(self.loc[-1].weight)
        nn.init.zeros_(self.loc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.tanh(self.loc(x))
        rot = self.max_rot * p[:, 0]
        scale = 1.0 + self.max_scale * p[:, 1]
        tx = self.max_trans * p[:, 2]
        ty = self.max_trans * p[:, 3]
        cos, sin = torch.cos(rot) / scale, torch.sin(rot) / scale
        theta = torch.stack([
            torch.stack([cos, -sin, tx], dim=1),
            torch.stack([sin, cos, ty], dim=1),
        ], dim=1)
        x = F.pad(x, (self.pad,) * 4, mode="reflect")
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


# --------------------------------------------------------------------------------------
# 小型残差基线
# --------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SEBlock(cout) if se else nn.Identity()
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + idt, inplace=True)


class FERResNet(nn.Module):
    """48×48 单通道输入的四个残差阶段小网络（第一轮基线，注意力版本复用）。"""

    def __init__(self, widths=(32, 64, 128, 256), blocks: int = 2, dropout: float = 0.25,
                 se: bool = False, in_ch: int = 1, cbam: bool = False, stn: bool = False,
                 width_mult: float = 1.0):
        super().__init__()
        widths = tuple(int(round(w * width_mult / 8)) * 8 for w in widths)
        self.stn = BoundedSTN(in_ch) if stn else nn.Identity()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers = []
        cin = widths[0]
        for i, w in enumerate(widths):
            for b in range(blocks):
                stride = 2 if (i > 0 and b == 0) else 1
                layers.append(BasicBlock(cin, w, stride, se))
                cin = w
        self.stages = nn.Sequential(*layers)
        self.cbam = CBAM(cin) if cbam else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, NUM_CLASSES)
        self.out_dim = cin

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x = self.stn(x)
        x = self.stem(x)
        x = self.stages(x)
        if self.cbam is not None:
            x = x + self.cbam.gamma * self.cbam(x)
        feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if return_feat:
            return feat
        return self.fc(self.drop(feat))


# --------------------------------------------------------------------------------------
# 迁移学习骨干
# --------------------------------------------------------------------------------------
class TorchvisionBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool, in_ch: int, dropout: float, stn: bool):
        super().__init__()
        import torchvision
        self.name = name
        self.stn = BoundedSTN(in_ch) if stn else nn.Identity()
        if name == "resnet18":
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            if in_ch != 3:
                m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            self.body = nn.Sequential(*list(m.children())[:-1])  # 到 avgpool 为止
            self.out_dim = 512
        elif name == "convnext_tiny":
            m = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
            self.body = m.features
            self.out_dim = 768
            self.pool = nn.Sequential(m.avgpool, m.classifier[0])
        elif name == "efficientnet_b0":
            m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            self.body = m.features
            self.out_dim = 1280
            self.pool = m.avgpool
        else:
            raise ValueError(f"未知的 torchvision 骨干：{name}")
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.out_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x = self.stn(x)
        x = self.body(x)
        if hasattr(self, "pool"):
            x = self.pool(x)
        x = torch.flatten(x, 1)
        if return_feat:
            return x
        return self.fc(self.drop(x))


class TimmBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool, in_ch: int, dropout: float, stn: bool,
                 pretrained_path: Optional[str] = None):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError("使用 timm 骨干需要安装 timm（pip install timm==0.9.16）") from e
        self.stn = BoundedSTN(in_ch) if stn else nn.Identity()
        kwargs = {"num_classes": 0, "in_chans": in_ch}
        if pretrained_path:
            kwargs["checkpoint_path"] = str(pretrained_path)
            pretrained = True
        elif pretrained:
            kwargs["pretrained"] = True
        self.body = timm.create_model(name, **kwargs)
        self.out_dim = int(getattr(self.body, "num_features", 0))
        if self.out_dim == 0:
            raise ValueError(f"{name} 没有 num_features，无法建立分类头")
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.out_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x = self.stn(x)
        x = self.body(x)
        x = torch.flatten(x, 1)
        if return_feat:
            return x
        return self.fc(self.drop(x))


TIMM_BACKBONES = {"rexnet_150", "rexnet_100", "efficientnet_b0_timm", "mobilenetv3_large_100"}


def build_model(cfg: dict) -> Tuple[nn.Module, Dict]:
    mcfg = cfg["model"]
    name = str(mcfg.get("name", "ferresnet"))
    in_ch = int(mcfg.get("in_ch", 1))
    pretrained = bool(mcfg.get("pretrained", False))
    dropout = float(mcfg.get("dropout", 0.25))
    stn = bool(mcfg.get("stn", False))
    path = mcfg.get("pretrained_path")

    if name == "ferresnet":
        model = FERResNet(widths=(32, 64, 128, 256), blocks=2, dropout=dropout,
                          se=bool(mcfg.get("se", False)), in_ch=in_ch,
                          cbam=bool(mcfg.get("cbam", False)), stn=stn,
                          width_mult=float(mcfg.get("width_mult", 1.0)))
    elif name in ("resnet18", "convnext_tiny", "efficientnet_b0"):
        model = TorchvisionBackbone(name, pretrained, in_ch, dropout, stn)
    elif name in TIMM_BACKBONES or name in ("efficientnet_b0_face",):
        timm_name = "efficientnet_b0" if name == "efficientnet_b0_face" else name
        model = TimmBackbone(timm_name, pretrained, in_ch, dropout, stn, pretrained_path=path)
    else:
        raise ValueError(f"未知模型：{name}")

    n_param = sum(p.numel() for p in model.parameters())
    info = {"name": name, "params": n_param, "out_dim": getattr(model, "out_dim", None),
            "pretrained": pretrained, "pretrained_path": str(path) if path else None}
    return model, info


def param_groups(model: nn.Module, base_lr: float, head_lr_mult: float, weight_decay: float):
    """骨干与分类头分离学习率；BN/偏置不做权重衰减。"""
    head_ids = set()
    head = getattr(model, "fc", None)
    if head is not None:
        head_ids = {id(p) for p in head.parameters()}
    decay_b, decay_h, nodecay = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            (decay_h if p.ndim > 1 else nodecay).append(p)
        else:
            (decay_b if p.ndim > 1 else nodecay).append(p)
    groups = [
        {"params": decay_b, "lr": base_lr, "weight_decay": weight_decay},
        {"params": decay_h, "lr": base_lr * head_lr_mult, "weight_decay": weight_decay},
        {"params": nodecay, "lr": base_lr, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor, flip_tta: bool = True) -> torch.Tensor:
    """原图与水平翻转图的概率平均（报告中的统一 TTA 口径）。"""
    p = F.softmax(model(x), dim=1)
    if flip_tta:
        p = p + F.softmax(model(torch.flip(x, dims=[3])), dim=1)
        p = p / 2.0
    return p


def load_state_dict_file(path: str | Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "model_state", "net"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise ValueError(f"{path} 不是可识别的权重文件")
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in obj.items()}
