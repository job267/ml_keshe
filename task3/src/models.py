"""模型：ResNet 特征提取 + 热力图头（黄斑区域检测 / 中心凹定位共用）。

提供两种特征提取结构，用 `model.arch` 切换，便于做消融：

* ``arch: fpn``（默认）—— ResNet + 轻量 FPN，把 1/4、1/8、1/16、1/32 四级特征自顶向下融合，
  在 **1/4 分辨率**输出热力图。每格 ≈ 3.9px（768 宽输入），量化误差小。
* ``arch: dilated`` —— **不用 FPN**：用空洞卷积把深层特征保持在各向同性的 1/8 分辨率
  （layer3/layer4 不再下采样），再简单上采样到 1/4。没有侧向连接与金字塔，
  属于"精度优先"的 DeepLab 式结构，可作为 FPN 的对照组。

两种结构输出的热力图分辨率相同（都是 1/4），所以对比时只有"多尺度融合方式"这一个变量。

其余设计：
* 热力图经 **soft-argmax**（质量归一化求期望）得到亚像素级可导坐标；
* 可选 coord_head（全局池化回归）与 vis_head（中心凹可见性）；
* stage2 复用同一网络，输入换成以粗定位结果为中心的局部 patch。
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------- #
# 主干网络
# --------------------------------------------------------------------------- #
_RESNET_EXP = {"resnet50": 4, "resnet101": 4, "resnet152": 4}


def _build_resnet(name: str, pretrained: bool, in_ch: int = 3, **kwargs) -> nn.Module:
    """加载 torchvision ResNet（可带空洞卷积 / 多通道输入），失败时退化为随机初始化。"""
    if not hasattr(torchvision.models, name):
        raise ValueError(f"不支持的 backbone: {name}")
    try:
        m = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None, **kwargs)
    except Exception as e:  # 离线 / 无缓存 / 下载失败
        if not pretrained:
            raise
        warnings.warn(f"预训练权重加载失败({type(e).__name__}: {e})，改用随机初始化。")
        m = getattr(torchvision.models, name)(weights=None, **kwargs)
    return _adapt_first_conv(m, in_ch)


def _adapt_first_conv(model: nn.Module, in_ch: int) -> nn.Module:
    """把 conv1 从 3 通道扩展到 in_ch 通道，并尽量保留预训练特征。

    做法：新通道的权重 = 原 3 通道权重 / 3ch 组数。这样每个 3 通道视图上的响应量级与
    单图输入一致（patch 与全局缩略图各占一半权重），不会因为通道翻倍把激活值放大一倍。
    """
    if in_ch == 3:
        return model
    old = model.conv1
    w = old.weight.data                     # (64, 3, 7, 7)
    new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    groups = in_ch // 3
    with torch.no_grad():
        for g in range(groups):
            new.weight.data[:, 3 * g: 3 * (g + 1)] = w / groups
        if in_ch % 3:                        # 非常规通道数：剩余通道复制第一组
            new.weight.data[:, 3 * groups:] = w[:, : in_ch - 3 * groups] / groups
    model.conv1 = new
    return model


def _init_new_convs(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ResNetFPN(nn.Module):
    """ResNet + FPN，输出 1/4 分辨率的融合特征（默认 arch=fpn）。"""

    def __init__(self, name: str = "resnet18", pretrained: bool = True, hidden: int = 128,
                 in_ch: int = 3):
        super().__init__()
        m = _build_resnet(name, pretrained, in_ch=in_ch)

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4

        exp = _RESNET_EXP.get(name, 1)
        chs = [64 * exp, 128 * exp, 256 * exp, 512 * exp]

        self.lat = nn.ModuleList([nn.Conv2d(c, hidden, 1) for c in chs])
        self.smooth = nn.ModuleList([nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(4)])
        self.out_channels = hidden
        self.norm = nn.GroupNorm(8, hidden)
        _init_new_convs(nn.ModuleList([*self.lat, *self.smooth]))

    @staticmethod
    def _up(x: torch.Tensor, size) -> torch.Tensor:
        """双线性上采样。

        注意：这里**不能用 nearest**——最近邻 + 相加会在热力图上留下棋盘格伪影，
        热力图回归会被这些网格主导（实测中会让定位几乎没有进展）。
        """
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.layer1(self.stem(x))     # 1/4
        f2 = self.layer2(f1)               # 1/8
        f3 = self.layer3(f2)               # 1/16
        f4 = self.layer4(f3)               # 1/32

        p4 = self.lat[3](f4)
        p3 = self.smooth[2](self.lat[2](f3) + self._up(p4, f3.shape[-2:]))
        p2 = self.smooth[1](self.lat[1](f2) + self._up(p3, f2.shape[-2:]))
        p1 = self.smooth[0](self.lat[0](f1) + self._up(p2, f1.shape[-2:]))
        return self.norm(F.relu(p1))


class DilatedResNet(nn.Module):
    """ResNet + 空洞卷积，输出 1/4 分辨率特征（默认 arch=dilated，无 FPN）。

    layer3/layer4 的步长被换成空洞卷积，特征始终维持在 1/8（各向同性、无金字塔），
    然后一次性上采样到 1/4。相比 FPN，少了侧向连接和逐级融合，但保留了高分辨率。

    注意：只有 Bottleneck 结构（resnet50/101/152）支持 dilation > 1，
    resnet18/34 的 BasicBlock 不支持，所以 dilated 只接受 50 以上的骨干。
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True, hidden: int = 128,
                 in_ch: int = 3):
        super().__init__()
        if name not in _RESNET_EXP:
            raise ValueError(
                f"arch=dilated 只支持 Bottleneck 骨干（resnet50/101/152），收到 {name}；"
                f"resnet18/34 的 BasicBlock 不支持空洞卷积。"
            )
        m = _build_resnet(name, pretrained, in_ch=in_ch,
                          replace_stride_with_dilation=[False, True, True])

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        c = 512 * _RESNET_EXP.get(name, 1)

        self.head = nn.Sequential(
            nn.Conv2d(c, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
        )
        self.out_channels = hidden
        _init_new_convs(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)                    # 1/4
        f = self.layer1(f)                  # 1/4
        f = self.layer2(f)                  # 1/8
        f = self.layer3(f)                  # 1/8（空洞）
        f = self.layer4(f)                  # 1/8（空洞）
        f = self.head(f)                    # 1/8
        return F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)  # 1/4


class PlainResNet(nn.Module):
    """无 FPN（最省算力的"不要金字塔"版本）：直接取最深层 1/32 特征，一次上采样 ×8 到 1/4。

    对照组的价值在于说明：去掉多尺度融合后，热力图的"信息分辨率"退回到 1/32
    （768 宽输入时每格 ≈ 24px），上采样只是把粗网格插值变光滑，并不会恢复定位精度。
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True, hidden: int = 128,
                 in_ch: int = 3):
        super().__init__()
        m = _build_resnet(name, pretrained, in_ch=in_ch)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        c = 512 * _RESNET_EXP.get(name, 1)
        self.head = nn.Sequential(
            nn.Conv2d(c, hidden, 1), nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
        )
        self.out_channels = hidden
        _init_new_convs(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)
        f = self.layer1(f); f = self.layer2(f)
        f = self.layer3(f); f = self.layer4(f)       # 1/32
        f = self.head(f)
        return F.interpolate(f, scale_factor=8, mode="bilinear", align_corners=False)  # 1/4


# --------------------------------------------------------------------------- #
# 热力图网络
# --------------------------------------------------------------------------- #
class HeatmapNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        hidden: int = 128,
        out_stride: int = 4,
        arch: str = "fpn",
        in_ch: int = 3,
        use_coord_head: bool = True,
        use_vis_head: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert arch in ("fpn", "dilated", "plain"), "arch 支持 fpn / dilated / plain"
        assert out_stride in (2, 4), "out_stride 支持 2 / 4（更粗的分辨率没必要）"
        self.out_stride = out_stride
        self.arch = arch
        if arch == "fpn":
            self.backbone = ResNetFPN(backbone, pretrained, hidden, in_ch)
        elif arch == "dilated":
            self.backbone = DilatedResNet(backbone, pretrained, hidden, in_ch)
        else:
            self.backbone = PlainResNet(backbone, pretrained, hidden, in_ch)
        c = self.backbone.out_channels

        if out_stride == 2:
            self.pre_heat = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            )
        else:
            self.pre_heat = nn.Identity()

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
        )
        nn.init.zeros_(self.heatmap_head[2].weight)          # 最后 1x1 零初始化
        nn.init.constant_(self.heatmap_head[2].bias, -4.0)   # 初始概率 ~0.018，加速收敛

        self.use_coord_head = use_coord_head
        self.use_vis_head = use_vis_head
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if use_coord_head:
            self.coord_head = nn.Linear(c, 2)
            nn.init.zeros_(self.coord_head.weight)
            nn.init.zeros_(self.coord_head.bias)          # sigmoid(0)=0.5，接近数据均值
        if use_vis_head:
            self.vis_head = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.pre_heat(self.backbone(x))
        hm = self.heatmap_head(feat)                       # logits
        out: Dict[str, Optional[torch.Tensor]] = {
            "heatmap": hm,
            "xy_soft": soft_argmax(torch.sigmoid(hm)),
        }
        pooled = self.drop(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        if self.use_coord_head:
            out["xy_head"] = torch.sigmoid(self.coord_head(pooled))
        else:
            out["xy_head"] = None
        if self.use_vis_head:
            out["vis_logit"] = self.vis_head(pooled).squeeze(1)
        else:
            out["vis_logit"] = None
        return out


# --------------------------------------------------------------------------- #
# soft-argmax / 坐标解码
# --------------------------------------------------------------------------- #
def soft_argmax(prob: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """prob: (B,1,H,W) 非负热力图 -> (B,2) 归一化坐标 ∈[0,1]。

    对热力图做 **质量归一化后求期望**（integral regression / DSNT 做法）：
        p = prob^beta / Σ prob^beta ,  x = Σ p * grid_x
    注意不能直接对概率图做 softmax——概率背景值(0.02)远大于 0，
    softmax 后背景会淹没峰值，预测会塌缩到网格中心。

    网格坐标约定：第 i 个像素中心对应 (i+0.5)/W，
    因此热力图标签也按 `gt*W - 0.5` 生成，两者严格一致（见 losses.make_heatmap_target）。
    """
    b, _, h, w = prob.shape
    p = prob.reshape(b, -1).clamp_min(0.0)
    if beta != 1.0:
        p = p.pow(beta)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
    p = p.reshape(b, 1, h, w)
    gx = (torch.arange(w, device=prob.device, dtype=prob.dtype) + 0.5) / w
    gy = (torch.arange(h, device=prob.device, dtype=prob.dtype) + 0.5) / h
    x = (p.sum(dim=2) * gx).sum(dim=2).squeeze(1)
    y = (p.sum(dim=3) * gy).sum(dim=2).squeeze(1)
    return torch.stack([x, y], dim=1)


def argmax_coords(prob: torch.Tensor) -> torch.Tensor:
    """硬 argmax 解码（定位评价时可与 soft-argmax 对比）。"""
    b, _, h, w = prob.shape
    idx = prob.reshape(b, -1).argmax(dim=1)
    x = ((idx % w).float() + 0.5) / w
    y = ((idx // w).float() + 0.5) / h
    return torch.stack([x, y], dim=1)


def select_xy(out: Dict[str, Optional[torch.Tensor]], source: str = "soft") -> torch.Tensor:
    """从网络输出中选择最终坐标：soft(soft-argmax) / head(回归头) / mean(两者平均)。"""
    soft, head = out["xy_soft"], out.get("xy_head")
    if source == "head" and head is not None:
        return head
    if source == "mean" and head is not None:
        return (soft + head) / 2.0
    return soft


@torch.no_grad()
def predict_coords(model: nn.Module, x: torch.Tensor, tta: bool = False) -> Dict[str, torch.Tensor]:
    """推理接口：返回 xy_soft / xy_head / vis_prob（可选水平翻转 TTA）。"""
    return predict_tta(model, x, {"flip": bool(tta), "scales": [1.0], "shifts": [(0.0, 0.0)]})


# --------------------------------------------------------------------------- #
# 测试时增强（多尺度 / 翻转 / 平移）
# --------------------------------------------------------------------------- #
def _resize_input(x: torch.Tensor, scale: float) -> torch.Tensor:
    """按比例缩放输入（尺寸取 32 的倍数，FPN 各层才能对齐）。"""
    if abs(scale - 1.0) < 1e-6:
        return x
    h, w = x.shape[-2:]
    nh = max(32, int(round(h * scale / 32.0)) * 32)
    nw = max(32, int(round(w * scale / 32.0)) * 32)
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)


def _translate(x: torch.Tensor, fx: float, fy: float) -> Tuple[torch.Tensor, Tuple[float, float]]:
    """把图像内容平移 (fx, fy)（占宽高比例），返回平移后的图与实际位移。

    内容右移 fx 后，同一特征在归一化坐标里从 x 变成 x+fx，所以预测值要减去 fx 修正。
    边缘用 0 填充——眼底图边缘本来就是黑的，不会引入突兀边界。
    """
    h, w = x.shape[-2:]
    dx, dy = int(round(fx * w)), int(round(fy * h))
    if dx == 0 and dy == 0:
        return x, (0.0, 0.0)
    out = torch.zeros_like(x)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[:, :, sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = x[:, :, sy0:sy1, sx0:sx1]
    return out, (dx / w, dy / h)


@torch.no_grad()
def predict_tta(model: nn.Module, x: torch.Tensor, spec: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """多视角推理并平均坐标。spec = {flip, scales, shifts}。

    * scales：相对输入尺寸的缩放（如 1.15 / 0.87），对抗"只认某个尺度"的模型；
    * shifts：平移比例，对抗位置敏感/边缘效应；
    * 所有视角的预测都换算回 **原图归一化坐标** 再平均，因此坐标系不变。
    """
    spec = spec or {}
    scales = [float(s) for s in (spec.get("scales") or [1.0])]
    shifts = [(float(a), float(b)) for a, b in (spec.get("shifts") or [(0.0, 0.0)])]
    do_flip = bool(spec.get("flip", True))

    softs, heads, viss = [], [], []
    for sc in scales:
        for fx, fy in shifts:
            for flipped in ([False, True] if do_flip else [False]):
                xi = _resize_input(x, sc)
                xi, (sx, sy) = _translate(xi, fx, fy)
                if flipped:
                    xi = torch.flip(xi, dims=[3])
                out = model(xi)

                soft = out["xy_soft"].clone()
                if flipped:
                    soft[:, 0] = 1.0 - soft[:, 0]
                soft[:, 0] -= sx
                soft[:, 1] -= sy
                softs.append(soft)

                head = out.get("xy_head")
                if head is not None:
                    hd = head.clone()
                    if flipped:
                        hd[:, 0] = 1.0 - hd[:, 0]
                    hd[:, 0] -= sx
                    hd[:, 1] -= sy
                    heads.append(hd)
                vis = out.get("vis_logit")
                if vis is not None:
                    viss.append(vis)

    return {
        "xy_soft": torch.stack(softs).mean(0),
        "xy_head": torch.stack(heads).mean(0) if heads else None,
        "vis_prob": torch.sigmoid(torch.stack(viss).mean(0)) if viss else None,
        "n_views": len(softs),
    }


# --------------------------------------------------------------------------- #
def build_model(cfg: Dict, device: Optional[torch.device] = None) -> HeatmapNet:
    mcfg = cfg["model"]
    model = HeatmapNet(
        backbone=mcfg["backbone"],
        pretrained=mcfg["pretrained"],
        hidden=int(mcfg["hidden"]),
        out_stride=int(mcfg["out_stride"]),
        arch=str(mcfg.get("arch", "fpn")),
        in_ch=int(mcfg.get("in_ch", 3)),
        use_coord_head=bool(mcfg["use_coord_head"]),
        use_vis_head=bool(mcfg["use_vis_head"]),
        dropout=float(mcfg["dropout"]),
    )
    if device is not None:
        model = model.to(device)
    return model
