"""模型定义与注册表。

共用同一套编解码骨架，通过组合三种"注意力/主干"开关得到不同结构：

* ``unet``            : 经典 U-Net（Ronneberger et al., 2015）
* ``resunet``         : 残差块替代双卷积
* ``att_unet``        : U-Net + 注意力门控（Attention Gate, Oktay et al. 2018）
* ``scse_unet``       : U-Net + SCSE 通道/空间注意力（Roy et al. 2018）
* ``att_scse_unet``   : 两者都加（本项目第三轮的主实验）
* ``att_resunet``     : 残差主干 + 注意力门控

新增模型只需写一个返回 ``nn.Module`` 的函数并注册到 ``MODELS``。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基础模块
# --------------------------------------------------------------------------- #
def make_norm(name: str, channels: int) -> nn.Module:
    name = (name or "bn").lower()
    if name == "bn":
        return nn.BatchNorm2d(channels)
    if name == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    if name == "gn":
        groups = 8 if channels >= 8 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"未知归一化层: {name}")


def conv3x3(cin, cout):
    return nn.Conv2d(cin, cout, 3, padding=1, bias=False)


class ConvBlock(nn.Module):
    """两个 3x3 卷积 + 归一化 + ReLU。"""

    def __init__(self, cin, cout, norm="bn", dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(cin, cout), make_norm(norm, cout), nn.ReLU(inplace=True),
            conv3x3(cout, cout), make_norm(norm, cout), nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.block(x))


class ResBlock(nn.Module):
    """残差块：Conv-Norm-ReLU-Conv-Norm + shortcut。"""

    def __init__(self, cin, cout, norm="bn", dropout=0.0):
        super().__init__()
        self.c1 = conv3x3(cin, cout)
        self.n1 = make_norm(norm, cout)
        self.c2 = conv3x3(cout, cout)
        self.n2 = make_norm(norm, cout)
        self.act = nn.ReLU(inplace=True)
        self.short = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.drop(y)
        y = self.n2(self.c2(y))
        return self.act(y + self.short(x))


class AttentionGate(nn.Module):
    """加性注意力门控（Attention Gate, Oktay et al. 2018, arXiv:1804.03999）。

    用解码器传来的门控信号 ``g`` 给编码器跳连特征 ``x`` 逐像素打分::

        alpha = sigmoid( psi( relu( W_g(g) + W_x(x) ) ) )
        out   = x * alpha

    ``psi`` 的权重初始化为 0，于是初始 ``alpha = 0.5``：训练起点等价于
    "跳连特征整体乘 0.5"，不会把预训练权重一下子打乱 —— 这对
    ``--init_from`` 热启动很关键。
    """

    def __init__(self, f_gate, f_skip, f_int=None, norm="bn"):
        super().__init__()
        f_int = f_int or max(min(f_gate, f_skip) // 2, 8)
        self.wg = nn.Sequential(nn.Conv2d(f_gate, f_int, 1, bias=False),
                                make_norm(norm, f_int))
        self.wx = nn.Sequential(nn.Conv2d(f_skip, f_int, 1, bias=False),
                                make_norm(norm, f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        nn.init.zeros_(self.psi[0].weight)
        nn.init.zeros_(self.psi[0].bias)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(self.relu(self.wg(g) + self.wx(x)))
        return x * a


class SCSEBlock(nn.Module):
    """SCSE：通道注意力 + 空间注意力并行，再用逐元素相加融合。

    * 通道分支（cSE）：全局平均池化 → 1x1 卷积降维再升维 → sigmoid
    * 空间分支（sSE）：1x1 卷积压到 1 通道 → sigmoid

    两者都输出与 ``x`` 同形状的权重图，直接相乘后相加，属于"不改形状"的
    轻量模块（参数量只有 C²/r 级别），可以安全地插在每一层后面。
    """

    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(ch // r, 4)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=True), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #
class UNet(nn.Module):
    """可配置深度/注意力模块的 U-Net。

    输入尺寸建议为 ``2**depth`` 的整数倍（depth=4 时 16 的倍数）。

    :param block: ``"basic"``=双 3x3 卷积，``"res"``=残差块
    :param attn:  是否在每个跳连上插入 ``AttentionGate``
    :param scse:  是否在每个编码/解码块后接 ``SCSEBlock``
    """

    def __init__(self, in_channels=3, out_channels=1, base=64, depth=4,
                 norm="bn", dropout=0.0, block="basic", bilinear=True,
                 attn=False, scse=False):
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.bilinear = bilinear
        self.use_attn = attn
        self.use_scse = scse
        Block = ConvBlock if block == "basic" else ResBlock

        # ---- 编码器 ----
        self.enc = nn.ModuleList()
        c = in_channels
        chans = []
        for i in range(depth):
            o = base * (2 ** i)
            self.enc.append(Block(c, o, norm, dropout))
            chans.append(o)
            c = o
        self.pool = nn.MaxPool2d(2)
        self.enc_se = nn.ModuleList([SCSEBlock(o) for o in chans]) if scse else None

        # ---- 瓶颈 ----
        bot = base * (2 ** depth)
        self.bottleneck = Block(c, bot, norm, dropout)
        self.bot_se = SCSEBlock(bot) if scse else None

        # ---- 解码器 ----
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.att = nn.ModuleList() if attn else None
        self.dec_se = nn.ModuleList() if scse else None
        c = bot
        for i in reversed(range(depth)):
            o = base * (2 ** i)
            if bilinear:
                # 双线性上采样 + 1x1 卷积对齐通道（比转置卷积更少棋盘伪影）
                self.up.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(c, o, 1, bias=False), make_norm(norm, o),
                    nn.ReLU(inplace=True)))
            else:
                self.up.append(nn.ConvTranspose2d(c, o, kernel_size=2, stride=2))
            if attn:
                # 门控信号来自上采样后的特征，跳连特征来自编码器（通道都是 o）
                self.att.append(AttentionGate(f_gate=o, f_skip=o, norm=norm))
            # 拼接 skip 后通道数为 o + o
            self.dec.append(Block(o * 2, o, norm, dropout))
            if scse:
                self.dec_se.append(SCSEBlock(o))
            c = o

        self.head = nn.Conv2d(base, out_channels, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 注意力门控的 psi 必须重新置零（上面的循环会把它改成 kaiming 初值）
        if self.att is not None:
            for ag in self.att:
                nn.init.zeros_(ag.psi[0].weight)
                nn.init.zeros_(ag.psi[0].bias)

    def forward(self, x):
        feats = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if self.enc_se is not None:
                x = self.enc_se[i](x)
            feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        if self.bot_se is not None:
            x = self.bot_se(x)
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            skip = feats[-1 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.att is not None:
                # 用解码器特征当门控，抑制跳连里的背景/噪声响应
                skip = self.att[i](skip, x)
            x = dec(torch.cat([x, skip], dim=1))
            if self.dec_se is not None:
                x = self.dec_se[i](x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def _build(cfg, block="basic", attn=False, scse=False):
    return UNet(in_channels=len(cfg.channel_list), out_channels=1,
                base=cfg.base_channels, depth=cfg.depth, norm=cfg.norm,
                dropout=cfg.dropout, block=block, bilinear=cfg.bilinear,
                attn=attn, scse=scse)


def build_unet(cfg):
    return _build(cfg, block="basic")


def build_resunet(cfg):
    return _build(cfg, block="res")


def build_att_unet(cfg):
    """U-Net + 注意力门控（跳连上做空间注意力）。"""
    return _build(cfg, block="basic", attn=True)


def build_scse_unet(cfg):
    """U-Net + SCSE（通道 + 空间注意力）。"""
    return _build(cfg, block="basic", scse=True)


def build_att_scse_unet(cfg):
    """注意力门控 + SCSE（第三轮主实验：attention 全开）。"""
    return _build(cfg, block="basic", attn=True, scse=True)


def build_att_resunet(cfg):
    """残差主干 + 注意力门控（换主干时用）。"""
    return _build(cfg, block="res", attn=True)


def build_fr_unet(cfg):
    from .advanced_models import build_advanced
    return build_advanced(cfg, 'fr_unet')


def build_aux_unet(cfg):
    from .advanced_models import build_advanced
    return build_advanced(cfg, 'unet_aux')


def build_dsconv_unet(cfg):
    from .advanced_models import build_advanced
    return build_advanced(cfg, 'unet_dsconv')


MODELS = {
    "unet": build_unet,
    "resunet": build_resunet,
    "att_unet": build_att_unet,
    "scse_unet": build_scse_unet,
    "att_scse_unet": build_att_scse_unet,
    "att_resunet": build_att_resunet,
    "fr_unet": build_fr_unet,
    "unet_aux": build_aux_unet,
    "unet_dsconv": build_dsconv_unet,
}


def build_model(cfg, name=None) -> nn.Module:
    key = (name or cfg.model).lower()
    if key not in MODELS:
        raise ValueError(f"未知模型 '{key}'，可选: {sorted(MODELS)}")
    return MODELS[key](cfg)


def load_weights(model, path, device="cpu"):
    """把权重载入**已有**模型（用于外部数据预训练后的热启动）。

    兼容两种文件：训练脚本保存的完整 checkpoint（含 ``model`` 键）或裸 state_dict。
    shape 不匹配的张量会被跳过，方便改网络宽度后仍能部分加载。

    :return: (载入的张量数, 形状不匹配而被跳过的键, 模型里没被覆盖的键)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    own = model.state_dict()
    matched, skipped = {}, []
    for k, v in state.items():
        # Explicitly map the existing U-Net body/head into the two local extensions.
        if k not in own and k.startswith('head.') and 'head.region.weight' in own:
            k = k.replace('head.', 'head.region.', 1)
        if k not in own and k.startswith('enc.0.') and 'enc.0.standard.block.0.weight' in own:
            k = k.replace('enc.0.', 'enc.0.standard.', 1)
        if k in own and own[k].shape == v.shape:
            matched[k] = v
        elif k in own:
            skipped.append(k)
    missing = [k for k in own if k not in matched]
    model.load_state_dict(matched, strict=False)
    return len(matched), skipped, missing


def load_checkpoint(path, device="cpu"):
    """读取权重，返回 (model, cfg_dict, extra)。"""
    from .config import Config

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = Config.from_dict(cfg_dict) if cfg_dict else Config()
    model = build_model(cfg)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg, {k: v for k, v in ckpt.items() if k not in ("model", "cfg")}
