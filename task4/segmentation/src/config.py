"""超参数集中管理。

所有可调参数都定义在 :class:`Config` 中，训练/推理脚本通过
:func:`parse_config` 自动生成命令行接口，例如::

    python -m src.train --epochs 300 --lr 5e-4 --patch_size 256

注意：本文件不使用 ``from __future__ import annotations``，
因为 :func:`build_parser` 依赖真实的类型对象来生成 argparse 参数。
"""

import argparse
from dataclasses import dataclass, field, fields, asdict


# --------------------------------------------------------------------------- #
# 参数帮助信息（可选，仅用于 --help 显示）
# --------------------------------------------------------------------------- #
HELP = {
    "train_image_dir": "训练图像目录",
    "train_label_dir": "训练标签目录（血管=255，背景=0）",
    "test_image_dir": "测试图像目录",
    "out_dir": "训练输出目录（权重、日志）",
    "pred_dir": "测试集预测掩膜输出目录（默认为官方脚本约定的 ./image）",
    "ids": "参与训练的图片 id，逗号分隔；留空=全部",
    "val_ids": "留作验证的图片 id，逗号分隔；留空=不划分验证集",
    "patch_size": "训练 patch 边长",
    "scale": "整图放大倍率：>1 时训练/推理都在放大后的尺度上进行（提高细血管分辨率）",
    "channels": "输入通道组成，可选 clahe/green/highpass",
    "clahe_clip": "CLAHE 裁剪系数",
    "clahe_grid": "CLAHE 网格数",
    "label_threshold": "标签二值化阈值（> 该值视为血管）",
    "init_from": "预训练权重路径（外部数据预训练后热启动）",
    "external_dirs": "外部数据目录（逗号分隔，每个含 image/ 与 label/）",
    "pretrain_out": "预训练权重输出路径",
    "extra_train_dirs": "联合训练时额外加入的外部数据目录（逗号分隔）",
    "extra_repeat": "DRIVE 训练集重复次数（提高其在 patch 采样中的占比）",
    "model": "模型名：unet / resunet / att_unet / scse_unet / att_scse_unet / att_resunet / fr_unet / unet_aux / unet_dsconv",
    "strong_aug": "一键开启强增强预设（弹性形变/遮挡/对比度/逐通道增益/锐化）",
    "aug_elastic": "弹性形变概率（-1=跟随 strong_aug）",
    "aug_cutout": "随机遮挡概率（-1=跟随 strong_aug）",
    "aug_contrast": "对比度扰动幅度（-1=跟随 strong_aug）",
    "aug_cgap": "逐通道增益幅度（-1=跟随 strong_aug）",
    "aug_sharpen": "锐化概率（-1=跟随 strong_aug）",
    "aug_copy_paste": "血管片段复制粘贴概率（-1=跟随 strong_aug）",
    "cache_dtype": "整图缓存精度：float32（默认）/ float16（外部数据多时省一半内存）",
    "norm": "归一化层：bn / in / gn",
    "base_channels": "U-Net 第一层通道数",
    "depth": "U-Net 下采样层数",
    "dropout": "Dropout 概率",
    "loss": "损失函数：bce_dice / focal_dice / dice / bce_dice_cldice",
    "epochs": "训练轮数",
    "batch_size": "批大小",
    "patches_per_epoch": "每个 epoch 采样的 patch 数",
    "lr": "初始学习率",
    "weight_decay": "AdamW 权重衰减",
    "warmup_epochs": "学习率 warmup 轮数",
    "min_lr_ratio": "余弦退火的最低学习率比例",
    "amp": "是否使用混合精度训练",
    "num_workers": "DataLoader 进程数（缓存小，0 即可）",
    "seed": "随机种子",
    "deterministic": "开启 cudnn 确定性算法（可比特级复现，但更慢）",
    "device": "auto / cuda / cpu",
    "val_every": "每隔多少轮做一次整图验证",
    "threshold": "概率转二值的阈值",
    "min_size": "剔除小于该面积的连通域（像素数）",
    "use_fov": "是否用视场（FOV）掩膜裁剪预测",
    "fill_holes": "是否填充血管区域内部空洞",
    "tta": "推理时是否使用翻转 TTA",
    "invert_output": "输出取反（仅在评分方要求血管=0 时才用）",
    "checkpoint": "推理使用的权重路径",
}


@dataclass
class Config:
    # ----------------------------- 路径 ----------------------------- #
    train_image_dir: str = "train/image"
    train_label_dir: str = "train/label"
    test_image_dir: str = "test/image"
    out_dir: str = "outputs"
    pred_dir: str = "image"

    # ----------------------------- 数据 ----------------------------- #
    ids: str = ""
    val_ids: str = "17,18,19,20"
    patch_size: int = 192
    scale: float = 1.0                  # 训练/推理前把整图放大到该倍率（提高分辨率）
    preprocess_mode: str = "rgb_first"  # 新训练先缩放 RGB；旧权重显式回退 legacy
    channels: str = "clahe,green,highpass"
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    label_threshold: int = 127
    # 光度归一化常数（与 preprocess.to_channels 的输出尺度对应）
    norm_mean: str = "0.5,0.5,0.0"
    norm_std: str = "0.25,0.25,0.25"

    # 数据增强强度
    vessel_center_prob: float = 0.5     # 以血管像素为中心裁剪的概率
    thin_center_prob: float = 0.0       # 替换部分普通前景采样，不挤占背景采样
    thin_max_radius: float = 1.5        # scale=1 像素下的血管半径上限
    scale_range: str = "0.85,1.25"      # 随机缩放范围
    rot_angle: float = 45.0             # 随机旋转角度上限（度）
    aug_gain: float = 0.20              # 亮度增益 ±
    aug_bias: float = 0.08              # 亮度偏移 ±
    aug_noise: float = 0.03             # 高斯噪声 sigma
    aug_gamma: float = 0.25             # gamma 扰动幅度
    # ---- 强增强（第五轮新增）：-1 表示跟随 --strong_aug 预设 ----
    strong_aug: bool = False            # 一键开启强增强预设（见 augment.STRONG_AUG）
    aug_elastic: float = -1.0           # 弹性形变概率
    aug_cutout: float = -1.0            # 随机遮挡概率
    aug_contrast: float = -1.0          # 对比度扰动幅度
    aug_cgap: float = -1.0              # 逐通道增益幅度
    aug_sharpen: float = -1.0           # 锐化概率
    aug_copy_paste: float = -1.0        # 血管片段复制粘贴概率
    cache_dtype: str = "float32"        # 整图缓存精度：float32 / float16（省一半内存）

    # ----------------------------- 模型 ----------------------------- #
    model: str = "unet"
    norm: str = "bn"
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.0
    bilinear: bool = True
    fr_feature_scale: int = 2
    fr_dropout: float = 0.2
    ds_kernel_size: int = 9
    aux_weight: float = 0.05            # each branch, ramped up during initial epochs
    aux_warmup_epochs: int = 10

    # ----------------------------- 训练 ----------------------------- #
    loss: str = "bce_dice"
    cldice_weight: float = 0.1
    cldice_iters: int = 10
    epochs: int = 250
    batch_size: int = 8
    patches_per_epoch: int = 480
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01
    amp: bool = True
    num_workers: int = 0
    seed: int = 42
    split_seed: int = 42
    split_strategy: str = "stratified"  # legacy 仅用于复现；split_file 优先
    deterministic: bool = False
    device: str = "auto"
    val_every: int = 10
    selection_metric: str = "dice"      # dice / dice_512_proxy
    selection_tta: bool = False
    checkpoint_policy: str = "best"     # last 避免按外层标签挑 epoch
    resume_from: str = ""
    checkpoint_every: int = 10

    # -------------------------- 推理 / 后处理 ------------------------ #
    threshold: float = 0.45       # 由 scripts/tune_postprocess.py 在验证集上选出
    min_size: int = 10
    use_fov: bool = True
    fill_holes: bool = False
    tta: bool = True
    invert_output: bool = False   # 若评分方坚持"血管=0、其他=255"，置 True 再重跑推理
    checkpoint: str = "outputs/best.pt"
    init_from: str = ""           # 用外部数据预训练好的权重热启动（空=从头训）
    pretrain_out: str = "outputs/pretrained.pt"
    external_dirs: str = ("data_external/stare,data_external/chase,data_external/hrf,"
                          "data_external/fives640")
    extra_train_dirs: str = ""    # 联合训练时额外加入的外部数据（空=只用 DRIVE）
    extra_repeat: int = 1         # 训练集重复次数（提高 DRIVE 采样权重）

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in d.items() if k in names}
        # 老 checkpoint 保持旧的推理行为，不能悄悄改变既有提交。
        values.setdefault("preprocess_mode", "legacy")
        return cls(**values)

    # ---- 便捷解析 ---- #
    @property
    def channel_list(self):
        return [c.strip() for c in self.channels.split(",") if c.strip()]

    @property
    def norm_mean_list(self):
        return [float(v) for v in self.norm_mean.split(",")]

    @property
    def norm_std_list(self):
        return [float(v) for v in self.norm_std.split(",")]

    @property
    def scale_range_tuple(self):
        a, b = [float(v) for v in self.scale_range.split(",")]
        return (min(a, b), max(a, b))


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def build_parser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    for f in fields(Config):
        default = f.default
        help_text = HELP.get(f.name, "")
        if f.type is bool:
            p.add_argument(f"--{f.name}", type=_str2bool, nargs="?", const=True,
                           default=default, help=help_text)
        else:
            p.add_argument(f"--{f.name}", type=f.type, default=default, help=help_text)
    return p


def parse_config(argv=None, description: str = "") -> Config:
    args = build_parser(description).parse_args(argv)
    return Config(**vars(args))
