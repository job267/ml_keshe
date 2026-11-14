"""2026task3 —— 眼底图像黄斑中央凹（Fovea）坐标定位。

基础深度学习框架：
    stage1  整图 -> 黄斑区域热力图 -> 中心凹粗坐标
    stage2  以 stage1 粗坐标为中心裁切高清 patch -> 中心凹精坐标

约定（全局统一，非常重要）：
    所有坐标在模型内部都用"归一化坐标" (x/W, y/H) ∈ [0, 1] 表示，
    其中 W、H 为**原始图像**的宽高。这样不同分辨率的图像可以共享同一套网络与损失。
    只有在计算评价指标 / 生成提交文件时才乘回原始宽高变成像素坐标。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Windows 控制台默认 GBK，中文日志容易乱码；统一用 UTF-8 输出。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

# torchvision 预训练权重默认下载到 ~/.cache/torch，某些环境下该目录不可写，
# 因此统一放到项目内的 .cache/torch。必须在 import torchvision 之前设置。
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
