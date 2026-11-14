"""提交转换的确定部分与隐藏 GT 处理的两种假设，供训练/离线共用。"""

import numpy as np
from PIL import Image


def resize_binary(mask, resample):
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(arr).resize((512, 512), resample)) > 0


def official_binarize(mask):
    """官方对预测的 BILINEAR -> >0；不代表已确认 GT 处理。"""
    return resize_binary(mask, Image.Resampling.BILINEAR)


def gt_bilinear(mask):
    """隐藏 GT 采用同样转换的假设。"""
    return official_binarize(mask)


def gt_nearest(mask):
    return resize_binary(mask, Image.Resampling.NEAREST)
