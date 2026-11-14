"""训练期数据增强。

DRIVE 训练集只有 20 张图，必须靠强增强抑制过拟合。这里不依赖
albumentations，直接用 numpy / OpenCV 实现，保证零额外依赖。

几何增强：随机裁剪（可围绕血管像素）、随机缩放、随机旋转、8 种二面体变换、
          **弹性形变**（血管粗细/走向抖动）、**随机遮挡**、**血管片段复制粘贴**。
光度增强：亮度增益、亮度偏移、gamma、对比度、**逐通道增益**、高斯噪声、
          高斯模糊、**锐化**。

``STRONG_AUG`` 是第五轮用的一组强增强默认强度，通过 ``--strong_aug 1`` 开启，
单独的参数（``--aug_elastic`` 等）可以再覆盖。
"""

import math

import cv2
import numpy as np


# 强增强预设：--strong_aug 1 时生效（各字段仍可单独用命令行覆盖）
STRONG_AUG = dict(
    elastic=0.35,      # 弹性形变概率
    cutout=0.20,       # 随机遮挡概率
    contrast=0.30,     # 对比度扰动幅度
    cgap=0.20,         # 逐通道增益幅度
    sharpen=0.15,      # 锐化概率
    copy_paste=0.0,    # 血管片段复制粘贴概率（需要 donor 池，默认关闭）
)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _resize_chw(img: np.ndarray, size: int) -> np.ndarray:
    """(C,H,W) -> (C,size,size)，逐通道双线性插值。"""
    out = [cv2.resize(img[c], (size, size), interpolation=cv2.INTER_LINEAR)
           for c in range(img.shape[0])]
    return np.stack(out, axis=0)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    m = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.float32)


# --------------------------------------------------------------------------- #
# 几何增强
# --------------------------------------------------------------------------- #
def sample_patch(img: np.ndarray, mask: np.ndarray, size: int, rng: np.random.RandomState,
                 vessel_center_prob: float = 0.5, scale_range=(0.85, 1.25),
                 center=None):
    """从整图中裁剪一个 ``size x size`` 的 patch（可能先随机缩放）。

    Parameters
    ----------
    img : (C,H,W) float32
    mask : (H,W) float32, 0/1
    vessel_center_prob : float
        以随机血管像素为中心裁剪的概率，提高细血管的采样密度。
    scale_range : (float, float)
        裁剪边长的缩放范围（>1 表示放大局部，即缩小视野）。
    """
    _, H, W = img.shape

    # 1) 选取中心点
    if center is not None:
        cy, cx = center
        cy, cx = cy + rng.randint(-16, 17), cx + rng.randint(-16, 17)
    elif rng.rand() < vessel_center_prob and mask.any():
        ys, xs = np.nonzero(mask > 0.5)
        k = rng.randint(len(ys))
        cy = ys[k] + rng.randint(-16, 17)
        cx = xs[k] + rng.randint(-16, 17)
    else:
        cy, cx = rng.randint(H), rng.randint(W)

    # 2) 裁剪边长
    s = int(round(size * rng.uniform(*scale_range)))
    s = int(np.clip(s, 32, min(H, W)))

    y0 = int(np.clip(cy - s // 2, 0, H - s))
    x0 = int(np.clip(cx - s // 2, 0, W - s))
    p = img[:, y0:y0 + s, x0:x0 + s]
    m = mask[y0:y0 + s, x0:x0 + s]

    if s != size:
        p = _resize_chw(np.ascontiguousarray(p), size)
        m = _resize_mask(np.ascontiguousarray(m), size)
    return np.ascontiguousarray(p), np.ascontiguousarray(m)


def random_dihedral(img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState):
    """8 种二面体变换（4 旋转 x 是否水平翻转），不引入插值。"""
    k = rng.randint(4)
    if k:
        img = np.rot90(img, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(0, 1))
    if rng.rand() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def random_rotate(img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState,
                  max_angle: float = 45.0):
    """随机旋转（连续角度），用反射边界避免黑边干扰。"""
    if max_angle <= 0:
        return img, mask
    ang = rng.uniform(-max_angle, max_angle)
    if abs(ang) < 1.0:
        return img, mask
    H, W = mask.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
    hwc = np.transpose(img, (1, 2, 0))
    hwc = cv2.warpAffine(hwc, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    m = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(np.transpose(hwc, (2, 0, 1))), np.ascontiguousarray(m)


# --------------------------------------------------------------------------- #
# 强增强（第五轮新增）
# --------------------------------------------------------------------------- #
def elastic_deform(img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState,
                   alpha: float = 9.0, sigma: float = 6.0):
    """弹性形变（Simard 2003）：对位移场做高斯平滑后 remap。

    血管分割里非常有用——它能让血管的**粗细、弯曲程度**发生连续变化，
    等价于在没有新标注的情况下造出大量"解剖形态"不同的样本。
    """
    _, H, W = img.shape
    dx = rng.randn(H, W).astype(np.float32) * alpha
    dy = rng.randn(H, W).astype(np.float32) * alpha
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mapx = gx + dx
    mapy = gy + dy
    hwc = np.transpose(img, (1, 2, 0))
    hwc = cv2.remap(hwc, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    m = cv2.remap(mask, mapx, mapy, interpolation=cv2.INTER_NEAREST,
                  borderMode=cv2.BORDER_REFLECT_101)
    out = np.transpose(hwc, (2, 0, 1))
    return np.ascontiguousarray(out), np.ascontiguousarray(m)


def random_cutout(img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState,
                  n_holes: int = 1, len_ratio: float = 0.25):
    """随机矩形遮挡：图像与标签同时置 0（教模型"没证据就别猜"）。"""
    _, H, W = img.shape
    img = img.copy()
    mask = mask.copy()
    for _ in range(n_holes):
        lh = max(4, int(H * len_ratio * rng.uniform(0.5, 1.0)))
        lw = max(4, int(W * len_ratio * rng.uniform(0.5, 1.0)))
        y0 = rng.randint(0, max(1, H - lh))
        x0 = rng.randint(0, max(1, W - lw))
        img[:, y0:y0 + lh, x0:x0 + lw] = 0.0
        mask[y0:y0 + lh, x0:x0 + lw] = 0.0
    return img, mask


def copy_paste(img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState,
               donor_img: np.ndarray, donor_mask: np.ndarray, feather: int = 2):
    """把 donor 里的血管片段贴到当前 patch 的随机位置。

    只在"donor 血管膨胀 feather 像素"的区域做硬替换（边缘轻微羽化），
    并把目标 mask 对应位置置 1——这是小样本血管分割里最常用的造数手段之一。
    """
    dh, dw = donor_mask.shape[-2:]
    _, H, W = img.shape
    if dh > H or dw > W:
        return img, mask

    k = np.ones((2 * feather + 1, 2 * feather + 1), np.uint8)
    region = cv2.dilate((donor_mask > 0.5).astype(np.uint8), k)
    if region.sum() < 20:
        return img, mask

    y0 = rng.randint(0, H - dh + 1)
    x0 = rng.randint(0, W - dw + 1)
    sel = region.astype(bool)
    img = img.copy()
    mask = mask.copy()
    patch = img[:, y0:y0 + dh, x0:x0 + dw]
    patch[:, sel] = donor_img[:, sel]
    img[:, y0:y0 + dh, x0:x0 + dw] = patch
    m = mask[y0:y0 + dh, x0:x0 + dw]
    m[sel] = 1.0
    mask[y0:y0 + dh, x0:x0 + dw] = m
    return img, mask


def _sharpen(hwc: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(hwc, (0, 0), 1.2)
    return cv2.addWeighted(hwc, 1.6, blur, -0.6, 0.0)


def photometric(img: np.ndarray, rng: np.random.RandomState,
                bounded=None, gain: float = 0.2, bias: float = 0.08,
                noise: float = 0.03, gamma: float = 0.25,
                contrast: float = 0.0, cgap: float = 0.0, sharpen: float = 0.0):
    """对 (C,H,W) 通道栈施加温和的光度扰动。

    ``bounded`` 标记哪些通道取值在 [0,1]（可以安全地做 gamma 与截断）。
    """
    C = img.shape[0]
    if bounded is None:
        bounded = np.ones(C, dtype=bool)

    # 亮度增益 / 偏移（线性变换，对所有通道都安全）
    if rng.rand() < 0.8:
        a = 1.0 + rng.uniform(-gain, gain)
        b = rng.uniform(-bias, bias)
        img = img * a + b

    # 逐通道增益（模拟不同相机/曝光下的颜色差异）
    if cgap > 0 and rng.rand() < 0.6:
        g = 1.0 + rng.uniform(-cgap, cgap, size=(C, 1, 1)).astype(np.float32)
        img = img * g

    # 对比度（围绕通道均值缩放，只对有界通道）
    if contrast > 0 and rng.rand() < 0.6 and bounded.any():
        idx = np.nonzero(bounded)[0]
        a = 1.0 + rng.uniform(-contrast, contrast)
        sub = img[idx]
        mu = sub.mean(axis=(1, 2), keepdims=True)
        img[idx] = (sub - mu) * a + mu

    # gamma（只对 [0,1] 通道）
    if rng.rand() < 0.5 and bounded.any():
        g = math.exp(rng.uniform(-gamma, gamma))
        idx = np.nonzero(bounded)[0]
        sel = np.clip(img[idx], 1e-4, None) ** g
        img[idx] = sel

    # 高斯噪声
    if noise > 0 and rng.rand() < 0.3:
        std = rng.uniform(0, noise)
        img = img + rng.randn(*img.shape).astype(np.float32) * std

    # 轻微模糊
    if rng.rand() < 0.15:
        hwc = np.transpose(img, (1, 2, 0))
        hwc = cv2.GaussianBlur(hwc, (3, 3), 0.6)
        img = np.transpose(hwc, (2, 0, 1))

    # 锐化（只对有界通道，模拟 JPEG/降噪后的硬边缘）
    if sharpen > 0 and rng.rand() < sharpen and bounded.any():
        idx = np.nonzero(bounded)[0]
        hwc = np.transpose(np.ascontiguousarray(img[idx]), (1, 2, 0))
        img = img.copy()
        img[idx] = np.transpose(_sharpen(np.ascontiguousarray(hwc)), (2, 0, 1))

    # [0,1] 通道截断回合法范围（无界通道保持原样）
    if bounded.any():
        idx = np.nonzero(bounded)[0]
        img[idx] = np.clip(img[idx], 0.0, 1.0)

    return np.ascontiguousarray(img.astype(np.float32))


def augment_pair(img: np.ndarray, mask: np.ndarray, rng: np.random.RandomState,
                 bounded=None, rot_angle: float = 45.0, elastic: float = 0.0,
                 cutout: float = 0.0, contrast: float = 0.0, cgap: float = 0.0,
                 sharpen: float = 0.0, copy_paste_p: float = 0.0,
                 donor=None, **photo_kwargs):
    """几何 + 光度增强的组合入口。

    Parameters
    ----------
    elastic : float
        弹性形变概率（0 表示关闭）。
    cutout : float
        随机遮挡概率。
    copy_paste_p : float
        血管片段复制粘贴概率，需要传入 ``donor=(img, mask)``。
    """
    img, mask = random_dihedral(img, mask, rng)
    img, mask = random_rotate(img, mask, rng, rot_angle)
    if elastic > 0 and rng.rand() < elastic:
        img, mask = elastic_deform(img, mask, rng)
    if copy_paste_p > 0 and donor is not None and rng.rand() < copy_paste_p:
        img, mask = copy_paste(img, mask, rng, donor[0], donor[1])
    if cutout > 0 and rng.rand() < cutout:
        img, mask = random_cutout(img, mask, rng, n_holes=1, len_ratio=0.22)
    img = photometric(img, rng, bounded=bounded, contrast=contrast,
                      cgap=cgap, sharpen=sharpen, **photo_kwargs)
    return np.ascontiguousarray(img), (mask > 0.5).astype(np.float32)
