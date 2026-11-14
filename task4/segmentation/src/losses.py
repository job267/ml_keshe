"""损失函数。

血管像素只占约 8%，属于强类别不平衡的分割任务。
推荐 ``BCE + Dice``：BCE 提供稳定的逐像素梯度，Dice 直接优化重叠度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """1 - soft Dice（在 batch 维度取平均）。"""

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + self.eps
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        return 1.0 - (num / den).mean()


class FocalLoss(nn.Module):
    """二分类 focal loss。"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logit, target):
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        p = torch.sigmoid(logit)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class BCEDiceLoss(nn.Module):
    """weighted BCE + Dice。"""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: float = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss()
        self.pos_weight = pos_weight

    def forward(self, logit, target):
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logit.device, dtype=logit.dtype)
        bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)
        d = self.dice(logit, target)
        return self.bce_weight * bce + self.dice_weight * d


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = SoftDiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logit, target):
        return self.focal_weight * self.focal(logit, target) + \
            self.dice_weight * self.dice(logit, target)


LOSSES = {
    "bce_dice": BCEDiceLoss,
    "focal_dice": FocalDiceLoss,
    "dice": SoftDiceLoss,
}


def soft_erode(image):
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeleton(image, iterations=10):
    """可微形态学骨架，参考 clDice 原始算法；不做硬阈值。"""
    opened = F.max_pool2d(soft_erode(image), 3, stride=1, padding=1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, stride=1, padding=1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class SoftClDiceLoss(nn.Module):
    """1-clDice，float32 骨架计算避免 AMP 下细结构数值问题。"""

    def __init__(self, iterations=10, smooth=1.0):
        super().__init__()
        self.iterations, self.smooth = iterations, smooth

    def forward(self, logit, target):
        prob, target = torch.sigmoid(logit.float()), target.float()
        skel_p = soft_skeleton(prob, self.iterations)
        with torch.no_grad():
            skel_g = soft_skeleton(target, self.iterations)
        dims = (1, 2, 3)
        precision = ((skel_p * target).sum(dims) + self.smooth) / (skel_p.sum(dims) + self.smooth)
        recall = ((skel_g * prob).sum(dims) + self.smooth) / (skel_g.sum(dims) + self.smooth)
        return (1 - 2 * precision * recall / (precision + recall).clamp_min(1e-7)).mean()


class BCEDiceClDiceLoss(nn.Module):
    def __init__(self, weight=0.1, iterations=10):
        super().__init__()
        if weight < 0 or iterations < 1:
            raise ValueError("cldice_weight >=0，cldice_iters >=1")
        self.region = BCEDiceLoss()
        self.topology = SoftClDiceLoss(iterations)
        self.weight = weight

    def forward(self, logit, target):
        return self.region(logit, target) + self.weight * self.topology(logit, target)


def build_loss(name: str, cfg=None) -> nn.Module:
    key = (name or "bce_dice").lower()
    if key == "bce_dice_cldice":
        return BCEDiceClDiceLoss(getattr(cfg, "cldice_weight", 0.1),
                                getattr(cfg, "cldice_iters", 10))
    if key not in LOSSES:
        raise ValueError(f"未知损失 '{key}'，可选: {sorted(LOSSES)}")
    return LOSSES[key]()


def region_logits(output):
    return output['region'] if isinstance(output, dict) else output


def training_loss(output, target, criterion, cfg, auxiliary=None, epoch=1):
    loss = criterion(region_logits(output), target)
    if not isinstance(output, dict):
        return loss
    if auxiliary is None or 'skeleton' not in auxiliary:
        raise ValueError('Auxiliary model training requires skeleton labels from the training patch')
    if cfg.aux_weight < 0 or cfg.aux_warmup_epochs < 0:
        raise ValueError('aux_weight and aux_warmup_epochs must be nonnegative')
    ramp = min(1., epoch / max(1, cfg.aux_warmup_epochs))
    boundary = F.max_pool2d(target, 3, 1, 1) + F.max_pool2d(-target, 3, 1, 1)
    skeleton = auxiliary['skeleton'].to(target.device)
    auxiliary_criterion = BCEDiceLoss()
    aux = auxiliary_criterion(output['boundary'], boundary)
    aux = aux + auxiliary_criterion(output['skeleton'], skeleton)
    return loss + cfg.aux_weight * ramp * aux
