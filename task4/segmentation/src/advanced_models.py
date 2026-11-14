"""Round8 candidates: region-only inference, explicit auxiliary training outputs."""

import torch
from torch import nn
from .models import UNet
from .vendor.fr_unet import FR_UNet
from .vendor.dsconv import DSConv_pro


class AuxiliaryHead(nn.Module):
    def __init__(self, region, channels):
        super().__init__()
        self.region = region
        hidden = max(4, channels // 2)
        def branch():
            return nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1),
                                 nn.BatchNorm2d(hidden), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        self.boundary, self.skeleton = branch(), branch()

    def forward(self, x):
        region = self.region(x)
        if not self.training:
            return region
        return {'region': region, 'boundary': self.boundary(x), 'skeleton': self.skeleton(x)}


class SnakeInputBlock(nn.Module):
    """Identity-initialized fusion keeps a pretrained U-Net's initial function."""
    def __init__(self, standard, in_channels, width, kernel):
        super().__init__()
        if width < 4 or width % 4 or kernel < 3 or kernel % 2 == 0:
            raise ValueError('DSConv requires width divisible by 4 and odd kernel >=3')
        self.standard = standard
        self.snake_x = DSConv_pro(in_channels, width, kernel_size=kernel, morph=0, device='cpu')
        self.snake_y = DSConv_pro(in_channels, width, kernel_size=kernel, morph=1, device='cpu')
        self.fuse = nn.Conv2d(3*width, width, 1, bias=False)
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.weight[:, :width, 0, 0].copy_(torch.eye(width))

    def forward(self, image):
        return self.fuse(torch.cat([self.standard(image), self.snake_x(image), self.snake_y(image)], 1))


def build_advanced(cfg, name):
    if name == 'fr_unet':
        if cfg.depth != 4:
            raise ValueError('FR-UNet uses its fixed topology; keep depth=4 for common padding')
        return FR_UNet(num_channels=len(cfg.channel_list), num_classes=1,
                       feature_scale=cfg.fr_feature_scale, dropout=cfg.fr_dropout)
    model = UNet(in_channels=len(cfg.channel_list), out_channels=1, base=cfg.base_channels,
                 depth=cfg.depth, norm=cfg.norm, dropout=cfg.dropout, bilinear=cfg.bilinear)
    if name == 'unet_aux':
        model.head = AuxiliaryHead(model.head, cfg.base_channels)
    elif name == 'unet_dsconv':
        model.enc[0] = SnakeInputBlock(model.enc[0], len(cfg.channel_list), cfg.base_channels, cfg.ds_kernel_size)
    else:
        raise ValueError(name)
    return model
