import math

import torch.nn as nn
import torch.nn.functional as F


class NoiseHead(nn.Module):
    """
    Predicts a positive per-pixel noise scale from the shared map-classifier trunk.

    The convolution geometry mirrors the final heatmap convolution so the sigma
    branch sees the same receptive field. The output is softplus(raw) + sigma_min.
    """

    def __init__(self, in_ch, kernel_size, padding, dilation=1, init_sigma=0.7, sigma_min=0.1):
        super().__init__()
        assert init_sigma > sigma_min
        self.sigma_min = sigma_min
        self.conv = nn.Conv2d(in_ch, 1, kernel_size, padding=padding, dilation=dilation, bias=True)
        nn.init.normal_(self.conv.weight, std=1e-3)
        nn.init.constant_(self.conv.bias, math.log(math.expm1(init_sigma - sigma_min)))

    def forward(self, h, out_shape):
        raw = F.interpolate(self.conv(h), out_shape, mode='bilinear')
        return F.softplus(raw) + self.sigma_min
