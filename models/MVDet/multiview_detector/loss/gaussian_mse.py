import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GaussianMSE(nn.Module):
    """
    Original MVDet heatmap loss: MSE between prediction and the Gaussian-smoothed GT.

    Optional per-pixel hooks (both None -> original code path):
      - bg_weight:   weight on the regression term. GaussianMSE has no positive/background
                     split, so it applies to every pixel; callers keep it at 1 on GT blobs.
      - ignore_mask: pixels that contribute nothing (still counted in the mean).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel, bg_weight=None, ignore_mask=None):
        target = self._traget_transform(x, target, kernel)
        if bg_weight is None and ignore_mask is None:
            return F.mse_loss(x, target)
        weight = bg_weight if bg_weight is not None else torch.ones_like(x)
        if ignore_mask is not None:
            weight = weight * (~ignore_mask).to(x.dtype)
        return (weight * (x - target).pow(2)).sum() / x.numel()

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target
