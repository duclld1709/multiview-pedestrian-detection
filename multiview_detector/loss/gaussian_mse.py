import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def pseudo_peak_masks(soft_gt, pseudo_target=None, pseudo_weight=None):
    """Pseudo peak cells and the scale applied to the GT regression penalty.

    Peaks (pseudo_target == 1 off the GT peaks) are supervised by ``pseudo_peak_mse_loss``
    and removed from the GT/confuse branches; the rest of the pseudo blob scales
    the GT penalty by ``1 - pseudo_weight``.
    """
    if pseudo_target is None:
        return None, torch.ones_like(soft_gt)
    pseudo_target = pseudo_target.to(soft_gt.device, dtype=soft_gt.dtype)
    pseudo_weight = pseudo_weight.to(soft_gt.device, dtype=soft_gt.dtype)
    peak_inds = pseudo_target.eq(1).to(soft_gt.dtype) * soft_gt.lt(1).to(soft_gt.dtype)
    neg_scale = (1.0 - pseudo_weight).clamp(0.0, 1.0) * (1.0 - peak_inds)
    return peak_inds, neg_scale


def pseudo_peak_mse_loss(x, soft_gt, pseudo_target, pseudo_weight, normalizer):
    """Score-weighted MSE toward 1 at pseudo peaks, normalised like the GT MSE (by ``normalizer``)."""
    pseudo_weight = pseudo_weight.to(x.device, dtype=x.dtype)
    peak_inds, _ = pseudo_peak_masks(soft_gt, pseudo_target, pseudo_weight)
    loss = pseudo_weight * peak_inds * (x - 1.0).pow(2)
    return loss.sum() / max(float(normalizer), 1.0)


class GaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel, pseudo_target=None, pseudo_weight=None):
        target = self._traget_transform(x, target, kernel)
        if pseudo_target is None:
            return F.mse_loss(x, target)
        _, neg_scale = pseudo_peak_masks(target, pseudo_target, pseudo_weight)
        return (neg_scale * (x - target).pow(2)).sum() / x.numel()

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target
