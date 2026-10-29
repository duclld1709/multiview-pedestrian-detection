import torch.nn.functional as F
from torch import nn


class HeatmapLoss(nn.Module):
    """
    Base heatmap loss (GaussianMSE / BRLGaussianMSE) plus an optional PseudoLabelLoss.

    Returns {'base': ..., 'pseudo': ...}. Without a pseudo module or pseudo inputs the
    base loss runs its original code path and 'pseudo' is 0.
    """

    def __init__(self, base, pseudo=None):
        super().__init__()
        self.base = base
        self.pseudo = pseudo

    def forward(self, x, target, kernel, pseudo_target=None, pseudo_weight=None):
        if self.pseudo is None or pseudo_target is None:
            return {'base': self.base(x, target, kernel), 'pseudo': x.new_zeros(())}

        pseudo_target = pseudo_target.to(x.device, dtype=x.dtype)
        pseudo_weight = pseudo_weight.to(x.device, dtype=x.dtype)
        if pseudo_target.shape[-2:] != x.shape[-2:]:
            # nearest keeps the exact 1.0 peaks that the peak mask relies on
            pseudo_target = F.interpolate(pseudo_target, size=x.shape[-2:], mode='nearest')
            pseudo_weight = F.interpolate(pseudo_weight, size=x.shape[-2:], mode='nearest')

        soft_gt = self.base._traget_transform(x, target, kernel)
        peak_mask, bg_weight, ignore_mask = self.pseudo.masks(soft_gt, pseudo_target, pseudo_weight)
        return {'base': self.base(x, target, kernel, bg_weight=bg_weight, ignore_mask=ignore_mask),
                'pseudo': self.pseudo(x, peak_mask, pseudo_weight)}
