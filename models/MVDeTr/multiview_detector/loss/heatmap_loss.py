import torch.nn.functional as F
from torch import nn


class HeatmapLoss(nn.Module):
    """
    Base heatmap loss (FocalLoss / BRLFocalLoss) plus an optional PseudoLabelLoss.

    Returns {'base': ..., 'pseudo': ...}. Without a pseudo module or pseudo inputs the
    base loss runs its original code path and 'pseudo' is 0.
    """

    def __init__(self, base, pseudo=None):
        super().__init__()
        self.base = base
        self.pseudo = pseudo

    def forward(self, output, target, mask=None, pseudo_target=None, pseudo_weight=None):
        if self.pseudo is None or pseudo_target is None:
            return {'base': self.base(output, target, mask), 'pseudo': output.new_zeros(())}

        target = target.to(output.device)
        pseudo_target = pseudo_target.to(output.device, dtype=output.dtype)
        pseudo_weight = pseudo_weight.to(output.device, dtype=output.dtype)
        if pseudo_target.shape[-2:] != output.shape[-2:]:
            # nearest keeps the exact 1.0 peaks that the peak mask relies on
            pseudo_target = F.interpolate(pseudo_target, size=output.shape[-2:], mode='nearest')
            pseudo_weight = F.interpolate(pseudo_weight, size=output.shape[-2:], mode='nearest')

        peak_mask, neg_weight, ignore_mask = self.pseudo.masks(target, pseudo_target, pseudo_weight)
        return {'base': self.base(output, target, mask, neg_weight=neg_weight, ignore_mask=ignore_mask),
                'pseudo': self.pseudo(output, target, peak_mask, pseudo_weight)}
