import torch
from torch import nn


class PseudoLabelLoss(nn.Module):
    """
    Offline-detector pseudo labels on the BEV heatmap.

    pseudo_target is a max of Gaussian blobs (peak = 1) around projected detections,
    pseudo_weight the same blobs scaled by detector score; both are zero near kept GT.
    Two parts, switchable separately:
      - neg_reweight: scale the GT background term by (1 - pseudo_weight)
      - positive_region: peak (default) or all nonzero Gaussian support; selected cells leave the base negative/confuse terms
      - peak_loss:    pull selected pseudo cells toward 1 with a score-weighted MSE (x peak_weight);
                      the selected cells are then taken out of the GT background / confuse terms
    """

    def __init__(self, peak_weight=0.01, neg_reweight=True, peak_loss=True, positive_region='peak'):
        super().__init__()
        if not (neg_reweight or peak_loss):
            raise ValueError('PseudoLabelLoss needs neg_reweight and/or peak_loss')
        if positive_region not in ('peak', 'gaussian'):
            raise ValueError('pseudo_positive_region must be peak or gaussian')
        if positive_region == 'gaussian' and not peak_loss:
            raise ValueError('pseudo_positive_region=gaussian requires the pseudo positive loss')
        self.positive_region = positive_region
        self.peak_weight = peak_weight
        self.neg_reweight = neg_reweight
        self.peak_loss = peak_loss

    def masks(self, soft_gt, pseudo_target, pseudo_weight):
        """Returns (peak_mask, bg_weight, ignore_mask) for the base loss hooks."""
        # In gaussian mode all rasterized support uses the positive branch.
        # Gaussian x detector confidence remains the weight, not the target.
        support = pseudo_target.gt(0) if self.positive_region == 'gaussian' else pseudo_target.eq(1)
        peak_mask = support & soft_gt.lt(1)
        bg_weight = (1.0 - pseudo_weight).clamp(0.0, 1.0) if self.neg_reweight else None
        ignore_mask = peak_mask if self.peak_loss else None
        return peak_mask, bg_weight, ignore_mask

    def forward(self, x, peak_mask, pseudo_weight):
        if not self.peak_loss:
            return x.new_zeros(())
        # normalised by the pixel count, like the GT term
        loss = (pseudo_weight * peak_mask.to(x.dtype) * (x - 1.0).pow(2)).sum() / x.numel()
        return self.peak_weight * loss
