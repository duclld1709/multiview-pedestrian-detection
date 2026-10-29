import torch
from torch import nn
from multiview_detector.utils.tensor_utils import _sigmoid


class PseudoLabelLoss(nn.Module):
    """
    Offline-detector pseudo labels on the BEV heatmap (focal version).

    pseudo_target is a max of Gaussian blobs (peak = 1) around projected detections,
    pseudo_weight the same blobs scaled by detector score; both are zero near kept GT.
    Two parts, switchable separately:
      - neg_reweight: scale the GT negative term by (1 - pseudo_weight)
      - peak_loss:    score-weighted positive focal term at pseudo peaks (x peak_weight);
                      the peaks are then taken out of the GT negative / confuse terms
    """

    def __init__(self, peak_weight=0.01, neg_reweight=True, peak_loss=True):
        super().__init__()
        if not (neg_reweight or peak_loss):
            raise ValueError('PseudoLabelLoss needs neg_reweight and/or peak_loss')
        self.peak_weight = peak_weight
        self.neg_reweight = neg_reweight
        self.peak_loss = peak_loss

    def masks(self, target, pseudo_target, pseudo_weight):
        """Returns (peak_mask, neg_weight, ignore_mask) for the base loss hooks."""
        peak_mask = pseudo_target.eq(1) & target.lt(1)
        neg_weight = (1.0 - pseudo_weight).clamp(0.0, 1.0) if self.neg_reweight else None
        ignore_mask = peak_mask if self.peak_loss else None
        return peak_mask, neg_weight, ignore_mask

    def forward(self, output, target, peak_mask, pseudo_weight):
        """``output`` is logits, as for FocalLoss."""
        if not self.peak_loss:
            return output.new_zeros(())
        output = _sigmoid(output)
        loss = torch.log(output) * torch.pow(1 - output, 2) * pseudo_weight * peak_mask.float()
        # normalised by the number of GT peaks, like the GT focal loss
        return self.peak_weight * -loss.sum() / target.eq(1).float().sum().clamp(min=1)
