import torch
from torch import nn
import torch.nn.functional as F
from multiview_detector.loss.gaussian_mse import pseudo_peak_masks


class BRLGaussianMSE(nn.Module):
    """
    Heatmap adaptation of Background Recalibration Loss (BRL / Selective-IoU idea).

    Soft GT is built like GaussianMSE. Pixels are split into:
      - positive: soft_gt >= pos_thr
      - confuse:  background where prediction is high (possible missing annotation)
      - easy neg: remaining background

    On confuse pixels, either:
      - mirror=True: pull toward 1 (do not force background), scaled by beta
      - mirror=False: keep MSE-to-GT but down-weighted by beta

    Optional pseudo targets scale the easy-negative branch by ``1 - pseudo_weight``
    and take pseudo peaks out of the easy-negative and confuse branches.
    """

    def __init__(self, pos_thr=0.1, confuse_pred_thr=0.3, beta=0.1, mirror=True):
        super().__init__()
        self.pos_thr = pos_thr
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def forward(self, x, target, kernel, pseudo_target=None, pseudo_weight=None):
        soft_gt = self._traget_transform(x, target, kernel)

        pos_mask = soft_gt >= self.pos_thr
        bg_mask = ~pos_mask
        peak_inds, neg_scale = pseudo_peak_masks(soft_gt, pseudo_target, pseudo_weight)

        # assignment must not backprop through the thresholding
        confuse_mask = bg_mask & (x.detach() >= self.confuse_pred_thr)
        if peak_inds is not None:
            confuse_mask = confuse_mask & ~peak_inds.bool()
        easy_neg_mask = bg_mask & ~confuse_mask

        loss = x.new_zeros(())
        count = x.new_zeros(())

        if pos_mask.any():
            loss = loss + F.mse_loss(x[pos_mask], soft_gt[pos_mask], reduction="sum")
            count = count + pos_mask.sum()

        if easy_neg_mask.any():
            easy_err = (x[easy_neg_mask] - soft_gt[easy_neg_mask]).pow(2)
            loss = loss + (neg_scale[easy_neg_mask] * easy_err).sum()
            count = count + easy_neg_mask.sum()

        if confuse_mask.any():
            if self.mirror:
                mirror_tgt = torch.ones_like(x[confuse_mask])
                loss = loss + self.beta * F.mse_loss(
                    x[confuse_mask], mirror_tgt, reduction="sum"
                )
            else:
                loss = loss + self.beta * F.mse_loss(
                    x[confuse_mask], soft_gt[confuse_mask], reduction="sum"
                )
            count = count + confuse_mask.sum()

        return loss / count.clamp(min=1).float()

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(
                target,
                kernel.float().to(target.device),
                padding=int((kernel.shape[-1] - 1) / 2),
            )
        return target
