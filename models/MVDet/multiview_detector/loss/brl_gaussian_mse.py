import torch
from torch import nn
import torch.nn.functional as F


class BRLGaussianMSE(nn.Module):
    """
    Heatmap adaptation of Background Recalibration Loss (BRL / Selective-IoU idea).

    Soft GT is built like GaussianMSE. Pixels are split into:
      - positive: soft_gt >= pos_thr
      - confuse:  background where prediction is high (possible missing annotation)
      - easy neg: remaining background

    On confuse pixels, either:
      - mirror=True:  pull toward 1 (do not force background), scaled by beta
      - mirror=False: keep MSE to soft GT but down-weighted by beta

    Optional per-pixel hooks (both None -> original code path):
      - bg_weight:   weight on the easy-negative term
      - ignore_mask: background pixels removed from easy-neg and confuse
                     (still counted in the normaliser, so the loss stays a per-pixel mean)
    """

    def __init__(self, pos_thr=0.1, confuse_pred_thr=0.3, beta=0.1, mirror=True):
        super().__init__()
        self.pos_thr = pos_thr
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def forward(self, x, target, kernel, bg_weight=None, ignore_mask=None):
        soft_gt = self._traget_transform(x, target, kernel)

        pos_mask = soft_gt >= self.pos_thr
        bg_mask = ~pos_mask
        if ignore_mask is not None:
            ignored_mask = bg_mask & ignore_mask
            bg_mask = bg_mask & ~ignore_mask

        # assignment must not backprop through the thresholding
        confuse_mask = bg_mask & (x.detach() >= self.confuse_pred_thr)
        easy_neg_mask = bg_mask & ~confuse_mask

        loss = x.new_zeros(())
        count = x.new_zeros(())

        if pos_mask.any():
            loss = loss + F.mse_loss(x[pos_mask], soft_gt[pos_mask], reduction="sum")
            count = count + pos_mask.sum()

        if easy_neg_mask.any():
            if bg_weight is None:
                loss = loss + F.mse_loss(
                    x[easy_neg_mask], soft_gt[easy_neg_mask], reduction="sum"
                )
            else:
                easy_err = (x[easy_neg_mask] - soft_gt[easy_neg_mask]).pow(2)
                loss = loss + (bg_weight[easy_neg_mask] * easy_err).sum()
            count = count + easy_neg_mask.sum()

        if confuse_mask.any():
            confuse_tgt = torch.ones_like(x[confuse_mask]) if self.mirror else soft_gt[confuse_mask]
            loss = loss + self.beta * F.mse_loss(
                x[confuse_mask], confuse_tgt, reduction="sum"
            )
            count = count + confuse_mask.sum()

        if ignore_mask is not None:
            count = count + ignored_mask.sum()

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
