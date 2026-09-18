import torch
from torch import nn
import torch.nn.functional as F


class BRLGaussianMSEv2(nn.Module):
    """
    Heatmap BRL (advisor formulation) — three disjoint regions, one threshold:

      1. positive          : pixels that are ground-truth (hard GT peaks)
      2. confused positive : among the rest, pred >= c
      3. background        : among the rest, pred < c

    Only ``confuse_pred_thr`` (c) is used for the non-GT split.
    Positive / background: MSE toward soft (Gaussian) GT.
    Confuse: mirror toward 1 (or down-weighted MSE-to-GT if mirror=False), × beta.
    """

    def __init__(self, confuse_pred_thr=0.3, beta=0.1, mirror=True):
        super().__init__()
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def forward(self, x, target, kernel):
        # hard GT at heatmap resolution, then soft Gaussian target
        hard_gt = F.adaptive_max_pool2d(target, x.shape[2:])
        soft_gt = self._gaussian_target(hard_gt, kernel)

        # 1) GT points first → positive
        pos_mask = hard_gt > 0

        # 2)–3) one threshold on the remaining pixels
        rest_mask = ~pos_mask
        confuse_mask = rest_mask & (x.detach() >= self.confuse_pred_thr)
        bg_mask = rest_mask & ~confuse_mask

        loss = x.new_zeros(())
        count = x.new_zeros(())

        if pos_mask.any():
            loss = loss + F.mse_loss(x[pos_mask], soft_gt[pos_mask], reduction="sum")
            count = count + pos_mask.sum()

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

        if bg_mask.any():
            loss = loss + F.mse_loss(x[bg_mask], soft_gt[bg_mask], reduction="sum")
            count = count + bg_mask.sum()

        return loss / count.clamp(min=1).float()

    def _traget_transform(self, x, target, kernel):
        """Same API as GaussianMSE — used by trainer visualization."""
        hard_gt = F.adaptive_max_pool2d(target, x.shape[2:])
        return self._gaussian_target(hard_gt, kernel)

    def _gaussian_target(self, hard_gt, kernel):
        with torch.no_grad():
            soft_gt = F.conv2d(
                hard_gt,
                kernel.float().to(hard_gt.device),
                padding=int((kernel.shape[-1] - 1) / 2),
            )
        return soft_gt
