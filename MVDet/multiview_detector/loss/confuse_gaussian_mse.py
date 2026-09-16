import torch
from torch import nn
import torch.nn.functional as F


class ConfuseGaussianMSE(nn.Module):
    """
    Keeps the Gaussian soft target (identical to GaussianMSE), but does NOT use
    pos_thr or any other threshold on soft_gt to gate which pixels count as
    "positive" vs "background" vs "confuse".

    Confuse-candidate status is determined purely by the model's own
    prediction: pred >= c (self.confuse_pred_thr). The STRENGTH of the confuse
    "forgiveness" then scales continuously with (1 - soft_gt):
      - soft_gt -> 1 (pixel near a KNOWN/KEPT person): forgiveness -> 0, loss
        falls back to plain MSE-to-soft_gt -- known people are protected
        automatically, with no separate threshold needed.
      - soft_gt -> 0 (far from any known annotation): forgiveness -> full
        strength -- a confident prediction there is most likely a genuinely
        missed detection under partial annotation.

    So the only threshold anywhere in this loss is c itself -- no pos_thr, no
    other hidden cutoff.
    """

    def __init__(self, confuse_pred_thr=0.3, beta=0.1, mirror=True):
        super().__init__()
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(
                target,
                kernel.float().to(target.device),
                padding=int((kernel.shape[-1] - 1) / 2),
            )
        return target

    def forward(self, x, target, kernel):
        soft_gt = self._traget_transform(x, target, kernel)

        normal_sq = (x - soft_gt) ** 2
        if self.mirror:
            confuse_sq = (x - torch.ones_like(x)) ** 2
        else:
            confuse_sq = normal_sq

        # assignment must not backprop through the thresholding
        confuse_mask = (x.detach() >= self.confuse_pred_thr).float()

        # At confuse pixels: as soft_gt -> 1 (near a KNOWN person), the (1 - soft_gt) factor
        # kills the confuse_sq term and what remains is soft_gt * normal_sq ~= normal_sq --
        # full ordinary supervision, auto-protected, no threshold needed. As soft_gt -> 0
        # (far from anyone known), it reduces to beta * confuse_sq -- full confuse strength.
        confuse_term = (1 - soft_gt) * self.beta * confuse_sq + soft_gt * normal_sq
        blended = confuse_mask * confuse_term + (1 - confuse_mask) * normal_sq

        return blended.mean()
