import torch
from torch import nn


class ConfuseGaussianMSE(nn.Module):
    """
    MVDeTr port of multiview_detector/loss/confuse_gaussian_mse.py from the MVDet fork.

    Difference from the MVDet version: MVDet builds its Gaussian soft target inside the loss
    itself (_traget_transform convolves a hard occupancy map with a kernel at every call).
    MVDeTr instead bakes the final Gaussian heatmap once per frame in frameDataset.get_gt()
    (draw_umich_gaussian, CenterNet-style, peak=1 decaying outward) -- so `target` passed in
    here already IS soft_gt, no kernel argument needed.

    Same math otherwise, no pos_thr gate anywhere. A pixel is confuse-candidate purely by the
    model's own prediction: pred >= c (self.confuse_pred_thr). The confuse "forgiveness"
    strength then scales continuously with (1 - soft_gt):
      - soft_gt -> 1 (pixel at/near a KNOWN/KEPT person): forgiveness -> 0, falls back to
        plain MSE-to-soft_gt -- known people are protected automatically, no separate
        threshold needed.
      - soft_gt -> 0 (far from any known annotation): forgiveness -> full strength -- a
        confident prediction there is most likely a genuinely missed detection under
        partial annotation.
    """

    def __init__(self, confuse_pred_thr=0.3, beta=0.1, mirror=True):
        super().__init__()
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def forward(self, x, target):
        soft_gt = target.to(x.device)

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
