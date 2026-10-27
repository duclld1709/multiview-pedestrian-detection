import torch.nn.functional as F
from multiview_detector.loss.gaussian_mse import pseudo_peak_mse_loss


def prepare_pseudo_maps(map_res, pseudo_target=None, pseudo_weight=None, loss_weight=0.0):
    """Move pseudo maps onto ``map_res``; returns (None, None) when the pseudo branch is off.

    Same handling as MVDet_PL's ``PerspectiveTrainer._loss_components``.
    """
    if pseudo_target is None or pseudo_weight is None or loss_weight <= 0:
        return None, None
    pseudo_target = pseudo_target.to(map_res.device, dtype=map_res.dtype)
    pseudo_weight = pseudo_weight.to(map_res.device, dtype=map_res.dtype)
    if pseudo_target.shape[-2:] != map_res.shape[-2:]:
        # nearest keeps the exact 1.0 peaks that pseudo_peak_masks relies on
        pseudo_target = F.interpolate(pseudo_target, size=map_res.shape[-2:], mode='nearest')
        pseudo_weight = F.interpolate(pseudo_weight, size=map_res.shape[-2:], mode='nearest')
    return pseudo_target, pseudo_weight


def pseudo_heatmap_loss(map_res, map_gt, map_kernel, criterion, pseudo_target=None, pseudo_weight=None,
                        loss_weight=0.0):
    """``loss_weight`` x score-weighted pseudo-peak MSE; expects maps from ``prepare_pseudo_maps``.

    Both GaussianMSE and BRLGaussianMSE normalise the BEV term by the pixel count.
    """
    if pseudo_target is None:
        return map_res.new_zeros(())
    soft_gt = criterion._traget_transform(map_res, map_gt.to(map_res.device), map_kernel)
    return loss_weight * pseudo_peak_mse_loss(map_res, soft_gt, pseudo_target, pseudo_weight, map_res.numel())
