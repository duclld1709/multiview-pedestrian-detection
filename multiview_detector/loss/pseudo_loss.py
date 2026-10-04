import torch.nn.functional as F


def pseudo_heatmap_loss(map_res, pseudo_target=None, pseudo_weight=None, loss_weight=0.0):
    """Confidence-weighted MSE between the BEV heatmap and projected detector pseudo labels.

    Same formulation as MVDet's pseudo branch: only cells with weight > 0 count toward the mean.
    """
    if pseudo_target is None or pseudo_weight is None or loss_weight <= 0:
        return map_res.new_zeros(())
    target = pseudo_target.to(map_res.device, dtype=map_res.dtype)
    weights = pseudo_weight.to(map_res.device, dtype=map_res.dtype)
    if target.shape[-2:] != map_res.shape[-2:]:
        target = F.interpolate(target, size=map_res.shape[-2:], mode='bilinear', align_corners=False)
        weights = F.interpolate(weights, size=map_res.shape[-2:], mode='bilinear', align_corners=False)
    weighted_error = weights * (map_res - target).pow(2)
    active_count = (weights > 0).sum().clamp(min=1).to(map_res.dtype)
    return loss_weight * weighted_error.sum() / active_count
