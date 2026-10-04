# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from multiview_detector.utils.tensor_utils import _transpose_and_gather_feat, _sigmoid
import torch.nn.functional as F


def pseudo_peak_masks(target, pseudo_target=None, pseudo_weight=None):
    """Pseudo peak cells and the scale applied to the GT negative branch.

    Peaks (pseudo_target == 1 off the GT peaks) are supervised by ``pseudo_peak_loss``
    and removed from the negative/confuse branches; the rest of the pseudo blob scales
    the negative penalty by ``1 - pseudo_weight``.
    """
    if pseudo_target is None:
        return None, torch.ones_like(target)
    pseudo_target = pseudo_target.to(target.device)
    pseudo_weight = pseudo_weight.to(target.device)
    peak_inds = pseudo_target.eq(1).float() * target.lt(1).float()
    neg_scale = (1.0 - pseudo_weight).clamp(0.0, 1.0) * (1.0 - peak_inds)
    return peak_inds, neg_scale


def pseudo_peak_loss(output, target, pseudo_target, pseudo_weight):
    """Score-weighted CornerNet positive term at pseudo peaks, normalised like the GT focal loss.

    ``output`` is logits, as for ``FocalLoss``.
    """
    output = _sigmoid(output)
    target = target.to(output.device)
    pseudo_weight = pseudo_weight.to(output.device)
    peak_inds, _ = pseudo_peak_masks(target, pseudo_target, pseudo_weight)
    loss = torch.log(output) * torch.pow(1 - output, 2) * pseudo_weight * peak_inds
    return -loss.sum() / target.eq(1).float().sum().clamp(min=1)


class BRLFocalLoss_v2(nn.Module):
    """CornerNet FocalLoss + Background Recalibration (BRL) for missing annotations.

    Pixels with soft GT below ``pos_thr`` are background. Among them, locations where
    the prediction (detached) exceeds ``confuse_pred_thr`` are treated as *confuse*
    (possible unlabeled objects): either mirror the positive focal branch toward 1
    (``mirror=True``) or keep the negative branch but scale by ``beta``.
    Soft Gaussian rings (pos_thr <= target < 1) keep standard CornerNet neg weights.
    Optional pseudo targets scale the easy-negative branch by ``1 - pseudo_weight``
    and take pseudo peaks out of the easy-negative and confuse branches.
    """

    def __init__(self, pos_thr=0.1, confuse_pred_thr=0.3, beta=0.1):
        super(BRLFocalLoss_v2, self).__init__()
        self.pos_thr = pos_thr
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta

    def forward(self, output, target, mask=None, pseudo_target=None, pseudo_weight=None):
        if mask is None:
            mask = torch.ones_like(target)
        output = _sigmoid(output)
        target = target.to(output.device)
        mask = mask.to(output.device)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        bg_inds = (target < self.pos_thr).float()
        peak_inds, neg_scale = pseudo_peak_masks(target, pseudo_target, pseudo_weight)
        neg_weights = torch.pow(1 - target, 4)

        # assignment must not receive gradients
        confuse_inds = bg_inds * (output.detach() >= self.confuse_pred_thr).float()
        if peak_inds is not None:
            confuse_inds = confuse_inds * (1.0 - peak_inds)
        easy_neg_inds = neg_inds * (1.0 - confuse_inds) * neg_scale

        pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * easy_neg_inds

        confuse_loss = torch.log(output) * torch.pow(1 - output, 2) * confuse_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = (neg_loss * mask).sum()
        confuse_loss = self.beta * (confuse_loss * mask).sum()

        if num_pos == 0:
            loss = -(neg_loss + confuse_loss)
        else:
            loss = -(pos_loss + neg_loss + confuse_loss) / num_pos
        return loss


# class BRLFocalLoss(nn.Module):
#     """
#     MVDeTr heatmap BRL: CornerNet FocalLoss + Background Recalibration
#     (same region idea as MVDetBRL ``BRLGaussianMSE``).

#     GT heatmap is already a soft Gaussian (from ``get_gt``), so no kernel
#     transform is needed. Pixels are split into:

#       - positive : soft GT peak ``target == 1`` (CornerNet centers)
#       - confuse  : background (``target < pos_thr``) where pred >= c
#                    (possible missing annotation)
#       - easy neg : remaining negatives (incl. Gaussian ring)

#     On confuse pixels:
#       - mirror=True : positive focal branch (pull toward 1), scaled by beta
#       - mirror=False: negative focal branch, scaled by beta
#     """

#     def __init__(self, confuse_pred_thr=0.1, beta=0.5):
#         super(BRLFocalLoss, self).__init__()
#         self.confuse_pred_thr = confuse_pred_thr
#         self.beta = beta
#     def forward(self, output, target, mask=None):
#         """
#         Arguments:
#           output (batch x c x h x w) — logits
#           target (batch x c x h x w) — soft Gaussian heatmap GT
#         """
#         if mask is None:
#             mask = torch.ones_like(target)
#         output = _sigmoid(output)
#         target = target.to(output.device)
#         mask = mask.to(output.device)

#         # 1) positive: GT centers (target == 1)
#         pos_inds = target.eq(1).float()
#         # all non-center pixels (Gaussian ring + true background)
#         neg_inds = target.lt(1).float()
#         neg_weights = torch.pow(1 - target, 4)

#         # assignment must not backprop through the thresholding
#         confuse_inds = neg_inds * (output.detach() >= self.confuse_pred_thr).float()
#         easy_neg_inds = neg_inds * (1.0 - confuse_inds)

#         # positive focal (CornerNet)
#         pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
#         confuse_loss = torch.log(output) * torch.pow(1 - output, 2) * confuse_inds
#         # easy negatives: standard focal neg (ring keeps (1-y)^4)
#         neg_loss = (torch.log(1 - output) * torch.pow(output, 2) * neg_weights * easy_neg_inds)

#         num_pos = pos_inds.float().sum()
#         num_confuse = confuse_inds.float().sum()
#         pos_loss = pos_loss.sum()
#         confuse_loss = self.beta * confuse_loss.sum()
#         neg_loss = (neg_loss * mask).sum()
     
#         if num_pos == 0:
#             loss = -neg_loss
#         else:
#             loss = -(pos_loss + neg_loss + confuse_loss) / (num_pos + num_confuse)
#         return loss



class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, output, target, mask=None, pseudo_target=None, pseudo_weight=None):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            output (batch x c x h x w)
            target (batch x c x h x w)
            pseudo_target / pseudo_weight (optional, batch x c x h x w):
              soften the negative penalty on pseudo-labelled regions
        '''
        if mask is None:
            mask = torch.ones_like(target)
        output = _sigmoid(output)
        target = target.to(output.device)
        mask = mask.to(output.device)
        pos_inds = target.eq(1).float()
        _, neg_scale = pseudo_peak_masks(target, pseudo_target, pseudo_weight)
        neg_inds = target.lt(1).float() * neg_scale

        neg_weights = torch.pow(1 - target, 4)

        pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = (neg_loss * mask).sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        mask, ind, target = mask.to(output.device), ind.to(output.device), target.to(output.device)
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegCELoss(nn.Module):
    def __init__(self):
        super(RegCELoss, self).__init__()

    def forward(self, output, mask, ind, target):
        mask, ind, target = mask.to(output.device), ind.to(output.device), target.to(output.device)
        pred = _transpose_and_gather_feat(output, ind)
        if len(target[mask]) != 0:
            loss = F.cross_entropy(pred[mask], target[mask], reduction='sum')
            loss = loss / (mask.sum() + 1e-4)
        else:
            loss = 0
        return loss
