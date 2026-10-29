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


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, output, target, mask=None, neg_weight=None, ignore_mask=None):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            output (batch x c x h x w)
            target (batch x c x h x w)
            neg_weight (optional, batch x c x h x w): weight on the negative term
            ignore_mask (optional, bool, batch x c x h x w): pixels removed from the negative term
            (both None -> original code path)
        '''
        if mask is None:
            mask = torch.ones_like(target)
        output = _sigmoid(output)
        target = target.to(output.device)
        mask = mask.to(output.device)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        if ignore_mask is not None:
            neg_inds = neg_inds * (~ignore_mask).float()

        neg_weights = torch.pow(1 - target, 4)

        pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * neg_inds
        if neg_weight is not None:
            neg_loss = neg_loss * neg_weight

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = (neg_loss * mask).sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class BRLFocalLoss(nn.Module):
    '''CornerNet focal loss + Background Recalibration (BRL) for missing annotations.

    The GT heatmap is already a soft Gaussian (peak = 1). Pixels are split into:
      - positive: target == 1 (GT peaks), positive focal branch
      - confuse:  background (target < pos_thr) where the detached prediction >= confuse_pred_thr
                  (possible unlabelled person), scaled by beta:
                    mirror=True:  positive focal branch, pulled toward 1
                    mirror=False: negative focal branch, only down-weighted
      - easy neg: all other target < 1 pixels, standard focal negative branch; this includes the
                  Gaussian rings (pos_thr <= target < 1) around labelled people
    Normalised by the number of GT peaks as in FocalLoss. Without confuse pixels it equals FocalLoss.

    Optional per-pixel hooks (both None -> original code path):
      - neg_weight:  weight on the easy-negative term
      - ignore_mask: background pixels removed from the easy-negative and confuse terms
    '''

    def __init__(self, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05, mirror=True):
        super(BRLFocalLoss, self).__init__()
        self.pos_thr = pos_thr
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def forward(self, output, target, mask=None, neg_weight=None, ignore_mask=None):
        if mask is None:
            mask = torch.ones_like(target)
        output = _sigmoid(output)
        target = target.to(output.device)
        mask = mask.to(output.device)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        bg_inds = target.lt(self.pos_thr).float()

        neg_weights = torch.pow(1 - target, 4)

        # assignment must not receive gradients
        confuse_inds = bg_inds * (output.detach() >= self.confuse_pred_thr).float()
        if ignore_mask is not None:
            keep_inds = (~ignore_mask).float()
            confuse_inds = confuse_inds * keep_inds
            easy_neg_inds = neg_inds * (1.0 - confuse_inds) * keep_inds
        else:
            easy_neg_inds = neg_inds * (1.0 - confuse_inds)

        pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * easy_neg_inds
        if neg_weight is not None:
            neg_loss = neg_loss * neg_weight
        if self.mirror:
            confuse_loss = torch.log(output) * torch.pow(1 - output, 2) * confuse_inds
        else:
            confuse_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * confuse_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = (neg_loss * mask).sum()
        confuse_loss = self.beta * (confuse_loss * mask).sum()

        if num_pos == 0:
            loss = -(neg_loss + confuse_loss)
        else:
            loss = -(pos_loss + neg_loss + confuse_loss) / num_pos
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
