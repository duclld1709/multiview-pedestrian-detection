import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import basic


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = basic.reduce_masked_mean(loss, valid)
        return loss


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self, use_distance_weight=False):
        super(FocalLoss, self).__init__()
        self.use_distance_weight = use_distance_weight

    def forward(self, pred, gt, neg_weight=None, ignore_mask=None):
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
                pred (batch x c x h x w)
                gt_regr (batch x c x h x w)
                neg_weight (optional, batch x c x h x w): extra weight on the negative term
                ignore_mask (optional, bool, batch x c x h x w): pixels removed from the negative term
            With both hooks None this is the original code path.
        """
        # find pos indices and neg indices
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        if ignore_mask is not None:
            neg_inds = neg_inds * (~ignore_mask).float()

        distance_weight = torch.ones_like(gt)
        if self.use_distance_weight:
            w, h = gt.shape[-2:]
            xs = torch.linspace(-1, 1, steps=h, device=gt.device)
            ys = torch.linspace(-1, 1, steps=w, device=gt.device)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            distance_weight = 9 * torch.sin(torch.sqrt(x * x + y * y)) + 1

        # following paper alpha 2, beta 4
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * distance_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * distance_weight
        if neg_weight is not None:
            neg_loss = neg_loss * neg_weight

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class BRLFocalLoss(torch.nn.Module):
    '''CornerNet focal loss + Background Recalibration (BRL) for missing annotations.

    Same contract as FocalLoss: ``pred`` is already sigmoided. Pixels are split into:
      - positive: gt == 1 (GT peaks), positive focal branch
      - confuse:  background (gt < pos_thr) where the detached prediction >= confuse_pred_thr
                  (possible unlabelled person), scaled by beta:
                    mirror=True:  positive focal branch, pulled toward 1
                    mirror=False: negative focal branch, only down-weighted
      - easy neg: all other gt < 1 pixels, standard focal negative branch; this includes the
                  Gaussian rings (pos_thr <= gt < 1) around labelled people
    Normalised by the number of GT peaks as in FocalLoss. Without confuse pixels it equals FocalLoss.

    Optional per-pixel hooks (both None -> original code path):
      - neg_weight:  weight on the easy-negative term
      - ignore_mask: pixels removed from the easy-negative and confuse terms
    '''

    def __init__(self, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05, mirror=True, use_distance_weight=False):
        super(BRLFocalLoss, self).__init__()
        self.pos_thr = pos_thr
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror
        self.use_distance_weight = use_distance_weight

    def forward(self, pred, gt, neg_weight=None, ignore_mask=None):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        bg_inds = gt.lt(self.pos_thr).float()

        distance_weight = torch.ones_like(gt)
        if self.use_distance_weight:
            w, h = gt.shape[-2:]
            xs = torch.linspace(-1, 1, steps=h, device=gt.device)
            ys = torch.linspace(-1, 1, steps=w, device=gt.device)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            distance_weight = 9 * torch.sin(torch.sqrt(x * x + y * y)) + 1

        neg_weights = torch.pow(1 - gt, 4)

        # assignment must not receive gradients
        confuse_inds = bg_inds * (pred.detach() >= self.confuse_pred_thr).float()
        if ignore_mask is not None:
            keep_inds = (~ignore_mask).float()
            confuse_inds = confuse_inds * keep_inds
            easy_neg_inds = neg_inds * (1.0 - confuse_inds) * keep_inds
        else:
            easy_neg_inds = neg_inds * (1.0 - confuse_inds)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * distance_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * easy_neg_inds * distance_weight
        if neg_weight is not None:
            neg_loss = neg_loss * neg_weight
        if self.mirror:
            confuse_loss = torch.log(pred) * torch.pow(1 - pred, 2) * confuse_inds * distance_weight
        else:
            confuse_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * confuse_inds * distance_weight

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        confuse_loss = self.beta * confuse_loss.sum()

        if num_pos == 0:
            loss = loss - (neg_loss + confuse_loss)
        else:
            loss = loss - (pos_loss + neg_loss + confuse_loss) / num_pos
        return loss


class PseudoLabelLoss(torch.nn.Module):
    '''Offline-detector pseudo labels on the BEV center heatmap (focal version).

    pseudo_target is a max of Gaussian blobs (peak = 1) around projected detections,
    pseudo_weight the same blobs scaled by detector score; both are zero near kept GT.
    Two parts, switchable separately:
      - neg_reweight: scale the GT negative term by (1 - pseudo_weight)
      - peak_loss:    score-weighted positive focal term at pseudo peaks (x peak_weight);
                      the peaks are then taken out of the GT negative / confuse terms
    '''

    def __init__(self, peak_weight=0.01, neg_reweight=True, peak_loss=True):
        super(PseudoLabelLoss, self).__init__()
        if not (neg_reweight or peak_loss):
            raise ValueError('PseudoLabelLoss needs neg_reweight and/or peak_loss')
        self.peak_weight = peak_weight
        self.neg_reweight = neg_reweight
        self.peak_loss = peak_loss

    def masks(self, gt, pseudo_target, pseudo_weight):
        '''Returns (peak_mask, neg_weight, ignore_mask) for the base loss hooks.'''
        peak_mask = pseudo_target.eq(1) & gt.lt(1)
        neg_weight = (1.0 - pseudo_weight).clamp(0.0, 1.0) if self.neg_reweight else None
        ignore_mask = peak_mask if self.peak_loss else None
        return peak_mask, neg_weight, ignore_mask

    def forward(self, pred, gt, peak_mask, pseudo_weight):
        '''``pred`` is already sigmoided, as for FocalLoss.'''
        if not self.peak_loss:
            return pred.new_zeros(())
        loss = torch.log(pred) * torch.pow(1 - pred, 2) * pseudo_weight * peak_mask.float()
        # normalised by the number of GT peaks, like the GT focal loss
        return self.peak_weight * -loss.sum() / gt.eq(1).float().sum().clamp(min=1)


class HeatmapLoss(torch.nn.Module):
    '''Base center-heatmap loss (FocalLoss / BRLFocalLoss) plus an optional PseudoLabelLoss.

    Returns {'base': ..., 'pseudo': ...}. Without a pseudo module or pseudo inputs the
    base loss runs its original code path and 'pseudo' is 0.
    '''

    def __init__(self, base, pseudo=None):
        super(HeatmapLoss, self).__init__()
        self.base = base
        self.pseudo = pseudo

    def forward(self, pred, gt, pseudo_target=None, pseudo_weight=None):
        if self.pseudo is None or pseudo_target is None:
            return {'base': self.base(pred, gt), 'pseudo': pred.new_zeros(())}
        pseudo_target = pseudo_target.to(pred.device, dtype=pred.dtype)
        pseudo_weight = pseudo_weight.to(pred.device, dtype=pred.dtype)
        peak_mask, neg_weight, ignore_mask = self.pseudo.masks(gt, pseudo_target, pseudo_weight)
        return {'base': self.base(pred, gt, neg_weight=neg_weight, ignore_mask=ignore_mask),
                'pseudo': self.pseudo(pred, gt, peak_mask, pseudo_weight)}


def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = basic.reduce_masked_mean(mse_loss, pos_mask * valid)
    neg_loss = basic.reduce_masked_mean(mse_loss, neg_mask * valid)
    loss = (pos_loss + neg_loss) * 0.5

    return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, rotbin, rotres):
        loss = compute_rot_loss(output, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')


def compute_bin_loss(output, target, mask):
    # mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

