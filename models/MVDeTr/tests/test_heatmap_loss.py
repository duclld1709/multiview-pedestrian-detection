"""Regression tests for HeatmapLoss (focal / BRL focal + pseudo labels).

Run from the repo root:  python tests/test_heatmap_loss.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_detector.loss import FocalLoss, BRLFocalLoss, PseudoLabelLoss, HeatmapLoss
from multiview_detector.utils.image_utils import draw_umich_gaussian
from multiview_detector.utils.tensor_utils import _sigmoid


# ---- reference implementations ------------------------------------------------

def ref_focal(output, target, mask=None):
    """FocalLoss at tag add-aabl."""
    if mask is None:
        mask = torch.ones_like(target)
    output = _sigmoid(output)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)
    pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
    neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = (neg_loss * mask).sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def ref_brl(output, target, mask=None, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05, mirror=True):
    """BRLFocalLoss after --brl_no_mirror."""
    if mask is None:
        mask = torch.ones_like(target)
    output = _sigmoid(output)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    bg_inds = target.lt(pos_thr).float()
    neg_weights = torch.pow(1 - target, 4)
    confuse_inds = bg_inds * (output.detach() >= confuse_pred_thr).float()
    easy_neg_inds = neg_inds * (1.0 - confuse_inds)
    pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
    neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * easy_neg_inds
    if mirror:
        confuse_loss = torch.log(output) * torch.pow(1 - output, 2) * confuse_inds
    else:
        confuse_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * confuse_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = (neg_loss * mask).sum()
    confuse_loss = beta * (confuse_loss * mask).sum()
    if num_pos == 0:
        return -(neg_loss + confuse_loss)
    return -(pos_loss + neg_loss + confuse_loss) / num_pos


def ref_pl_masks(target, pseudo_target, pseudo_weight):
    """pseudo_peak_masks from aabl_pl/MVDeTr."""
    peak_inds = pseudo_target.eq(1).float() * target.lt(1).float()
    neg_scale = (1.0 - pseudo_weight).clamp(0.0, 1.0) * (1.0 - peak_inds)
    return peak_inds, neg_scale


def ref_pl_focal(output, target, pseudo_target, pseudo_weight):
    """FocalLoss with pseudo targets from aabl_pl/MVDeTr."""
    output = _sigmoid(output)
    pos_inds = target.eq(1).float()
    _, neg_scale = ref_pl_masks(target, pseudo_target, pseudo_weight)
    neg_inds = target.lt(1).float() * neg_scale
    neg_weights = torch.pow(1 - target, 4)
    pos_loss = (torch.log(output) * torch.pow(1 - output, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - output) * torch.pow(output, 2) * neg_weights * neg_inds).sum()
    num_pos = pos_inds.sum()
    return -(pos_loss + neg_loss) / num_pos if num_pos > 0 else -neg_loss


def ref_pl_brl(output, target, pseudo_target, pseudo_weight, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05):
    """BRLFocalLoss_v2 with pseudo targets from aabl_pl/MVDeTr."""
    output = _sigmoid(output)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    bg_inds = (target < pos_thr).float()
    peak_inds, neg_scale = ref_pl_masks(target, pseudo_target, pseudo_weight)
    neg_weights = torch.pow(1 - target, 4)
    confuse_inds = bg_inds * (output.detach() >= confuse_pred_thr).float() * (1.0 - peak_inds)
    easy_neg_inds = neg_inds * (1.0 - confuse_inds) * neg_scale
    pos_loss = (torch.log(output) * torch.pow(1 - output, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - output) * torch.pow(output, 2) * neg_weights * easy_neg_inds).sum()
    confuse_loss = beta * (torch.log(output) * torch.pow(1 - output, 2) * confuse_inds).sum()
    num_pos = pos_inds.sum()
    return -(pos_loss + neg_loss + confuse_loss) / num_pos if num_pos > 0 else -(neg_loss + confuse_loss)


def ref_pl_peak(output, target, pseudo_target, pseudo_weight):
    """pseudo_peak_loss from aabl_pl/MVDeTr (before lambda)."""
    output = _sigmoid(output)
    peak_inds, _ = ref_pl_masks(target, pseudo_target, pseudo_weight)
    loss = torch.log(output) * torch.pow(1 - output, 2) * pseudo_weight * peak_inds
    return -loss.sum() / target.eq(1).float().sum().clamp(min=1)


# ---- fixtures -----------------------------------------------------------------

def make_inputs(seed=0, height=60, width=90):
    g = torch.Generator().manual_seed(seed)
    heatmap = np.zeros([height, width], dtype=np.float32)
    for r, c in [(10, 15), (30, 50), (45, 20), (12, 70)]:
        draw_umich_gaussian(heatmap, np.array([c, r], dtype=np.int32), 2.5)
    target = torch.from_numpy(heatmap)[None, None]
    logits = (torch.randn(1, 1, height, width, generator=g) * 1.5 - 1.0).requires_grad_()

    # pseudo blobs (sigma 5 cells, peak exactly 1) far from GT, as built by frameDataset
    pseudo_target = torch.zeros(1, 1, height, width)
    pseudo_weight = torch.zeros(1, 1, height, width)
    rows = torch.arange(height, dtype=torch.float32).view(-1, 1)
    cols = torch.arange(width, dtype=torch.float32).view(1, -1)
    for (r, c), score in [((50, 75), 0.9), ((25, 30), 0.6), ((26, 32), 0.7)]:  # last two: one person, two cams
        blob = torch.exp(-((rows - r) ** 2 + (cols - c) ** 2) / (2 * 5.0 ** 2)).view(1, 1, height, width)
        pseudo_target = torch.maximum(pseudo_target, blob)
        pseudo_weight = torch.maximum(pseudo_weight, blob * score)
    return logits, target, pseudo_target, pseudo_weight


def loss_and_grad(fn, x):
    x.grad = None
    loss = fn()
    loss.backward()
    return loss.detach().clone(), x.grad.clone()


def assert_same(a, b, exact=True, name=''):
    (la, ga), (lb, gb) = a, b
    if exact:
        assert torch.equal(la, lb), f'{name}: loss {la.item()} != {lb.item()}'
        assert torch.equal(ga, gb), f'{name}: grad differs'
    else:
        assert torch.allclose(la, lb, rtol=1e-6, atol=1e-8), f'{name}: loss {la.item()} vs {lb.item()}'
        assert torch.allclose(ga, gb, rtol=1e-5, atol=1e-10), f'{name}: grad differs'


# ---- tests --------------------------------------------------------------------

def test_without_pseudo_is_unchanged():
    for seed in range(3):
        x, target, _, _ = make_inputs(seed)
        mask = (torch.rand(target.shape, generator=torch.Generator().manual_seed(seed)) > 0.2).float()
        cases = [(FocalLoss(), ref_focal), (BRLFocalLoss(), ref_brl),
                 (BRLFocalLoss(mirror=False), lambda *a: ref_brl(*a, mirror=False))]
        for base, ref in cases:
            for m in (None, mask):
                for criterion in (base, HeatmapLoss(base)):
                    out = loss_and_grad(lambda: _base(criterion(x, target, m)), x)
                    assert_same(out, loss_and_grad(lambda: ref(x, target, m), x), name=type(base).__name__)


def test_pseudo_module_ignored_without_pseudo_inputs():
    """Test batches and per-view maps carry no pseudo targets even when pseudo labels are on."""
    x, target, _, _ = make_inputs()
    out = HeatmapLoss(BRLFocalLoss(), PseudoLabelLoss())(x, target)
    assert torch.equal(out['base'], ref_brl(x, target))
    assert out['pseudo'].item() == 0


def test_pseudo_matches_aabl_pl():
    x, target, pt, pw = make_inputs()
    peaks = int((pt.eq(1) & target.lt(1)).sum())
    assert peaks == 3, f'fixture should have 3 pseudo peaks, got {peaks}'
    for base, ref in ((FocalLoss(), ref_pl_focal), (BRLFocalLoss(), ref_pl_brl)):
        criterion = HeatmapLoss(base, PseudoLabelLoss(peak_weight=0.5))
        out = loss_and_grad(lambda: criterion(x, target, None, pt, pw)['base'], x)
        assert_same(out, loss_and_grad(lambda: ref(x, target, pt, pw), x),
                    exact=False, name=f'{type(base).__name__} + pseudo base')
        out = loss_and_grad(lambda: criterion(x, target, None, pt, pw)['pseudo'], x)
        assert_same(out, loss_and_grad(lambda: 0.5 * ref_pl_peak(x, target, pt, pw), x),
                    exact=False, name=f'{type(base).__name__} + pseudo peak')


def test_ablation_switches():
    x, target, pt, pw = make_inputs()
    p = _sigmoid(x)
    neg_term = torch.log(1 - p) * torch.pow(p, 2) * torch.pow(1 - target, 4) * target.lt(1).float()
    pos_term = (torch.log(p) * torch.pow(1 - p, 2) * target.eq(1).float()).sum()
    num_pos = target.eq(1).float().sum()
    # peak loss off: no pseudo term, peaks stay in the negative term with only (1 - w) scaling
    out = HeatmapLoss(FocalLoss(), PseudoLabelLoss(peak_loss=False))(x, target, None, pt, pw)
    assert out['pseudo'].item() == 0
    assert torch.allclose(out['base'], -(pos_term + ((1 - pw) * neg_term).sum()) / num_pos)
    # neg reweight off: peaks removed from the negative term, everything else full weight
    peak = (pt.eq(1) & target.lt(1)).float()
    out = HeatmapLoss(FocalLoss(), PseudoLabelLoss(neg_reweight=False))(x, target, None, pt, pw)
    assert torch.allclose(out['base'], -(pos_term + ((1 - peak) * neg_term).sum()) / num_pos)
    assert out['pseudo'].item() > 0
    try:
        PseudoLabelLoss(neg_reweight=False, peak_loss=False)
    except ValueError:
        pass
    else:
        raise AssertionError('both pseudo parts disabled must raise')


def test_peak_term_pulls_up():
    x, target, pt, pw = make_inputs()
    peak = pt.eq(1) & target.lt(1)
    _, g = loss_and_grad(lambda: HeatmapLoss(FocalLoss(), PseudoLabelLoss())(x, target, None, pt, pw)['pseudo'], x)
    assert (g[peak] < 0).all() and (g[~peak] == 0).all()


def _base(out):
    return out['base'] if isinstance(out, dict) else out


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
