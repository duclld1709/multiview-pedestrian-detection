"""Regression tests for BRLFocalLoss.

Run from the repo root:  python tests/test_brl_focal_loss.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_detector.loss.losses import FocalLoss, BRLFocalLoss
from multiview_detector.utils.image_utils import draw_umich_gaussian
from multiview_detector.utils.tensor_utils import _sigmoid


# ---- reference implementation -------------------------------------------------

def ref_brl_v2(output, target, mask=None, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05):
    """BRLFocalLoss_v2 from aabl/MVDeTr."""
    if mask is None:
        mask = torch.ones_like(target)
    output = _sigmoid(output)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    bg_inds = (target < pos_thr).float()
    neg_weights = torch.pow(1 - target, 4)
    confuse_inds = bg_inds * (output.detach() >= confuse_pred_thr).float()
    easy_neg_inds = neg_inds * (1.0 - confuse_inds)
    pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
    neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * easy_neg_inds
    confuse_loss = torch.log(output) * torch.pow(1 - output, 2) * confuse_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = (neg_loss * mask).sum()
    confuse_loss = beta * (confuse_loss * mask).sum()
    if num_pos == 0:
        return -(neg_loss + confuse_loss)
    return -(pos_loss + neg_loss + confuse_loss) / num_pos


# ---- fixtures -----------------------------------------------------------------

def make_inputs(seed=0, height=60, width=90, centers=((10, 15), (30, 50), (45, 20), (12, 70)), radius=2.5):
    g = torch.Generator().manual_seed(seed)
    heatmap = np.zeros([height, width], dtype=np.float32)
    for r, c in centers:
        draw_umich_gaussian(heatmap, np.array([c, r], dtype=np.int32), radius)
    target = torch.from_numpy(heatmap)[None, None]
    # logits spread so that sigmoid covers both sides of the confuse threshold
    logits = (torch.randn(1, 1, height, width, generator=g) * 1.5 - 1.0).requires_grad_()
    return logits, target


def loss_and_grad(fn, x):
    x.grad = None
    loss = fn()
    loss.backward()
    return loss.detach().clone(), x.grad.clone()


def assert_same(a, b, name):
    (la, ga), (lb, gb) = a, b
    assert torch.equal(la, lb), f'{name}: loss {la.item()} != {lb.item()}'
    assert torch.equal(ga, gb), f'{name}: grad differs'


def confuse_count(logits, target, pos_thr=0.1, confuse_pred_thr=0.4):
    return int(((target < pos_thr) & (_sigmoid(logits.detach()) >= confuse_pred_thr)).sum())


# ---- tests --------------------------------------------------------------------

def test_equals_focal_without_confuse():
    """confuse_pred_thr above the sigmoid clamp: no pixel can be confuse."""
    for seed in range(3):
        x, target = make_inputs(seed)
        out = loss_and_grad(lambda: BRLFocalLoss(confuse_pred_thr=1.0)(x, target), x)
        assert_same(out, loss_and_grad(lambda: FocalLoss()(x, target), x), 'no confuse')


def test_equals_focal_at_init_bias():
    """Heatmap heads start at bias -2.19 (p ~ 0.1 < 0.4), so BRL starts identical to focal."""
    x, target = make_inputs()
    x = torch.full_like(x, -2.19).requires_grad_()
    assert confuse_count(x, target) == 0
    out = loss_and_grad(lambda: BRLFocalLoss()(x, target), x)
    assert_same(out, loss_and_grad(lambda: FocalLoss()(x, target), x), 'init bias')


def test_matches_aabl_formulation():
    for seed in range(3):
        x, target = make_inputs(seed)
        assert confuse_count(x, target) > 0, 'fixture must contain confuse pixels'
        mask = (torch.rand(target.shape, generator=torch.Generator().manual_seed(seed)) > 0.2).float()
        for m in (None, mask):
            out = loss_and_grad(lambda: BRLFocalLoss()(x, target, m), x)
            assert_same(out, loss_and_grad(lambda: ref_brl_v2(x, target, m), x), 'aabl v2')


def test_no_gt_frame():
    """Frames with every person dropped: no division by num_pos."""
    x, _ = make_inputs()
    target = torch.zeros_like(x)
    out = loss_and_grad(lambda: BRLFocalLoss()(x, target), x)
    assert_same(out, loss_and_grad(lambda: ref_brl_v2(x, target), x), 'no gt')
    assert torch.isfinite(out[0])


def test_rings_are_never_confuse():
    """Gaussian rings around labelled people (pos_thr <= target < 1) stay standard negatives."""
    x, target = make_inputs()
    ring = (target >= 0.1) & (target < 1)
    assert ring.any()
    x_high = x.detach().clone()
    x_high[ring] = 5.0  # p ~ 1 on every ring pixel
    x_high.requires_grad_()
    out = loss_and_grad(lambda: BRLFocalLoss()(x_high, target), x_high)
    # remove the confuse pixels outside rings: the result must equal focal on the same input
    x_low = x_high.detach().clone()
    x_low[(target < 0.1) & (_sigmoid(x_low) >= 0.4)] = -5.0
    x_low.requires_grad_()
    brl = BRLFocalLoss()(x_low, target)
    assert torch.equal(brl, FocalLoss()(x_low, target)), 'ring pixels were treated as confuse'
    assert out[1][ring].abs().sum() > 0


def test_confuse_term_pulls_up():
    """On confuse pixels the gradient pushes logits up (toward 1), unlike the focal negative."""
    x, target = make_inputs()
    confuse = (target < 0.1) & (_sigmoid(x.detach()) >= 0.4)
    _, g_brl = loss_and_grad(lambda: BRLFocalLoss()(x, target), x)
    _, g_focal = loss_and_grad(lambda: FocalLoss()(x, target), x)
    assert (g_brl[confuse] < 0).all(), 'gradient descent must increase confuse logits'
    assert (g_focal[confuse] > 0).all()


def test_no_mirror():
    """mirror=False: confuse pixels stay on the focal negative branch, scaled by beta."""
    x, target = make_inputs()
    p = _sigmoid(x)
    confuse = ((target < 0.1) & (p.detach() >= 0.4)).float()
    neg = target.lt(1).float()
    neg_term = torch.log(1 - p) * torch.pow(p, 2) * torch.pow(1 - target, 4)
    pos_term = (torch.log(p) * torch.pow(1 - p, 2) * target.eq(1).float()).sum()
    expected = -(pos_term + (neg_term * neg * (1 - confuse)).sum() + 0.05 * (neg_term * confuse).sum()) \
        / target.eq(1).float().sum()
    assert torch.allclose(BRLFocalLoss(mirror=False)(x, target), expected)
    _, g = loss_and_grad(lambda: BRLFocalLoss(mirror=False)(x, target), x)
    assert (g[confuse.bool()] > 0).all(), 'no-mirror must still push confuse logits down'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
