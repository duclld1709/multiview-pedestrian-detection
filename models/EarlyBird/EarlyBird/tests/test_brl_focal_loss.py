"""Regression tests for BRLFocalLoss.

Run from EarlyBird/EarlyBird:  python tests/test_brl_focal_loss.py
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import basic

# load models/loss.py directly: models/__init__.py pulls in the full network (kornia, timm, ...)
_spec = importlib.util.spec_from_file_location('eb_loss', os.path.join(ROOT, 'models', 'loss.py'))
_loss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loss)
FocalLoss, BRLFocalLoss = _loss.FocalLoss, _loss.BRLFocalLoss


# ---- reference implementation -------------------------------------------------

def ref_brl(pred, gt, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05):
    """BRLFocalLoss from aabl/EarlyBird (use_distance_weight=False)."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    bg_inds = (gt < pos_thr).float()
    neg_weights = torch.pow(1 - gt, 4)
    confuse_inds = bg_inds * (pred.detach() >= confuse_pred_thr).float()
    easy_neg_inds = neg_inds * (1.0 - confuse_inds)
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * easy_neg_inds).sum()
    confuse_loss = beta * (torch.log(pred) * torch.pow(1 - pred, 2) * confuse_inds).sum()
    num_pos = pos_inds.sum()
    if num_pos == 0:
        return -(neg_loss + confuse_loss)
    return -(pos_loss + neg_loss + confuse_loss) / num_pos


# ---- fixtures -----------------------------------------------------------------

def make_inputs(seed=0, height=60, width=90):
    """GT drawn like PedestrianDataset.get_bev_gt (kernel_size 1.5); logits as the center head outputs."""
    g = torch.Generator().manual_seed(seed)
    gt = torch.zeros((1, 1, height, width))
    for r, c in [(10, 15), (30, 50), (45, 20), (12, 70)]:
        basic.draw_umich_gaussian(gt[0, 0], torch.tensor([c, r]), 1.5)
    logits = (torch.randn(1, 1, height, width, generator=g) * 1.5 - 1.0).requires_grad_()
    return logits, gt


def loss_and_grad(fn, x):
    x.grad = None
    loss = fn(basic._sigmoid(x))
    loss.backward()
    return loss.detach().clone(), x.grad.clone()


def assert_same(a, b, name):
    (la, ga), (lb, gb) = a, b
    assert torch.equal(la, lb), f'{name}: loss {la.item()} != {lb.item()}'
    assert torch.equal(ga, gb), f'{name}: grad differs'


def confuse_mask(x, gt, pos_thr=0.1, confuse_pred_thr=0.4):
    return (gt < pos_thr) & (basic._sigmoid(x.detach()) >= confuse_pred_thr)


# ---- tests --------------------------------------------------------------------

def test_equals_focal_without_confuse():
    """confuse_pred_thr above the sigmoid clamp: no pixel can be confuse."""
    for seed in range(3):
        x, gt = make_inputs(seed)
        for dw in (False, True):
            out = loss_and_grad(lambda p: BRLFocalLoss(confuse_pred_thr=1.0, use_distance_weight=dw)(p, gt), x)
            assert_same(out, loss_and_grad(lambda p: FocalLoss(use_distance_weight=dw)(p, gt), x), 'no confuse')


def test_equals_focal_at_init_bias():
    """The center head starts at bias -2.19 (p ~ 0.1 < 0.4), so BRL starts identical to focal."""
    _, gt = make_inputs()
    x = torch.full_like(gt, -2.19).requires_grad_()
    assert not confuse_mask(x, gt).any()
    out = loss_and_grad(lambda p: BRLFocalLoss()(p, gt), x)
    assert_same(out, loss_and_grad(lambda p: FocalLoss()(p, gt), x), 'init bias')


def test_matches_aabl_formulation():
    for seed in range(3):
        x, gt = make_inputs(seed)
        assert confuse_mask(x, gt).any(), 'fixture must contain confuse pixels'
        out = loss_and_grad(lambda p: BRLFocalLoss()(p, gt), x)
        assert_same(out, loss_and_grad(lambda p: ref_brl(p, gt), x), 'aabl')


def test_no_gt_frame():
    """Frames with every person dropped: no division by num_pos."""
    x, _ = make_inputs()
    gt = torch.zeros_like(x)
    out = loss_and_grad(lambda p: BRLFocalLoss()(p, gt), x)
    assert_same(out, loss_and_grad(lambda p: ref_brl(p, gt), x), 'no gt')
    assert torch.isfinite(out[0])


def test_rings_are_never_confuse():
    """Gaussian rings around labelled people (pos_thr <= gt < 1) stay standard negatives."""
    x, gt = make_inputs()
    ring = (gt >= 0.1) & (gt < 1)
    assert ring.any()
    x_ring = x.detach().clone()
    x_ring[ring] = 5.0  # p ~ 1 on every ring pixel
    x_ring[confuse_mask(x_ring, gt)] = -5.0  # no confuse pixel outside rings
    p = basic._sigmoid(x_ring)
    assert torch.equal(BRLFocalLoss()(p, gt), FocalLoss()(p, gt)), 'ring pixels were treated as confuse'


def test_confuse_term_pulls_up():
    """On confuse pixels the gradient pushes logits up (toward 1), unlike the focal negative."""
    x, gt = make_inputs()
    confuse = confuse_mask(x, gt)
    _, g_brl = loss_and_grad(lambda p: BRLFocalLoss()(p, gt), x)
    _, g_focal = loss_and_grad(lambda p: FocalLoss()(p, gt), x)
    assert (g_brl[confuse] < 0).all(), 'gradient descent must increase confuse logits'
    assert (g_focal[confuse] > 0).all()


def test_no_mirror():
    """mirror=False: confuse pixels stay on the focal negative branch, scaled by beta."""
    x, gt = make_inputs()
    p = basic._sigmoid(x)
    confuse = confuse_mask(x, gt).float()
    neg_term = torch.log(1 - p) * torch.pow(p, 2) * torch.pow(1 - gt, 4) * gt.lt(1).float()
    pos_term = (torch.log(p) * torch.pow(1 - p, 2) * gt.eq(1).float()).sum()
    expected = -(pos_term + (neg_term * (1 - confuse)).sum() + 0.05 * (neg_term * confuse).sum()) / gt.eq(1).sum()
    assert torch.allclose(BRLFocalLoss(mirror=False)(p, gt), expected)
    _, g = loss_and_grad(lambda q: BRLFocalLoss(mirror=False)(q, gt), x)
    assert (g[confuse.bool()] > 0).all(), 'no-mirror must still push confuse logits down'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
