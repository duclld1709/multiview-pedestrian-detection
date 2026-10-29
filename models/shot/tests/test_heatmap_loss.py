"""Regression tests for the heatmap losses.

Run from the repo root:  python tests/test_heatmap_loss.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.brl_gaussian_mse import BRLGaussianMSE
from multiview_detector.loss.pseudo_label import PseudoLabelLoss
from multiview_detector.loss.heatmap_loss import HeatmapLoss


# ---- reference implementations ------------------------------------------------

def _soft_gt(x, target, kernel):
    target = F.adaptive_max_pool2d(target, x.shape[2:])
    return F.conv2d(target, kernel.float(), padding=int((kernel.shape[-1] - 1) / 2))


def ref_mse(x, target, kernel):
    """GaussianMSE before pseudo labels."""
    return F.mse_loss(x, _soft_gt(x, target, kernel))


def ref_brl(x, target, kernel, pos_thr=0.1, confuse_pred_thr=0.3, beta=0.1):
    """BRLGaussianMSE from commit 43c90ea."""
    soft_gt = _soft_gt(x, target, kernel)
    pos_mask = soft_gt >= pos_thr
    bg_mask = ~pos_mask
    confuse_mask = bg_mask & (x.detach() >= confuse_pred_thr)
    easy_neg_mask = bg_mask & ~confuse_mask
    loss = x.new_zeros(())
    count = x.new_zeros(())
    if pos_mask.any():
        loss = loss + F.mse_loss(x[pos_mask], soft_gt[pos_mask], reduction="sum")
        count = count + pos_mask.sum()
    if easy_neg_mask.any():
        loss = loss + F.mse_loss(x[easy_neg_mask], soft_gt[easy_neg_mask], reduction="sum")
        count = count + easy_neg_mask.sum()
    if confuse_mask.any():
        loss = loss + beta * F.mse_loss(x[confuse_mask], torch.ones_like(x[confuse_mask]), reduction="sum")
        count = count + confuse_mask.sum()
    return loss / count.clamp(min=1).float()


def ref_pseudo_masks(soft_gt, pseudo_target, pseudo_weight):
    """pseudo_peak_masks from aabl_pl/MVDet_PL."""
    peak_inds = pseudo_target.eq(1).to(soft_gt.dtype) * soft_gt.lt(1).to(soft_gt.dtype)
    neg_scale = (1.0 - pseudo_weight).clamp(0.0, 1.0) * (1.0 - peak_inds)
    return peak_inds, neg_scale


def ref_pl_mse(x, target, kernel, pseudo_target, pseudo_weight):
    soft_gt = _soft_gt(x, target, kernel)
    _, neg_scale = ref_pseudo_masks(soft_gt, pseudo_target, pseudo_weight)
    return (neg_scale * (x - soft_gt).pow(2)).sum() / x.numel()


def ref_pl_brl(x, target, kernel, pseudo_target, pseudo_weight, pos_thr=0.1, confuse_pred_thr=0.3, beta=0.1):
    soft_gt = _soft_gt(x, target, kernel)
    pos_mask = soft_gt >= pos_thr
    bg_mask = ~pos_mask
    peak_inds, neg_scale = ref_pseudo_masks(soft_gt, pseudo_target, pseudo_weight)
    confuse_mask = bg_mask & (x.detach() >= confuse_pred_thr) & ~peak_inds.bool()
    easy_neg_mask = bg_mask & ~confuse_mask
    loss = F.mse_loss(x[pos_mask], soft_gt[pos_mask], reduction="sum")
    loss = loss + (neg_scale[easy_neg_mask] * (x[easy_neg_mask] - soft_gt[easy_neg_mask]).pow(2)).sum()
    loss = loss + beta * F.mse_loss(x[confuse_mask], torch.ones_like(x[confuse_mask]), reduction="sum")
    count = pos_mask.sum() + easy_neg_mask.sum() + confuse_mask.sum()
    return loss / count.float()


def ref_pl_peak(x, target, kernel, pseudo_target, pseudo_weight, peak_weight):
    peak_inds, _ = ref_pseudo_masks(_soft_gt(x, target, kernel), pseudo_target, pseudo_weight)
    return peak_weight * (pseudo_weight * peak_inds * (x - 1.0).pow(2)).sum() / x.numel()


# ---- fixtures -----------------------------------------------------------------

def make_inputs(seed=0, height=60, width=90):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(1, 1, height, width, generator=g) * 1.3 - 0.15).requires_grad_()
    target = torch.zeros(1, 1, height, width)
    for r, c in [(10, 15), (30, 50), (45, 20), (12, 70)]:
        target[0, 0, r, c] = 1
    axis = torch.arange(-10, 11, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 5.0 ** 2))
    kernel = (kernel / kernel.max()).view(1, 1, 21, 21)

    # two pseudo blobs far from GT, peak exactly 1
    pseudo_target = torch.zeros(1, 1, height, width)
    pseudo_weight = torch.zeros(1, 1, height, width)
    rows = torch.arange(height, dtype=torch.float32).view(-1, 1)
    cols = torch.arange(width, dtype=torch.float32).view(1, -1)
    for (r, c), score in [((50, 75), 0.9), ((25, 30), 0.6)]:
        blob = torch.exp(-((rows - r) ** 2 + (cols - c) ** 2) / (2 * 5.0 ** 2)).view(1, 1, height, width)
        pseudo_target = torch.maximum(pseudo_target, blob)
        pseudo_weight = torch.maximum(pseudo_weight, blob * score)
    return x, target, kernel, pseudo_target, pseudo_weight


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

def test_mse_without_pseudo_is_unchanged():
    x, target, kernel, _, _ = make_inputs()
    for criterion in (GaussianMSE(), HeatmapLoss(GaussianMSE())):
        out = loss_and_grad(lambda: _base(criterion(x, target, kernel)), x)
        assert_same(out, loss_and_grad(lambda: ref_mse(x, target, kernel), x), name='mse')


def test_brl_without_pseudo_is_unchanged():
    for seed in range(3):
        x, target, kernel, _, _ = make_inputs(seed)
        for criterion in (BRLGaussianMSE(), HeatmapLoss(BRLGaussianMSE())):
            out = loss_and_grad(lambda: _base(criterion(x, target, kernel)), x)
            assert_same(out, loss_and_grad(lambda: ref_brl(x, target, kernel), x), name='brl')


def test_pseudo_module_ignored_without_pseudo_inputs():
    """Validation batches carry no pseudo targets even when pseudo labels are on."""
    x, target, kernel, _, _ = make_inputs()
    out = HeatmapLoss(BRLGaussianMSE(), PseudoLabelLoss())(x, target, kernel)
    assert torch.equal(out['base'], ref_brl(x, target, kernel))
    assert out['pseudo'].item() == 0


def test_pseudo_matches_mvdet_pl():
    x, target, kernel, pt, pw = make_inputs()
    for base, ref in ((GaussianMSE(), ref_pl_mse), (BRLGaussianMSE(), ref_pl_brl)):
        criterion = HeatmapLoss(base, PseudoLabelLoss(peak_weight=0.5))
        out = loss_and_grad(lambda: _base(criterion(x, target, kernel, pt, pw)), x)
        assert_same(out, loss_and_grad(lambda: ref(x, target, kernel, pt, pw), x),
                    exact=False, name=f'{type(base).__name__} + pseudo base')
        out = loss_and_grad(lambda: criterion(x, target, kernel, pt, pw)['pseudo'], x)
        assert_same(out, loss_and_grad(lambda: ref_pl_peak(x, target, kernel, pt, pw, 0.5), x),
                    exact=False, name=f'{type(base).__name__} + pseudo peak')


def test_ablation_switches():
    x, target, kernel, pt, pw = make_inputs()
    # peak loss off: no pseudo term, peaks stay in the base loss with only (1 - w) scaling
    out = HeatmapLoss(GaussianMSE(), PseudoLabelLoss(peak_loss=False))(x, target, kernel, pt, pw)
    assert out['pseudo'].item() == 0
    soft_gt = _soft_gt(x, target, kernel)
    expected = ((1 - pw) * (x - soft_gt).pow(2)).sum() / x.numel()
    assert torch.allclose(out['base'], expected)
    # neg reweight off and no pseudo blobs: base loss falls back to the original value
    zeros = torch.zeros_like(pt)
    out = HeatmapLoss(BRLGaussianMSE(), PseudoLabelLoss(neg_reweight=False))(x, target, kernel, zeros, zeros)
    assert torch.allclose(out['base'], ref_brl(x, target, kernel))
    try:
        PseudoLabelLoss(neg_reweight=False, peak_loss=False)
    except ValueError:
        pass
    else:
        raise AssertionError('both pseudo parts disabled must raise')


def test_brl_no_mirror():
    x, target, kernel, _, _ = make_inputs()
    soft_gt = _soft_gt(x, target, kernel)
    pos = soft_gt >= 0.1
    confuse = ~pos & (x.detach() >= 0.3)
    easy = ~pos & ~confuse
    err = (x - soft_gt).pow(2)
    expected = (err[pos].sum() + err[easy].sum() + 0.1 * err[confuse].sum()) / x.numel()
    assert torch.allclose(BRLGaussianMSE(mirror=False)(x, target, kernel), expected)


def _base(out):
    return out['base'] if isinstance(out, dict) else out


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
