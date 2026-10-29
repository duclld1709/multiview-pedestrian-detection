"""Regression tests for HeatmapLoss (focal / BRL focal + pseudo labels) and the pseudo BEV target.

Run from TrackTacular/WorldTrack:  python tests/test_heatmap_loss.py
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import basic, geom, vox

# load models/loss.py directly: models/__init__.py pulls in the full network (kornia, timm, ...)
_spec = importlib.util.spec_from_file_location('eb_loss', os.path.join(ROOT, 'models', 'loss.py'))
_loss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loss)
FocalLoss, BRLFocalLoss = _loss.FocalLoss, _loss.BRLFocalLoss
PseudoLabelLoss, HeatmapLoss = _loss.PseudoLabelLoss, _loss.HeatmapLoss


# ---- reference implementations ------------------------------------------------

def ref_focal(pred, gt):
    """FocalLoss at tag add-aabl (use_distance_weight=False)."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds).sum()
    num_pos = pos_inds.sum()
    return 0 - neg_loss if num_pos == 0 else 0 - (pos_loss + neg_loss) / num_pos


def ref_brl(pred, gt, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05, mirror=True):
    """BRLFocalLoss after brl_no_mirror."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    bg_inds = gt.lt(pos_thr).float()
    neg_weights = torch.pow(1 - gt, 4)
    confuse_inds = bg_inds * (pred.detach() >= confuse_pred_thr).float()
    easy_neg_inds = neg_inds * (1.0 - confuse_inds)
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * easy_neg_inds).sum()
    if mirror:
        confuse_loss = torch.log(pred) * torch.pow(1 - pred, 2) * confuse_inds
    else:
        confuse_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * confuse_inds
    confuse_loss = beta * confuse_loss.sum()
    num_pos = pos_inds.sum()
    if num_pos == 0:
        return 0 - (neg_loss + confuse_loss)
    return 0 - (pos_loss + neg_loss + confuse_loss) / num_pos


def ref_pl_masks(gt, pseudo_target, pseudo_weight):
    """pseudo_peak_masks from aabl_pl/TrackTacular."""
    peak_inds = pseudo_target.eq(1).float() * gt.lt(1).float()
    neg_scale = (1.0 - pseudo_weight).clamp(0.0, 1.0) * (1.0 - peak_inds)
    return peak_inds, neg_scale


def ref_pl_focal(pred, gt, pseudo_target, pseudo_weight):
    """FocalLoss with pseudo targets from aabl_pl/TrackTacular."""
    pos_inds = gt.eq(1).float()
    _, neg_scale = ref_pl_masks(gt, pseudo_target, pseudo_weight)
    neg_inds = gt.lt(1).float() * neg_scale
    neg_weights = torch.pow(1 - gt, 4)
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds).sum()
    return -(pos_loss + neg_loss) / pos_inds.sum()


def ref_pl_brl(pred, gt, pseudo_target, pseudo_weight, pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05):
    """BRLFocalLoss with pseudo targets from aabl_pl/TrackTacular."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    bg_inds = (gt < pos_thr).float()
    peak_inds, neg_scale = ref_pl_masks(gt, pseudo_target, pseudo_weight)
    neg_weights = torch.pow(1 - gt, 4)
    confuse_inds = bg_inds * (pred.detach() >= confuse_pred_thr).float() * (1.0 - peak_inds)
    easy_neg_inds = neg_inds * (1.0 - confuse_inds) * neg_scale
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * easy_neg_inds).sum()
    confuse_loss = beta * (torch.log(pred) * torch.pow(1 - pred, 2) * confuse_inds).sum()
    return -(pos_loss + neg_loss + confuse_loss) / pos_inds.sum()


def ref_pl_peak(pred, gt, pseudo_target, pseudo_weight):
    """pseudo_peak_loss from aabl_pl/TrackTacular (before lambda and 10 * center_factor)."""
    peak_inds, _ = ref_pl_masks(gt, pseudo_target, pseudo_weight)
    loss = torch.log(pred) * torch.pow(1 - pred, 2) * pseudo_weight * peak_inds
    return -loss.sum() / gt.eq(1).float().sum().clamp(min=1)


# ---- fixtures -----------------------------------------------------------------

def make_inputs(seed=0, height=60, width=90):
    g = torch.Generator().manual_seed(seed)
    gt = torch.zeros((1, 1, height, width))
    for r, c in [(10, 15), (30, 50), (45, 20), (12, 70)]:
        basic.draw_umich_gaussian(gt[0, 0], torch.tensor([c, r]), 1.5)
    logits = (torch.randn(1, 1, height, width, generator=g) * 1.5 - 1.0).requires_grad_()

    # pseudo blobs (sigma 3 cells, peak exactly 1) far from GT, as built by the dataset
    pseudo_target = torch.zeros(1, 1, height, width)
    pseudo_weight = torch.zeros(1, 1, height, width)
    rows = torch.arange(height, dtype=torch.float32).view(-1, 1)
    cols = torch.arange(width, dtype=torch.float32).view(1, -1)
    for (r, c), score in [((50, 75), 0.9), ((25, 30), 0.6), ((26, 32), 0.7)]:  # last two: one person, two cams
        blob = torch.exp(-((rows - r) ** 2 + (cols - c) ** 2) / (2 * 3.0 ** 2)).view(1, 1, height, width)
        pseudo_target = torch.maximum(pseudo_target, blob)
        pseudo_weight = torch.maximum(pseudo_weight, blob * score)
    return logits, gt, pseudo_target, pseudo_weight


def loss_and_grad(fn, x):
    x.grad = None
    loss = fn(basic.sigmoid(x))
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


# ---- loss tests ---------------------------------------------------------------

def test_without_pseudo_is_unchanged():
    for seed in range(3):
        x, gt, _, _ = make_inputs(seed)
        cases = [(FocalLoss(), ref_focal), (BRLFocalLoss(), ref_brl),
                 (BRLFocalLoss(mirror=False), lambda p, g: ref_brl(p, g, mirror=False))]
        for base, ref in cases:
            for criterion in (base, lambda p, g, b=base: HeatmapLoss(b)(p, g)['base']):
                out = loss_and_grad(lambda p: criterion(p, gt), x)
                assert_same(out, loss_and_grad(lambda p: ref(p, gt), x), name=type(base).__name__)


def test_pseudo_module_ignored_without_pseudo_inputs():
    """Validation batches and per-view maps carry no pseudo targets even when pseudo labels are on."""
    x, gt, _, _ = make_inputs()
    p = basic.sigmoid(x)
    out = HeatmapLoss(BRLFocalLoss(), PseudoLabelLoss())(p, gt)
    assert torch.equal(out['base'], ref_brl(p, gt))
    assert out['pseudo'].item() == 0


def test_pseudo_matches_aabl_pl():
    x, gt, pt, pw = make_inputs()
    peaks = int((pt.eq(1) & gt.lt(1)).sum())
    assert peaks == 3, f'fixture should have 3 pseudo peaks, got {peaks}'
    for base, ref in ((FocalLoss(), ref_pl_focal), (BRLFocalLoss(), ref_pl_brl)):
        criterion = HeatmapLoss(base, PseudoLabelLoss(peak_weight=0.5))
        out = loss_and_grad(lambda p: criterion(p, gt, pt, pw)['base'], x)
        assert_same(out, loss_and_grad(lambda p: ref(p, gt, pt, pw), x),
                    exact=False, name=f'{type(base).__name__} + pseudo base')
        out = loss_and_grad(lambda p: criterion(p, gt, pt, pw)['pseudo'], x)
        assert_same(out, loss_and_grad(lambda p: 0.5 * ref_pl_peak(p, gt, pt, pw), x),
                    exact=False, name=f'{type(base).__name__} + pseudo peak')


def test_ablation_switches():
    x, gt, pt, pw = make_inputs()
    p = basic.sigmoid(x)
    neg_term = torch.log(1 - p) * torch.pow(p, 2) * torch.pow(1 - gt, 4) * gt.lt(1).float()
    pos_term = (torch.log(p) * torch.pow(1 - p, 2) * gt.eq(1).float()).sum()
    num_pos = gt.eq(1).float().sum()
    # peak loss off: no pseudo term, peaks stay in the negative term with only (1 - w) scaling
    out = HeatmapLoss(FocalLoss(), PseudoLabelLoss(peak_loss=False))(p, gt, pt, pw)
    assert out['pseudo'].item() == 0
    assert torch.allclose(out['base'], -(pos_term + ((1 - pw) * neg_term).sum()) / num_pos)
    # neg reweight off: peaks removed from the negative term, everything else full weight
    peak = (pt.eq(1) & gt.lt(1)).float()
    out = HeatmapLoss(FocalLoss(), PseudoLabelLoss(neg_reweight=False))(p, gt, pt, pw)
    assert torch.allclose(out['base'], -(pos_term + ((1 - peak) * neg_term).sum()) / num_pos)
    assert out['pseudo'].item() > 0
    try:
        PseudoLabelLoss(neg_reweight=False, peak_loss=False)
    except ValueError:
        pass
    else:
        raise AssertionError('both pseudo parts disabled must raise')


def test_peak_term_pulls_up():
    x, gt, pt, pw = make_inputs()
    peak = pt.eq(1) & gt.lt(1)
    _, g = loss_and_grad(lambda p: HeatmapLoss(FocalLoss(), PseudoLabelLoss())(p, gt, pt, pw)['pseudo'], x)
    assert (g[peak] < 0).all() and (g[~peak] == 0).all()


# ---- pseudo BEV target --------------------------------------------------------

def _dataset_stub():
    """PedestrianDataset with Wildtrack geometry (TrackTacular resolution) and a camera that maps
    image (u, v) to world (u, v, 0)."""
    if 'lightning' not in sys.modules:  # datasets/__init__.py imports the datamodule
        try:
            import lightning  # noqa: F401
        except ImportError:
            stub = types.ModuleType('lightning')
            stub.LightningDataModule = object
            sys.modules['lightning'] = stub
    from datasets.pedestrian_dataset import PedestrianDataset
    ds = PedestrianDataset.__new__(PedestrianDataset)
    ds.Y, ds.Z, ds.X = 120, 4, 360
    ds.bounds = (0, 1440, 0, 480, 0, 200)
    ds.num_cam = 1
    ds.base = types.SimpleNamespace(
        __name__='Wildtrack', intrinsic_matrices=[np.eye(3)],
        extrinsic_matrices=[np.hstack([np.eye(3), [[0.], [0.], [1.]]])],
        worldcoord_from_worldgrid_mat=np.array([[0, 2.5, -300], [2.5, 0, -900], [0, 0, 1]]))
    ds.vox_util = vox.VoxelUtil(ds.Y, ds.Z, ds.X, scene_centroid=torch.zeros(1, 3), bounds=ds.bounds,
                                assert_cube=False)
    ds.pseudo_conf_threshold, ds.pseudo_sigma_m, ds.pseudo_suppress_radius_m = 0.5, 0.3, 1.0
    return ds


def _gt_path(ds, worldgrid_T_worldcoord, augment, world_xy):
    """The GT path of __getitem__: world -> grid (unaugmented), then augment the grid points, Ref2Mem."""
    xyz = torch.zeros((1, 1, 3))
    xyz[0, 0, :2] = torch.tensor(world_xy, dtype=torch.float32)
    grid = geom.apply_4x4(worldgrid_T_worldcoord.unsqueeze(0), xyz)
    grid = geom.apply_4x4(augment.unsqueeze(0), grid)
    return ds.vox_util.Ref2Mem(grid, ds.Y, ds.Z, ds.X)


def test_pseudo_bev_target_projection():
    ds = _dataset_stub()
    worldcoord_from_worldgrid = torch.eye(4)
    mat = torch.tensor(ds.base.worldcoord_from_worldgrid_mat, dtype=torch.float32)
    worldcoord_from_worldgrid[:2, :2], worldcoord_from_worldgrid[:2, 3] = mat[:2, :2], mat[:2, 2]
    worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

    gt_world, far_world, near_world = (0., 0.), (500., 1500.), (40., 0.)  # cm; near is 0.4 m from the GT
    shift = torch.eye(4)
    shift[:2, 3] = torch.tensor([-8., 12.])  # grid-space augment (a translation of 2 x 3 voxels)
    for augment in (torch.eye(4), shift):
        gt_mem = _gt_path(ds, worldgrid_T_worldcoord, augment, gt_world)
        ds.pseudo_detections = {'7': {'0': [
            [far_world[0] - 10, far_world[1] - 100, far_world[0] + 10, far_world[1], 0.9],  # foot = far_world
            [near_world[0] - 10, near_world[1] - 100, near_world[0] + 10, near_world[1], 0.95],
            [far_world[0] - 10, far_world[1] - 300, far_world[0] + 10, far_world[1] - 200, 0.3],  # low score
        ]}}
        # the dataset passes the augmented transform, augment @ worldgrid_T_worldcoord
        pc, pw = ds.get_pseudo_bev_target(7, torch.matmul(augment, worldgrid_T_worldcoord), gt_mem)
        expected = _gt_path(ds, worldgrid_T_worldcoord, augment, far_world)[0, 0]
        peaks = torch.nonzero(pc[0] == 1).tolist()
        assert peaks == [[int(expected[1]), int(expected[0])]], f'peaks {peaks}, expected {expected.tolist()}'
        assert abs(pw[0, int(expected[1]), int(expected[0])].item() - 0.9) < 1e-6
        # sigma 0.3 m = 3 voxels: value 3 cells away is exp(-0.5)
        assert abs(pc[0, int(expected[1]), int(expected[0]) + 3].item() - np.exp(-0.5)) < 1e-5
        # every cell within 1 m of the GT stays zero (the near detection is suppressed)
        yy, xx = torch.meshgrid(torch.arange(ds.Y).float(), torch.arange(ds.X).float(), indexing='ij')
        near = ((xx - gt_mem[0, 0, 0]) ** 2 + (yy - gt_mem[0, 0, 1]) ** 2) * 0.1 ** 2 <= 1.0
        assert (pc[0][near] == 0).all()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
