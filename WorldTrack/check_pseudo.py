"""Sanity checks for the focal pseudo-label loss and the pseudo BEV targets.

    python check_pseudo.py loss
    python check_pseudo.py data --dataset wildtrack --data-dir /path/Wildtrack \
        --pseudo-cache /path/wildtrack.json --drop-ratio 60 --out-dir pseudo_check
"""

import argparse
import os

import numpy as np
import torch

from models.loss import BRLFocalLoss, FocalLoss, pseudo_peak_loss

DATASET_GRIDS = {
    'wildtrack': dict(resolution=(120, 4, 360), bounds=(0, 1440, 0, 480, 0, 200)),
    'multiviewx': dict(resolution=(160, 2, 250), bounds=(0, 1000, 0, 640, 0, 2)),
}


def check_loss():
    torch.manual_seed(0)
    shape = (2, 1, 40, 60)
    pred = torch.rand(shape).clamp(1e-4, 1 - 1e-4)
    gt = torch.zeros(shape)
    gt[0, 0, 10, 10] = 1.0
    gt[1, 0, 30, 50] = 1.0
    gt[0, 0, 10, 11] = 0.5

    pseudo_target = torch.zeros(shape)
    pseudo_weight = torch.zeros(shape)
    yy, xx = torch.meshgrid(torch.arange(40.), torch.arange(60.), indexing='ij')
    blob = torch.exp(-0.5 * (((xx - 40) / 3) ** 2 + ((yy - 20) / 3) ** 2))
    pseudo_target[0, 0] = blob
    pseudo_weight[0, 0] = blob * 0.8

    for loss_fn in (FocalLoss(), BRLFocalLoss(pos_thr=0.1, confuse_pred_thr=0.4, beta=0.05)):
        name = type(loss_fn).__name__
        # Without pseudo inputs the loss must match the original formulation.
        base = loss_fn(pred, gt)
        zero = loss_fn(pred, gt, torch.zeros(shape), torch.zeros(shape))
        assert torch.allclose(base, zero), f'{name}: empty pseudo target changed the loss'

        # Pseudo blob must only lower the negative penalty.
        relieved = loss_fn(pred, gt, pseudo_target, pseudo_weight)
        assert relieved < base, f'{name}: pseudo relief did not reduce the loss'
        print(f'{name}: base={base.item():.4f} with_pseudo={relieved.item():.4f}  OK')

    p = pred.clone().requires_grad_(True)
    peak = pseudo_peak_loss(p, gt, pseudo_target, pseudo_weight)
    peak.backward()
    assert p.grad[0, 0, 20, 40] < 0, 'pseudo peak gradient should push the prediction up'
    off_peak = p.grad.clone()
    off_peak[0, 0, 20, 40] = 0
    assert off_peak.abs().max() == 0, 'pseudo peak loss must only touch peak cells'
    print(f'pseudo_peak_loss={peak.item():.4f}, grad at peak={p.grad[0, 0, 20, 40].item():.4f}  OK')


def check_data(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from datasets.multiviewx_dataset import MultiviewX
    from datasets.pedestrian_dataset import PedestrianDataset
    from datasets.wildtrack_dataset import Wildtrack

    base_cls = Wildtrack if args.dataset == 'wildtrack' else MultiviewX
    base = base_cls(os.path.expanduser(args.data_dir))
    grid = DATASET_GRIDS[args.dataset]
    dropped = PedestrianDataset(
        base, is_train=True, drop_ratio=args.drop_ratio, pseudo_cache=args.pseudo_cache,
        pseudo_conf_threshold=args.conf, pseudo_sigma_m=args.sigma_m,
        pseudo_fuse_radius_m=args.fuse_m, pseudo_min_views=args.min_views, **grid)
    full = PedestrianDataset(base, is_train=True, drop_ratio=0, **grid)

    Y, Z, X = grid['resolution']
    bounds = grid['bounds']
    worldgrid_mat = np.asarray(base.worldcoord_from_worldgrid_mat, dtype=np.float64)
    units_to_m = 0.01 if args.dataset == 'wildtrack' else 1.0
    vox_x = (bounds[1] - bounds[0]) / X * np.linalg.norm(worldgrid_mat[:2, 0]) * units_to_m
    vox_y = (bounds[3] - bounds[2]) / Y * np.linalg.norm(worldgrid_mat[:2, 1]) * units_to_m

    def to_mem(pts):
        xyz = torch.cat((pts, torch.zeros_like(pts[:, :1])), dim=1).unsqueeze(0)
        return full.vox_util.Ref2Mem(xyz, Y, Z, X)[0, :, :2]

    def near(a, b, radius_m):
        if len(a) == 0 or len(b) == 0:
            return torch.zeros(len(a), dtype=torch.bool)
        d2 = ((a[:, None, 0] - b[None, :, 0]) * vox_x) ** 2 + ((a[:, None, 1] - b[None, :, 1]) * vox_y) ** 2
        return (d2 <= radius_m ** 2).any(dim=1)

    os.makedirs(args.out_dir, exist_ok=True)
    indices = np.linspace(0, len(dropped) - 1, num=min(args.samples, len(dropped))).astype(int)
    n_peaks = n_peaks_correct = n_peaks_dropped = n_dropped = n_dropped_hit = 0
    for i, index in enumerate(indices):
        frame = list(dropped.world_gt.keys())[index]
        _, target = dropped[index]
        peaks = torch.nonzero(target['pseudo_center_bev'][0] == 1)[:, [1, 0]].float()  # (x, y)

        full_pts, full_pids = full.world_gt[frame]
        _, kept_pids = dropped.world_gt[frame]
        is_dropped = ~torch.isin(full_pids, kept_pids)
        full_mem = to_mem(full_pts)
        dropped_mem = full_mem[is_dropped]

        n_peaks += len(peaks)
        n_peaks_correct += int(near(peaks, full_mem, args.match_m).sum())
        n_peaks_dropped += int(near(peaks, dropped_mem, args.match_m).sum())
        n_dropped += len(dropped_mem)
        n_dropped_hit += int(near(dropped_mem, peaks, args.match_m).sum())

        if i < args.plots:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(target['pseudo_center_bev'][0].numpy(), cmap='viridis')
            kept_mem = full_mem[~is_dropped]
            ax.scatter(kept_mem[:, 0], kept_mem[:, 1], s=30, facecolors='none', edgecolors='lime', label='kept GT')
            ax.scatter(dropped_mem[:, 0], dropped_mem[:, 1], s=30, marker='x', c='red', label='dropped GT')
            ax.legend(loc='upper right')
            ax.set_title(f'frame {frame}: pseudo_center_bev')
            fig.savefig(os.path.join(args.out_dir, f'frame_{frame:05d}.png'), bbox_inches='tight')
            plt.close(fig)

    def ratio(a, b):
        return a / b if b else float('nan')

    print(f'{len(indices)} frames, voxel {vox_x:.3f} x {vox_y:.3f} m, match radius {args.match_m} m, '
          f'fuse radius {args.fuse_m} m, min views {args.min_views}')
    print(f'pseudo peaks: {n_peaks} ({ratio(n_peaks, len(indices)):.1f}/frame)')
    print(f'  near any GT person : {ratio(n_peaks_correct, n_peaks):.3f}')
    print(f'  near a dropped GT  : {ratio(n_peaks_dropped, n_peaks):.3f}')
    print(f'dropped GT recovered by a pseudo peak: {n_dropped_hit}/{n_dropped} = {ratio(n_dropped_hit, n_dropped):.3f}')
    print(f'pseudo peaks per recovered dropped GT: {ratio(n_peaks_dropped, n_dropped_hit):.2f}  (ideal: 1.0)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('loss', help='Unit-check the focal pseudo loss (no data needed)')
    data = sub.add_parser('data', help='Compare pseudo peaks with full and dropped annotations')
    data.add_argument('--dataset', choices=tuple(DATASET_GRIDS), required=True)
    data.add_argument('--data-dir', required=True)
    data.add_argument('--pseudo-cache', required=True)
    data.add_argument('--drop-ratio', type=int, default=60)
    data.add_argument('--conf', type=float, default=0.5)
    data.add_argument('--sigma-m', type=float, default=0.3)
    data.add_argument('--match-m', type=float, default=0.5)
    data.add_argument('--fuse-m', type=float, default=0.5, help='0 = no cross-view fusion (old behaviour)')
    data.add_argument('--min-views', type=int, default=1)
    data.add_argument('--samples', type=int, default=50)
    data.add_argument('--plots', type=int, default=5)
    data.add_argument('--out-dir', default='pseudo_check')
    args = parser.parse_args()

    if args.command == 'loss':
        check_loss()
    else:
        check_data(args)


if __name__ == '__main__':
    main()
