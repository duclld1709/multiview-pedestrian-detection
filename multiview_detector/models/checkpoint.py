"""Load BEVNet ground-plane weights for a calibrated inference scene."""

import torch

from .bevnet import BEVNet
from ..utils.projection import BevGrid


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def load_bevnet(checkpoint, device, *, roi=None, view_count=None, foreign=False):
    if checkpoint.get('architecture', 'bevnet') != 'bevnet':
        raise ValueError('Select a BEVNet checkpoint.')
    trained_grid = BevGrid(**checkpoint['grid'])
    grid = trained_grid
    if roi is not None and (foreign or grid.x0 > roi.x0 or grid.y0 > roi.y0
                            or grid.x1 < roi.x1 or grid.y1 < roi.y1):
        # Benchmark ROIs are much larger than WaveTrack. Use their usual output grid stride
        # for transferred weights, and use the saved cell size for native weights.
        cell = 4 * roi.cell if foreign else trained_grid.cell
        grid = BevGrid.covering([roi], cell=cell, margin=0)

    fusion = checkpoint.get('fusion', 'concat')
    if fusion not in ('meanmax', 'concat'):
        raise ValueError(f'Unsupported BEVNet fusion mode: {fusion}')
    state = {k: v for k, v in checkpoint['model'].items()
             if k not in ('grids', 'valid')}
    channels = state['reduce.0.weight'].shape[0]
    n_views = None
    if fusion == 'concat':
        n_views, remainder = divmod(state['head.0.weight'].shape[1] - 2, channels + 1)
        if remainder or not n_views or (view_count is not None and n_views != view_count):
            raise ValueError(f'This concat checkpoint requires {n_views} cameras.')

    aux_view = checkpoint.get('aux_view', any(k.startswith('aux_head.') for k in state))
    model = BEVNet(fusion, n_views, grid.shape, channels, pretrained=False, aux_view=aux_view)
    if grid != trained_grid or 'coords' not in state:
        state['coords'] = model.coords
    model.load_state_dict(state)
    return model.to(device).eval(), grid
