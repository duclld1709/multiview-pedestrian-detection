"""Frame-level pseudo supervision, independent of detector architecture.

Kept annotations remain supervised. Missing annotation files are unknown, never
background ground truth. Only their BEV heatmap receives pseudo supervision.
This module is mirrored in the five independently runnable model packages.
"""
import json
import math
import os

import torch
from torch import nn
import torch.nn.functional as F


def validate_frame_pseudo(drop_mode, drop_ratio, use_pseudo_labels,
                          neg_reweight=None, no_neg_reweight=None,
                          no_peak_loss=False, weight=0.01, positive_region='peak'):
    if positive_region not in ('peak', 'gaussian'):
        raise ValueError('pseudo_positive_region must be peak or gaussian')
    if positive_region == 'gaussian':
        if drop_mode != 'instance' or not use_pseudo_labels:
            raise ValueError('pseudo_positive_region=gaussian requires instance mode and use_pseudo_labels')
        if no_peak_loss:
            raise ValueError('pseudo_positive_region=gaussian requires the pseudo positive loss')
    active = drop_mode == 'frame' and drop_ratio > 0 and use_pseudo_labels
    if neg_reweight is True and no_neg_reweight is True:
        raise ValueError('Conflicting pseudo neg_reweight options')
    if active:
        if neg_reweight is True or no_neg_reweight is False:
            raise ValueError('frame-drop + pseudo labels does not support neg_reweight; '
                             'omit the option enabling it (or set pseudo_no_neg_reweight=true)')
        if no_peak_loss:
            raise ValueError('frame-drop + pseudo labels uses the full BEV loss; '
                             'pseudo_no_peak_loss is not supported')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('pseudo_loss_weight must be finite and non-negative')
    return active


def annotation_files(root, anno_dir, include_unlabeled):
    # Only filenames are read from the full set; hidden labels are never loaded.
    directory = os.path.join(root, 'annotations_positions') if include_unlabeled else anno_dir
    return sorted(name for name in os.listdir(directory) if name.endswith('.json'))


def load_annotation(anno_dir, name):
    path = os.path.join(anno_dir, name)
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


class FramePseudoLoss(nn.Module):
    """Mean of per-frame native BEV losses, scaled once by the frame weight.

    Gaussian targets already have peaks of one: do not blur them a second time
    or scale them by detector confidence. Empty pseudo frames have no loss.
    Focal normalization uses each frame's pseudo peak count, never true GT.
    """
    def __init__(self, kind='focal_logits', weight=0.01):
        super().__init__()
        if kind not in ('mse', 'focal_logits', 'focal_probs'):
            raise ValueError(f'Unknown frame pseudo loss: {kind}')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('pseudo_loss_weight must be finite and non-negative')
        self.kind = kind
        self.weight = weight

    def forward(self, prediction, target):
        target = target.to(prediction.device, dtype=prediction.dtype)
        if target.shape[-2:] != prediction.shape[-2:]:
            target = F.interpolate(target, size=prediction.shape[-2:], mode='nearest')
        valid = target.eq(1).flatten(1).any(1)
        if not valid.any():
            return prediction.sum() * 0
        prediction, target = prediction[valid], target[valid]
        if self.kind == 'mse':
            losses = (prediction - target).square().flatten(1).mean(1)
        else:
            probability = prediction.sigmoid() if self.kind == 'focal_logits' else prediction
            probability = probability.clamp(1e-4, 1 - 1e-4)
            positive = target.eq(1)
            pos = probability.log() * (1 - probability).square() * positive
            neg = (1 - probability).log() * probability.square() * (1 - target).pow(4) * ~positive
            losses = -(pos + neg).flatten(1).sum(1) / positive.flatten(1).sum(1)
        return self.weight * losses.mean()


def select_supervised(target, output, mask):
    """Slice BEV B and flattened per-camera B*S outputs consistently."""
    batch_size, cameras = target['center_img'].shape[:2]
    selected_target = {key: value[mask.to(value.device)] if torch.is_tensor(value) else value
                       for key, value in target.items()
                       if key not in ('frame_pseudo', 'pseudo_center_bev', 'pseudo_weight_bev')}
    selected_output = {}
    for key, value in output.items():
        if not torch.is_tensor(value):
            selected_output[key] = value
        elif key.startswith('img_'):
            shaped = value.reshape(batch_size, cameras, *value.shape[1:])
            selected_output[key] = shaped[mask.to(value.device)].flatten(0, 1)
        else:
            selected_output[key] = value[mask.to(value.device)]
    return selected_target, selected_output
