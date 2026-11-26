"""BEVNet with optional per-view supervision, with its original parameter names."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.resnet import resnet18
from ..utils.projection import BevGrid


class BEVNet(nn.Module):
    """fusion "meanmax": mean + max of the BEV features over the views that see each cell, plus the number
    of such views; works for any number of cameras, so recordings with 1 and 2 cameras train one model.
    fusion "concat": MVDet's per-view concatenation, fixed `n_views` (checkpoints from before meanmax)."""

    def __init__(self, fusion="meanmax", n_views=None, grid_shape=None, feat_ch=128, pretrained=True, aux_view=False):
        super().__init__()
        self.backbone = resnet18(pretrained)
        self.reduce = nn.Sequential(nn.Conv2d(512, feat_ch, 1), nn.ReLU(inplace=True))
        self.fusion, self.n_views = fusion, n_views
        rows, cols = grid_shape or BevGrid().shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, rows), torch.linspace(-1, 1, cols), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None])  # (1, 2, rows, cols)
        in_ch = (2 * feat_ch + 1 if fusion == "meanmax" else n_views * (feat_ch + 1)) + 2
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, padding=1),
        )
        nn.init.constant_(self.head[-1].bias, -2.19)  # CenterNet prior: sigmoid ~ 0.1

        self.aux_view = aux_view
        self.aux_kind = 'mvdet'
        self.aux_head = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 2, 1, bias=False)) if aux_view else None

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():  # tiny batches -> keep ImageNet BN statistics frozen
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, imgs, grids, return_aux=False, aux_size=None):
        """imgs: (B, N, 3, H, W) normalised; grids: (B, N, rows, cols, 2) from projection_grid (-2 = unseen)."""
        B, N = imgs.shape[:2]
        features = self.backbone(imgs.flatten(0, 1))
        feats = self.reduce(features)  # (B*N, C, h, w)
        logits = self.ground_logits(feats, grids)
        if not return_aux:
            return logits
        if self.aux_head is None:
            raise ValueError('Auxiliary outputs require aux_view=True')
        auxiliary = self.auxiliary_outputs(features, feats, imgs.shape[-2:], aux_size)
        if isinstance(auxiliary, dict):
            return logits, {key: value.view(B, N, *value.shape[1:]) for key, value in auxiliary.items()}
        return logits, auxiliary.view(B, N, 2, *auxiliary.shape[-2:])

    def auxiliary_outputs(self, features, reduced, image_size, aux_size=None):
        auxiliary = self.aux_head[0](features)
        if aux_size is None:
            aux_size = (image_size[0] // 4, image_size[1] // 4)
        auxiliary = F.interpolate(auxiliary, aux_size, mode='bilinear', align_corners=False)
        auxiliary = self.aux_head[2](self.aux_head[1](auxiliary))
        return auxiliary

    def ground_logits(self, feats, grids):
        B, N = grids.shape[:2]
        bev = F.grid_sample(feats, grids.flatten(0, 1).to(feats.dtype), align_corners=True)  # (B*N, C, rows, cols)
        bev = bev.view(B, N, *bev.shape[1:])
        valid = (grids.abs() <= 1).all(-1)[:, :, None].to(bev.dtype)  # (B, N, 1, rows, cols)
        x = self.fuse_views(bev, valid, grids)
        x = torch.cat([x, self.coords.expand(B, -1, -1, -1).to(x.dtype)], 1)
        return self.head(x)

    def fuse_views(self, bev, valid, grids):
        if self.fusion == "meanmax":
            n = valid.sum(1)
            bev = bev * valid  # features are >= 0 (ReLU), so a masked max is the max over the seeing views
            x = torch.cat([bev.sum(1) / n.clamp(min=1), bev.amax(1), n], 1)
        else:
            x = torch.cat([bev, valid], 2).flatten(1, 2)
        return x
