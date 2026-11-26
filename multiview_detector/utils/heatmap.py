"""BEV training targets, loss, and peak decoding."""
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_target(grid, XYs, sigma_cells=2.0):
    rows, cols = grid.shape
    target = np.zeros((rows, cols), np.float32)
    rr, cc = np.mgrid[0:rows, 0:cols]
    for r, c in grid.to_rc(XYs):
        if -3 <= r < rows + 3 and -3 <= c < cols + 3:
            target = np.maximum(target, np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma_cells ** 2)))
    for r, c in np.round(grid.to_rc(XYs)).astype(int):  # exact 1.0 at the peak cell (focal-loss positive)
        if 0 <= r < rows and 0 <= c < cols:
            target[r, c] = 1.0
    return target


def focal_loss(logits, target, alpha=2, beta=4):
    """CenterNet penalty-reduced focal loss on a Gaussian target."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos.clamp(min=1)


def decode(heat, grid, thresh=0.4, nms_kernel=5, max_det=20, min_dist=0.3):
    """heat: (rows, cols) probabilities (tensor). Returns [(X, Y, score)], sub-cell refined (3x3 centroid).
    Peaks closer than `min_dist` metres to a stronger one are dropped (two people can't overlap). People standing
    side by side are labelled as close as 0.35 m apart, so 0.4 m (and a 7-cell NMS window) dropped one of them."""
    h = heat[None, None]
    keep = (F.max_pool2d(h, nms_kernel, 1, nms_kernel // 2) == h)[0, 0] & (heat >= thresh)
    idx = torch.nonzero(keep)
    scores = heat[keep]
    order = torch.argsort(scores, descending=True)[:max_det]
    hn = heat.cpu().numpy()
    rows, cols = hn.shape
    dets = []
    for i in order.tolist():
        r, c = idx[i].tolist()
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, rows), max(c - 1, 0), min(c + 2, cols)
        w = hn[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        rc = np.array([(w * rr).sum() / w.sum(), (w * cc).sum() / w.sum()])
        X, Y = grid.to_xy(rc)[0]
        if all(np.hypot(X - d[0], Y - d[1]) >= min_dist for d in dets):  # scores are descending
            dets.append((float(X), float(Y), float(scores[i])))
    return dets
