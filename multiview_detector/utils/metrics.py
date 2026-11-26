"""World-distance matching and MODA/MODP."""
import numpy as np


def match(dets, gts, max_dist=0.5):
    """Greedy distance matching (enough for few people). Returns list of (det_i, gt_j, dist)."""
    pairs = sorted(((np.hypot(d[0] - g[0], d[1] - g[1]), i, j) for i, d in enumerate(dets)
                    for j, g in enumerate(gts)), key=lambda t: t[0])
    used_d, used_g, out = set(), set(), []
    for dist, i, j in pairs:
        if dist <= max_dist and i not in used_d and j not in used_g:
            used_d.add(i), used_g.add(j), out.append((i, j, dist))
    return out


def moda_modp(all_dets, all_gts, max_dist=0.5):
    """Wildtrack-style metrics over frames. all_dets/all_gts: lists (per frame) of [(X, Y, ...)]."""
    tp = fp = fn = 0
    dists = []
    for dets, gts in zip(all_dets, all_gts):
        m = match(dets, gts, max_dist)
        tp += len(m)
        fp += len(dets) - len(m)
        fn += len(gts) - len(m)
        dists += [d for _, _, d in m]
    n_gt = tp + fn
    return {
        "MODA": 1 - (fp + fn) / max(n_gt, 1),
        "MODP": float(np.mean([1 - d / max_dist for d in dists])) if dists else 0.0,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(n_gt, 1),
        "err_cm_mean": float(np.mean(dists) * 100) if dists else float("nan"),
        "err_cm_median": float(np.median(dists) * 100) if dists else float("nan"),
        "TP": tp, "FP": fp, "FN": fn,
    }
