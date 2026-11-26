"""Cylinder projection helpers shared by inference visualisation and label export."""
import cv2
import numpy as np

CYL_COLOR = (80, 255, 80)


class CylinderColors:
    """Distinct display colours, matched by nearby ground positions between frames.

    This is visual continuity only: colours may switch during crossings/occlusions.
    No person IDs are inferred or written to annotations.
    """
    PALETTE_HEX = ['#E53935', '#2196F3', '#43A047', '#FFB300', '#8E24AA', '#00BCD4',
                   '#F06292', '#F57C00', '#ADCB20', '#546E7A', '#795548', '#004D40']
    PALETTE = [tuple(int(h[i:i+2], 16) for i in (5, 3, 1)) for h in PALETTE_HEX]

    def __init__(self, max_distance=1.0):
        self.previous = []
        self.max_distance = max_distance

    def assign(self, people):
        assigned, used_old = {}, set()
        pairs = sorted((float(np.hypot(p[0] - q[0], p[1] - q[1])), i, j)
                       for i, p in enumerate(people) for j, (q, _) in enumerate(self.previous))
        for distance, i, j in pairs:
            if distance <= self.max_distance and i not in assigned and j not in used_old:
                assigned[i] = self.previous[j][1]
                used_old.add(j)
        used = set(assigned.values())
        for i in range(len(people)):
            if i not in assigned:
                slot = next(k for k in range(len(people) + len(used) + 1) if k not in used)
                assigned[i] = slot
                used.add(slot)
        self.previous = [(p[:2], assigned[i]) for i, p in enumerate(people)]
        return [self.color(assigned[i]) for i in range(len(people))]

    @classmethod
    def color(cls, slot):
        if slot < len(cls.PALETTE):
            return cls.PALETTE[slot]
        hsv = np.uint8([[[int((slot * 137.508) % 360 / 2), 190, 255]]])
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def cylinder_rings(cam, xy, radius, height):
    angle = np.linspace(0, 2 * np.pi, 65)
    base = np.column_stack([xy[0] + radius * np.cos(angle), xy[1] + radius * np.sin(angle)])
    return [cam.project(np.column_stack([base, np.full(len(base), z)])) for z in (0, height)]


def cylinder_box(cam, xy, radius, height):
    rings = cylinder_rings(cam, xy, radius, height)
    uv = np.concatenate([p for p, _ in rings])
    if not all(ok.all() for _, ok in rings) or not np.isfinite(uv).all():
        return None, False
    lo, hi = uv.min(0), uv.max(0)
    w, h = cam.image_size
    if hi[0] < 0 or hi[1] < 0 or lo[0] >= w or lo[1] >= h:
        return None, False
    return np.array([max(0, lo[0]), max(0, lo[1]), min(w - 1, hi[0]), min(h - 1, hi[1])]), True


def draw_person_box(image, cam, xy, radius, height, color):
    """Draw only the outline of the calibrated person's projected bounding box."""
    box, visible = cylinder_box(cam, xy, radius, height)
    if visible:
        xmin, ymin, xmax, ymax = np.rint(box).astype(int)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3, cv2.LINE_AA)


def draw_cylinder_3d(image, cam, xy, radius, height, color, alpha=.38):
    """Shaded translucent cylinder, using only its camera-facing side surface."""
    xy = np.asarray(xy, dtype=float)
    theta = np.linspace(0, 2*np.pi, 65)
    ring = xy + radius*np.column_stack([np.cos(theta), np.sin(theta)])
    levels = np.linspace(0, height, 6)
    points = np.concatenate([np.column_stack([ring, np.full(len(ring), z)]) for z in levels])
    uv, valid = cam.project(points)
    if not valid.all() or not np.isfinite(uv).all():
        return
    h, w = image.shape[:2]
    lo = np.maximum(np.floor(uv.min(0)).astype(int)-3, [0, 0])
    hi = np.minimum(np.ceil(uv.max(0)).astype(int)+3, [w, h])
    if np.any(hi <= lo):
        return
    uv = uv.reshape(len(levels), len(theta), 2)
    mid = (theta[:-1]+theta[1:])/2
    normals = np.column_stack([np.cos(mid), np.sin(mid)])
    facing = np.sum(normals*(cam.center[:2]-xy-radius*normals), axis=1) > 0
    to_cam = cam.center[:2]-xy
    distance = np.linalg.norm(to_cam)
    if distance <= radius:
        return
    to_cam /= distance
    angle = np.deg2rad(45)
    light = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ to_cam
    crop = image[lo[1]:hi[1], lo[0]:hi[0]]
    layer = crop.copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    for k in np.flatnonzero(facing):
        shade = .65+.35*max(0, float(normals[k] @ light))
        polygon = np.rint(np.vstack([uv[:, k], uv[::-1, k+1]])-lo).astype(np.int32)
        tint = tuple(int(channel*shade) for channel in color)
        cv2.fillPoly(layer, [polygon], tint, cv2.LINE_AA)
        cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
    if cam.center[2] > height:
        polygon = np.rint(uv[-1]-lo).astype(np.int32)
        cv2.fillPoly(layer, [polygon], color, cv2.LINE_AA)
        cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
    opacity = mask[..., None].astype(np.float32)*(alpha/255)
    crop[:] = (crop*(1-opacity)+layer*opacity).astype(np.uint8)
    for level in (0, -1):
        for k in range(64):
            if facing[k] or k % 2 == 0:
                p, q = np.rint(uv[level, k:k+2]).astype(int)
                cv2.line(image, tuple(p), tuple(q), color, 2 if facing[k] else 1, cv2.LINE_AA)
    for k in range(64):
        if facing[k] != facing[(k+1) % 64]:
            cv2.polylines(image, [np.rint(uv[:, k+1]).astype(np.int32)], False, color, 2, cv2.LINE_AA)
