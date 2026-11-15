"""Wildtrack-style cylinder labelling tool.

Each person is a vertical cylinder standing on the ground plane (Z=0), centred on a ROI grid
node (2.5 cm, like Wildtrack's positionID grid). From above it is a circle on the BEV; in the
camera it is the axis-aligned rectangle bounding the projection of the cylinder (top and bottom
circles projected with the full distortion model).

Reads all shared filenames from Image_subsets/C1, C2, ... using the dataset's
zero-distortion calibration. Raw MP4s are not used. The original OpenCV camera
layout, BEV and cylinder controls are retained. Each locked frame creates
annotations/<image-frame>.json containing positionID, personID (integer or null), and views.
QA annotations open as unlocked drafts; locking replaces their JSON with WaveTrack records.
Unlocking keeps the converted JSON and persists the unlocked state separately.
Draft edits exist only in memory until locked. The copy action copies positions
with their person IDs. Assign the same ID manually across frames to track a person.

Mouse
  Camera   double-click to focus; double-click again or Esc returns to the overview
  BEV      left-drag a cylinder (click empty space = move selected there),
           right-click empty space = add, right-click a cylinder = delete it
  Camera   left-click / drag = put the selected cylinder's feet at that pixel (Wildtrack style), in any view;
           right-click a person's feet = add a cylinder there
  Add      [Add] button or n, then click the floor in the BEV or a person's feet in a camera (Esc cancels)
  Buttons  [LOCK] [Add] [Delete] [Person] [Layout] [Reset layout] [BBox] [Grid] [Outline] [Info] [Site map] in the collapsible Controls tab (t)
Keys
  a / d  prev / next label frame   A / D  -/+ 10 label frames   f  next unlocked frame
  space  play / pause               Enter / l  lock + save        u  unlock (keeps saved file)
  arrows nudge selected by 1 cell   n  add cylinder   x / Delete  delete selected   Tab  next cylinder
  c  copy previous label frame
  r  assign / change selected person ID (blank clears it; unlock before editing)
  Ctrl+Z  undo annotation edit (current frame, since last lock)
  v  show/hide selected person on all cameras (BEV marker stays selectable)
  j  toggle live camera crops of the selected person in the left sidebar
  t  open / close Controls tab
  p  edit / lock overview layout; drag panels, drag bottom-right corner to resize proportionally
  [ / ]  radius -/+ 1 cm            - / =  height -/+ 5 cm
  1-7  focus camera (Esc returns to overview)
  Wheel zoom at cursor; middle-drag pan; Home/0 fit to window; z native pixels (scale shown as % of source pixels)
  g  floor grid    b  rectangle / cylinder    o  cylinder outline    i  info panel    h  help    q  quit
  Esc  cancel Add / leave focused camera / quit overview
"""
import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np

from bev import BevCanvas, draw_floor_grid, fit_window
from recording import Recording
from dataset import atomic_json, POSITION_CONVENTION
from viewport import Viewport
from layout import OverviewLayout
from annotation_startup import choose_startup_paths
from frame_picker import FramePicker

WIN = "Wave Tool"
# cv2.waitKeyEx codes on Windows
K_LEFT, K_UP, K_RIGHT, K_DOWN = 2424832, 2490368, 2555904, 2621440
K_DELETE = 3014656
LOCK_COLOR = (60, 60, 230)
DRAFT_COLOR = (0, 200, 255)
CYL_COLOR = (255, 120, 0)
# Per-frame display colours (BGR), not identities; avoid selection white, grid cyan, ROI yellow
PALETTE = [(255, 120, 0), (40, 200, 40), (50, 50, 235), (0, 140, 255), (200, 70, 150), (140, 140, 0)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("auto", "wavetrack", "wildtrack"), default="auto",
                    help="auto-detect WILDTRACK calibration; WILDTRACK is always read-only")
    ap.add_argument("--source", default=None, help="suggested dataset root or Image_subsets directory in the startup dialog")
    ap.add_argument("--calib", default=None, help="suggested calibration folder with extrinsic/, intrinsic_zero/, grid.json")
    ap.add_argument("--labels-dir", default=None, help="suggested annotation folder in the startup dialog")
    ap.add_argument("--fps-label", type=float, default=2.0, help="preview playback speed only; every subset image is labelled")
    ap.add_argument("--height-m", type=float, default=1.8, help="cylinder height (Wildtrack/multicam-gt: 1.8)")
    ap.add_argument("--radius-m", type=float, default=0.25, help="cylinder radius (multicam-gt width 0.5)")
    ap.add_argument("--panel-height", type=int, default=None, help="default: 720 (1 camera), 960 (more)")
    ap.add_argument("--site-map", default=str(Path(__file__).with_name('wavetrack-site-map.json')),
                    help="floor-plan JSON for the BEV background (display only)")
    args = ap.parse_args()
    if args.fps_label <= 0 or args.radius_m <= 0 or args.height_m <= 0:
        ap.error("Playback fps and cylinder dimensions must be positive")
    if args.panel_height is not None and args.panel_height < 400:
        ap.error("--panel-height must be at least 400")
    return args


class Cylinder:
    def __init__(self, XY, color=0, person_id=None):
        self.XY = np.asarray(XY, dtype=np.float64)
        self.person_id = person_id
        self.visible = True  # display only; hidden people are still exported
        self.color = color  # index into PALETTE; display only, not saved


def free_color(cyls):
    """First palette colour not used by `cyls` (cycles when there are more cylinders than colours)."""
    used = {c.color for c in cyls}
    return next((i for i in range(len(PALETTE)) if i not in used), len(cyls) % len(PALETTE))


def validate_person_ids(cyls):
    used = set()
    for cylinder in cyls:
        value = cylinder.person_id
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise ValueError('personID must be a non-negative integer or null')
        if value in used:
            raise ValueError(f'Duplicate personID {value} in this frame')
        used.add(value)


def ask_person_id(initial):
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    try:
        return simpledialog.askstring('Person ID', 'Non-negative integer (blank clears ID):',
                                      initialvalue=initial, parent=root)
    finally:
        root.destroy()


class FrameLabel:
    def __init__(self, cyls, radius, height, locked=False, imported=False):
        self.cyls, self.radius, self.height, self.locked = cyls, radius, height, locked
        self.imported = imported
        self.undo_history = []


def cylinder_points(XY, radius, height, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([XY[0] + radius * np.cos(t), XY[1] + radius * np.sin(t)])
    bottom = np.column_stack([ring, np.zeros(n)])
    top = np.column_stack([ring, np.full(n, height)])
    return bottom, top


def cylinder_box(cam, XY, radius, height):
    """Returns (box [x1,y1,x2,y2] clipped to image or None, bottom uv, top uv, truncated flag)."""
    bottom, top = cylinder_points(XY, radius, height)
    uv, front = cam.project(np.vstack([bottom, top]))
    if not front.any():
        return None, None, None, True
    w, h = cam.image_size
    pts = uv[front]
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    truncated = (not front.all()) or x1 < 0 or y1 < 0 or x2 > w - 1 or y2 > h - 1
    box = np.array([max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)])
    if box[2] <= box[0] or box[3] <= box[1]:
        box = None
    n = len(bottom)
    return box, np.where(front[:n, None], uv[:n], np.nan), np.where(front[n:, None], uv[n:], np.nan), truncated


def _shade(color, k):
    return tuple(int(np.clip(c * k, 0, 255)) for c in color)


def draw_cylinder_3d(img, cam, XY, radius, height, color, selected=False, alpha=0.38, n_seg=64, n_z=6):
    """Shaded, solid-looking cylinder: Lambert-shaded camera-facing side strips, exact silhouette
    generators, visible rim arcs solid / hidden arcs dashed, soft contact shadow on the floor.
    Every edge is sampled densely so lens distortion bends it like the real image."""
    XY = np.asarray(XY, dtype=np.float64)
    C = cam.center
    to_cam = C[:2] - XY
    dist = np.linalg.norm(to_cam)
    if dist <= radius:
        return
    to_cam /= dist
    # Light from upper-left of the viewer: to-camera direction rotated by +50 deg in the ground plane
    a = np.deg2rad(50)
    light = np.array([to_cam[0] * np.cos(a) - to_cam[1] * np.sin(a), to_cam[0] * np.sin(a) + to_cam[1] * np.cos(a)])

    th = np.linspace(0, 2 * np.pi, n_seg + 1)
    ring = XY + radius * np.column_stack([np.cos(th), np.sin(th)])
    mid = (th[:-1] + th[1:]) / 2
    normals = np.column_stack([np.cos(mid), np.sin(mid)])
    ring_mid = XY + radius * normals
    view = C[:2] - ring_mid
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing_cos = (normals * view).sum(1)
    facing = facing_cos > 0
    top_visible = C[2] > height

    z = np.linspace(0, height, n_z)
    # side surface grid: (n_seg+1) angles x n_z heights
    grid = np.concatenate([np.column_stack([ring, np.full(len(ring), zz)]) for zz in z])
    uv, front = cam.project(grid)
    if not front.all():
        return
    uv = uv.reshape(n_z, n_seg + 1, 2)
    h_img, w_img = img.shape[:2]
    pts_all = uv.reshape(-1, 2)
    x0, y0 = np.floor(pts_all.min(0)).astype(int) - 4
    x1, y1 = np.ceil(pts_all.max(0)).astype(int) + 4
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w_img), min(y1, h_img)
    if x1 <= x0 or y1 <= y0:
        return
    off = np.array([x0, y0])

    # Contact shadow on the floor (drawn straight onto the image, under the body)
    sh_ring = XY + 1.25 * radius * np.column_stack([np.cos(th), np.sin(th)]) - 0.06 * light
    sh_uv, sh_ok = cam.project(np.column_stack([sh_ring, np.zeros(len(sh_ring))]))
    if sh_ok.all():
        k = min(max(3, int(0.2 * np.ptp(sh_uv[:, 0]))), 151) | 1  # blur kernel; pad the crop by it so the blur isn't clipped
        sx0, sy0 = np.floor(sh_uv.min(0)).astype(int) - k
        sx1, sy1 = np.ceil(sh_uv.max(0)).astype(int) + k
        sx0, sy0, sx1, sy1 = max(sx0, 0), max(sy0, 0), min(sx1, w_img), min(sy1, h_img)
        if sx1 > sx0 and sy1 > sy0:
            mask = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
            cv2.fillPoly(mask, [(sh_uv - [sx0, sy0]).astype(np.int32)], 1.0, cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None] * 0.4
            roi = img[sy0:sy1, sx0:sx1]
            roi[:] = (roi * (1 - mask)).astype(np.uint8)

    crop = img[y0:y1, x0:x1]
    layer = crop.copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    # Side strips, back-to-front isn't needed: only camera-facing strips of a convex body are drawn
    for i in np.flatnonzero(facing):
        lam = max(0.0, float(normals[i] @ light))
        k = 0.35 + 0.45 * lam + 0.35 * facing_cos[i] ** 2
        poly = np.vstack([uv[:, i], uv[::-1, i + 1]]) - off
        poly = poly.astype(np.int32)
        cv2.fillPoly(layer, [poly], _shade(color, k))
        cv2.fillPoly(mask, [poly], 255)
    if top_visible:
        cap = (uv[-1] - off).astype(np.int32)
        cv2.fillPoly(layer, [cap], _shade(color, 1.25))
        cv2.fillPoly(mask, [cap], 255)
    m = (mask[..., None] / 255.0) * alpha
    crop[:] = (crop * (1 - m) + layer * m).astype(np.uint8)

    # Edges
    edge = _shade(color, 0.55) if not selected else (255, 255, 255)
    t_vis = 1  # Match the thin dashed rim, including selected cylinders
    for level, cap_visible in ((0, False), (n_z - 1, top_visible)):
        r = uv[level]
        for i in range(n_seg):
            p, q = tuple(r[i].astype(int)), tuple(r[i + 1].astype(int))
            if facing[i] or cap_visible:
                cv2.line(img, p, q, edge, t_vis, cv2.LINE_AA)
            elif i % 2 == 0:  # hidden rim -> dashed, thin
                cv2.line(img, p, q, edge, 1, cv2.LINE_AA)
    # Silhouette generators: where facing flips between neighbouring strips
    for i in range(n_seg):
        if facing[i] != facing[(i + 1) % n_seg]:
            cv2.polylines(img, [uv[:, i + 1].astype(np.int32)], False, edge, t_vis, cv2.LINE_AA)


class WaveTool:
    def __init__(self, args):
        self.viewport = Viewport()
        self.display_canvas = None
        self.rec = Recording(args.source, args.calib, getattr(args, "dataset", "auto"))
        self._read_only = self.rec.dataset.read_only
        self.cams, self.roi, self.pitch = self.rec.cams, self.rec.roi, self.rec.pitch
        self.reader = self.rec.reader()
        print(self.rec)
        # Keep camera frames at their native pixel size in the composed image. Overview has
        # three columns and two rows; the BEV occupies the last slot in row two.
        n = len(self.cams)
        self.ph = args.panel_height or max(c.image_size[1] for c in self.cams)
        self.scales = [1.0] * n
        self.bev = BevCanvas(self.cams, self.roi, self.pitch, height=self.ph,
                             site_map_path=None if self.read_only else getattr(args, 'site_map', Path(__file__).with_name('wavetrack-site-map.json')))
        self.view_mode = None
        self.overview = None
        self.configure_layout()

        self.fps_label = args.fps_label
        self.step = 1
        self.n_label = len(self.rec)
        self.default_radius, self.default_height = args.radius_m, args.height_m
        self.sync_cam = None  # camera whose time offset , / . adjust (click a view to pick)

        self.labels_dir = Path(args.labels_dir) if args.labels_dir else self.rec.labels
        if not self.read_only:
            self.labels_dir.mkdir(parents=True, exist_ok=True)
        elif not self.labels_dir.is_dir():
            raise ValueError(f"Missing WILDTRACK annotations folder: {self.labels_dir}")
        self.layout_path = self.labels_dir / ".editor" / "layout.json"
        self.overview.load(self.layout_path)
        self.configure_layout()
        self.frames = {}  # label index k -> FrameLabel
        self.annotation_paths = {}
        self.load_locked()
        print(f"Image subset: {self.n_label} frames (playback {args.fps_label:g} fps), "
              f"{self.n_locked()} already locked in {self.labels_dir}")

        self.k, self.sel, self.playing = 0, 0, False
        self.frame_picker = FramePicker(self.rec.frame_ids)
        self.frame_picker_dirty = False
        self.show_grid, self.show_bbox, self.show_outline = not self.read_only, self.read_only, True
        self.show_info = False  # text panel at the top left of the camera views
        self.drag = None  # "bev" | camera index
        self.camera_drag_origin = None
        self.undo_drag = None
        self.adding = False  # Add pressed: the next left click places a new cylinder
        self.msg, self.msg_until = "", 0.0
        self.dirty = True
        self.ensure_frame(self.k)

        self.controls_open = True
        self.show_crops = True
        self.crop_resize = None
        self.controls_width = 320
        self.controls_scroll = 0
        self.window_size = None
        self.control_groups = []
        self.buttons = []
        y = 76
        for title, rows in (
                ('ANNOTATION', (('lock',),) if self.read_only else (('lock',), ('Add', 'Delete'), ('Person ID',))),
                ('PERSON DISPLAY', (('Person', 'Crops'), ('BBox', 'Outline'))),
                ('VIEW', (('Grid', 'Info'), ('Site map',))),
                ('WORKSPACE LAYOUT', (('Layout', 'Reset layout'),))):
            top = y
            y += 28
            for row in rows:
                width = (280 - 8 * (len(row)-1)) // len(row)
                for column, name in enumerate(row):
                    x = 20 + column * (width + 8)
                    self.buttons.append((name, (x, y, x+width, y+34)))
                y += 40
            self.control_groups.append((title, (12, top, 308, y)))
            y += 12
        self.shortcut_groups = (
            ('FRAMES', (('a / d', 'Previous / next frame'),
                        ('Shift+A/D', 'Back / forward 10 frames'),
                        ('f', 'Next unlocked frame'),
                        ('Space', 'Play / pause'),
                        ('Enter / l', 'Lock + save'),
                        ('u / c', 'Unlock / copy previous'))),
            ('PERSON', (('n / x, Del', 'Add / delete person'),
                        ('r', 'Assign / change person ID'),
                        ('Ctrl+Z', 'Undo annotation edit'),
                        ('Tab', 'Select next person'),
                        ('Arrows', f'Move {self.roi.cell * 100:g} cm per press'),
                        ('[ / ]', 'Radius - / + 1 cm'),
                        ('- / =', 'Height - / + 5 cm'),
                        ('v', 'Hide / show person'))),
            ('VIEW & LAYOUT', (('b / o', 'BBox / outline'),
                               ('j', 'Live person crops'),
                               ('g / i', 'Camera grid / info'),
                               ('p', 'Edit / lock layout'),
                               ('t / h', 'Controls / console help'),
                               ('1-5', 'Focus camera C1-C5'),
                               ('Home, 0 / z', 'Fit / native pixels'),
                               ('Esc', 'Cancel / back / quit'),
                               ('q', 'Quit'))),
            ('MOUSE', (('Left drag', 'Move person'),
                       ('Right click', 'Add / delete person'),
                       ('Double click', 'Focus / back to overview'),
                       ('Wheel', 'Zoom / scroll Controls'),
                       ('Middle drag', 'Pan view'))))
        self.frame_lists_top = y
        self.frame_lists_open = False
        self.frame_list_pages = {'draft': 0, 'unlabeled': 0}
        self.shortcuts_open = False
        if self.read_only:
            self.shortcut_groups = (
                ('WILDTRACK READ ONLY', (('a / d', 'Previous / next frame'),
                                         ('Shift+A/D', 'Back / forward 10 frames'),
                                         ('Space', 'Play / pause'),
                                         ('Tab', 'Select next person'),
                                         ('1 - 7', 'Focus camera'),
                                         ('b / o', 'BBox / outline'),
                                         ('j / v', 'Crops / person visibility'),
                                         ('Wheel', 'Zoom / scroll Controls'),
                                         ('Middle drag', 'Pan view'),
                                         ('Double click', 'Focus / back to overview'))),)

    @property
    def frame_lists_height(self):
        return 402 if self.frame_lists_open else 46

    @property
    def shortcuts_top(self):
        return self.frame_lists_top + self.frame_lists_height

    @property
    def shortcuts_rect(self):
        return (12, self.shortcuts_top, 308, self.shortcuts_top + 34)

    def frame_lists(self):
        """Return label indices; merely visiting an empty frame is not labeling it."""
        draft, unlabeled = [], []
        for k in range(self.n_label):
            frame = self.frames.get(k)
            if frame is not None and frame.locked:
                continue
            (draft if frame is not None and (frame.cyls or frame.imported) else unlabeled).append(k)
        return {'draft': draft, 'unlabeled': unlabeled}

    def frame_list_controls(self):
        """Visible controls shared by rendering and hit testing, with bounded pages."""
        y = self.frame_lists_top
        items = [('toggle', None, (12, y, 308, y + 34),
                  ('[-]' if self.frame_lists_open else '[+]') + ' Frame lists')]
        if not self.frame_lists_open:
            return items
        groups = self.frame_lists()
        for group, title in (('draft', 'Labeled, not locked'), ('unlabeled', 'Unlabeled')):
            y += 38
            indices = groups[group]
            pages = max(1, (len(indices) + 4) // 5)
            page = min(self.frame_list_pages[group], pages - 1)
            self.frame_list_pages[group] = page
            items.append(('title', None, (12, y, 308, y + 24), f'{title}: {len(indices)}'))
            y += 26
            for row, k in enumerate(indices[page * 5:page * 5 + 5]):
                text = f'{self.video_frame(k):08d}'
                if group == 'draft':
                    text += f'   ({len(self.frames[k].cyls)} people)'
                items.append(('frame', k, (20, y + row * 24, 300, y + row * 24 + 22), text))
            if not indices:
                items.append(('title', None, (20, y, 300, y + 22), 'None'))
            y += 122
            items.extend([
                ('page', (group, -1), (20, y, 78, y + 24), '< Prev'),
                ('title', None, (90, y, 230, y + 24), f'{page + 1} / {pages}'),
                ('page', (group, 1), (242, y, 300, y + 24), 'Next >')])
        return items

    @property
    def controls_height(self):
        body = sum(30 + 22 * len(rows) for _, rows in self.shortcut_groups) if self.shortcuts_open else 0
        return self.shortcuts_top + 46 + body + 12

    def toggle_controls(self):
        self.controls_open = not self.controls_open
        self.drag = self.camera_drag_origin = None
        self.overview.drag = None
        self.viewport.pan_start = None
        self.dirty = True

    def content_size(self, size):
        w, h = size
        return max(1, w - (self.controls_width if self.controls_open else 0) - self.crop_width(size)), h

    def crop_width(self, size):
        available = max(1, size[0] - (self.controls_width if self.controls_open else 0))
        requested = self.overview.crop_sidebar_width
        if requested is None:
            requested = min(260, available // 4)
        return min(requested, max(1, available - min(200, available // 2))) if self.show_crops else 0

    def toggle_crops(self):
        self.crop_resize = None
        self.show_crops = not self.show_crops
        self.drag = self.camera_drag_origin = None
        self.overview.drag = None
        self.viewport.pan_start = None
        self.dirty = True

    @property
    def read_only(self):
        return getattr(self, '_read_only', False)

    def person_box(self, cam, person, fl):
        if self.read_only:
            return person.boxes.get(cam.number - 1, [-1] * 4)
        return cam.box(person.XY, fl.radius, fl.height)

    def selected_crops(self):
        """Crop native camera pixels using the current selected cylinder geometry."""
        person = self.selected
        if person is None:
            return [None] * len(self.cams)
        images = self.reader.get(self.video_frame(self.k))
        crops = []
        for cam, image in zip(self.cams, images):
            x1, y1, x2, y2 = self.person_box(cam, person, self.cur)
            h, w = image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crops.append(image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None)
        return crops

    def draw_crops(self, width, height):
        panel = np.full((height, width, 3), (30, 34, 39), np.uint8)
        if width < 24 or height < 70:
            return panel
        label = (f'Person {self.sel + 1} | ID: {self.selected.person_id if self.selected.person_id is not None else "unassigned"}'
                 if self.selected is not None else 'Select a person')
        cv2.putText(panel, label, (8, 49), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (235, 235, 235), 1, cv2.LINE_AA)
        row_height = (height - 60) // max(1, len(self.cams))
        for i, (cam, crop) in enumerate(zip(self.cams, self.selected_crops())):
            top = 60 + i * row_height
            cv2.putText(panel, cam.name, (8, top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, (220, 220, 220), 1, cv2.LINE_AA)
            if row_height < 32:
                continue
            if crop is None:
                cv2.putText(panel, 'No visible crop', (8, top + 34), cv2.FONT_HERSHEY_SIMPLEX,
                            .35, (160, 160, 160), 1, cv2.LINE_AA)
                continue
            ch, cw = crop.shape[:2]
            scale = min((width - 16) / cw, (row_height - 26) / ch)
            tw, th = max(1, round(cw * scale)), max(1, round(ch * scale))
            tile = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            x, y = (width - tw) // 2, top + 22
            panel[y:y+th, x:x+tw] = tile
        return panel

    def controls_mouse(self, event, x, y):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for kind, value, (x1, y1, x2, y2), _ in self.frame_list_controls():
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            if kind == 'toggle':
                self.frame_lists_open = not self.frame_lists_open
            elif kind == 'frame' and not self.overview.editing:
                self.playing = False
                self.drag = self.camera_drag_origin = None
                self.adding = False
                self.set_label_frame(value)
            elif kind == 'page':
                group, delta = value
                pages = max(1, (len(self.frame_lists()[group]) + 4) // 5)
                self.frame_list_pages[group] = int(np.clip(self.frame_list_pages[group] + delta, 0, pages - 1))
            if self.window_size:
                self.controls_scroll = min(self.controls_scroll, max(0, self.controls_height - self.window_size[1]))
            self.dirty = True
            return
        x1, y1, x2, y2 = self.shortcuts_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.shortcuts_open = not self.shortcuts_open
            if self.window_size:
                self.controls_scroll = min(self.controls_scroll, max(0, self.controls_height-self.window_size[1]))
            self.dirty = True
            return
        for name, (x1, y1, x2, y2) in self.buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if not self.overview.editing or name in ("Layout", "Reset layout"):
                    self.on_button(name)
                return

    def configure_layout(self):
        """Set source-pixel camera rectangles and the BEV position for the active layout."""
        if self.view_mode is not None:
            v = self.view_mode
            w, h = self.cams[v].image_size
            self.camera_rects = {v: (0, 0, w, h)}
            self.bev_origin = (w, 0)
            self.canvas_w = w + self.bev.w
            self.canvas_h = max(h, self.bev.h)
            return

        # Camera columns use the widest native camera frame. Give the BEV its own width in
        # the last column, rather than padding every camera out to the BEV's often larger width.
        self.cell_w = max(c.image_size[0] for c in self.cams)
        self.cell_h = max(max(c.image_size[1] for c in self.cams), self.bev.h)
        self.camera_rects = {}
        for v, cam in enumerate(self.cams):
            x, y = (v % 3) * self.cell_w, (v // 3) * self.cell_h
            self.camera_rects[v] = (x, y, *cam.image_size)
        rows = (len(self.cams) + 3) // 3
        self.bev_origin = ((len(self.cams) % 3) * self.cell_w, (len(self.cams) // 3) * self.cell_h)
        self.canvas_w, self.canvas_h = 2 * self.cell_w + max(self.cell_w, self.bev.w), rows * self.cell_h
        if self.overview is None:
            sizes = {cam.name: cam.image_size for cam in self.cams}
            sizes['BEV'] = (self.bev.w, self.bev.h)
            rects = {self.cams[v].name: rect for v, rect in self.camera_rects.items()}
            rects['BEV'] = (*self.bev_origin, self.bev.w, self.bev.h)
            self.overview = OverviewLayout(sizes, rects, (self.canvas_w, self.canvas_h))
        self.camera_rects = {v: self.overview.rects[cam.name] for v, cam in enumerate(self.cams)}
        self.bev_origin = self.overview.rects['BEV'][:2]
        self.canvas_w, self.canvas_h = self.overview.canvas_size

    # --- label storage ------------------------------------------------------------------------
    def video_frame(self, k):
        return self.rec.frame_ids[k]

    def label_path(self, k):
        return self.annotation_paths.get(k, self.labels_dir / f"{self.video_frame(k):08d}.json")

    def n_locked(self):
        return sum(f.locked for f in self.frames.values())

    def load_locked(self):
        for p in sorted(self.labels_dir.glob("*.json")):
            if not p.stem.isdigit():
                continue
            frame = int(p.stem)
            if frame not in self.rec.frame_indices:
                print(f"  skip {p.name}: no matching images in all cameras")
                continue
            k = self.rec.frame_indices[frame]
            if k in self.annotation_paths:
                raise ValueError(f"{p}: duplicate annotation for frame {frame}")
            with open(p, encoding="utf-8-sig") as f:
                people = json.load(f)
            meta = self.labels_dir / ".editor" / p.name
            settings = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() and not self.read_only else {}
            native = settings.get("position_convention") == POSITION_CONVENTION
            if meta.exists() and not native and not self.read_only:
                raise ValueError(f"{p}: legacy annotation; run wave_tool/scripts/migrate_grid_labels.py before opening")
            radius = settings.get("radius", self.default_radius)
            height = settings.get("height", self.default_height)
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                   not np.isfinite(v) or v <= 0 for v in (radius, height)):
                raise ValueError(f"{p}: invalid cylinder dimensions")
            if not isinstance(people, list):
                raise ValueError(f"{p}: expected a list of people")
            cyls = []
            for index, person in enumerate(people):
                if not isinstance(person, dict):
                    raise ValueError(f"{p}: person {index + 1}: expected an object")
                if native and set(person) != {"positionID", "personID", "views"}:
                    raise ValueError(f"{p}: person {index + 1}: expected positionID, personID, views")
                if not self.read_only and not native and 'position_m' in person:
                    XY = person['position_m']
                    if (not isinstance(XY, list) or len(XY) != 2 or
                            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                                not np.isfinite(v) for v in XY)):
                        raise ValueError(f"{p}: person {index + 1}: invalid position_m")
                else:
                    position = person.get("positionID")
                    if isinstance(position, bool) or not isinstance(position, int):
                        raise ValueError(f"{p}: person {index + 1}: invalid positionID {position!r}")
                    XY = self.rec.dataset.xy(position)
                cylinder = Cylinder(XY, free_color(cyls), person_id=person.get('personID'))
                if self.read_only:
                    cylinder.boxes = {}
                    for view in person.get('views', []):
                        number = view.get('viewNum')
                        box = [view.get(key) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
                        if (type(number) is not int or not 0 <= number < len(self.cams)
                                or number in cylinder.boxes or any(type(v) is not int for v in box)
                                or (box != [-1] * 4 and (box[2] < box[0] or box[3] < box[1]))):
                            raise ValueError(f'{p}: invalid WILDTRACK view bbox')
                        cylinder.boxes[number] = box
                cyls.append(cylinder)
            if not self.read_only:
                try:
                    validate_person_ids(cyls)
                except ValueError as exc:
                    raise ValueError(f'{p}: {exc}') from exc
            locked = True if self.read_only else settings.get('locked', True) if native else False
            if not isinstance(locked, bool):
                raise ValueError(f"{meta}: locked must be true or false")
            self.annotation_paths[k] = p
            self.frames[k] = FrameLabel(cyls, radius, height, locked=locked, imported=not native)

    def save_frame_settings(self, k, locked):
        if self.read_only:
            raise PermissionError("WILDTRACK is read-only")
        fl = self.frames[k]
        atomic_json(self.labels_dir / '.editor' / self.label_path(k).name,
                    {'radius': fl.radius, 'height': fl.height,
                     'position_convention': POSITION_CONVENTION, 'locked': locked})

    def save_frame(self, k):
        if self.read_only:
            raise PermissionError("WILDTRACK is read-only")
        fl = self.frames[k]
        validate_person_ids(fl.cyls)
        people = []
        positions = []
        for c in fl.cyls:
            try:
                pos = self.rec.dataset.position(c.XY)
            except ValueError:
                raise ValueError("A detection is outside the ROI; move it inside before locking") from None
            XY = self.rec.dataset.xy(pos)
            positions.append(XY)
            views = []
            for cam in self.cams:
                xmin, ymin, xmax, ymax = cam.box(XY, fl.radius, fl.height)
                views.append({"viewNum": cam.number - 1, "xmin": xmin, "ymin": ymin,
                              "xmax": xmax, "ymax": ymax})
            people.append({"positionID": pos, "personID": c.person_id, "views": views})
        # Editing metadata is separate from the Wildtrack detection schema.
        atomic_json(self.label_path(k), people)
        self.save_frame_settings(k, locked=True)
        for c, XY in zip(fl.cyls, positions):
            c.XY = XY
        fl.imported = False

    # --- frame state --------------------------------------------------------------------------
    @property
    def cur(self):
        return self.frames[self.k]

    @property
    def cyls(self):
        return self.cur.cyls

    def prev_labelled(self, k):
        for j in range(k - 1, -1, -1):
            if j in self.frames:
                return self.frames[j]
        return None

    def ensure_frame(self, k):
        if k not in self.frames:
            prev = self.prev_labelled(k)
            radius = prev.radius if prev else self.default_radius
            height = prev.height if prev else self.default_height
            self.frames[k] = FrameLabel([], radius, height, locked=self.read_only)
        self.sel = min(self.sel, len(self.cyls) - 1)
        if self.sel < 0 and self.cyls:
            self.sel = 0

    def editable(self):
        if self.read_only:
            self.flash("WILDTRACK READ ONLY - editing and unlock disabled")
            return False
        if self.cur.locked:
            self.flash("Frame LOCKED - press u to unlock")
            return False
        return True

    def remember_edit(self):
        """Keep one snapshot per edit, grouping a continuous mouse drag."""
        if self.drag is not None and self.undo_drag == self.k:
            return
        self.cur.undo_history.append((copy.deepcopy(self.cyls), self.cur.radius,
                                      self.cur.height, self.sel))
        del self.cur.undo_history[:-100]
        if self.drag is not None:
            self.undo_drag = self.k

    def undo(self):
        self.playing = False
        self.drag = self.camera_drag_origin = self.undo_drag = None
        self.adding = False
        if not self.editable():
            return
        if not self.cur.undo_history:
            self.flash('Nothing to undo in this frame')
            return
        self.cur.cyls, self.cur.radius, self.cur.height, self.sel = self.cur.undo_history.pop()
        self.flash('Undone (draft; lock to save)')

    def flash(self, text, sec=1.5):
        self.msg, self.msg_until = text, time.time() + sec
        self.dirty = True

    def lock(self):
        if self.read_only:
            self.flash("WILDTRACK READ ONLY")
            return
        if self.cur.locked:
            return
        try:
            self.save_frame(self.k)
        except (OSError, ValueError) as error:
            self.flash(f"Save failed: {error}", sec=5)
            return
        self.cur.locked = True
        self.cur.undo_history.clear()
        self.flash(f"Locked + saved {self.label_path(self.k).name}")

    def unlock(self):
        if self.read_only:
            self.flash("WILDTRACK READ ONLY - unlock disabled")
            return
        if not self.cur.locked:
            return
        try:
            self.save_frame_settings(self.k, locked=False)
        except OSError as error:
            self.flash(f"Unlock failed: {error}", sec=5)
            return
        self.cur.locked = False
        self.flash("Unlocked (WaveTrack file kept)")

    def shift_sync(self, frames):
        self.flash("Image subsets are matched by filename; video sync is disabled")

    def set_person_id(self, value):
        if not self.editable():
            return False
        person = self.selected
        if person is None:
            self.flash('Select a person first')
            return False
        if value is not None and (type(value) is not int or value < 0):
            self.flash('Person ID must be a non-negative integer, or blank to clear')
            return False
        if value is not None and any(c is not person and c.person_id == value for c in self.cyls):
            self.flash(f'Person ID {value} is already assigned in this frame')
            return False
        if person.person_id == value:
            return True
        self.remember_edit()
        person.person_id = value
        self.flash(f'Person ID: {value if value is not None else "unassigned"} (lock to save)')
        return True

    def edit_person_id(self):
        if not self.editable():
            return
        if self.selected is None:
            self.flash('Select a person first')
            return
        self.playing = False
        self.drag = self.camera_drag_origin = self.undo_drag = None
        self.adding = False
        current = self.selected.person_id
        try:
            value = ask_person_id('' if current is None else str(current))
        except Exception as exc:
            self.flash(f'Cannot open Person ID dialog: {exc}', sec=5)
            return
        if value is None:
            return
        value = value.strip()
        if value and not (value.isascii() and value.isdecimal()):
            self.flash('Person ID must be a non-negative integer, or blank to clear')
            return
        try:
            self.set_person_id(int(value) if value else None)
        except ValueError:
            self.flash('Invalid person ID')

    # --- geometry helpers ---------------------------------------------------------------------
    def snap(self, XY):
        """Snap to a grid node without a half-cell offset. The grid is extended
        beyond the ROI so people standing just outside it are not dragged onto the ROI border."""
        r = self.roi
        row, col = r.to_cell(XY[0], XY[1])
        return np.array([r.x0 + col * r.cell, r.y0 + row * r.cell])

    def cell_info(self, XY):
        """(row, col, positionID); positionID = -1 outside the ROI grid."""
        row, col = (int(v) for v in self.roi.to_cell(XY[0], XY[1]))
        rows, cols = self.roi.shape
        return row, col, row * cols + col if 0 <= row < rows and 0 <= col < cols else -1

    @property
    def selected(self):
        return self.cyls[self.sel] if 0 <= self.sel < len(self.cyls) else None

    def add_cylinder(self, XY):
        if not self.editable():
            return
        self.remember_edit()
        self.cyls.append(Cylinder(self.snap(XY), free_color(self.cyls)))
        self.sel = len(self.cyls) - 1
        self.dirty = True

    def delete_selected(self):
        if not self.editable():
            return
        if self.selected is None:
            self.flash("No cylinder selected")
            return
        self.remember_edit()
        self.cyls.pop(self.sel)
        self.sel = min(self.sel, len(self.cyls) - 1)
        self.dirty = True

    def cylinder_at(self, XY):
        """Index of the cylinder under a BEV point, or None."""
        d = [np.linalg.norm(c.XY - XY) for c in self.cyls]
        return int(np.argmin(d)) if d and min(d) < max(self.cur.radius, 0.15) else None

    def move_selected(self, XY):
        if not self.editable():
            return
        if self.selected is None:
            self.add_cylinder(XY)
        elif not np.array_equal(self.selected.XY, self.snap(XY)):
            self.remember_edit()
            self.selected.XY = self.snap(XY)
        self.dirty = True

    # --- input --------------------------------------------------------------------------------
    def toggle_person(self):
        if self.selected is not None:
            self.selected.visible = not self.selected.visible
        else:
            self.flash("Select a person on the BEV first")
        self.dirty = True

    def toggle_layout(self):
        if self.view_mode is not None:
            self.flash("Return to overview (Esc) to edit the layout")
            return
        if self.overview.editing and not self.save_layout():
            return
        self.overview.editing = not self.overview.editing
        self.overview.drag = None
        self.crop_resize = None
        self.drag = self.camera_drag_origin = None
        self.adding = False
        self.playing = False
        self.dirty = True

    def save_layout(self):
        """Persist the overview, even while layout editing or camera focus is active."""
        if self.read_only:
            return True
        try:
            self.overview.save(self.layout_path)
        except OSError as exc:
            self.flash(f"Cannot save layout: {exc}")
            return False
        return True

    def prepare_layout_workspace(self, size):
        if (self.view_mode is None and self.overview.editing and self.overview.drag is None and self.crop_resize is None
                and self.viewport.zoom == 1 and self.overview.expand_to_window(size)):
            self.configure_layout()
            self.viewport.reset()
            self.save_layout()
            self.dirty = True

    def on_button(self, name):
        if name == "Person ID":
            self.edit_person_id()
        elif name == "Crops":
            self.toggle_crops()
        elif name == "Person":
            self.toggle_person()
        elif name == "Layout":
            self.toggle_layout()
        elif name == "Reset layout":
            self.overview.crop_sidebar_width = None
            self.overview.rects = dict(self.overview.defaults)
            self.overview.order = list(self.overview.defaults)
            self.overview.canvas_size = self.overview.default_canvas_size
            self.overview.editing = False
            self.overview.drag = None
            self.configure_layout()
            self.viewport.reset()
            self.save_layout()
            self.dirty = True
            return
        elif name == "lock":
            self.unlock() if self.cur.locked else self.lock()
        elif name == "Add":
            self.toggle_add()
        elif name == "Delete":
            if self.editable():
                self.delete_selected()
        elif name == "BBox":
            self.show_bbox = not self.show_bbox
        elif name == "Grid":
            self.show_grid = not self.show_grid
        elif name == "Outline":
            self.show_outline = not self.show_outline
        elif name == "Info":
            self.show_info = not self.show_info
        elif name == "Site map":
            self.bev.toggle_site_map()
            if self.bev.site_map is None:
                self.flash("Site map unavailable; using classic BEV")
        self.dirty = True

    def view_at(self, x, y):
        if self.view_mode is None:
            key = self.overview.hit(x, y)
            return next((v for v, cam in enumerate(self.cams) if cam.name == key), None)
        for v, (x0, y0, w, h) in self.camera_rects.items():
            if x0 <= x < x0+w and y0 <= y < y0+h:
                return v
        return None

    def bev_at(self, x, y):
        if self.view_mode is None:
            return self.overview.hit(x, y) == 'BEV'
        bx, by = self.bev_origin
        return bx <= x < bx+self.bev.w and by <= y < by+self.bev.h

    def bev_point(self, x, y):
        if self.view_mode is None:
            return self.overview.local('BEV', x, y)
        bx, by = self.bev_origin
        return x-bx, y-by

    def camera_point(self, view, x, y):
        x0, y0, w, h = self.camera_rects[view]
        sw, sh = self.cams[view].image_size
        return (x-x0)*sw/w, (y-y0)*sh/h

    def floor_at(self, x, y, view=None):
        """Floor point (m) under a window pixel: in the BEV panel, or in camera `view` (default: the view
        under the pixel) where the pixel is taken as a person's feet. None if that ray misses the floor."""
        if view is None and self.bev_at(x, y):
            return self.bev.from_px(*self.bev_point(x, y))
        v = self.view_at(x, y) if view is None else view
        if v is None:
            return None
        x0, y0, w, h = self.camera_rects[v]
        u, v_px = self.camera_point(v, x, y)
        sw, sh = self.cams[v].image_size
        u, v_px = np.clip(u, 0, sw-1), np.clip(v_px, 0, sh-1)
        XY = self.cams[v].image_to_ground(np.array([[u, v_px]]))[0]
        return XY if np.isfinite(XY).all() else None

    def toggle_add(self):
        if not self.adding and not self.editable():
            return
        self.adding = not self.adding
        self.dirty = True

    def on_mouse(self, event, x, y, flags, _):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            self.undo_drag = None
        view = self.view_at(x, y)
        in_bev = self.bev_at(x, y)
        if self.view_mode is None and self.overview.editing:
            handle = max(40, 16 / max(self.viewport.scale, .01))
            finished_drag = event == cv2.EVENT_LBUTTONUP and self.overview.drag is not None
            self.overview.mouse(event, x, y, handle)
            self.configure_layout()
            if finished_drag:
                self.save_layout()
            self.dirty = True
            return
        if event == cv2.EVENT_LBUTTONUP:
            self.drag = None
            self.camera_drag_origin = None
            return
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if view is not None and not self.adding:
                if self.view_mode is None:
                    self.focus_camera(view)
                else:
                    self.drag = None
                    self.camera_drag_origin = None
                    self.view_mode = None
                    self.configure_layout()
                    self.dirty = True
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.editable():
                XY = self.floor_at(x, y)
                hit = None if view is not None or XY is None else self.cylinder_at(XY)
                if hit is not None:
                    self.sel = hit
                    self.delete_selected()
                elif XY is not None:
                    self.add_cylinder(XY)
                self.dirty = True
            return
        if event == cv2.EVENT_LBUTTONDOWN and self.adding:
            XY = self.floor_at(x, y)
            if XY is None:
                self.flash("That point is not on the floor - click a person's feet")
            elif self.editable():
                self.add_cylinder(XY)
                self.adding = False
            self.dirty = True
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.camera_drag_origin = None
            # In overview, grab a cylinder's body without moving it on mouse-down.
            # This also keeps double-click focus from changing the annotation.
            if view is not None and self.view_mode is None and not self.adding:
                self.drag = None
                x0, y0, _, _ = self.camera_rects[view]
                u, v_px = self.camera_point(view, x, y)
                hits = []
                for i, cyl in enumerate(self.cyls):
                    if not cyl.visible:
                        continue
                    left, top, right, bottom = self.person_box(self.cams[view], cyl, self.cur)
                    if left <= u <= right and top <= v_px <= bottom:
                        hits.append(i)
                if hits:
                    self.sel = self.sel if self.sel in hits else hits[-1]
                    self.dirty = True
                    if not self.cur.locked:
                        foot = self.cams[view].ground_to_image([self.selected.XY])[0]
                        if np.isfinite(foot).all():
                            self.drag = view
                            self.camera_drag_origin = (x, y, x0 + foot[0]*self.camera_rects[view][2]/self.cams[view].image_size[0],
                                                       y0 + foot[1]*self.camera_rects[view][3]/self.cams[view].image_size[1])
                if view > 0:
                    self.sync_cam = view
                return
            if view is not None and view > 0:
                self.sync_cam = view
            if in_bev:
                hit = self.cylinder_at(self.bev.from_px(*self.bev_point(x, y)))
                if hit is not None:
                    self.sel = hit  # selecting is allowed on locked frames
                    self.dirty = True
                    self.drag = "bev" if not self.cur.locked else None
                    return
            if not self.editable():
                return
            self.drag = view if view is not None else "bev" if in_bev else None
        if self.drag is None or event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            return
        if self.camera_drag_origin is not None:
            start_x, start_y, foot_x, foot_y = self.camera_drag_origin
            x, y = foot_x + x - start_x, foot_y + y - start_y
        XY = (self.bev.from_px(*self.bev_point(x, y)) if self.drag == "bev"
              else self.floor_at(x, y, view=self.drag))
        if XY is not None:
            self.move_selected(XY)

    def focus_camera(self, view):
        if not 0 <= view < len(self.cams) or self.adding or self.overview.editing:
            return
        self.drag = None
        self.camera_drag_origin = None
        self.view_mode = view
        self.configure_layout()
        if view > 0:
            self.sync_cam = view
        self.dirty = True

    def on_key(self, k):
        picker = getattr(self, 'frame_picker', None)
        if picker is not None:
            handled, selected = picker.key(k)
            if handled:
                self.frame_picker_dirty = True
                if selected is not None:
                    self.playing = False
                    self.set_label_frame(selected)
                return True
        if k == 26:  # OpenCV Ctrl+Z
            if not self.overview.editing:
                self.undo()
            return True
        if ord("1") <= k <= ord("9") and k - ord("1") < len(self.cams):
            self.focus_camera(k - ord("1"))
            return True
        if k == ord("z"):
            vp = self.viewport
            w, h = vp.size
            sw, sh = vp.source_size
            vp.zoom = max(1.0, 1.0 / min(w / sw, h / sh))
            vp.transform()
            self.dirty = True
            return True
        if k in (2359296, 65360, ord("0")):  # Home or 0
            self.viewport.reset()
            self.dirty = True
            return True
        if k < 0:
            return True
        ch = chr(k) if k < 256 else ""
        if ch == 't':
            self.toggle_controls()
            return True
        if ch == 'j':
            self.toggle_crops()
            return True
        if ch == 'p':
            self.toggle_layout()
            return True
        if ch == 'v':
            self.toggle_person()
            return True
        if self.overview.editing and self.view_mode is None and ch not in ('q', 'h', '\x1b'):
            return True
        step = self.roi.cell
        if ch == "\x1b":
            if self.overview.editing:
                self.toggle_layout()
                return True
            if self.adding:
                self.adding = False
                self.dirty = True
                return True
            if self.view_mode is not None:
                self.view_mode = None
                self.configure_layout()
                self.dirty = True
                return True
            return not self.confirm_exit()
        if ch == "q":
            return not self.confirm_exit()
        moves = {K_LEFT: (-step, 0), K_RIGHT: (step, 0), K_UP: (0, step), K_DOWN: (0, -step)}
        edits = set("x[]-=c") | set(moves) | {K_DELETE}
        if (k in edits or ch in edits) and not self.editable():
            return True
        if k in moves and self.selected is not None:
            self.move_selected(self.selected.XY + np.array(moves[k]))
        elif ch and ch in "aAdD":
            self.playing = False
            self.set_label_frame(self.k + {"a": -1, "d": 1, "A": -10, "D": 10}[ch])
        elif ch == "f":
            self.playing = False
            todo = [j for j in range(self.n_label) if not (j in self.frames and self.frames[j].locked)]
            nxt = next((j for j in todo if j > self.k), todo[0] if todo else None)
            if nxt is None:
                self.flash("All label frames are locked")
            else:
                self.set_label_frame(nxt)
        elif ch == " ":
            self.playing = not self.playing
        elif ch in ("\r", "\n", "l"):
            self.lock()
        elif ch == "u":
            self.unlock()
        elif ch == "n":
            self.toggle_add()
        elif ch == "r":
            self.edit_person_id()
        elif ch == "x" or k == K_DELETE:
            self.delete_selected()
        elif ch == "c":
            prev = self.prev_labelled(self.k)
            if prev:
                self.remember_edit()
                self.cur.cyls = copy.deepcopy(prev.cyls)
                self.sel = 0
        elif ch == "\t" and self.cyls:
            self.sel = (self.sel + 1) % len(self.cyls)
        elif ch == "[":
            if self.cur.radius > 0.05:
                self.remember_edit()
            self.cur.radius = max(0.05, self.cur.radius - 0.01)
        elif ch == "]":
            self.remember_edit()
            self.cur.radius += 0.01
        elif ch == "-":
            if self.cur.height > 0.5:
                self.remember_edit()
            self.cur.height = max(0.5, self.cur.height - 0.05)
        elif ch == "=":
            self.remember_edit()
            self.cur.height += 0.05
        elif ch in (",", ".", "<", ">") and self.sync_cam is not None:
            self.shift_sync({",": -1, ".": 1, "<": -10, ">": 10}[ch])
        elif ch == "g":
            self.show_grid = not self.show_grid
        elif ch == "b":
            self.show_bbox = not self.show_bbox
        elif ch == "o":
            self.show_outline = not self.show_outline
        elif ch == "i":
            self.show_info = not self.show_info
        elif ch == "h":
            print(__doc__)
        else:
            return True
        self.dirty = True
        return True

    def process_pending_keys(self, first_key):
        """Apply queued input before the next expensive render, preserving key order."""
        key = first_key
        for index in range(128):
            if key < 0:
                break
            if not self.on_key(key):
                return False
            if index == 127:
                break  # Leave the next event in the OS queue for the next batch.
            key = cv2.pollKey()
        return True

    def set_label_frame(self, k):
        k = int(np.clip(k, 0, self.n_label - 1))
        if k != self.k:
            self.drag = self.camera_drag_origin = self.undo_drag = None
            self.k = k
            self.ensure_frame(k)
        self.dirty = True

    def frame_picker_area(self, size):
        start = self.crop_width(size)
        content_w, _ = self.content_size(size)
        end = min(start + content_w, size[0] - 168)
        return start, max(start + 2, end)

    def draw_frame_picker(self, display):
        start, end = self.frame_picker_area(display.shape[1::-1])
        self.frame_picker.draw(display, start, end, self.k)

    @staticmethod
    def confirm_exit():
        """Ask before leaving Wave Tool. Default to staying if the dialog fails."""
        try:
            import ctypes
            answer = ctypes.windll.user32.MessageBoxW(
                None,
                "Bạn có chắc chắn muốn thoát ứng dụng không?",
                WIN,
                0x00000004 | 0x00000020 | 0x00000100,  # Yes/No, question icon, default No
            )
            return answer == 6  # IDYES
        except (AttributeError, OSError):
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                try:
                    return messagebox.askyesno(
                        WIN,
                        "Bạn có chắc chắn muốn thoát ứng dụng không?",
                        default=messagebox.NO,
                        parent=root,
                    )
                finally:
                    root.destroy()
            except Exception:
                return False

    # --- rendering ----------------------------------------------------------------------------
    def render_view(self, v, img, fl):
        """Draw manual annotations as cylinders or unfilled exported bounding boxes."""
        cam = self.cams[v]
        vis = img.copy()
        if self.show_grid:
            draw_floor_grid(vis, cam, self.roi, self.pitch)
        if self.show_outline:
            # far -> near so closer cylinders are painted over farther ones
            order = sorted(range(len(fl.cyls)), key=lambda j: -np.linalg.norm(fl.cyls[j].XY - cam.center[:2]))
            for j in order:
                c = fl.cyls[j]
                if not c.visible:
                    continue
                if self.show_bbox:
                    box = self.person_box(cam, c, fl)
                    if box == [-1] * 4:
                        continue
                    color = (255, 255, 255) if j == self.sel else PALETTE[c.color]
                    cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), color, 2, cv2.LINE_AA)
                else:
                    draw_cylinder_3d(vis, cam, c.XY, fl.radius, fl.height, PALETTE[c.color], selected=j == self.sel)
                if c.person_id is not None:
                    box = self.person_box(cam, c, fl)
                    if box != [-1] * 4:
                        cv2.putText(vis, f'ID {c.person_id}',
                                    (max(0, box[0]), max(18, box[1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6,
                                    (255, 255, 255) if j == self.sel else PALETTE[c.color], 2, cv2.LINE_AA)
        for c in (c for c in fl.cyls if c.visible):
            # Keep the ground-centre marker visible in both cylinder and bbox modes.
            if self.show_bbox and self.person_box(cam, c, fl) == [-1] * 4:
                continue
            foot = cam.ground_to_image(c.XY[None])[0]
            if np.isfinite(foot).all():
                cv2.drawMarker(vis, tuple(foot.astype(int)), PALETTE[c.color], cv2.MARKER_CROSS, 24, 1)
        w, h = cam.image_size
        tile = vis
        if len(self.cams) > 1:
            f = self.video_frame(self.k)
            text = f"{cam.name}  image {f:08d}  undistorted"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            bg = tile[h - th - 22:h - 4, 4:min(w - 4, tw + 20)]
            bg[:] = (bg * 0.35).astype(np.uint8)
            cv2.putText(tile, text, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2,
                        cv2.LINE_AA)
        return tile

    def render(self):
        fl = self.cur
        vframe = self.video_frame(self.k)
        imgs = self.reader.get(vframe)
        bev_img = self.bev.background.copy()
        rad = int(round(fl.radius * self.bev.scale))
        tiles = {v: self.render_view(v, img, fl)
                 for v, img in enumerate(imgs) if self.view_mode is None or v == self.view_mode}

        info = []
        for i, c in enumerate(fl.cyls):
            color = PALETTE[c.color]
            is_sel = i == self.sel
            # BEV circle = top view of the cylinder
            u, v = self.bev.to_px(c.XY)[0]
            overlay = bev_img.copy()
            if c.visible:
                cv2.circle(overlay, (int(u), int(v)), rad, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.6, bev_img, 0.4, 0, bev_img)  # opaque enough to keep the hue on green
            cv2.circle(bev_img, (int(u), int(v)), rad, color, 2, cv2.LINE_AA)
            if is_sel:
                cv2.circle(bev_img, (int(u), int(v)), rad + 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(bev_img, (int(u), int(v)), 2, color, -1)
            if c.person_id is not None:
                cv2.putText(bev_img, f'ID {c.person_id}', (int(u) + rad + 4, int(v) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255) if is_sel else color, 1, cv2.LINE_AA)
            if not c.visible:
                cv2.putText(bev_img, 'hidden', (int(u)+rad+4, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (180, 180, 180), 1, cv2.LINE_AA)

            if is_sel:
                row, col, pos = self.cell_info(c.XY)
                info.append(f"selected  X={c.XY[0]:.3f} Y={c.XY[1]:.3f} m  cell=({row},{col}) posID={pos}"
                            f"{'  OUTSIDE ROI' if pos < 0 else ''}")
                for cam in self.cams:
                    if self.read_only:
                        box, trunc = self.person_box(cam, c, fl), False
                    else:
                        box, _, _, trunc = cylinder_box(cam, c.XY, fl.radius, fl.height)
                    if box is not None:
                        info.append(f"{cam.name} box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})"
                                    f"{'  TRUNCATED' if trunc else ''}")

        state_color = LOCK_COLOR if fl.locked else DRAFT_COLOR
        hud = [f"Frame {vframe} / {self.rec.frame_ids[-1]}"
               f"  {'PLAY' if self.playing else 'PAUSE'}",
               f"{'READ ONLY' if self.read_only else 'LOCKED' if fl.locked else 'DRAFT'}   locked {self.n_locked()}/{self.n_label}"
               f"   persons: {len(fl.cyls)}   cylinder H={fl.height:.2f} m R={fl.radius:.2f} m"] + info
        if not self.show_info:
            hud = []  # flash messages below still show; DRAFT / LOCKED stays visible as the border colour
        if time.time() < self.msg_until:
            hud.append(">> " + self.msg)
        if self.adding:
            hud.append(">> ADD: click the floor in the BEV or a person's feet in a camera (Esc: cancel)")
        cv2.rectangle(bev_img, (0, 0), (self.bev.w - 1, self.ph - 1), (0, 0, 255), 4)

        # Progress strip: red = locked, yellow = draft, white = current.
        y0, y1 = 28, 36
        for j in range(self.n_label):
            xa = 8 + int(j * (self.bev.w - 16) / self.n_label)
            xb = 8 + int((j + 1) * (self.bev.w - 16) / self.n_label)
            fj = self.frames.get(j)
            col = LOCK_COLOR if fj and fj.locked else DRAFT_COLOR if fj else (60, 60, 60)
            cv2.rectangle(bev_img, (xa, y0), (max(xa, xb - 1), y1), col, -1)
        xc = 8 + int((self.k + 0.5) * (self.bev.w - 16) / self.n_label)
        cv2.line(bev_img, (xc, y0 - 4), (xc, y1 + 4), (255, 255, 255), 2)

        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), np.uint8)
        panel_tiles = {}
        for v, tile in tiles.items():
            tile = tile.copy()
            th, tw = tile.shape[:2]
            cv2.rectangle(tile, (0, 0), (tw-1, th-1), state_color, 4)
            if self.view_mode is None and not self.overview.editing:
                cv2.putText(tile, f"{self.cams[v].name}  double-click to focus", (18, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 200, 0), 2, cv2.LINE_AA)
            panel_tiles[self.cams[v].name] = tile
        panel_tiles['BEV'] = bev_img
        if self.view_mode is None:
            self.overview.compose(canvas, panel_tiles)
        else:
            cam = self.cams[self.view_mode]
            w, h = cam.image_size
            canvas[:h, :w] = panel_tiles[cam.name]
            bx, by = self.bev_origin
            canvas[by:by+self.ph, bx:bx+self.bev.w] = bev_img
        if self.overview.editing:
            hud = ['LAYOUT EDIT: drag panels | yellow corner: resize | p / Esc: lock and save']
        if hud:
            tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0] for t in hud)
            bg = canvas[4:14+24*len(hud), 4:24+tw]
            bg[:] = (bg*.35).astype(np.uint8)
            for i, t in enumerate(hud):
                cv2.putText(canvas, t, (12, 26+24*i), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            state_color if self.show_info and i == 1 else (255, 255, 255),
                            1, cv2.LINE_AA)
        return canvas

    def draw_controls(self, height):
        fl = self.cur
        panel = np.full((max(height, self.controls_height), self.controls_width, 3), (30, 34, 39), np.uint8)
        cv2.putText(panel, "CONTROLS", (16, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (230, 230, 230), 1, cv2.LINE_AA)
        for title, (x1, y1, x2, y2) in self.control_groups:
            cv2.rectangle(panel, (x1, y1), (x2, y2), (39, 43, 49), -1)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (65, 70, 78), 1)
            cv2.putText(panel, title, (x1+8, y1+18), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, (180, 190, 200), 1, cv2.LINE_AA)
        for name, (bx1, by1, bx2, by2) in self.buttons:
            if name == "lock":
                fill = (40, 40, 160) if fl.locked else (40, 130, 40)
                text = "READ ONLY" if self.read_only else "LOCKED (u)" if fl.locked else "LOCK (Enter)"
                scale = 0.6
            elif name == "Add":
                fill = (200, 120, 0) if self.adding else (70, 70, 70) if fl.locked else (120, 90, 40)
                text = "Add (n)"
                scale = 0.6
            elif name == "Delete":
                can = not fl.locked and self.selected is not None
                fill = (50, 50, 170) if can else (70, 70, 70)
                text = "Delete (x)"
                scale = 0.6
            elif name == "Person ID":
                person = self.selected
                value = person.person_id if person is not None else None
                text = f'Person ID: {value if value is not None else "unassigned"} (r)'
                fill = (70, 70, 70) if fl.locked or person is None else (120, 90, 40)
                scale = .5
            elif name == "Person":
                on = self.selected is not None and self.selected.visible
                text = "Person: " + ("ON (v)" if on else "OFF (v)")
                fill, scale = ((70, 150, 70) if on else (70, 70, 70)), .6
            elif name == "Layout":
                text = "Layout: " + ("EDIT (p)" if self.overview.editing else "LOCK (p)")
                fill, scale = ((0, 130, 170) if self.overview.editing else (70, 70, 70)), .6
            elif name == "Reset layout":
                text, fill, scale = "Reset layout", (70, 70, 70), .6
            else:
                on = {"BBox": self.show_bbox, "Grid": self.show_grid, "Outline": self.show_outline,
                      "Info": self.show_info, "Site map": self.bev.show_site_map,
                      "Crops": self.show_crops}[name]
                fill = (70, 150, 70) if on else (70, 70, 70)
                text = f"{name}: {'ON' if on else 'OFF'}"
                scale = 0.6
            if bx2-bx1 < 200:
                scale = .47
            if self.overview.editing and name not in ('Layout', 'Reset layout'):
                fill = (55, 58, 63)
            cv2.rectangle(panel, (bx1, by1), (bx2, by2), fill, -1)
            on_edge = name == "Add" and self.adding
            cv2.rectangle(panel, (bx1, by1), (bx2, by2), (255, 255, 255) if on_edge else (220, 220, 220),
                          2 if on_edge else 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            tx = (bx1 + bx2 - tw) // 2 if name in ("lock", "Add", "Delete") else bx1 + 8
            cv2.putText(panel, text, (tx, (by1 + by2 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
        for kind, value, (x1, y1, x2, y2), text in self.frame_list_controls():
            active = kind == 'frame' and value == self.k
            if kind != 'title':
                fill = (80, 110, 55) if active else (55, 60, 68)
                cv2.rectangle(panel, (x1, y1), (x2, y2), fill, -1)
                cv2.rectangle(panel, (x1, y1), (x2, y2), (90, 95, 105), 1)
            cv2.putText(panel, text, (x1 + 6, y1 + (23 if kind == 'toggle' else 17)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5 if kind == 'toggle' else .4,
                        (230, 230, 230), 1, cv2.LINE_AA)
        x1, y1, x2, y2 = self.shortcuts_rect
        cv2.rectangle(panel, (x1, y1), (x2, y2), (55, 60, 68), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (90, 95, 105), 1)
        cv2.putText(panel, ('[-]' if self.shortcuts_open else '[+]') + ' Shortcuts',
                    (x1+10, y1+23), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        if not self.shortcuts_open:
            return panel
        y = self.shortcuts_top + 46
        for title, rows in self.shortcut_groups:
            cv2.line(panel, (16, y), (304, y), (65, 70, 78), 1)
            cv2.putText(panel, title, (20, y+19), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, (180, 190, 200), 1, cv2.LINE_AA)
            y += 30
            for key, action in rows:
                cv2.putText(panel, key, (20, y+14), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (225, 225, 225), 1, cv2.LINE_AA)
                cv2.putText(panel, action, (116, y+14), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (180, 185, 190), 1, cv2.LINE_AA)
                y += 22
        return panel

    def draw_display(self, size):
        w, h = size
        content_w, _ = self.content_size(size)
        crop_w = self.crop_width(size)
        controls_x = crop_w + content_w
        display = np.full((h, w, 3), (18, 18, 18), np.uint8)
        display[:, crop_w:controls_x] = self.viewport.draw(self.display_canvas, (content_w, h))
        if crop_w:
            display[:, :crop_w] = self.draw_crops(crop_w, h)
            if self.overview.editing:
                cv2.rectangle(display, (max(0, crop_w - 5), 0), (crop_w - 1, h - 1), (0, 190, 255), -1)
                cv2.putText(display, 'Drag edge to resize', (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (230, 230, 230), 1, cv2.LINE_AA)
        if self.controls_open:
            panel = self.draw_controls(h)
            self.controls_scroll = min(self.controls_scroll, max(0, panel.shape[0] - h))
            if self.controls_scroll and h > 64:
                panel[64:h] = panel[64 + self.controls_scroll:h + self.controls_scroll].copy()
            if h < self.controls_height:
                cv2.putText(panel, "Wheel to scroll", (16, 59), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (190, 190, 190), 1, cv2.LINE_AA)
            width = min(self.controls_width, w - controls_x)
            display[:, controls_x:controls_x + width] = panel[:h, :width]
        x = max(0, w - 168)
        cv2.rectangle(display, (x, 32), (w - 1, 62), (65, 75, 85), -1)
        cv2.putText(display, f"Crops [J]: {'ON' if self.show_crops else 'OFF'}", (x + 8, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(display, (x, 0), (w - 1, 30), (65, 75, 85), -1)
        text = "Controls [T] " + ("< close" if self.controls_open else "> open")
        cv2.putText(display, text, (x + 8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (255, 255, 255), 1, cv2.LINE_AA)
        self.draw_frame_picker(display)
        return display

    def on_display_mouse(self, event, x, y, flags, param):
        vp = self.viewport
        w, h = self.window_size or self.display_size()
        start, end = self.frame_picker_area((w, h))
        was_editing = self.frame_picker.editing
        handled, selected = self.frame_picker.mouse(event, x, y, flags, start, end, h, self.k)
        if handled or was_editing != self.frame_picker.editing:
            self.frame_picker_dirty = True
        if handled:
            self.playing = False
            if selected is not None:
                self.set_label_frame(selected)
            return
        if self.crop_resize is not None:
            if event == cv2.EVENT_MOUSEMOVE:
                start_x, start_width = self.crop_resize
                available = max(1, w - (self.controls_width if self.controls_open else 0))
                limit = max(1, available - min(200, available // 2))
                self.overview.crop_sidebar_width = int(np.clip(start_width + x - start_x, min(120, limit), limit))
                self.dirty = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.crop_resize = None
                self.save_layout()
                self.dirty = True
            return
        if event == cv2.EVENT_LBUTTONUP:
            if self.overview.drag is not None:
                self.save_layout()
            self.drag = self.camera_drag_origin = None
            self.overview.drag = None
            vp.pan_start = None
        if x >= max(0, w - 168) and y <= 30:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.toggle_controls()
            return
        if x >= max(0, w - 168) and 32 <= y <= 62:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.toggle_crops()
            return
        content_w, _ = self.content_size((w, h))
        crop_w = self.crop_width((w, h))
        controls_x = crop_w + content_w
        if self.overview.editing and crop_w and abs(x - crop_w) <= 7 and event == cv2.EVENT_LBUTTONDOWN:
            self.crop_resize = (x, crop_w)
            self.viewport.pan_start = None
            self.drag = self.camera_drag_origin = None
            return
        if self.controls_open and x >= controls_x:
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = (flags >> 16) & 0xffff
                delta = delta - 65536 if delta >= 32768 else delta
                if delta:
                    self.controls_scroll = int(np.clip(self.controls_scroll + (-44 if delta > 0 else 44),
                                                       0, max(0, self.controls_height - h)))
                    self.dirty = True
            elif y >= 64:
                self.controls_mouse(event, x - controls_x, y + self.controls_scroll)
            return
        if x < crop_w:
            return
        x -= crop_w
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xffff
            if delta >= 32768:
                delta -= 65536
            if delta:
                vp.zoom_at(x, y, 1.25 if delta > 0 else 0.8)
                self.dirty = True
            return
        if event == cv2.EVENT_MBUTTONDOWN:
            vp.pan_start = (np.array([x, y]), vp.center.copy())
            return
        if event == cv2.EVENT_MBUTTONUP:
            vp.pan_start = None
            return
        if event == cv2.EVENT_MOUSEMOVE and vp.pan_start is not None:
            start, center = vp.pan_start
            vp.center = center - (np.array([x, y]) - start) / vp.scale
            vp.transform()
            self.dirty = True
            return
        sx, sy = vp.source_point(x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag = None
            self.camera_drag_origin = None
            self.overview.drag = None
        if self.view_mode is None and self.overview.editing and self.overview.drag is not None:
            self.on_mouse(event, int(sx), int(sy), flags, param)
            return
        if 0 <= sx < self.canvas_w and 0 <= sy < self.canvas_h and y >= 28:
            self.on_mouse(event, int(sx), int(sy), flags, param)

    def display_size(self):
        try:
            _, _, w, h = cv2.getWindowImageRect(WIN)
            if w > 0 and h > 0:
                return w, h
        except cv2.error:
            pass
        return 1280, 720

    def open_window(self):
        fit_window(WIN, self.canvas_w, self.canvas_h)
        cv2.setMouseCallback(WIN, self.on_display_mouse)

    def run(self):
        self.open_window()
        print(__doc__)
        delay = max(1, int(1000 / self.fps_label))
        msg_shown = False
        while True:
            if self.playing:
                if self.k >= self.n_label - 1:
                    self.playing = False
                else:
                    self.set_label_frame(self.k + 1)
            if msg_shown and time.time() >= self.msg_until:
                self.dirty = True  # clear the flash message
            size = self.display_size()
            resized = size != self.window_size
            self.window_size = size
            if self.dirty or resized:
                self.prepare_layout_workspace(self.content_size(size))
            if self.dirty or resized or self.frame_picker_dirty:
                msg_shown = time.time() < self.msg_until
                if self.dirty or self.display_canvas is None:
                    self.display_canvas = self.render()
                cv2.imshow(WIN, self.draw_display(size))
                self.dirty = False
                self.frame_picker_dirty = False
            if not self.process_pending_keys(cv2.waitKeyEx(delay if self.playing else 15)):
                break
            try:
                window_closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                window_closed = True
            if window_closed:
                if self.confirm_exit():
                    break
                cv2.destroyAllWindows()
                self.open_window()
                self.dirty = True
        self.save_layout()
        cv2.destroyAllWindows()
        self.reader.release()
        drafts = len(self.frames) - self.n_locked()
        if drafts:
            print(f"Note: {drafts} draft frame(s) were not locked and are not saved.")
        print(f"Locked {self.n_locked()}/{self.n_label} label frames in {self.labels_dir}")


if __name__ == "__main__":
    args = parse_args()
    paths = choose_startup_paths(args)
    if paths is not None:
        args.source = str(paths['source'])
        args.calib = str(paths['calib'])
        args.labels_dir = str(paths['annotations_dir'])
        WaveTool(args).run()
