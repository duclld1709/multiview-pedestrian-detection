"""Movable overview panels with proportional resizing in an expandable workspace."""
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import atomic_json


class OverviewLayout:
    def __init__(self, sizes, rects, canvas_size):
        self.sizes = sizes
        self.defaults = dict(rects)
        self.rects = dict(rects)
        self.order = list(rects)
        self.canvas_size = canvas_size
        self.default_canvas_size = canvas_size
        self.editing = False
        self.drag = None
        self.crop_sidebar_width = None

    def hit(self, x, y):
        for key in reversed(self.order):
            px, py, w, h = self.rects[key]
            if px <= x < px + w and py <= y < py + h:
                return key
        return None

    def local(self, key, x, y):
        px, py, w, h = self.rects[key]
        sw, sh = self.sizes[key]
        return (x-px)*sw/w, (y-py)*sh/h

    def mouse(self, event, x, y, handle=40):
        if event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            key = self.hit(x, y)
            if key is not None:
                rect = self.rects[key]
                px, py, w, h = rect
                resize = x >= px+w-handle and y >= py+h-handle
                self.drag = key, x, y, rect, resize
                self.order.remove(key)
                self.order.append(key)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            key, sx, sy, (px, py, w, h), resize = self.drag
            cw, ch = self.canvas_size
            if resize:
                sw, sh = self.sizes[key]
                # Project mouse motion onto the panel diagonal; one shared scale.
                factor = 1 + ((x-sx)*w + (y-sy)*h)/(w*w+h*h)
                scale = np.clip(factor*w/sw, min(120/sw, 120/sh),
                                min((cw-px)/sw, (ch-py)/sh))
                self.rects[key] = (px, py, max(1, round(sw*scale)), max(1, round(sh*scale)))
            else:
                self.rects[key] = (int(np.clip(px+x-sx, 0, cw-w)),
                                   int(np.clip(py+y-sy, 0, ch-h)), w, h)

    def save(self, path):
        atomic_json(path, {'rects': self.rects, 'order': self.order, 'canvas_size': self.canvas_size,
                           'crop_sidebar_width': self.crop_sidebar_width})

    def expand_to_window(self, size):
        """Turn fit-to-window padding into usable space without shrinking any panel."""
        ww, wh = size
        cw, ch = self.canvas_size
        scale = min(ww/cw, wh/ch)
        nw, nh = max(cw, round(ww/scale)), max(ch, round(wh/scale))
        if (nw, nh) == (cw, ch):
            return False
        dx, dy = (nw-cw)//2, (nh-ch)//2
        self.rects = {key: (x+dx, y+dy, w, h) for key, (x, y, w, h) in self.rects.items()}
        self.canvas_size = nw, nh
        self.drag = None
        return True

    def load(self, path):
        if not Path(path).exists():
            return
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            rects = data['rects']
            order = data['order']
            if set(rects) != set(self.sizes) or set(order) != set(self.sizes) or len(order) != len(self.sizes):
                raise ValueError('Panel set differs')
            canvas_size = tuple(data.get('canvas_size', self.default_canvas_size))
            if len(canvas_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in canvas_size):
                raise ValueError('Invalid workspace size')
            cw, ch = canvas_size
            for key, rect in rects.items():
                x, y, w, h = rect
                sw, sh = self.sizes[key]
                if (not all(isinstance(v, int) for v in rect) or min(x, y) < 0
                        or min(w, h) < 1 or x+w > cw or y+h > ch
                        or abs(w/sw-h/sh) > 1/min(sw, sh)):
                    raise ValueError('Invalid panel rectangle')
            self.rects = {key: tuple(rect) for key, rect in rects.items()}
            self.order = order
            self.canvas_size = canvas_size
            width = data.get('crop_sidebar_width')
            if isinstance(width, int) and not isinstance(width, bool) and 120 <= width <= 10000:
                self.crop_sidebar_width = width
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f'Ignoring saved overview layout: {exc}')

    def compose(self, canvas, tiles):
        for key in self.order:
            x, y, w, h = self.rects[key]
            tile = tiles[key]
            if tile.shape[1::-1] != (w, h):
                tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_AREA if w < tile.shape[1] else cv2.INTER_LINEAR)
            canvas[y:y+h, x:x+w] = tile
            if self.editing:
                cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), (0, 220, 255), 3)
                cv2.rectangle(canvas, (x+w-40, y+h-40), (x+w-1, y+h-1), (0, 220, 255), -1)
                cv2.putText(canvas, f'{key}: drag | corner: resize', (x+10, y+28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2, cv2.LINE_AA)
