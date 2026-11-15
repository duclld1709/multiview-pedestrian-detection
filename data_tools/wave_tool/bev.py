"""BEV and floor-grid helpers for the OpenCV Wave Tool."""
import json
from pathlib import Path

import cv2
import numpy as np

def load_site_map(path, roi):
    """Read only drawing geometry; map orientation never changes the label axes."""
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        bounds = np.asarray(value['bounds'], dtype=float)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or not (bounds[0] <= roi.x0 < roi.x1 <= bounds[1]
                        and bounds[2] <= roi.y0 < roi.y1 <= bounds[3])
                or max(bounds[1]-bounds[0], bounds[3]-bounds[2]) > 100):
            raise ValueError('map bounds must contain the label ROI and span at most 100 m')
        def points(raw, minimum):
            p = np.asarray(raw, dtype=float)
            if (p.ndim != 2 or p.shape[1] != 2 or not minimum <= len(p) <= 100
                    or not np.isfinite(p).all()
                    or (p < bounds[[0, 2]]).any() or (p > bounds[[1, 3]]).any()):
                raise ValueError('invalid map points')
            return p.tolist()
        objects = value.get('objects', [])
        floors = value.get('floor_regions', [])
        if len(objects) > 300 or len(floors) > 30:
            raise ValueError('too many map objects')
        for item in objects:
            if item['kind'] == 'wall':
                item['points'] = points(item['points'], 2)
                size, limit = float(item['width']), 2
                item['width'] = size
            elif item['kind'] == 'pillar':
                item['center'] = points([item['center']], 1)[0]
                size, limit = float(item['radius']), 5
                item['radius'] = size
            else:
                raise ValueError('unknown map object')
            if not np.isfinite(size) or not 0.02 <= size <= limit:
                raise ValueError('invalid map object size')
        for region in floors:
            region['points'] = points(region['points'], 3)
        value['bounds'] = bounds.tolist()
        return value
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        print(f'Site map unavailable; using classic BEV: {exc}')
        return None


class BevCanvas:
    def __init__(self, cams, roi, pitch, height=1080, site_map_path=None):
        self.h = height
        self.site_map = load_site_map(site_map_path, roi) if site_map_path else None
        self.show_site_map = self.site_map is not None
        self.bounds = self.site_map['bounds'] if self.site_map else [roi.x0, roi.x1, roi.y0, roi.y1]
        x0, x1, y0, y1 = self.bounds
        self.scale = (height-100)/(y1-y0)
        self.w = max(500, int((x1-x0)*self.scale)+100)
        self.roi = roi
        self.origin = np.array([40, height-35])
        self._classic = np.full((self.h,self.w,3), (38,57,38), np.uint8)
        self._site = None
        for x in np.arange(roi.x0,roi.x1+1e-6,pitch[0]):
            self.line((x,roi.y0),(x,roi.y1),(65,90,65))
        for y in np.arange(roi.y0,roi.y1+1e-6,pitch[1]):
            self.line((roi.x0,y),(roi.x1,y),(65,90,65))
        a,b=self.to_px([[roi.x0,roi.y1],[roi.x1,roi.y0]]).astype(int)
        cv2.rectangle(self.background,tuple(a),tuple(b),(0,0,255),2)
        cv2.putText(self.background,'BEV | X ->  Y up',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1,cv2.LINE_AA)
        if self.site_map:
            self._site = self._draw_site(cams)

    @property
    def background(self):
        return self._site if self.show_site_map and self._site is not None else self._classic

    def toggle_site_map(self):
        if self.site_map is not None:
            self.show_site_map = not self.show_site_map

    def _draw_site(self, cams):
        image = np.full((self.h, self.w, 3), (238, 236, 231), np.uint8)
        x0, x1, y0, y1 = self.bounds
        def px(points):
            return np.rint(self.to_px(points)).astype(np.int32)
        mask = np.zeros((self.h, self.w), np.uint8)
        floors = self.site_map.get('floor_regions', [])
        if floors:
            for floor in floors:
                cv2.fillPoly(mask, [px(floor['points'])], 255)
        else:
            a, b = px([[x0, y1], [x1, y0]])
            cv2.rectangle(mask, tuple(a), tuple(b), 255, -1)
        image[mask > 0] = (255, 255, 255)
        grid = image.copy()
        grid_step = 0.6  # metres; display only, independent of annotation snap
        for x in np.arange(np.ceil(x0/grid_step)*grid_step, x1+1e-6, grid_step):
            p, q = px([[x, y0], [x, y1]])
            cv2.line(grid, tuple(p), tuple(q), (225, 228, 226), 1, cv2.LINE_AA)
        for y in np.arange(np.ceil(y0/grid_step)*grid_step, y1+1e-6, grid_step):
            p, q = px([[x0, y], [x1, y]])
            cv2.line(grid, tuple(p), tuple(q), (225, 228, 226), 1, cv2.LINE_AA)
        image[mask > 0] = grid[mask > 0]
        a, b = px([[self.roi.x0, self.roi.y1], [self.roi.x1, self.roi.y0]])
        overlay = image.copy()
        cv2.rectangle(overlay, tuple(a), tuple(b), (100, 90, 235), -1)
        cv2.addWeighted(overlay, .06, image, .94, 0, image)
        cv2.rectangle(image, tuple(a), tuple(b), (65, 49, 197), 2, cv2.LINE_AA)
        cv2.putText(image, f'ROI MODEL {self.roi.x1-self.roi.x0:.2f} x {self.roi.y1-self.roi.y0:.2f} m',
                    tuple(a+[7, 20]), cv2.FONT_HERSHEY_SIMPLEX, .42, (65, 49, 197), 1, cv2.LINE_AA)
        for item in self.site_map.get('objects', []):
            if item['kind'] == 'wall':
                color = (36, 97, 150) if item.get('role') == 'door' else (30, 28, 23)
                if item.get('occludes') is False:
                    for p, q in zip(px(item['points'])[:-1], px(item['points'])[1:]):
                        length = np.linalg.norm(q-p)
                        for start in np.arange(0, length, 15):
                            u = p+(q-p)*start/length
                            v = p+(q-p)*min(start+9, length)/length
                            cv2.line(image, tuple(u.astype(int)), tuple(v.astype(int)), color, 2, cv2.LINE_AA)
                else:
                    cv2.polylines(image, [px(item['points'])], False, color,
                                  max(3, round(float(item['width'])*self.scale)), cv2.LINE_AA)
            else:
                center = tuple(px([item['center']])[0])
                radius = max(2, round(float(item['radius'])*self.scale))
                cv2.circle(image, center, radius, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(image, center, radius, (0, 0, 0), 2, cv2.LINE_AA)
        colors = [(160, 135, 15), (35, 135, 240), (80, 155, 30), (70, 60, 210), (180, 110, 150)]
        for i, cam in enumerate(cams):
            center = np.asarray(cam.center)[:2]
            if not (x0 <= center[0] <= x1 and y0 <= center[1] <= y1):
                continue
            p = self.to_px(center)[0]
            direction = np.asarray(cam.R)[2, :2]*[1, -1]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction = direction/norm
            side = np.array([-direction[1], direction[0]])
            triangle = np.rint([p+direction*14, p-direction*9+side*8,
                               p-direction*9-side*8]).astype(np.int32)
            color = colors[i % len(colors)]
            cv2.fillPoly(image, [triangle], color, cv2.LINE_AA)
            cv2.polylines(image, [triangle], True, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(image, f'C{cam.number}', tuple(np.rint(p+[12, -9]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        a, b = px([[x0, y1], [x1, y0]])
        cv2.rectangle(image, tuple(a), tuple(b), (170, 180, 180), 1)
        for x in np.arange(np.ceil(x0/2)*2, x1+1e-6, 2):
            p = px([[x, y1]])[0]
            cv2.putText(image, f'x {x:g}', tuple(p+[2, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                        .36, (95, 110, 115), 1, cv2.LINE_AA)
        for y in np.arange(np.ceil(y0/2)*2, y1+1e-6, 2):
            p = px([[x1, y]])[0]
            cv2.putText(image, f'{y:g}', tuple(p+[5, 4]), cv2.FONT_HERSHEY_SIMPLEX,
                        .36, (95, 110, 115), 1, cv2.LINE_AA)
        cv2.putText(image, 'Site map | X ->  Y up | grid 60 cm', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (70, 80, 80), 1, cv2.LINE_AA)
        return image

    def to_px(self, xy):
        return (np.atleast_2d(xy)-[self.bounds[0],self.bounds[2]])*[self.scale,-self.scale]+self.origin

    def from_px(self,x,y):
        return (np.array([x,y])-self.origin)/[self.scale,-self.scale]+[self.bounds[0],self.bounds[2]]

    def line(self,a,b,color):
        p,q=self.to_px([a,b]).astype(int)
        cv2.line(self.background,tuple(p),tuple(q),color,1,cv2.LINE_AA)


def draw_floor_grid(image,cam,roi,pitch):
    lines = [((x,roi.y0,0),(x,roi.y1,0)) for x in np.arange(roi.x0,roi.x1+1e-6,pitch[0])]
    lines += [((roi.x0,y,0),(roi.x1,y,0)) for y in np.arange(roi.y0,roi.y1+1e-6,pitch[1])]
    # Draw all four exact ROI edges last, even when pitch does not divide the ROI.
    corners = [(roi.x0,roi.y0,0), (roi.x1,roi.y0,0),
               (roi.x1,roi.y1,0), (roi.x0,roi.y1,0)]
    colored_lines = [(a,b,(150,150,50)) for a,b in lines]
    colored_lines += [(corners[i],corners[(i+1)%4],(0,0,255)) for i in range(4)]
    h,w=image.shape[:2]
    for a,b,color in colored_lines:
        uv,front=cam.project(np.linspace(a,b,80))
        for i in range(len(uv)-1):
            if front[i:i+2].all() and np.isfinite(uv[i:i+2]).all() and np.abs(uv[i:i+2]).max()<1e7:
                p,q=uv[i:i+2].astype(int)
                ok,p,q=cv2.clipLine((0,0,w,h),tuple(p),tuple(q))
                if ok: cv2.line(image,p,q,color,1,cv2.LINE_AA)


def fit_window(name,width,height):
    try:
        import ctypes
        # Prevent Windows DPI virtualization from blurring the whole GUI.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError,OSError):
            ctypes.windll.user32.SetProcessDPIAware()
        user32=ctypes.windll.user32
        sw,sh=user32.GetSystemMetrics(0),user32.GetSystemMetrics(1)
    except (AttributeError,OSError):
        sw,sh=1600,1000
    scale=min((sw-80)/width,(sh-150)/height,1)
    cv2.namedWindow(name,cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(name,int(width*scale),int(height*scale))
