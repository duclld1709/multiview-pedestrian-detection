"""Approved minimal five-camera video layout, with a calibrated site map."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cylinders import draw_cylinder_3d, draw_person_box


class SiteMap:
    BASE_SIZE = (1920, 1080)
    CARDS = {1: (8, 109, 752, 423), 2: (8, 548, 752, 423),
             4: (1288, 5, 624, 351), 5: (1288, 364, 624, 351), 3: (1288, 723, 624, 351)}
    BACKDROP = (22, 20, 18)
    PREVIEW_PADDING = 12
    ROUTE_EDGE = (55, 49, 41)
    ROUTE_ACTIVE = (219, 162, 78)
    ROUTE_INACTIVE = (150, 141, 134)

    def __init__(self, cams, roi, pitch, site_map, background_fill=True, *, wavetrack=True):
        self.cams, self.roi, self.pitch = list(cams), roi, pitch
        self.background_fill = background_fill
        self.numbers = [int(c.name[3:]) for c in self.cams]
        self.site = site_map if isinstance(site_map, dict) else json.loads(Path(site_map).read_text(encoding='utf-8'))
        self.wavetrack = wavetrack
        self.bounds = np.asarray(self.site['bounds'], dtype=float)
        if (self.bounds.shape != (4,) or not np.isfinite(self.bounds).all()
                or self.bounds[1] <= self.bounds[0] or self.bounds[3] <= self.bounds[2]):
            raise ValueError('Invalid site map bounds')
        if self.site.get('flip_horizontal') or self.site.get('flip_vertical'):
            raise ValueError('The approved site layout requires both map flips disabled')
        self.mx, self.my, self.extent = 768, 284, 512
        self.scale = self.extent / max(self.bounds[1]-self.bounds[0], self.bounds[3]-self.bounds[2])
        self.routes = {number: self._route(cam, number) for cam, number in zip(self.cams, self.numbers)
                       if wavetrack and number in self.CARDS}

    @staticmethod
    def _font(size):
        try:
            return ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf', size)
        except OSError:
            return ImageFont.load_default(size=size)

    def to_px(self, points):
        p = np.atleast_2d(points).astype(float)
        x0, _, _, y1 = self.bounds
        return np.column_stack([(p[:, 0]-x0)*self.scale+self.mx, (y1-p[:, 1])*self.scale+self.my])

    def _px(self, points):
        return np.rint(self.to_px(points)).astype(np.int32)

    def _route(self, cam, number):
        center = self.to_px(cam.center[:2])[0]
        x, _, width, _ = self.CARDS[number]
        edge_x = x + width if number in (1, 2) else x
        if number in (3, 4):
            bend_y = {3: 788, 4: 320}[number]
            points = [center, (center[0], bend_y), (edge_x, bend_y)]
        else:
            points = [center, (edge_x, center[1])]
        return np.rint(points).astype(np.int32)

    def _camera_markers(self):
        locations = self.to_px([cam.center[:2] for cam in self.cams])
        markers = locations.copy()
        if not self.wavetrack:
            groups = []
            for index, xy in enumerate(locations):
                nearby = [group for group in groups if np.any(np.linalg.norm(locations[group] - xy, axis=1) < 24)]
                group = [index]
                for members in nearby:
                    groups.remove(members)
                    group.extend(members)
                groups.append(sorted(group))
            for group in groups:
                if len(group) < 2:
                    continue
                step = 52
                width = step * (len(group) - 1)
                x = np.clip(locations[group, 0].mean() - width / 2, self.mx + 36, self.mx + self.extent - width - 36)
                markers[group, 0] = x + step * np.arange(len(group))
                markers[group, 1] = locations[group, 1].mean()
        return locations, markers

    def _background(self, show_roi=True, show_routes=True, active=None):
        active = self.numbers if active is None else active
        image = np.full((*self.BASE_SIZE[::-1], 3), self.BACKDROP, np.uint8)
        mx, my, extent = self.mx, self.my, self.extent
        roi, pitch = self.roi, self.pitch
        x0, x1, y0, y1 = self.bounds
        image[my:my+extent, mx:mx+extent] = (232, 231, 227)
        floor_mask = np.zeros(image.shape[:2], np.uint8)
        for floor in self.site.get('floor_regions', []):
            points = self._px(floor['points'])
            cv2.fillPoly(image, [points], (250, 250, 249), cv2.LINE_AA)
            cv2.fillPoly(floor_mask, [points], 255)
        grid = image.copy()
        for x in np.arange(np.ceil(x0/pitch[0])*pitch[0], x1+1e-6, pitch[0]):
            a, b = self._px([[x, y0], [x, y1]])
            cv2.line(grid, tuple(a), tuple(b), (222, 226, 224), 1, cv2.LINE_AA)
        for y in np.arange(np.ceil(y0/pitch[1])*pitch[1], y1+1e-6, pitch[1]):
            a, b = self._px([[x0, y], [x1, y]])
            cv2.line(grid, tuple(a), tuple(b), (222, 226, 224), 1, cv2.LINE_AA)
        image[floor_mask > 0] = grid[floor_mask > 0]
        if show_roi:
            polygon = self._px([[roi.x0, roi.y0], [roi.x1, roi.y0], [roi.x1, roi.y1], [roi.x0, roi.y1]])
            if self.background_fill:
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                overlay = image.copy()
                cv2.fillPoly(overlay, [polygon], (225, 207, 242))
                blended = cv2.addWeighted(overlay, .13, image, .87, 0)
                image[mask > 0] = blended[mask > 0]
            cv2.polylines(image, [polygon], True, (166, 122, 140), 2, cv2.LINE_AA)
        for obj in self.site['objects']:
            if obj['kind'] == 'pillar':
                cv2.circle(image, tuple(self._px(obj['center'])[0]), max(2, round(obj['radius']*self.scale)),
                           (49, 47, 43), -1, cv2.LINE_AA)
                continue
            points = self._px(obj['points'])
            color = (75, 133, 171) if obj.get('role') == 'door' else (63, 59, 52)
            if obj.get('role') == 'door' and obj.get('occludes') is False:
                for p, q in zip(points[:-1], points[1:]):
                    length = np.linalg.norm(q-p)
                    if not length:
                        continue
                    for start in np.arange(0, length, 11):
                        a = p+(q-p)*start/length
                        b = p+(q-p)*min(start+6, length)/length
                        cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), color, 2, cv2.LINE_AA)
            else:
                cv2.polylines(image, [points], False, color, max(3, round(obj['width']*self.scale)), cv2.LINE_AA)
        if show_routes:
            for number, route in self.routes.items():
                cv2.polylines(image, [route], False, self.ROUTE_EDGE, 4, cv2.LINE_AA)
                line = self.ROUTE_ACTIVE if number in active else self.ROUTE_INACTIVE
                cv2.polylines(image, [route], False, line, 2, cv2.LINE_AA)
        locations, markers = self._camera_markers()
        for location, marker in zip(locations, markers):
            if not np.array_equal(location, marker):
                a, b = [tuple(point) for point in np.rint([location, marker]).astype(int)]
                cv2.line(image, a, b, (138, 143, 137), 1, cv2.LINE_AA)
                cv2.circle(image, a, 2, (73, 83, 83), -1, cv2.LINE_AA)
        labels = []
        for cam, number, location, center in zip(self.cams, self.numbers, locations, markers):
            forward = self.to_px(cam.center[:2] + cam.forward[:2])[0] - location
            norm = np.linalg.norm(forward)
            forward = forward / norm if norm > 1e-9 else np.array([0., -1.])
            side = np.array([-forward[1], forward[0]])
            body = .72*np.array([[-15, -9], [4, -9], [7, -5], [14, -8], [14, 8], [7, 5], [4, 9], [-15, 9]])
            polygon = np.rint(center+body[:, 0, None]*forward+body[:, 1, None]*side).astype(np.int32)
            edges = [(0, cam.K[1, 2]), (cam.image_size[0]-1, cam.K[1, 2])]
            if np.any(cam.dist):
                edges = cam.undistort(edges)
            for u, v in edges:
                ray = cam.depth_sign * (cam.R.T @ np.linalg.solve(cam.K, [u, v, 1.]))
                direction = self.to_px(cam.center[:2] + ray[:2])[0] - location
                direction /= max(np.linalg.norm(direction), 1e-9)
                a, b = center+forward*10, center+direction*34
                cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), (164, 169, 164), 1, cv2.LINE_AA)
            body_color = (237, 178, 29) if number in active else self.ROUTE_INACTIVE
            cv2.fillPoly(image, [polygon], body_color, cv2.LINE_AA)
            cv2.polylines(image, [polygon], True, (53, 55, 53), 1, cv2.LINE_AA)
            if self.wavetrack:
                offset = {3: [-50, -7], 4: [-43, -8], 5: [-43, -8]}.get(number, [11, -20])
            else:
                a, b = (tuple(np.rint(center + forward * length).astype(int)) for length in (12, 29))
                color = (132, 96, 53) if number in active else (133, 134, 130)
                cv2.arrowedLine(image, a, b, color, 1, cv2.LINE_AA, tipLength=.35)
                direction = center[0] - location[0]
                place_left = direction < 0 or (direction == 0 and forward[0] > .35)
                offset = [-33, -17] if place_left else [12, -17]
            labels.append((number, tuple(np.rint(center+offset).astype(int))))
        return self._labels(image, labels, active)

    def _labels(self, image, labels, active):
        result = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(result)
        font = self._font(14)
        for number, xy in labels:
            label = f'C{number}'
            if not self.wavetrack or number in (3, 4, 5):
                a, b, c, d = draw.textbbox(xy, label, font=font)
                draw.rounded_rectangle((a-4, b-3, c+4, d+3), radius=3, fill=(250, 250, 249))
            fill = (45, 78, 98) if number in active else (103, 110, 114)
            draw.text(xy, label, font=font, fill=fill)
        return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

    def _endpoints(self, image, active):
        for number, route in self.routes.items():
            endpoint = tuple(route[-1])
            cv2.circle(image, endpoint, 4, self.ROUTE_EDGE, -1, cv2.LINE_AA)
            color = self.ROUTE_ACTIVE if number in active else self.ROUTE_INACTIVE
            cv2.circle(image, endpoint, 2, color, -1, cv2.LINE_AA)

    def preview(self, show_roi, show_routes, active):
        image = self._background(show_roi, show_routes, active)
        if show_routes:
            self._endpoints(image, active)
        pad = self.PREVIEW_PADDING
        crop = image[self.my-pad:self.my+self.extent+pad, self.mx-pad:self.mx+self.extent+pad]
        preview = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        preview[:, :, 3][np.all(crop == self.BACKDROP, axis=2)] = 0
        return preview


class SiteMapLayout(SiteMap):
    def __init__(self, cams, roi, pitch, site_map, size=(1920, 1080), background_fill=True):
        super().__init__(cams, roi, pitch, site_map, background_fill)
        self.size = tuple(size)
        if len(self.cams) != 5 or set(self.numbers) != {1, 2, 3, 4, 5}:
            raise ValueError('This site layout requires exactly Cam1 through Cam5')
        if len(self.size) != 2 or min(self.size) < 2:
            raise ValueError('Output canvas must have positive width and height')
        self.background = self._background()
        self.tags = {k: self._tag(f'C{k}') for k in self.numbers}

    @classmethod
    def _tag(cls, label):
        image = Image.new('RGB', (41, 29), (26, 30, 33))
        ImageDraw.Draw(image).text((9, 3), label, font=cls._font(17), fill=(233, 239, 242))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def compose(self, views, people, colors, radius_m=.25, height_m=1.8, person_style='cylinder'):
        if len(views) != len(self.cams) or len(people) != len(colors):
            raise ValueError('Images, cameras and detection colours must align')
        if person_style not in ('cylinder', 'bbox'):
            raise ValueError('Person style must be cylinder or bbox')
        draw_person = draw_person_box if person_style == 'bbox' else draw_cylinder_3d
        result = self.background.copy()
        for cam, number, view in zip(self.cams, self.numbers, views):
            image = view.copy()
            order = sorted(range(len(people)), key=lambda i: -np.linalg.norm(np.asarray(people[i][:2])-cam.center[:2]))
            for i in order:
                draw_person(image, cam, people[i][:2], radius_m, height_m, colors[i])
            x, y, w, h = self.CARDS[number]
            result[y:y+h, x:x+w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            result[y+8:y+37, x+8:x+49] = self.tags[number]
        self._endpoints(result, self.numbers)
        for person, color in zip(people, colors):
            point = tuple(self._px(person[:2])[0])
            # Clip outside-map points to avoid drawing on camera views or other panels.
            if not (self.mx <= point[0] < self.mx+self.extent and self.my <= point[1] < self.my+self.extent):
                continue
            cv2.circle(result, point, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(result, point, 6, color, -1, cv2.LINE_AA)
        if self.size == self.BASE_SIZE:
            return result
        factor = min(self.size[0]/1920, self.size[1]/1080)
        width, height = max(1, round(1920*factor)), max(1, round(1080*factor))
        output = np.full((self.size[1], self.size[0], 3), self.BACKDROP, np.uint8)
        x, y = (self.size[0]-width)//2, (self.size[1]-height)//2
        output[y:y+height, x:x+width] = cv2.resize(result, (width, height), interpolation=cv2.INTER_AREA)
        return output

