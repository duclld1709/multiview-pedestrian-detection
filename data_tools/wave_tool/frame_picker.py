"""Inline frame number entry and a scrollable list of available image frames."""
from bisect import bisect_left

import cv2
import numpy as np


class FramePicker:
    def __init__(self, frames):
        self.frames = frames
        self.indices = {frame: i for i, frame in enumerate(frames)}
        self.editing = self.open = self.select_all = self.scroll_drag = False
        self.text = ''
        self.error = False
        self.scroll = self.highlight = 0
        self.rows = 9

    def layout(self, start, end, height):
        width = max(1, end - start)
        field_left = start + min(60, width // 5)
        field_right = min(end - 28, field_left + 100)
        self.rows = max(1, min(9, (height - 66) // 28))
        return field_left, field_right, min(end, start + 280), 64 + self.rows * 28

    def begin(self, k, dropdown=False):
        self.editing = True
        self.text = str(self.frames[k])
        self.select_all = True
        self.error = False
        self.open = dropdown
        self.highlight = k
        self.scroll = max(0, min(k - self.rows // 2, len(self.frames) - self.rows))

    def cancel(self):
        self.editing = self.open = self.scroll_drag = self.error = False

    def key(self, key):
        """Return (handled, selected image index); invalid input never changes frames."""
        if not self.editing:
            return False, None
        if key < 0:
            return True, None
        if key == 27:
            self.cancel()
        elif key in (10, 13):
            index = self.indices.get(int(self.text)) if self.text else None
            if index is None:
                self.error = True
            else:
                self.cancel()
                return True, index
        elif key in (8, 127, 3014656):
            self.text = '' if self.select_all or key == 3014656 else self.text[:-1]
            self.select_all = self.error = False
        elif key == 1:  # Ctrl+A
            self.select_all = True
        elif ord('0') <= key <= ord('9'):
            self.text = chr(key) if self.select_all else (self.text + chr(key))[:12]
            self.select_all = self.error = False
            self.highlight = min(len(self.frames) - 1, bisect_left(self.frames, int(self.text)))
            self.scroll = max(0, min(self.highlight - self.rows // 2, len(self.frames) - self.rows))
        elif key in (2424832, 2490368, 2555904, 2621440, 2162688, 2228224):
            delta = {2424832: -1, 2490368: -1, 2555904: 1, 2621440: 1,
                     2162688: -self.rows, 2228224: self.rows}[key]
            self.open = True
            self.highlight = int(np.clip(self.highlight + delta, 0, len(self.frames) - 1))
            self.text = str(self.frames[self.highlight])
            self.select_all = True
            self.error = False
            self.scroll = max(0, min(self.highlight - self.rows // 2, len(self.frames) - self.rows))
        return True, None

    def mouse(self, event, x, y, flags, start, end, height, k):
        field_left, field_right, list_right, list_bottom = self.layout(start, end, height)
        if self.scroll_drag:
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                fraction = np.clip((y - 64) / max(1, list_bottom - 64), 0., 1.)
                self.scroll = int(round(fraction * max(0, len(self.frames) - self.rows)))
            if event == cv2.EVENT_LBUTTONUP:
                self.scroll_drag = False
            return True, None
        if self.open and start <= x < list_right and 64 <= y < list_bottom:
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = (flags >> 16) & 0xffff
                delta = delta - 65536 if delta >= 32768 else delta
                self.scroll = int(np.clip(self.scroll - int(delta / 120) * 5,
                                          0, max(0, len(self.frames) - self.rows)))
            elif event == cv2.EVENT_LBUTTONDOWN:
                if x >= list_right - 14:
                    self.scroll_drag = True
                    return self.mouse(cv2.EVENT_MOUSEMOVE, x, y, flags, start, end, height, k)
                index = self.scroll + (y - 64) // 28
                if index < len(self.frames):
                    self.cancel()
                    return True, index
            return True, None
        if start <= x < end and 32 <= y <= 62:
            if event == cv2.EVENT_LBUTTONDOWN:
                if field_left <= x < field_right:
                    self.begin(k)
                elif field_right <= x < field_right + 28:
                    if self.open:
                        self.cancel()
                    else:
                        self.begin(k, dropdown=True)
            elif event == cv2.EVENT_MOUSEWHEEL:
                if not self.open:
                    self.begin(k, dropdown=True)
                delta = (flags >> 16) & 0xffff
                delta = delta - 65536 if delta >= 32768 else delta
                self.scroll = int(np.clip(self.scroll - int(delta / 120) * 5,
                                          0, max(0, len(self.frames) - self.rows)))
            return True, None
        if event == cv2.EVENT_LBUTTONDOWN and self.editing:
            self.cancel()
        return False, None

    def draw(self, display, start, end, k):
        height = display.shape[0]
        field_left, field_right, list_right, list_bottom = self.layout(start, end, height)
        strip = display[32:63, start:end]
        strip[:] = (30, 30, 30)
        def text(target, value, x, y, color=(235, 235, 235), scale=.48):
            cv2.putText(target, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        text(strip, 'Frame', 8, 21)
        left, right = field_left - start, field_right - start
        cv2.rectangle(strip, (left, 2), (right, 28), (65, 65, 65), -1)
        if self.editing and self.select_all:
            cv2.rectangle(strip, (left + 2, 4), (right - 2, 26), (140, 85, 35), -1)
        border = (60, 60, 240) if self.error else (0, 200, 255) if self.editing else (110, 110, 110)
        cv2.rectangle(strip, (left, 2), (right, 28), border, 1)
        value = self.text if self.editing else str(self.frames[k])
        text(strip, value + ('|' if self.editing and not self.select_all else ''), left + 7, 21)
        cv2.rectangle(strip, (right + 1, 2), (right + 27, 28), (75, 75, 75), -1)
        text(strip, '^' if self.open else 'v', right + 8, 21)
        text(strip, f'/ {self.frames[-1]}', right + 36, 21)
        if self.error and end - field_right > 330:
            text(strip, 'Frame unavailable', right + 135, 21, (100, 120, 255), .42)
        if not self.open:
            return
        popup = display[64:list_bottom, start:list_right]
        popup[:] = (38, 38, 38)
        for row in range(self.rows):
            index = self.scroll + row
            if index >= len(self.frames):
                break
            y = row * 28
            if index == self.highlight:
                cv2.rectangle(popup, (0, y), (popup.shape[1] - 15, y + 27), (80, 65, 35), -1)
            text(popup, str(self.frames[index]), 12, y + 19,
                 (0, 210, 255) if index == k else (235, 235, 235))
        x = popup.shape[1] - 12
        cv2.rectangle(popup, (x, 0), (popup.shape[1] - 1, popup.shape[0] - 1), (65, 65, 65), -1)
        thumb = max(16, int(popup.shape[0] * min(1., self.rows / len(self.frames))))
        offset = int((popup.shape[0] - thumb) * self.scroll / max(1, len(self.frames) - self.rows))
        cv2.rectangle(popup, (x + 2, offset), (popup.shape[1] - 3, offset + thumb - 1),
                      (160, 160, 160), -1)
