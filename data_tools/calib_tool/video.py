"""
Trích khung hình hiệu chỉnh từ video.

Quy trình 2 lượt:
  1. Quét video, phân tích ~N khung/giây: nhận diện bảng, đo độ nét, đo tốc độ di chuyển của bảng
     (so với khung phân tích trước/sau), ước lượng tư thế (vị trí trong khung, kích thước, góc ngả).
  2. Lọc khung đủ góc + đủ nét + bảng gần như đứng yên, rồi chọn tập khung ĐA DẠNG nhất
     (farthest-point sampling trên vị trí/kích thước/góc ngả), giữ khoảng cách thời gian tối thiểu.
  3. Đọc lại video tuần tự và lưu các khung đã chọn ra thư mục ảnh.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

import core

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".mts", ".m2ts", ".webm", ".wmv")


@dataclass
class VideoOptions:
    target_frames: int = 40       # số khung muốn lấy
    analyze_fps: float = 6.0      # số khung phân tích mỗi giây video
    max_speed: float = 1.0        # px/khung — tốc độ tối đa của bảng để coi là "đứng yên"
    min_gap_s: float = 0.5        # khoảng cách thời gian tối thiểu giữa 2 khung được chọn
    min_corners: int = 8
    jpg_quality: int = 97
    out_dir: str = ""             # rỗng = "<tên video>_frames" cạnh file video


@dataclass
class FrameInfo:
    idx: int
    t: float
    n_corners: int = 0
    sharpness: float = 0.0
    speed: Optional[float] = None      # px/khung, max(trước, sau)
    center: tuple = (0.0, 0.0)         # chuẩn hóa 0..1
    scale: float = 0.0                 # cạnh lớn nhất của vùng góc / chiều rộng ảnh
    tilt_x: Optional[float] = None
    tilt_y: Optional[float] = None
    corners: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    status: str = ""                   # lý do loại / "chọn"


@dataclass
class VideoScan:
    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    n_frames: int = 0
    duration: float = 0.0
    frames: list = field(default_factory=list)       # FrameInfo đã phân tích
    selected: list = field(default_factory=list)     # chỉ số khung (idx) được chọn
    saved_paths: list = field(default_factory=list)
    out_dir: str = ""

    def counts(self) -> dict:
        f = self.frames
        det = [x for x in f if x.n_corners > 0]
        return {
            "analyzed": len(f),
            "detected": len(det),
            "enough": sum(1 for x in f if x.status not in ("không thấy bảng", "thiếu góc")),
            "still": sum(1 for x in f if x.status in ("ứng viên", "chọn", "trùng thời gian", "ít đa dạng")),
            "selected": len(self.selected),
        }


# --------------------------------------------------------------------------
# Mở video (xử lý đường dẫn Unicode trên Windows)
# --------------------------------------------------------------------------
def _short_path_windows(path: str) -> Optional[str]:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        fn = ctypes.windll.kernel32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        n = fn(path, buf, 1024)
        if n and buf.value and buf.value.isascii():
            return buf.value
    except Exception:
        pass
    return None


class OpenedVideo:
    """Context manager: mở video, tự dọn file tạm nếu phải copy sang đường dẫn ASCII."""

    def __init__(self, path: str):
        self.path = path
        self.tmp = None
        self.cap = None

    def __enter__(self):
        cands = [self.path]
        sp = _short_path_windows(self.path)
        if sp:
            cands.append(sp)
        for p in cands:
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                self.cap = cap
                return cap
            cap.release()
        if not self.path.isascii():
            d = tempfile.mkdtemp(prefix="charuco_")
            self.tmp = os.path.join(d, "video" + os.path.splitext(self.path)[1].lower())
            shutil.copyfile(self.path, self.tmp)
            cap = cv2.VideoCapture(self.tmp)
            if cap.isOpened():
                self.cap = cap
                return cap
        raise ValueError(f"Không mở được video:\n{self.path}\n"
                         "Định dạng có thể chưa được hỗ trợ (thử xuất lại video sang MP4/H.264).")

    def __exit__(self, *exc):
        if self.cap is not None:
            self.cap.release()
        if self.tmp:
            shutil.rmtree(os.path.dirname(self.tmp), ignore_errors=True)


class FrameReader:
    """Giữ video mở để duyệt từng khung (tiến / lùi) nhanh; có bộ nhớ đệm vài khung gần nhất."""

    def __init__(self, path: str, cache: int = 24):
        self.path = path
        self._ov = OpenedVideo(path)
        self.cap = self._ov.__enter__()
        info = video_info_cap(self.cap)
        self.fps, self.n = info["fps"], max(1, info["n_frames"])
        self.width, self.height = info["width"], info["height"]
        self._next = 0              # chỉ số khung mà cap.read() sẽ trả về
        self._cache: dict = {}
        self._cache_n = cache

    def read(self, idx: int) -> Optional[np.ndarray]:
        """Trả về khung RGB thứ idx (None nếu không đọc được)."""
        idx = int(min(max(idx, 0), self.n - 1))
        if idx in self._cache:
            fr = self._cache.pop(idx)
            self._cache[idx] = fr
            return fr
        if idx != self._next:
            # tiến ít khung: đọc tuần tự (nhanh hơn seek với H.264)
            if self._next < idx <= self._next + 15:
                while self._next < idx:
                    if not self.cap.grab():
                        break
                    self._next += 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._next = idx
        ok, fr = self.cap.read()
        if not ok or fr is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next = idx
            return None
        self._next = idx + 1
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        self._cache[idx] = rgb
        while len(self._cache) > self._cache_n:
            self._cache.pop(next(iter(self._cache)))
        return rgb

    def close(self):
        if self._ov is not None:
            self._ov.__exit__(None, None, None)
            self._ov = None


def video_info_cap(cap) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or fps > 1000:
        fps = 30.0
    return {"fps": fps, "n_frames": n, "width": w, "height": h, "duration": n / fps if n else 0.0}


def video_info(path: str) -> dict:
    with OpenedVideo(path) as cap:
        return video_info_cap(cap)


# --------------------------------------------------------------------------
# Lượt 1: phân tích
# --------------------------------------------------------------------------
def _match_speed(a: FrameInfo, b: FrameInfo) -> Optional[float]:
    """Tốc độ trung bình (px/khung) của các góc chung giữa hai khung phân tích."""
    if a.ids is None or b.ids is None or b.idx == a.idx:
        return None
    ia = {int(i): k for k, i in enumerate(a.ids.ravel())}
    pairs = [(ia[int(i)], k) for k, i in enumerate(b.ids.ravel()) if int(i) in ia]
    if len(pairs) < 4:
        return None
    pa = a.corners.reshape(-1, 2)[[p[0] for p in pairs]]
    pb = b.corners.reshape(-1, 2)[[p[1] for p in pairs]]
    d = np.linalg.norm(pb - pa, axis=1)
    return float(np.median(d)) / abs(b.idx - a.idx)


def _pose(fi: FrameInfo, board, K):
    obj, img = board.matchImagePoints(fi.corners, fi.ids)
    if obj is None or len(obj) < 6:
        return
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.zeros(5))
    except cv2.error:
        return
    if not ok:
        return
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]
    if n[2] < 0:
        n = -n
    fi.tilt_x = float(math.degrees(math.atan(-n[0] / n[2])))
    fi.tilt_y = float(math.degrees(math.atan(-n[1] / n[2])))


def scan_video(path: str, cfg: core.BoardConfig, opt: VideoOptions,
               progress: Optional[Callable[[str, float], None]] = None,
               cancel: Optional[Callable[[], bool]] = None) -> VideoScan:
    progress = progress or (lambda m, f: None)
    cancel = cancel or (lambda: False)
    info = video_info(path)
    scan = VideoScan(path=path, width=info["width"], height=info["height"], fps=info["fps"],
                     n_frames=info["n_frames"], duration=info["duration"])
    board = cfg.make_board()
    det = core.make_detector(board)
    stride = max(1, int(round(scan.fps / max(0.5, opt.analyze_fps))))
    K = None
    total = scan.n_frames or 1

    with OpenedVideo(path) as cap:
        idx = -1
        while True:
            if cancel():
                raise ValueError("Đã hủy.")
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            h, w = gray.shape[:2]
            if scan.width == 0 or (w, h) != (scan.width, scan.height):
                scan.width, scan.height = w, h        # tin kích thước thực (đã xoay theo metadata)
                K = None
            if K is None:
                K = core.estimate_K(w, h)
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if cap.get(cv2.CAP_PROP_POS_MSEC) > 0 else idx / scan.fps
            fi = FrameInfo(idx=idx, t=t)
            ch, ids, _, _ = det.detectBoard(gray)
            if ids is not None and len(ids) > 0:
                fi.n_corners = len(ids)
                fi.corners = ch.astype(np.float32)
                fi.ids = ids.astype(np.int32)
                p = fi.corners.reshape(-1, 2)
                fi.center = (float(p[:, 0].mean() / w), float(p[:, 1].mean() / h))
                fi.scale = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) / w)
                if fi.n_corners >= opt.min_corners:
                    fi.sharpness = core.board_sharpness(gray, fi.corners)
                    _pose(fi, board, K)
            if scan.frames:
                s = _match_speed(scan.frames[-1], fi)
                if s is not None:
                    prev = scan.frames[-1]
                    prev.speed = s if prev.speed is None else max(prev.speed, s)
                    fi.speed = s
            scan.frames.append(fi)
            if len(scan.frames) % 5 == 0:
                progress(f"Đang quét video: {t:5.1f} s / {scan.duration:.1f} s", 0.8 * min(1.0, idx / total))
    if scan.n_frames == 0 and scan.frames:
        scan.n_frames = scan.frames[-1].idx + 1
        scan.duration = scan.frames[-1].t
    return scan


# --------------------------------------------------------------------------
# Chọn khung
# --------------------------------------------------------------------------
def select_frames(scan: VideoScan, opt: VideoOptions) -> list:
    fr = scan.frames
    for f in fr:
        f.status = ""
    sharp_vals = [f.sharpness for f in fr if f.n_corners >= opt.min_corners and f.sharpness > 0]
    sharp_thr = core.SHARP_BAD
    if sharp_vals:
        sharp_thr = max(core.SHARP_BAD, 0.3 * float(np.median(sharp_vals)))
    cands = []
    for f in fr:
        if f.n_corners == 0:
            f.status = "không thấy bảng"
        elif f.n_corners < opt.min_corners:
            f.status = "thiếu góc"
        elif f.speed is not None and f.speed > opt.max_speed:
            f.status = "đang di chuyển"
        elif f.sharpness < sharp_thr:
            f.status = "mờ"
        elif f.speed is None:
            f.status = "không đo được chuyển động"
        else:
            f.status = "ứng viên"
            cands.append(f)
    if not cands:
        scan.selected = []
        return []

    # đặc trưng tư thế: vị trí (0..1), kích thước (log), góc ngả (/45°)
    def feat(f):
        tx = (f.tilt_x or 0.0) / 45.0
        ty = (f.tilt_y or 0.0) / 45.0
        return np.array([f.center[0] * 1.6, f.center[1] * 0.9 * 1.6, 0.6 * math.log(max(f.scale, 1e-3)), tx, ty])

    X = np.array([feat(f) for f in cands])
    q = np.array([f.sharpness for f in cands])
    qn = (q - q.min()) / (np.ptp(q) + 1e-9)                       # 0..1
    sp = np.array([f.speed for f in cands])
    still = 1.0 - np.clip(sp / max(opt.max_speed, 1e-6), 0, 1)     # 1 = đứng yên hẳn
    quality = 0.5 + 0.3 * qn + 0.2 * still + 0.02 * np.array([f.n_corners for f in cands]) / max(1, max(f.n_corners for f in cands))

    times = np.array([f.t for f in cands])
    chosen = [int(np.argmax(quality))]
    mind = np.linalg.norm(X - X[chosen[0]], axis=1)
    blocked = np.abs(times - times[chosen[0]]) < opt.min_gap_s
    while len(chosen) < opt.target_frames:
        score = mind * quality
        score[blocked] = -1
        score[chosen] = -1
        j = int(np.argmax(score))
        if score[j] <= 0:
            break
        chosen.append(j)
        mind = np.minimum(mind, np.linalg.norm(X - X[j], axis=1))
        blocked |= np.abs(times - times[j]) < opt.min_gap_s
    chosen_set = set(chosen)
    for k, f in enumerate(cands):
        if k in chosen_set:
            f.status = "chọn"
        elif blocked[k]:
            f.status = "trùng thời gian"
        else:
            f.status = "ít đa dạng"
    scan.selected = sorted(cands[k].idx for k in chosen)
    return scan.selected


# --------------------------------------------------------------------------
# Lượt 2: lưu khung
# --------------------------------------------------------------------------
def default_out_dir(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(path)), f"{stem}_frames")


def _write_jpg(path: str, bgr: np.ndarray, quality: int):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Không mã hóa được ảnh.")
    buf.tofile(path)            # tofile hỗ trợ đường dẫn Unicode trên Windows


def save_frames(scan: VideoScan, opt: VideoOptions,
                progress: Optional[Callable[[str, float], None]] = None,
                cancel: Optional[Callable[[], bool]] = None) -> list:
    progress = progress or (lambda m, f: None)
    cancel = cancel or (lambda: False)
    out = opt.out_dir or default_out_dir(scan.path)
    os.makedirs(out, exist_ok=True)
    scan.out_dir = out
    want = set(scan.selected)
    if not want:
        return []
    last = max(want)
    stem = os.path.splitext(os.path.basename(scan.path))[0]
    paths = []
    with OpenedVideo(scan.path) as cap:
        idx = -1
        while idx < last:
            if cancel():
                raise ValueError("Đã hủy.")
            if not cap.grab():
                break
            idx += 1
            if idx not in want:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            p = os.path.join(out, f"{stem}_f{idx:06d}.jpg")
            _write_jpg(p, frame, opt.jpg_quality)
            paths.append(p)
            progress(f"Đang lưu khung {len(paths)}/{len(want)}", 0.8 + 0.2 * len(paths) / len(want))
    scan.saved_paths = paths
    return paths


def process_video(path: str, cfg: core.BoardConfig, opt: VideoOptions,
                  progress: Optional[Callable[[str, float], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None) -> VideoScan:
    scan = scan_video(path, cfg, opt, progress, cancel)
    select_frames(scan, opt)
    if scan.selected:
        save_frames(scan, opt, progress, cancel)
    if progress:
        progress("Xong", 1.0)
    return scan


def scan_advice(scan: VideoScan, opt: VideoOptions) -> list:
    c = scan.counts()
    tips = []
    if c["analyzed"] == 0:
        return ["Không đọc được khung hình nào từ video."]
    if c["detected"] < 0.3 * c["analyzed"]:
        tips.append(f"Chỉ {c['detected']}/{c['analyzed']} khung thấy bảng — kiểm tra thông số bảng ở tab 1, "
                    "giữ bảng trong khung hình và đủ sáng.")
    moving = sum(1 for f in scan.frames if f.status == "đang di chuyển")
    blurry = sum(1 for f in scan.frames if f.status == "mờ")
    if c["still"] < opt.target_frames:
        tips.append(f"Chỉ có {c['still']} khung đủ nét và bảng đứng yên (cần {opt.target_frames}). "
                    "Khi quay: di chuyển chậm và DỪNG khoảng 1 giây ở mỗi tư thế. "
                    f"Có thể tăng “ngưỡng đứng yên” (hiện {opt.max_speed:g} px/khung) nếu cần.")
    if moving > 0.6 * max(1, c["enough"]) and c["still"] < opt.target_frames:
        tips.append("Phần lớn khung bị loại vì bảng đang di chuyển — rolling shutter của điện thoại làm méo bảng khi chuyển động.")
    if blurry > 0.3 * max(1, c["enough"]):
        tips.append("Nhiều khung bị mờ dù bảng đứng yên — kiểm tra khóa nét (bảng phải nằm trong vùng nét), thêm ánh sáng.")
    if not tips:
        tips.append(f"Đã chọn {c['selected']} khung đa dạng từ {c['still']} khung tốt.")
    return tips
