"""
Lõi hiệu chỉnh extrinsic nhiều camera trên sàn lát gạch.

Hệ tọa độ thế giới: gốc tại một góc gạch, X và Y chạy theo đường ron, Z hướng lên, mặt sàn Z = 0, đơn vị mét.
Hệ phải (đứng ở gốc nhìn theo +X thì +Y ở bên TRÁI).

Nguồn điểm mốc cho mỗi camera:
  - marker ArUco đặt trên sàn tại góc gạch đã biết (tự nhận diện, tự ghép giữa các camera)
  - góc gạch (giao điểm ron) — dự đoán từ tư thế hiện có rồi tinh chỉnh dưới 1 pixel
  - điểm thủ công có tọa độ XYZ
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

import core
import video as vid

ARROWS = {"+X": 0.0, "+Y": 90.0, "-X": 180.0, "-Y": 270.0}
FLOOR_DICTS = ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_50", "DICT_APRILTAG_36h11"]
FLOOR_DICTS = [d for d in FLOOR_DICTS if hasattr(cv2.aruco, d)]
SOURCES = {"marker": "Marker", "tile": "Góc gạch", "manual": "Thủ công"}
PERSON_H = 1.7


# ==========================================================================
# Hệ tọa độ & marker sàn
# ==========================================================================
@dataclass
class MarkerPlacement:
    id: int
    i: float = 0.0          # vị trí TÂM marker theo số viên gạch
    j: float = 0.0
    arrow: str = "+X"       # mũi tên in trên marker chỉ theo hướng nào của thế giới


@dataclass
class WorldSetup:
    pitch_x_mm: float = 600.0      # kích thước 1 viên tính cả ron, theo X
    pitch_y_mm: float = 600.0
    marker_dict: str = "DICT_4X4_50"
    marker_mm: float = 180.0       # cạnh ô đen ngoài cùng của marker
    markers: list = field(default_factory=list)

    @property
    def px(self) -> float:
        return self.pitch_x_mm / 1000.0

    @property
    def py(self) -> float:
        return self.pitch_y_mm / 1000.0

    def grid_to_world(self, i, j):
        return float(i) * self.px, float(j) * self.py

    def placement(self, mid: int) -> Optional[MarkerPlacement]:
        for m in self.markers:
            if int(m.id) == int(mid):
                return m
        return None

    def marker_world_corners(self, m: MarkerPlacement) -> np.ndarray:
        """4 góc theo thứ tự OpenCV (trên-trái, trên-phải, dưới-phải, dưới-trái của ảnh marker)."""
        s = self.marker_mm / 1000.0 / 2
        cx, cy = self.grid_to_world(m.i, m.j)
        # marker nằm ngửa trên sàn, nhìn từ trên xuống: trục x ảnh = mũi tên; trục y ảnh (xuống) = -Y cục bộ
        local = np.array([[-s, s], [s, s], [s, -s], [-s, -s]])
        a = math.radians(ARROWS.get(m.arrow, 0.0))
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = local @ R.T + [cx, cy]
        return np.hstack([p, np.zeros((4, 1))])

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d):
        ms = [MarkerPlacement(**m) for m in d.get("markers", [])]
        keys = {k: v for k, v in d.items() if k in WorldSetup.__dataclass_fields__ and k != "markers"}
        return WorldSetup(markers=ms, **keys)


def floor_dictionary(name):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def generate_marker_pdf(world: WorldSetup, ids: list, path: str, paper: str = "A4"):
    """Mỗi trang một marker, có đường chữ thập qua tâm (căn theo ron gạch), mũi tên hướng x, thước 100 mm."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    font, font_b, uni = core._register_fonts()
    txt = (lambda s: s) if uni else core._ascii
    pw, ph = core.PAPERS[paper]
    s = world.marker_mm
    if s > min(pw, ph) - 30:
        raise ValueError(f"Marker {s:g} mm quá lớn cho khổ {paper} (tối đa {min(pw, ph) - 30:g} mm).")
    d = floor_dictionary(world.marker_dict)
    nb = d.markerSize + 2
    c = canvas.Canvas(path, pagesize=(pw * mm, ph * mm))
    c.setTitle(core._ascii(f"Marker san {world.marker_dict} {s:g} mm"))
    for mid in ids:
        if mid >= d.bytesList.shape[0]:
            raise ValueError(f"ID {mid} vượt quá số marker của {world.marker_dict}.")
        bits = cv2.aruco.generateImageMarker(d, int(mid), nb, borderBits=1) < 128
        cx, cy = pw / 2, ph / 2 + 8           # tâm marker (mm, gốc dưới-trái của PDF)
        x0, y0 = cx - s / 2, cy - s / 2
        cell = s / nb
        # chữ thập qua tâm, kéo ra tới mép giấy
        c.setStrokeGray(0.35); c.setLineWidth(0.5)
        c.setDash(4, 3)
        c.line(4 * mm, cy * mm, x0 * mm - 3 * mm, cy * mm)
        c.line((x0 + s) * mm + 3 * mm, cy * mm, (pw - 4) * mm, cy * mm)
        c.line(cx * mm, 4 * mm, cx * mm, y0 * mm - 3 * mm)
        c.line(cx * mm, (y0 + s) * mm + 3 * mm, cx * mm, (ph - 4) * mm)
        c.setDash()
        c.setFillGray(0)
        for i in range(nb):
            for j in range(nb):
                if bits[i, j]:
                    c.rect((x0 + j * cell) * mm, (y0 + s - (i + 1) * cell) * mm, cell * mm + 0.01, cell * mm + 0.01,
                           stroke=0, fill=1)
        # mũi tên hướng x của marker (bên phải)
        ay = cy + s / 2 + 9
        c.setLineWidth(1.2); c.setStrokeGray(0)
        c.line(cx * mm, ay * mm, (cx + 40) * mm, ay * mm)
        p = c.beginPath(); p.moveTo((cx + 44) * mm, ay * mm); p.lineTo((cx + 38) * mm, (ay + 3) * mm)
        p.lineTo((cx + 38) * mm, (ay - 3) * mm); p.close(); c.drawPath(p, stroke=0, fill=1)
        c.setFont(font_b, 22)
        c.drawString(12 * mm, (ph - 18) * mm, f"ID {mid}")
        c.setFont(font, 9)
        c.drawString(12 * mm, (ph - 25) * mm, txt(f"{world.marker_dict} · cạnh ô đen {s:g} mm · IN 100% / ACTUAL SIZE"))
        c.drawString((cx + 46) * mm, (ay - 1.2) * mm, txt("hướng mũi tên"))
        c.drawString(12 * mm, 22 * mm, txt("Đặt TÂM marker trùng góc gạch; hai đường nét đứt trùng đường ron."))
        c.drawString(12 * mm, 17 * mm, txt("Nhập vị trí (i, j) và hướng mũi tên (+X/+Y/-X/-Y) vào tool. Đo lại cạnh ô đen."))
        rx = 12
        c.setLineWidth(0.6)
        c.line(rx * mm, 10 * mm, (rx + 100) * mm, 10 * mm)
        for k in range(11):
            c.line((rx + 10 * k) * mm, 10 * mm, (rx + 10 * k) * mm, (10 + (3 if k % 5 == 0 else 1.8)) * mm)
        c.drawString((rx + 103) * mm, 9.2 * mm, txt("= 100 mm"))
        c.showPage()
    c.save()


# ==========================================================================
# Camera
# ==========================================================================
@dataclass
class Corr:
    X: float
    Y: float
    Z: float
    u: float
    v: float
    source: str = "manual"
    label: str = ""
    enabled: bool = True
    gi: Optional[float] = None
    gj: Optional[float] = None
    err_px: Optional[float] = None
    err_cm: Optional[float] = None
    inlier: bool = True

    @property
    def world(self):
        return np.array([self.X, self.Y, self.Z], np.float64)

    def key(self):
        if self.source == "tile":
            return ("tile", self.gi, self.gj)
        if self.source == "marker":
            return ("marker", self.label)
        return ("manual", round(self.u, 1), round(self.v, 1))


@dataclass
class ExResult:
    rvec: list
    tvec: list
    rms: float = 0.0
    med_cm: float = 0.0
    max_cm: float = 0.0
    n_total: int = 0
    n_inliers: int = 0
    C: list = field(default_factory=list)
    C_std_cm: list = field(default_factory=list)
    cov6: Optional[list] = None
    warnings: list = field(default_factory=list)
    level: str = "ok"
    pitch_scale: Optional[float] = None


class CameraEx:
    def __init__(self, name: str):
        self.name = name
        self.intr_path = ""
        self.ref_path = ""
        self.K = None
        self.dist = None
        self.model = "pinhole5"
        self.intr_size = None
        self.image = None          # RGB ảnh tham chiếu (nguồn video: khung đang xem)
        self.video_path = ""       # nguồn video — duyệt khung khi gán điểm
        self.frame_idx = 0
        self._reader = None
        self.corrs: list[Corr] = []
        self.result: Optional[ExResult] = None
        self._nb = None

    # ---- nạp dữ liệu ----
    def load_intrinsics(self, path):
        d = core.load_calibration(path)
        self.intr_path = path
        self.K, self.dist = d["K"], np.asarray(d["dist"], np.float64).ravel()
        self.model = d.get("model") or "pinhole5"
        self.intr_size = d.get("size")
        self._nb = None

    def load_image(self, path):
        self.image = core.load_image(path)
        self.ref_path = path
        self.close_video()
        self.video_path = ""
        self._nb = None

    def load_video(self, path, frame_idx=0):
        """Dùng video làm nguồn ảnh: self.image luôn là khung frame_idx."""
        reader = vid.FrameReader(path)
        self.close_video()
        self._reader = reader
        self.video_path, self.ref_path = path, ""
        if not self.set_frame(frame_idx) and not self.set_frame(0):
            self.close_video()
            self.video_path = ""
            raise ValueError(f"Không đọc được khung hình nào từ video:\n{path}")

    def set_frame(self, idx) -> bool:
        if self._reader is None:
            return False
        idx = int(min(max(idx, 0), self._reader.n - 1))
        fr = self._reader.read(idx)
        if fr is None:
            return False
        if self.image is None or self.image.shape != fr.shape:
            self._nb = None
        self.image, self.frame_idx = fr, idx
        return True

    def close_video(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    @property
    def is_video(self):
        return self._reader is not None

    @property
    def n_frames(self):
        return self._reader.n if self._reader else 0

    @property
    def fps(self):
        return self._reader.fps if self._reader else 30.0

    @property
    def size(self):
        if self.image is not None:
            return self.image.shape[1], self.image.shape[0]
        return tuple(self.intr_size) if self.intr_size else (1920, 1080)

    @property
    def fisheye(self):
        return self.model == "fisheye"

    def ready(self) -> bool:
        return self.K is not None and self.image is not None

    def size_warning(self) -> str:
        if self.intr_size and self.image is not None and tuple(self.intr_size) != self.size:
            return (f"Ảnh tham chiếu {self.size[0]}×{self.size[1]} khác độ phân giải lúc hiệu chỉnh intrinsic "
                    f"{self.intr_size[0]}×{self.intr_size[1]}.")
        return ""

    @property
    def has_pose(self):
        return self.result is not None

    def pose(self):
        r = self.result
        return np.array(r.rvec, np.float64).reshape(3, 1), np.array(r.tvec, np.float64).reshape(3, 1)

    def R(self, rvec=None):
        if rvec is None:
            rvec, _ = self.pose()
        return cv2.Rodrigues(rvec)[0]

    def center(self, rvec=None, tvec=None):
        if rvec is None:
            rvec, tvec = self.pose()
        R = cv2.Rodrigues(rvec)[0]
        return (-R.T @ tvec).ravel()

    # ---- chiếu ----
    def project_raw(self, X, rvec, tvec):
        X = np.asarray(X, np.float64).reshape(-1, 3)
        if len(X) == 0:
            return np.zeros((0, 2))
        if self.fisheye:
            p, _ = cv2.fisheye.projectPoints(X.reshape(1, -1, 3), rvec, tvec, self.K, self.dist[:4])
        else:
            p, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rvec, tvec, self.K, self.dist)
        return p.reshape(-1, 2)

    def normalize(self, uv):
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        if len(uv) == 0:
            return np.zeros((0, 2))
        if self.fisheye:
            n = cv2.fisheye.undistortPoints(uv.reshape(1, -1, 2), self.K, self.dist[:4])
        else:
            n = cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.dist)
        return n.reshape(-1, 2)

    def norm_bounds(self):
        if self._nb is None:
            w, h = self.size
            t = np.linspace(0, 1, 60)
            b = np.vstack([np.c_[t * (w - 1), np.zeros_like(t)], np.c_[t * (w - 1), np.full_like(t, h - 1)],
                           np.c_[np.zeros_like(t), t * (h - 1)], np.c_[np.full_like(t, w - 1), t * (h - 1)]])
            n = self.normalize(b)
            self._nb = (n[:, 0].min(), n[:, 0].max(), n[:, 1].min(), n[:, 1].max())
        return self._nb

    def project(self, X, rvec=None, tvec=None, margin=0.0):
        """Chiếu an toàn: trả về (uv, mask) — mask = trước camera, trong vùng ảnh (đã tính méo)."""
        if rvec is None:
            rvec, tvec = self.pose()
        X = np.asarray(X, np.float64).reshape(-1, 3)
        R = cv2.Rodrigues(rvec)[0]
        Xc = X @ R.T + tvec.ravel()
        front = Xc[:, 2] > 1e-3
        uv = np.full((len(X), 2), np.nan)
        if not front.any():
            return uv, front
        xn = np.zeros((len(X), 2))
        xn[front] = Xc[front, :2] / Xc[front, 2:3]
        x0, x1, y0, y1 = self.norm_bounds()
        ex = 0.03 * max(x1 - x0, y1 - y0)
        ok = front & (xn[:, 0] > x0 - ex) & (xn[:, 0] < x1 + ex) & (xn[:, 1] > y0 - ex) & (xn[:, 1] < y1 + ex)
        if ok.any():
            uv[ok] = self.project_raw(X[ok], rvec, tvec)
        w, h = self.size
        inside = ok & (uv[:, 0] >= -margin) & (uv[:, 0] < w + margin) & (uv[:, 1] >= -margin) & (uv[:, 1] < h + margin)
        return uv, inside

    def backproject(self, uv, z=0.0, rvec=None, tvec=None):
        """Chiếu ngược điểm ảnh xuống mặt phẳng Z = z. Trả về (N,3), NaN nếu tia không cắt mặt phẳng."""
        if rvec is None:
            rvec, tvec = self.pose()
        n = self.normalize(uv)
        R = cv2.Rodrigues(rvec)[0]
        C = (-R.T @ tvec).ravel()
        d = np.c_[n, np.ones(len(n))] @ R          # = (R.T @ d_c).T
        out = np.full((len(n), 3), np.nan)
        ok = np.abs(d[:, 2]) > 1e-9
        s = np.full(len(n), np.nan)
        s[ok] = (z - C[2]) / d[ok, 2]
        ok &= s > 0
        out[ok] = C + s[ok, None] * d[ok]
        return out

    # ---- lưu ----
    def to_dict(self, base=""):
        def rel(p):
            if not p:
                return ""
            try:
                return os.path.relpath(p, base) if base else p
            except ValueError:
                return p
        return {"name": self.name, "intr_path": rel(self.intr_path), "ref_path": rel(self.ref_path),
                "video_path": rel(self.video_path), "frame_idx": int(self.frame_idx),
                "corrs": [asdict(c) for c in self.corrs],
                "result": asdict(self.result) if self.result else None}

    @staticmethod
    def from_dict(d, base=""):
        cam = CameraEx(d["name"])

        def ab(p):
            if not p:
                return ""
            p = p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
            if not os.path.exists(p):
                # dự án bị chuyển thư mục: thử tìm file cùng tên cạnh file dự án
                alt = os.path.join(base, os.path.basename(p.replace("\\", "/")))
                if os.path.exists(alt):
                    return alt
            return p
        cam.intr_path, cam.ref_path = ab(d.get("intr_path", "")), ab(d.get("ref_path", ""))
        video_path = ab(d.get("video_path", ""))
        errors = []
        if cam.intr_path:
            try:
                cam.load_intrinsics(cam.intr_path)
            except Exception as e:
                errors.append(f"{cam.name}: không đọc được intrinsic ({e})")
        if cam.ref_path:
            try:
                cam.load_image(cam.ref_path)
            except Exception as e:
                errors.append(f"{cam.name}: không mở được ảnh tham chiếu ({e})")
        if video_path:
            try:
                cam.load_video(video_path, int(d.get("frame_idx", 0)))
            except Exception as e:
                cam.video_path = video_path
                errors.append(f"{cam.name}: không mở được video ({e})")
        cam.corrs = [Corr(**c) for c in d.get("corrs", [])]
        if d.get("result"):
            if cam.K is not None:
                cam.result = ExResult(**d["result"])
            else:
                errors.append(f"{cam.name}: bỏ kết quả extrinsic đã lưu vì thiếu intrinsic — nạp lại intrinsic rồi giải lại")
        return cam, errors


# ==========================================================================
# Nhận diện marker sàn
# ==========================================================================
def detect_floor_markers(cam: CameraEx, world: WorldSetup):
    """Trả về (corrs, unknown_ids, min_side_px)."""
    gray = cv2.cvtColor(cam.image, cv2.COLOR_RGB2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 4
    det = cv2.aruco.ArucoDetector(floor_dictionary(world.marker_dict), params)
    corners, ids, _ = det.detectMarkers(gray)
    out, unknown, sides = [], [], []
    if ids is None:
        return out, unknown, None
    for c4, mid in zip(corners, ids.ravel()):
        m = world.placement(int(mid))
        c4 = c4.reshape(4, 2)
        sides.append(min(np.linalg.norm(c4[k] - c4[(k + 1) % 4]) for k in range(4)))
        if m is None:
            unknown.append(int(mid))
            continue
        W = world.marker_world_corners(m)
        for k in range(4):
            out.append(Corr(float(W[k, 0]), float(W[k, 1]), float(W[k, 2]), float(c4[k, 0]), float(c4[k, 1]),
                            source="marker", label=f"M{int(mid)}-{k}"))
    return out, unknown, (min(sides) if sides else None)


# ==========================================================================
# Lưới gạch: dự đoán & tinh chỉnh giao điểm ron
# ==========================================================================
def floor_footprint(cam: CameraEx, max_dist=30.0, n=240, rvec=None, tvec=None):
    """Đa giác vùng sàn camera nhìn thấy (chiếu biên ảnh xuống sàn, cắt ở max_dist)."""
    if rvec is None:
        rvec, tvec = cam.pose()
    w, h = cam.size
    t = np.linspace(0, 1, n // 4, endpoint=False)
    b = np.vstack([np.c_[t * w, np.zeros_like(t)], np.c_[np.full_like(t, w - 1), t * h],
                   np.c_[(1 - t) * w, np.full_like(t, h - 1)], np.c_[np.zeros_like(t), (1 - t) * h]])
    nrm = cam.normalize(b)
    R = cv2.Rodrigues(rvec)[0]
    C = (-R.T @ tvec).ravel()
    d = np.c_[nrm, np.ones(len(nrm))] @ R
    pts = []
    for di in d:
        hz = math.hypot(di[0], di[1]) + 1e-12
        if di[2] < -1e-9:
            s = -C[2] / di[2]
            p = C + s * di
            if math.hypot(p[0] - C[0], p[1] - C[1]) <= max_dist:
                pts.append(p[:2]); continue
        pts.append(C[:2] + max_dist * di[:2] / hz)
    return np.array(pts)


def predict_tiles(cam: CameraEx, world: WorldSetup, rvec, tvec, roi=None, min_spacing=14.0, max_dist=30.0):
    fp = floor_footprint(cam, max_dist, rvec=rvec, tvec=tvec)
    x0, y0 = fp.min(axis=0); x1, y1 = fp.max(axis=0)
    if roi is not None and roi.valid():
        m = 2.0
        x0, y0 = max(x0, roi.x0 - m), max(y0, roi.y0 - m)
        x1, y1 = min(x1, roi.x1 + m), min(y1, roi.y1 + m)
    px, py = world.px, world.py
    I = np.arange(math.floor(x0 / px) - 1, math.ceil(x1 / px) + 2)
    J = np.arange(math.floor(y0 / py) - 1, math.ceil(y1 / py) + 2)
    if len(I) * len(J) > 40000:
        return []
    gi, gj = np.meshgrid(I, J, indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    X = np.c_[gi * px, gj * py, np.zeros(len(gi))]
    uv, ok = cam.project(X, rvec, tvec, margin=-12)
    uvx, okx = cam.project(X + [px, 0, 0], rvec, tvec)
    uvy, oky = cam.project(X + [0, py, 0], rvec, tvec)
    sp = np.minimum(np.linalg.norm(uvx - uv, axis=1), np.linalg.norm(uvy - uv, axis=1))
    keep = ok & okx & oky & (sp >= min_spacing)
    return [(float(gi[k]), float(gj[k]), uv[k], float(sp[k])) for k in np.nonzero(keep)[0]]


def _profile_peak(p):
    """Đỉnh (âm hoặc dương) nổi bật nhất sau khi trừ xu hướng; trả về (vị trí dưới-pixel, độ tin cậy, dấu)."""
    p = cv2.GaussianBlur(p.reshape(1, -1).astype(np.float32), (0, 0), 1.0).ravel()
    x = np.arange(len(p))
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    r = p - A @ coef
    n = len(r)
    lo, hi = int(0.2 * n), int(0.8 * n)
    seg = r[lo:hi]
    k = int(np.argmax(np.abs(seg))) + lo
    sign = np.sign(r[k])
    noise = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
    conf = abs(r[k]) / noise
    if 0 < k < n - 1:
        a, b, c = r[k - 1], r[k], r[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        off = float(np.clip(off, -1, 1))
    else:
        off = 0.0
    return k + off, conf, sign


def refine_tile_corner(cam: CameraEx, gray, world: WorldSetup, rvec, tvec, gi, gj, spacing_px):
    """Tinh chỉnh một góc gạch: nắn thẳng vùng nhỏ quanh góc về góc nhìn từ trên xuống, tìm đường ron dọc/ngang."""
    X0, Y0 = world.grid_to_world(gi, gj)
    pitch = min(world.px, world.py)
    hsz = 0.22 * pitch
    ppm = spacing_px / pitch                    # pixel ảnh / mét tại vị trí này
    N = int(np.clip(2 * hsz * ppm, 21, 81)) | 1
    a = np.linspace(-hsz, hsz, N)
    A, B = np.meshgrid(a, a)                    # hàng: Y tăng xuống dưới trong patch
    Xw = np.c_[X0 + A.ravel(), Y0 + B.ravel(), np.zeros(A.size)]
    uv, ok = cam.project(Xw, rvec, tvec, margin=-2)
    if ok.mean() < 0.98:
        return None
    mx = uv[:, 0].reshape(N, N).astype(np.float32)
    my = uv[:, 1].reshape(N, N).astype(np.float32)
    patch = cv2.remap(gray, mx, my, cv2.INTER_LINEAR).astype(np.float32)
    col = patch.mean(axis=0)                    # đường ron dọc (X = const)
    row = patch.mean(axis=1)                    # đường ron ngang (Y = const)
    kx, cx, sx = _profile_peak(col)
    ky, cy, sy = _profile_peak(row)
    if min(cx, cy) < 3.5 or sx != sy:
        return None
    step = a[1] - a[0]
    dx, dy = -hsz + kx * step, -hsz + ky * step
    if max(abs(dx), abs(dy)) > 0.6 * hsz:
        return None
    p, okp = cam.project(np.array([[X0 + dx, Y0 + dy, 0.0]]), rvec, tvec)
    if not okp[0]:
        return None
    return p[0], float(min(cx, cy))


def snap_point(gray, u, v, win=5):
    """Bắt điểm click thủ công vào góc/giao điểm gần nhất (cornerSubPix)."""
    pt = np.array([[[u, v]]], np.float32)
    try:
        cv2.cornerSubPix(gray, pt, (win, win), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    except cv2.error:
        return u, v
    nu, nv = pt.ravel()
    if math.hypot(nu - u, nv - v) > win * 1.5:
        return u, v
    return float(nu), float(nv)


def auto_tiles(cam: CameraEx, world: WorldSetup, roi=None, iterations=3, progress=None):
    """Lặp: dự đoán lưới từ tư thế hiện tại → tinh chỉnh → giải lại. Giữ marker & điểm thủ công."""
    if not cam.has_pose:
        raise ValueError("Cần có tư thế ban đầu (giải từ marker hoặc ≥ 4 điểm thủ công) trước khi tìm lưới gạch.")
    gray = cv2.cvtColor(cam.image, cv2.COLOR_RGB2GRAY)
    disabled = {c.key() for c in cam.corrs if c.source == "tile" and not c.enabled}
    n_prev = -1
    for it in range(iterations):
        rvec, tvec = cam.pose()
        cands = predict_tiles(cam, world, rvec, tvec, roi)
        found = []
        for k, (gi, gj, uv0, sp) in enumerate(cands):
            r = refine_tile_corner(cam, gray, world, rvec, tvec, gi, gj, sp)
            if r is None:
                continue
            (u, v), conf = r
            X, Y = world.grid_to_world(gi, gj)
            c = Corr(X, Y, 0.0, float(u), float(v), source="tile", label=f"({gi:g},{gj:g})", gi=gi, gj=gj)
            c.enabled = c.key() not in disabled
            found.append(c)
            if progress and k % 40 == 0:
                progress(f"Vòng {it + 1}: tinh chỉnh góc gạch {k}/{len(cands)}", (it + k / max(1, len(cands))) / iterations)
        cam.corrs = [c for c in cam.corrs if c.source != "tile"] + found
        solve(cam, world)
        if len(found) == n_prev:
            break
        n_prev = len(found)
    return sum(1 for c in cam.corrs if c.source == "tile")


# ==========================================================================
# Giải tư thế
# ==========================================================================
def _solve_raw(cam: CameraEx, obj, img, init=None, ransac_px=4.0):
    obj = obj.astype(np.float64).reshape(-1, 3)
    img = img.astype(np.float64).reshape(-1, 2)
    if cam.fisheye:
        K, D = np.eye(3), None
        pts = cam.normalize(img)
        thr = ransac_px / cam.K[0, 0]
    else:
        K, D, pts, thr = cam.K, cam.dist, img, ransac_px
    cands = []
    planar = np.allclose(obj[:, 2], 0)
    if init is not None:
        cands.append(init)
    try:
        ok, rv, tv, inl = cv2.solvePnPRansac(obj, pts, K, D, iterationsCount=1000, reprojectionError=thr,
                                             confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            cands.append((rv, tv))
    except cv2.error:
        pass
    if planar and len(obj) >= 4:
        try:
            n, rvs, tvs, _ = cv2.solvePnPGeneric(obj, pts, K, D, flags=cv2.SOLVEPNP_IPPE)
            cands += list(zip(rvs, tvs))
        except cv2.error:
            pass
    if not cands:
        try:
            ok, rv, tv = cv2.solvePnP(obj, pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                cands.append((rv, tv))
        except cv2.error:
            pass
    best = None
    for rv, tv in cands:
        rv = np.asarray(rv, np.float64).reshape(3, 1); tv = np.asarray(tv, np.float64).reshape(3, 1)
        e = np.linalg.norm(cam.project_raw(obj, rv, tv) - img, axis=1)
        # ưu tiên nghiệm có camera ở trên sàn
        C = (-cv2.Rodrigues(rv)[0].T @ tv).ravel()
        inl = e < ransac_px
        score = (inl.sum(), -np.median(e))
        if best is None or score > best[0]:
            best = (score, rv, tv, inl)
    if best is None:
        raise ValueError("Không giải được tư thế camera từ các điểm hiện có.")
    _, rv, tv, inl = best
    if inl.sum() >= 4:
        try:
            rv, tv = cv2.solvePnPRefineLM(obj[inl], pts[inl], K, D, rv, tv)
        except cv2.error:
            pass
        e = np.linalg.norm(cam.project_raw(obj, rv, tv) - img, axis=1)
        inl = e < ransac_px
        if inl.sum() >= 4:
            try:
                rv, tv = cv2.solvePnPRefineLM(obj[inl], pts[inl], K, D, rv, tv)
            except cv2.error:
                pass
    return rv.reshape(3, 1), tv.reshape(3, 1)


def _num_jac(cam, obj, rv, tv, eps=1e-6):
    p0 = cam.project_raw(obj, rv, tv).ravel()
    J = np.zeros((len(p0), 6))
    x = np.r_[rv.ravel(), tv.ravel()]
    for k in range(6):
        d = np.zeros(6); d[k] = eps * (1 if k < 3 else max(1.0, np.linalg.norm(tv)))
        xp = x + d
        J[:, k] = (cam.project_raw(obj, xp[:3].reshape(3, 1), xp[3:].reshape(3, 1)).ravel() - p0) / d[k]
    return J


def ground_error_cm(cam, c: Corr, rv, tv):
    p = cam.backproject(np.array([[c.u, c.v]]), z=c.Z, rvec=rv, tvec=tv)[0]
    if np.isnan(p).any():
        return None
    return float(np.linalg.norm(p[:2] - [c.X, c.Y]) * 100)


def solve(cam: CameraEx, world: WorldSetup, roi=None, ransac_px=None) -> ExResult:
    if cam.K is None:
        raise ValueError(f"{cam.name}: chưa có intrinsic.")
    use = [c for c in cam.corrs if c.enabled]
    if len(use) < 4:
        raise ValueError(f"{cam.name}: cần ít nhất 4 điểm mốc đang bật (hiện có {len(use)}).")
    obj = np.array([c.world for c in use])
    img = np.array([[c.u, c.v] for c in use])
    if ransac_px is None:
        ransac_px = max(3.0, 0.004 * max(cam.size))
    init = cam.pose() if cam.has_pose else None
    rv, tv = _solve_raw(cam, obj, img, init, ransac_px)
    # sai số từng điểm (kể cả điểm đang tắt)
    for c in cam.corrs:
        p = cam.project_raw(c.world, rv, tv)[0]
        c.err_px = float(np.linalg.norm(p - [c.u, c.v]))
        c.err_cm = ground_error_cm(cam, c, rv, tv)
        c.inlier = c.err_px < ransac_px
    inl = [c for c in use if c.inlier]
    e = np.array([c.err_px for c in inl]) if inl else np.array([np.nan])
    ecm = np.array([c.err_cm for c in inl if c.err_cm is not None]) if inl else np.array([])
    R = cv2.Rodrigues(rv)[0]
    C = (-R.T @ tv).ravel()
    res = ExResult(rvec=rv.ravel().tolist(), tvec=tv.ravel().tolist(),
                   rms=float(np.sqrt(np.mean(e ** 2))) if inl else float("nan"),
                   med_cm=float(np.median(ecm)) if len(ecm) else float("nan"),
                   max_cm=float(np.max(ecm)) if len(ecm) else float("nan"),
                   n_total=len(use), n_inliers=len(inl), C=C.tolist())
    # độ bất định tư thế
    if len(inl) >= 6:
        oi = np.array([c.world for c in inl]); ii = np.array([[c.u, c.v] for c in inl])
        J = _num_jac(cam, oi, rv, tv)
        r = (cam.project_raw(oi, rv, tv) - ii).ravel()
        dof = max(1, len(r) - 6)
        s2 = float(r @ r) / dof
        try:
            cov = s2 * np.linalg.inv(J.T @ J)
            # lan truyền sang tâm camera C = -R^T t (số)
            x0 = np.r_[rv.ravel(), tv.ravel()]
            Jc = np.zeros((3, 6))
            for k in range(6):
                d = np.zeros(6); d[k] = 1e-6
                xp = x0 + d
                Jc[:, k] = (cam.center(xp[:3].reshape(3, 1), xp[3:].reshape(3, 1)) - C) / 1e-6
            cc = Jc @ cov @ Jc.T
            res.C_std_cm = (np.sqrt(np.clip(np.diag(cc), 0, None)) * 100).tolist()
            res.cov6 = cov.tolist()
        except np.linalg.LinAlgError:
            pass
    cam.result = res
    _warnings(cam, world, res, inl, roi)
    return res


def _warnings(cam, world, res: ExResult, inl, roi):
    W = res.warnings
    w, h = cam.size
    if res.C[2] < 0:
        W.append(("fail", "Camera nằm DƯỚI mặt sàn — nhiều khả năng trục X/Y bị đặt ngược (hệ tay trái) hoặc hướng "
                          "mũi tên marker nhập sai. Quy ước: đứng ở gốc nhìn theo +X thì +Y ở bên TRÁI."))
        res.level = "fail"
        return
    if res.n_inliers < 6:
        W.append(("fail", f"Chỉ có {res.n_inliers} điểm hợp lệ — cần ít nhất 6, nên có hàng chục điểm."))
    elif res.n_inliers < 15:
        W.append(("warn", f"Có {res.n_inliers} điểm hợp lệ — nên tìm thêm góc gạch để có hàng chục điểm."))
    out = res.n_total - res.n_inliers
    if out:
        W.append(("warn" if out > 0.2 * res.n_total else "ok", f"{out} điểm bị loại do lệch nhiều (click nhầm / ron bị che)."))
    if inl:
        uv = np.array([[c.u, c.v] for c in inl], np.float32)
        hull = cv2.convexHull(uv)
        frac = cv2.contourArea(hull) / (w * h)
        if frac < 0.15:
            W.append(("fail" if frac < 0.05 else "warn",
                      f"Điểm mốc chỉ phủ {frac * 100:.0f}% diện tích ảnh — cần trải rộng khắp vùng sàn nhìn thấy, cả gần lẫn xa."))
    if res.rms > 2.0:
        W.append(("warn", f"Sai số chiếu lại {res.rms:.2f} px khá lớn."))
    if not math.isnan(res.med_cm) and res.med_cm > 5:
        W.append(("warn", f"Sai số trên sàn trung vị {res.med_cm:.1f} cm."))
    if res.C_std_cm and max(res.C_std_cm) > 5:
        W.append(("warn", f"Vị trí camera chưa chắc chắn (± {max(res.C_std_cm):.1f} cm) — thêm điểm mốc ở xa và ở hai bên."))
    if roi is not None and roi.valid() and inl:
        cov = roi_point_coverage(cam, roi, inl)
        if cov is not None and cov < 0.5:
            W.append(("warn", f"Điểm mốc chỉ phủ {cov * 100:.0f}% phần vùng quay mà camera này nhìn thấy."))
    sw = cam.size_warning()
    if sw:
        W.append(("warn", sw))
    lv = [l for l, _ in W]
    res.level = "fail" if "fail" in lv else ("warn" if "warn" in lv else "ok")


def roi_point_coverage(cam, roi, inl):
    g = ground_points(roi, 0.25)
    _, vis = cam.project(g, margin=0)
    if vis.sum() == 0:
        return None
    pts = np.array([[c.X, c.Y] for c in inl if abs(c.Z) < 1e-6], np.float32)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts)
    gv = g[vis][:, :2].astype(np.float32)
    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in gv])
    return float(inside.mean())


# ==========================================================================
# Vùng quan tâm, độ phủ, độ bất định, ảnh ghép
# ==========================================================================
@dataclass
class ROI:
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    cell_cm: float = 2.5

    def valid(self):
        return self.x1 > self.x0 and self.y1 > self.y0

    def grid_shape(self):
        c = self.cell_cm / 100.0
        return int(round((self.y1 - self.y0) / c)), int(round((self.x1 - self.x0) / c))


def ground_points(roi: ROI, step, z=0.0):
    xs = np.arange(roi.x0 + step / 2, roi.x1, step)
    ys = np.arange(roi.y0 + step / 2, roi.y1, step)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]


def coverage_map(cams, roi: ROI, step=0.1, height=PERSON_H):
    """Số camera nhìn thấy cả chân (Z=0) và đầu (Z=height) tại mỗi ô."""
    xs = np.arange(roi.x0 + step / 2, roi.x1, step)
    ys = np.arange(roi.y0 + step / 2, roi.y1, step)
    X, Y = np.meshgrid(xs, ys)
    g0 = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    g1 = g0 + [0, 0, height]
    cnt = np.zeros(X.size, int)
    for c in cams:
        if not c.has_pose:
            continue
        _, a = c.project(g0)
        _, b = c.project(g1)
        cnt += (a & b)
    return xs, ys, cnt.reshape(X.shape)


def ground_uncertainty(cam: CameraEx, roi: ROI, step=0.2, samples=150, seed=0):
    """Độ bất định vị trí trên sàn (cm) do sai số tư thế camera — Monte Carlo trên hiệp phương sai."""
    r = cam.result
    if r is None or r.cov6 is None:
        return None
    xs = np.arange(roi.x0 + step / 2, roi.x1, step)
    ys = np.arange(roi.y0 + step / 2, roi.y1, step)
    X, Y = np.meshgrid(xs, ys)
    g = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    rv, tv = cam.pose()
    uv, vis = cam.project(g, rv, tv)
    out = np.full(X.size, np.nan)
    if vis.sum() == 0:
        return xs, ys, out.reshape(X.shape)
    rng = np.random.default_rng(seed)
    cov = np.array(r.cov6)
    x0 = np.r_[rv.ravel(), tv.ravel()]
    try:
        L = np.linalg.cholesky(cov + 1e-15 * np.eye(6))
    except np.linalg.LinAlgError:
        return None
    acc = []
    for _ in range(samples):
        x = x0 + L @ rng.standard_normal(6)
        p = cam.backproject(uv[vis], 0.0, x[:3].reshape(3, 1), x[3:].reshape(3, 1))
        acc.append(p[:, :2])
    A = np.stack(acc)
    sd = np.sqrt(np.nanvar(A[..., 0], axis=0) + np.nanvar(A[..., 1], axis=0)) * 100
    out[vis] = sd
    return xs, ys, out.reshape(X.shape)


def ground_mosaic(cams, roi: ROI, res_m=0.01, margin=0.5):
    """Ảnh ghép nhìn từ trên xuống mặt sàn (trung bình các camera nhìn thấy)."""
    x0, y0, x1, y1 = roi.x0 - margin, roi.y0 - margin, roi.x1 + margin, roi.y1 + margin
    W = int((x1 - x0) / res_m); H = int((y1 - y0) / res_m)
    scale = 1.0
    if W * H > 2_500_000:
        scale = math.sqrt(2_500_000 / (W * H))
        W, H = int(W * scale), int(H * scale)
    xs = np.linspace(x0, x1, W); ys = np.linspace(y1, y0, H)      # hàng đầu = Y lớn (Y hướng lên trên ảnh)
    X, Y = np.meshgrid(xs, ys)
    g = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    acc = np.zeros((H, W, 3), np.float64)
    n = np.zeros((H, W), np.float64)
    for c in cams:
        if not c.has_pose or c.image is None:
            continue
        uv, vis = c.project(g, margin=-2)
        mx = np.where(vis, uv[:, 0], -1).reshape(H, W).astype(np.float32)
        my = np.where(vis, uv[:, 1], -1).reshape(H, W).astype(np.float32)
        img = cv2.remap(c.image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = vis.reshape(H, W)
        acc[m] += img[m]
        n[m] += 1
    out = np.full((H, W, 3), 245, np.uint8)
    ok = n > 0
    out[ok] = (acc[ok] / n[ok, None]).astype(np.uint8)
    return out, (x0, y0, x1, y1), n


def footprint_polygon(cam, max_dist=30.0):
    return floor_footprint(cam, max_dist)


# ==========================================================================
# Điểm kiểm tra đa camera
# ==========================================================================
@dataclass
class TestPoint:
    name: str
    z: float = 0.0
    clicks: dict = field(default_factory=dict)      # tên camera -> [u, v]

    def estimates(self, cams):
        out = {}
        for c in cams:
            if c.name in self.clicks and c.has_pose:
                p = c.backproject(np.array([self.clicks[c.name]]), z=self.z)[0]
                if not np.isnan(p).any():
                    out[c.name] = p
        return out

    def summary(self, cams):
        est = self.estimates(cams)
        if not est:
            return None
        P = np.array(list(est.values()))
        mean = P.mean(axis=0)
        spread = float(np.max(np.linalg.norm(P[:, :2] - mean[:2], axis=1)) * 100) if len(P) > 1 else None
        pair = None
        if len(P) > 1:
            pair = float(max(np.linalg.norm(P[a, :2] - P[b, :2]) for a in range(len(P)) for b in range(a + 1, len(P))) * 100)
        return {"mean": mean, "n": len(P), "spread_cm": spread, "max_pair_cm": pair, "est": est}


def cylinder_points(center_xy, radius=0.25, height=PERSON_H, n=24):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.c_[center_xy[0] + radius * np.cos(t), center_xy[1] + radius * np.sin(t)]
    bot = np.c_[ring, np.zeros(n)]
    top = np.c_[ring, np.full(n, height)]
    return bot, top


# ==========================================================================
# Dự án
# ==========================================================================
def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class Project:
    def __init__(self):
        self.world = WorldSetup()
        self.cameras: list[CameraEx] = []
        self.roi = ROI()
        self.test_points: list[TestPoint] = []
        self.path = ""

    def camera(self, name):
        for c in self.cameras:
            if c.name == name:
                return c
        return None

    def unique_name(self, base="Cam"):
        k = len(self.cameras) + 1
        names = {c.name for c in self.cameras}
        while f"{base}{k}" in names:
            k += 1
        return f"{base}{k}"

    def save(self, path):
        base = os.path.dirname(os.path.abspath(path))
        d = {"version": 1, "world": self.world.to_dict(), "roi": asdict(self.roi),
             "cameras": [c.to_dict(base) for c in self.cameras],
             "test_points": [asdict(t) for t in self.test_points]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1, default=_json_default)
        self.path = path

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
        p = Project()
        p.world = WorldSetup.from_dict(d.get("world", {}))
        p.roi = ROI(**d.get("roi", {}))
        errors = []
        for cd in d.get("cameras", []):
            cam, err = CameraEx.from_dict(cd, base)
            p.cameras.append(cam)
            errors += err
        p.test_points = [TestPoint(**t) for t in d.get("test_points", [])]
        p.path = path
        return p, errors

    def ground_corr_bbox(self):
        pts = [(c.X, c.Y) for cam in self.cameras for c in cam.corrs if c.enabled and abs(c.Z) < 1e-6]
        if not pts:
            return None
        P = np.array(pts)
        return P.min(axis=0), P.max(axis=0)


# ==========================================================================
# Xuất
# ==========================================================================
def _xml_mat(name, M):
    M = np.asarray(M, np.float64)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    data = " ".join(repr(float(v)) for v in M.ravel())
    return (f'<{name} type_id="opencv-matrix">\n  <rows>{M.shape[0]}</rows>\n  <cols>{M.shape[1]}</cols>\n'
            f'  <dt>d</dt>\n  <data>{data}</data></{name}>\n')


def write_xml(path, mats: dict):
    body = "".join(_xml_mat(k, v) for k, v in mats.items())
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n<opencv_storage>\n' + body + "</opencv_storage>\n")


def camera_export_dict(cam: CameraEx):
    rv, tv = cam.pose()
    R = cv2.Rodrigues(rv)[0]
    P = cam.K @ np.hstack([R, tv])
    H = cam.K @ np.column_stack([R[:, 0], R[:, 1], tv.ravel()])      # sàn (X, Y, 1) → ảnh đã khử méo
    r = cam.result
    return {"name": cam.name, "image_size": list(cam.size), "model": cam.model,
            "camera_matrix": cam.K.tolist(), "dist_coeffs": cam.dist.tolist(),
            "rvec": rv.ravel().tolist(), "tvec_m": tv.ravel().tolist(), "R": R.tolist(),
            "camera_center_m": cam.center().tolist(), "P": P.tolist(), "H_ground_to_undistorted_image": H.tolist(),
            "rms_px": r.rms, "ground_err_median_cm": r.med_cm, "n_points": r.n_inliers,
            "camera_center_std_cm": r.C_std_cm}


def export_project(proj: Project, out_dir: str, unit: str = "m"):
    os.makedirs(out_dir, exist_ok=True)
    sc = 100.0 if unit == "cm" else 1.0
    files = []
    cams = [c for c in proj.cameras if c.has_pose and c.K is not None]
    for c in cams:
        rv, tv = c.pose()
        p1 = os.path.join(out_dir, f"intr_{c.name}.xml")
        p2 = os.path.join(out_dir, f"extr_{c.name}.xml")
        write_xml(p1, {"camera_matrix": c.K, "distortion_coefficients": c.dist})
        write_xml(p2, {"rvec": rv, "tvec": tv * sc})
        files += [p1, p2]
    d = {"units": {"world": "m", "extr_xml_tvec": unit},
         "world": proj.world.to_dict(),
         "roi": {**asdict(proj.roi), "grid_shape_rows_cols": list(proj.roi.grid_shape()) if proj.roi.valid() else None,
                 "note": "ô (row, col) có tâm tại X = x0 + (col+0.5)*cell, Y = y0 + (row+0.5)*cell"},
         "cameras": [camera_export_dict(c) for c in cams]}
    pj = os.path.join(out_dir, "calibration.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, default=_json_default)
    files.append(pj)
    return files
