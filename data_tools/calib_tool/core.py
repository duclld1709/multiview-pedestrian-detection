"""
Lõi xử lý của ChArUco Calibrator: cấu hình bảng, xuất PDF đúng kích thước,
đọc ảnh, nhận diện, đánh giá chất lượng, hiệu chỉnh và kiểm tra độ tin cậy.
Không phụ thuộc giao diện — có thể import dùng riêng.
"""
from __future__ import annotations

import json
import math
import os
import random
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Từ điển ArUco
# --------------------------------------------------------------------------
DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11",
]
DICT_NAMES = [n for n in DICT_NAMES if hasattr(cv2.aruco, n)]


def get_dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def dictionary_capacity(name: str) -> int:
    return int(get_dictionary(name).bytesList.shape[0])


# --------------------------------------------------------------------------
# Khổ giấy (mm, dọc: rộng x cao)
# --------------------------------------------------------------------------
PAPERS = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
    "Letter": (215.9, 279.4),
}
MARGIN_SIDE_MM = 12.0      # lề trái/phải tối thiểu
TEXT_BAND_MM = 20.0        # dải chữ phía trên
RULER_BAND_MM = 22.0       # dải thước phía dưới


# --------------------------------------------------------------------------
# Cấu hình bảng
# --------------------------------------------------------------------------
@dataclass
class BoardConfig:
    squares_x: int = 7
    squares_y: int = 5
    square_mm: float = 30.0
    marker_mm: float = 22.5
    dictionary: str = "DICT_5X5_100"
    legacy: bool = False
    paper: str = "A4"
    landscape: bool = True

    # ---- thông tin suy ra ----
    @property
    def n_markers(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def board_size_mm(self) -> tuple[float, float]:
        return self.squares_x * self.square_mm, self.squares_y * self.square_mm

    def paper_size_mm(self) -> tuple[float, float]:
        w, h = PAPERS[self.paper]
        return (h, w) if self.landscape else (w, h)

    def validate(self) -> list[str]:
        """Trả về danh sách lỗi (rỗng = hợp lệ)."""
        errs = []
        if self.squares_x < 3 or self.squares_y < 3:
            errs.append("Bảng cần ít nhất 3 × 3 ô.")
        if self.marker_mm >= self.square_mm:
            errs.append("Cạnh marker phải nhỏ hơn cạnh ô.")
        if self.marker_mm < 0.4 * self.square_mm:
            errs.append("Marker quá nhỏ so với ô (nên ≥ 0.6 × cạnh ô).")
        cap = dictionary_capacity(self.dictionary)
        if self.n_markers > cap:
            errs.append(f"Bảng cần {self.n_markers} marker nhưng {self.dictionary} chỉ có {cap}.")
        return errs

    def fits_paper(self) -> tuple[bool, str]:
        pw, ph = self.paper_size_mm()
        bw, bh = self.board_size_mm
        avail_w = pw - 2 * MARGIN_SIDE_MM
        avail_h = ph - TEXT_BAND_MM - RULER_BAND_MM
        if bw <= avail_w and bh <= avail_h:
            return True, ""
        return False, (f"Bảng {bw:.0f} × {bh:.0f} mm không vừa vùng in "
                       f"{avail_w:.0f} × {avail_h:.0f} mm của {self.paper} "
                       f"{'ngang' if self.landscape else 'dọc'}.")

    def max_square_for_paper(self, step_mm: float = 5.0) -> float:
        pw, ph = self.paper_size_mm()
        avail_w = pw - 2 * MARGIN_SIDE_MM
        avail_h = ph - TEXT_BAND_MM - RULER_BAND_MM
        s = min(avail_w / self.squares_x, avail_h / self.squares_y)
        return max(step_mm, math.floor(s / step_mm) * step_mm)

    # ---- tạo đối tượng OpenCV ----
    def make_board(self, unit: str = "m"):
        scale = 0.001 if unit == "m" else 1.0
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_mm * scale, self.marker_mm * scale,
            get_dictionary(self.dictionary))
        board.setLegacyPattern(self.legacy)
        return board

    def describe(self) -> str:
        bw, bh = self.board_size_mm
        return (f"ChArUco {self.squares_x}×{self.squares_y} ô · ô {self.square_mm:g} mm · "
                f"marker {self.marker_mm:g} mm · {self.dictionary} · bảng {bw:g} × {bh:g} mm")

    # ---- lưu / mở ----
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BoardConfig":
        fields = BoardConfig.__dataclass_fields__.keys()
        return BoardConfig(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "BoardConfig":
        with open(path, "r", encoding="utf-8") as f:
            return BoardConfig.from_dict(json.load(f))


# --------------------------------------------------------------------------
# Hình học bảng dạng vector (mm, gốc trên-trái, trục y hướng xuống)
# --------------------------------------------------------------------------
def board_geometry(cfg: BoardConfig):
    """Trả về (danh sách ô đen [x,y,w,h], danh sách marker [(x, y, cell_mm, ma trận bit)])."""
    board = cfg.make_board(unit="mm")
    dictionary = get_dictionary(cfg.dictionary)
    n_bits = dictionary.markerSize
    markers = []
    centers = []
    for mid, pts in zip(board.getIds().ravel(), board.getObjPoints()):
        p = np.asarray(pts).reshape(-1, 3)[:, :2]
        x0, y0 = p.min(axis=0)
        side = float(p[:, 0].max() - x0)
        img = cv2.aruco.generateImageMarker(dictionary, int(mid), n_bits + 2, borderBits=1)
        markers.append((float(x0), float(y0), side / (n_bits + 2), img < 128))
        centers.append(p.mean(axis=0))
    centers = np.array(centers)
    s = cfg.square_mm
    black = []
    for r in range(cfg.squares_y):
        for c in range(cfg.squares_x):
            cx, cy = (c + 0.5) * s, (r + 0.5) * s
            has_marker = np.any((np.abs(centers[:, 0] - cx) < s / 2) & (np.abs(centers[:, 1] - cy) < s / 2))
            if not has_marker:
                black.append((c * s, r * s, s, s))
    return black, markers


def render_board_image(cfg: BoardConfig, px_per_mm: float = 4.0, with_paper: bool = True) -> np.ndarray:
    """Ảnh xem trước (gray uint8) của bảng trên giấy, vẽ từ cùng hình học vector với PDF."""
    black, markers = board_geometry(cfg)
    bw, bh = cfg.board_size_mm
    if with_paper:
        pw, ph = cfg.paper_size_mm()
        ox, oy = (pw - bw) / 2, TEXT_BAND_MM + ((ph - TEXT_BAND_MM - RULER_BAND_MM) - bh) / 2
    else:
        pw, ph, ox, oy = bw, bh, 0.0, 0.0
    W, H = int(round(pw * px_per_mm)), int(round(ph * px_per_mm))
    img = np.full((H, W), 255, np.uint8)

    def rect(x, y, w, h):
        x0, y0 = int(round((ox + x) * px_per_mm)), int(round((oy + y) * px_per_mm))
        x1, y1 = int(round((ox + x + w) * px_per_mm)), int(round((oy + y + h) * px_per_mm))
        img[y0:y1, x0:x1] = 0

    for (x, y, w, h) in black:
        rect(x, y, w, h)
    for (x0, y0, cell, bits) in markers:
        for i in range(bits.shape[0]):
            for j in range(bits.shape[1]):
                if bits[i, j]:
                    rect(x0 + j * cell, y0 + i * cell, cell, cell)
    if with_paper:
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), 200, 1)
    return img


# --------------------------------------------------------------------------
# Font cho PDF (cần font có dấu tiếng Việt)
# --------------------------------------------------------------------------
_FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for reg, bold in _FONT_CANDIDATES:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("UIFont", reg))
                pdfmetrics.registerFont(TTFont("UIFontB", bold))
                return "UIFont", "UIFontB", True
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold", False


def _ascii(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D").replace("×", "x").replace("·", "-").replace("—", "-").replace("←", "<-")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def generate_pdf(cfg: BoardConfig, path: str):
    """Xuất PDF vector đúng kích thước thật, kèm thước kiểm tra 100 mm."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    errs = cfg.validate()
    if errs:
        raise ValueError("\n".join(errs))
    ok, msg = cfg.fits_paper()
    if not ok:
        raise ValueError(msg)

    font, font_b, unicode_ok = _register_fonts()
    txt = (lambda s: s) if unicode_ok else _ascii

    pw, ph = cfg.paper_size_mm()
    bw, bh = cfg.board_size_mm
    ox = (pw - bw) / 2
    oy = TEXT_BAND_MM + ((ph - TEXT_BAND_MM - RULER_BAND_MM) - bh) / 2   # từ mép trên

    c = canvas.Canvas(path, pagesize=(pw * mm, ph * mm))
    c.setTitle(_ascii(cfg.describe()))
    c.setFillGray(0)

    def rect(x, y, w, h):  # toạ độ bảng (mm, y xuống) -> PDF (pt, y lên)
        c.rect((ox + x) * mm, (ph - oy - y - h) * mm, w * mm + 0.01, h * mm + 0.01, stroke=0, fill=1)

    black, markers = board_geometry(cfg)
    for (x, y, w, h) in black:
        rect(x, y, w, h)
    for (x0, y0, cell, bits) in markers:
        for i in range(bits.shape[0]):
            j = 0
            while j < bits.shape[1]:
                if bits[i, j]:
                    s = j
                    while j < bits.shape[1] and bits[i, j]:
                        j += 1
                    rect(x0 + s * cell, y0 + i * cell, (j - s) * cell, cell)
                else:
                    j += 1

    left = ox * mm
    c.setFont(font_b, 10)
    c.drawString(left, (ph - 10) * mm, txt("IN 100% / ACTUAL SIZE — KHÔNG 'FIT TO PAGE' — "
                                            f"giấy {cfg.paper} {'ngang' if cfg.landscape else 'dọc'}"))
    c.setFont(font, 8.5)
    c.drawString(left, (ph - 15) * mm, txt(cfg.describe() + (" · legacy" if cfg.legacy else "")))

    ry = (ph - oy - bh - 12) * mm
    c.setLineWidth(0.6)
    c.setStrokeGray(0)
    c.line(left, ry, left + 100 * mm, ry)
    for i in range(11):
        t = (3 if i % 5 == 0 else 1.8) * mm
        c.line(left + i * 10 * mm, ry, left + i * 10 * mm, ry + t)
    c.drawString(left + 103 * mm, ry - 1 * mm, txt("← vạch này phải dài đúng 100 mm. Sai thì in lại."))
    c.showPage()
    c.save()


# --------------------------------------------------------------------------
# Đọc ảnh (hỗ trợ đường dẫn tiếng Việt trên Windows + xoay theo EXIF)
# --------------------------------------------------------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_image(path: str, gray: bool = False) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("L" if gray else "RGB")
        return np.array(im)


def save_image(path: str, rgb: np.ndarray):
    Image.fromarray(rgb).save(path, quality=95)


# --------------------------------------------------------------------------
# Kết quả từng ảnh
# --------------------------------------------------------------------------
SHARP_BAD = 60.0     # dưới mức này: ảnh mờ (đo trên vùng bảng)
SHARP_WARN = 150.0
AREA_WARN = 2.0      # % diện tích ảnh
TILT_FRONTAL = 12.0  # độ


@dataclass
class ImageResult:
    path: str
    width: int = 0
    height: int = 0
    n_markers: int = 0
    n_corners: int = 0
    corners: Optional[np.ndarray] = None    # (N,1,2) float32
    ids: Optional[np.ndarray] = None        # (N,1) int32
    sharpness: float = 0.0
    area_pct: float = 0.0
    tilt_deg: Optional[float] = None
    tilt_dir: str = ""
    roll_deg: Optional[float] = None
    tilt_x: Optional[float] = None   # độ, dương = mép phải xa hơn
    tilt_y: Optional[float] = None   # độ, dương = mép dưới xa hơn
    reproj_err: Optional[float] = None
    used: bool = True
    excluded_auto: bool = False
    board_sig: tuple = ()                   # thông số bảng lúc nhận diện
    error: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def detected(self) -> bool:
        return self.ids is not None and self.n_corners > 0

    def issues(self, min_corners: int = 6) -> list[tuple[str, str]]:
        """Danh sách (mức, nội dung): mức = 'fail' | 'warn'."""
        out = []
        if self.error:
            return [("fail", self.error)]
        if self.n_corners < min_corners:
            out.append(("fail", f"chỉ thấy {self.n_corners} góc"))
            return out
        if self.sharpness < SHARP_BAD:
            out.append(("fail", "ảnh mờ"))
        elif self.sharpness < SHARP_WARN:
            out.append(("warn", "hơi mờ"))
        if self.area_pct < AREA_WARN:
            out.append(("warn", "bảng quá nhỏ trong khung"))
        if self.excluded_auto:
            out.append(("warn", "sai số lớn, đã tự loại"))
        return out

    def status_text(self) -> str:
        iss = self.issues()
        if not iss:
            return "Tốt"
        return "; ".join(t for _, t in iss)

    def status_level(self) -> str:
        levels = [lv for lv, _ in self.issues()]
        return "fail" if "fail" in levels else ("warn" if "warn" in levels else "ok")


def board_signature(cfg: "BoardConfig") -> tuple:
    """Các thông số ảnh hưởng tới kết quả nhận diện (không gồm kích thước tuyệt đối)."""
    return (cfg.squares_x, cfg.squares_y, cfg.dictionary, bool(cfg.legacy),
            round(cfg.marker_mm / cfg.square_mm, 4))


def make_detector(board):
    cp = cv2.aruco.CharucoParameters()
    try:
        cp.tryRefineMarkers = True
    except Exception:
        pass
    dp = cv2.aruco.DetectorParameters()
    return cv2.aruco.CharucoDetector(board, cp, dp)


def board_sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
    """Phương sai Laplacian trên vùng bao các góc — đo độ nét của chính bảng."""
    pts = corners.reshape(-1, 2).astype(np.int32)
    hull = cv2.convexHull(pts)
    x, y, w, h = cv2.boundingRect(hull)
    if w < 8 or h < 8:
        return 0.0
    roi = gray[y:y + h, x:x + w]
    mask = np.zeros_like(roi, np.uint8)
    cv2.fillConvexPoly(mask, hull - [x, y], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    vals = lap[mask > 0]
    return float(vals.var()) if vals.size > 50 else 0.0


def detect_image(path: str, board, detector=None, sig: tuple = ()) -> ImageResult:
    res = ImageResult(path=path, board_sig=sig)
    try:
        gray = load_image(path, gray=True)
    except Exception as e:
        res.error = f"không đọc được ảnh ({e})"
        return res
    res.height, res.width = gray.shape[:2]
    if detector is None:
        detector = make_detector(board)
    ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)
    res.n_markers = 0 if m_ids is None else len(m_ids)
    if ch_ids is None or len(ch_ids) == 0:
        return res
    res.corners = ch_corners.astype(np.float32)
    res.ids = ch_ids.astype(np.int32)
    res.n_corners = len(ch_ids)
    pts = res.corners.reshape(-1, 2)
    if res.n_corners >= 3:
        hull = cv2.convexHull(pts)
        res.area_pct = 100.0 * cv2.contourArea(hull) / (res.width * res.height)
    res.sharpness = board_sharpness(gray, res.corners)
    return res


# --------------------------------------------------------------------------
# Góc ngả của bảng (cần K: dùng kết quả hiệu chỉnh hoặc ước lượng)
# --------------------------------------------------------------------------
def estimate_K(width: int, height: int) -> np.ndarray:
    f = 0.8 * max(width, height)   # gần đúng cho camera điện thoại / webcam thường
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)


def update_poses(results: list[ImageResult], board, K=None, dist=None, fisheye=False):
    for r in results:
        r.tilt_deg, r.tilt_dir, r.roll_deg, r.tilt_x, r.tilt_y = None, "", None, None, None
        if not r.detected or r.n_corners < 6:
            continue
        Ki = K if K is not None else estimate_K(r.width, r.height)
        di = dist if dist is not None else np.zeros(5)
        obj, img = board.matchImagePoints(r.corners, r.ids)
        if obj is None or len(obj) < 6:
            continue
        try:
            if fisheye:
                und = cv2.fisheye.undistortPoints(img.reshape(1, -1, 2).astype(np.float64), Ki, di, P=Ki)
                ok, rvec, tvec = cv2.solvePnP(obj, und.reshape(-1, 1, 2), Ki, None)
            else:
                ok, rvec, tvec = cv2.solvePnP(obj, img, Ki, di)
        except cv2.error:
            continue
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        n = R[:, 2]                         # pháp tuyến bảng trong hệ camera
        if n[2] < 0:
            n = -n
        r.tilt_deg = float(math.degrees(math.acos(min(1.0, abs(n[2])))))
        r.roll_deg = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))
        # độ dốc chiều sâu theo hướng ngang/dọc của ảnh
        gx, gy = -n[0] / n[2], -n[1] / n[2]
        r.tilt_x, r.tilt_y = float(math.degrees(math.atan(gx))), float(math.degrees(math.atan(gy)))
        if r.tilt_deg < TILT_FRONTAL:
            r.tilt_dir = "thẳng mặt"
        elif abs(gx) >= abs(gy):
            r.tilt_dir = "mép phải xa hơn" if gx > 0 else "mép trái xa hơn"
        else:
            r.tilt_dir = "mép dưới xa hơn" if gy > 0 else "mép trên xa hơn"


# --------------------------------------------------------------------------
# Độ phủ khung hình
# --------------------------------------------------------------------------
CELL_NAMES = [["trên-trái", "trên-giữa", "trên-phải"],
              ["giữa-trái", "chính giữa", "giữa-phải"],
              ["dưới-trái", "dưới-giữa", "dưới-phải"]]


def coverage_grid(results: list[ImageResult], n: int = 3) -> np.ndarray:
    grid = np.zeros((n, n), int)
    for r in results:
        if not (r.used and r.detected):
            continue
        for x, y in r.corners.reshape(-1, 2):
            i = min(int(y / r.height * n), n - 1)
            j = min(int(x / r.width * n), n - 1)
            grid[i, j] += 1
    return grid


def tilt_summary(results: list[ImageResult]) -> dict:
    keys = ["thẳng mặt", "mép trái xa hơn", "mép phải xa hơn", "mép trên xa hơn", "mép dưới xa hơn"]
    out = {k: 0 for k in keys}
    for r in results:
        if r.used and r.tilt_dir in out:
            out[r.tilt_dir] += 1
    return out


def dataset_suggestions(results: list[ImageResult]) -> list[str]:
    used = [r for r in results if r.used and r.detected and r.n_corners >= 6]
    tips = []
    if not used:
        return ["Chưa có ảnh nào nhận được bảng."]
    if len(used) < 15:
        tips.append(f"Mới có {len(used)} ảnh dùng được — nên có 20–30 ảnh.")
    grid = coverage_grid(results)
    total = grid.sum()
    weak = [CELL_NAMES[i][j] for i in range(3) for j in range(3) if grid[i, j] < 0.03 * total]
    if weak:
        tips.append("Thiếu dữ liệu ở vùng: " + ", ".join(weak) +
                    ". Đưa bảng đến các vùng này (giữ nguyên khoảng cách đã lấy nét).")
    corners_missing = [CELL_NAMES[i][j] for i, j in [(0, 0), (0, 2), (2, 0), (2, 2)] if grid[i, j] == 0]
    if corners_missing:
        tips.append("4 góc ảnh là nơi méo mạnh nhất — hiện chưa có điểm nào ở: " + ", ".join(corners_missing) + ".")
    ts = tilt_summary(results)
    n_tilt = sum(v for k, v in ts.items() if k != "thẳng mặt")
    if n_tilt < 0.5 * len(used):
        tips.append(f"Chỉ {n_tilt}/{len(used)} ảnh có ngả bảng (≥ {TILT_FRONTAL:.0f}°). Ngả bảng ra trước/sau "
                    "và xoay trái/phải khoảng 20–40° — không phải xoay trong mặt phẳng.")
    for k in ["mép trái xa hơn", "mép phải xa hơn", "mép trên xa hơn", "mép dưới xa hơn"]:
        if ts[k] < 2:
            tips.append(f"Cần thêm ảnh có {k} (hiện {ts[k]}).")
    blurry = [r.name for r in used if r.sharpness < SHARP_BAD]
    if blurry:
        tips.append(f"{len(blurry)} ảnh bị mờ — nên bỏ: " + ", ".join(blurry[:8]) + ("…" if len(blurry) > 8 else ""))
    small = [r for r in used if r.area_pct < AREA_WARN]
    if len(small) > 0.5 * len(used):
        tips.append("Bảng chiếm quá ít khung hình ở phần lớn ảnh. Nếu không thể đến gần hơn (vì vùng nét), "
                    "hãy in bảng khổ lớn hơn (A3/A2).")
    sizes = {(r.width, r.height) for r in used}
    if len(sizes) > 1:
        tips.append("Các ảnh có độ phân giải khác nhau: " + ", ".join(f"{w}×{h}" for w, h in sizes) +
                    ". Chỉ dùng ảnh cùng một chế độ chụp.")
    if not tips:
        tips.append("Bộ ảnh đạt yêu cầu về độ phủ, góc ngả và độ nét.")
    return tips


# --------------------------------------------------------------------------
# Hiệu chỉnh
# --------------------------------------------------------------------------
MODELS = {
    "pinhole_k3": "Pinhole – k1, k2, p1, p2 (k3 = 0) — khuyên dùng",
    "pinhole5": "Pinhole – 5 hệ số (k1, k2, p1, p2, k3)",
    "rational": "Rational – 8 hệ số (ống kính méo mạnh)",
    "fisheye": "Fisheye (ống kính mắt cá / góc siêu rộng)",
}


@dataclass
class CalibOptions:
    model: str = "pinhole_k3"
    fix_aspect: bool = False
    fix_principal: bool = False
    zero_tangent: bool = False
    auto_reject: bool = True
    reject_factor: float = 2.5
    min_keep: int = 8
    stability_runs: int = 20
    min_corners: int = 6


@dataclass
class Check:
    level: str     # ok | warn | fail
    text: str


@dataclass
class CalibResult:
    image_size: tuple[int, int]
    model: str
    K: np.ndarray
    dist: np.ndarray
    rms: float
    used_names: list[str]
    per_view: dict
    std_intrinsics: Optional[np.ndarray] = None   # fx fy cx cy k1 k2 p1 p2 k3 ...
    stability: dict = field(default_factory=dict)
    holdout_err: Optional[float] = None
    rejected: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    verdict: str = ""
    verdict_level: str = ""
    board: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    residuals: Optional[np.ndarray] = None        # (M,4): x, y, dx, dy

    @property
    def fx(self): return float(self.K[0, 0])
    @property
    def fy(self): return float(self.K[1, 1])
    @property
    def cx(self): return float(self.K[0, 2])
    @property
    def cy(self): return float(self.K[1, 2])

    def dist_names(self) -> list[str]:
        if self.model == "fisheye":
            return ["k1", "k2", "k3", "k4"]
        names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"]
        return names[:len(self.dist.ravel())]


def _fe(name: str) -> int:
    """Hằng số fisheye: OpenCV 4 đặt trong cv2.fisheye, OpenCV 5 đặt trong cv2 (giá trị khác nhau)."""
    v = getattr(cv2.fisheye, name, None)
    return int(v) if v is not None else int(getattr(cv2, name))


def _flags(opt: CalibOptions) -> int:
    if opt.model == "fisheye":
        f = _fe("CALIB_RECOMPUTE_EXTRINSIC") | _fe("CALIB_FIX_SKEW")
        if opt.fix_principal:
            f |= _fe("CALIB_FIX_PRINCIPAL_POINT")
        return f
    f = 0
    if opt.model == "pinhole_k3":
        f |= cv2.CALIB_FIX_K3
    if opt.model == "rational":
        f |= cv2.CALIB_RATIONAL_MODEL
    if opt.fix_aspect:
        f |= cv2.CALIB_FIX_ASPECT_RATIO
    if opt.fix_principal:
        f |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if opt.zero_tangent:
        f |= cv2.CALIB_ZERO_TANGENT_DIST
    return f


def _view_errors(obj_list, img_list, rvecs, tvecs, K, dist, fisheye):
    errs, resid = [], []
    for o, i, r, t in zip(obj_list, img_list, rvecs, tvecs):
        if fisheye:
            proj, _ = cv2.fisheye.projectPoints(o.reshape(1, -1, 3).astype(np.float64), r, t, K, dist)
        else:
            proj, _ = cv2.projectPoints(o, r, t, K, dist)
        d = proj.reshape(-1, 2) - i.reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))))
        resid.append(np.hstack([i.reshape(-1, 2), d]))
    return errs, resid


def _calib_once(obj_list, img_list, size, opt: CalibOptions, extended=False):
    flags = _flags(opt)
    if opt.model == "fisheye":
        K = np.eye(3)
        D = np.zeros((4, 1))
        o = [x.reshape(1, -1, 3).astype(np.float64) for x in obj_list]
        i = [x.reshape(1, -1, 2).astype(np.float64) for x in img_list]
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(o, i, size, K, D, flags=flags, criteria=crit)
        return rms, K, D, rvecs, tvecs, None
    K0 = np.eye(3)
    if extended:
        rms, K, D, rvecs, tvecs, std_in, _, _ = cv2.calibrateCameraExtended(
            obj_list, img_list, size, K0, None, flags=flags)
        return rms, K, D, rvecs, tvecs, std_in.ravel()
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_list, img_list, size, K0, None, flags=flags)
    return rms, K, D, rvecs, tvecs, None


def _radial_monotonic(K, dist, size, model) -> bool:
    """Kiểm tra hàm méo xuyên tâm còn đơn điệu đến góc ảnh (nếu không: mô hình 'bịa' ở rìa)."""
    if model == "fisheye":
        return True
    d = np.zeros(8)
    dd = dist.ravel()
    d[:min(8, len(dd))] = dd[:8]
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    w, h = size
    r_corner = max(math.hypot((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1])
                   for x, y in [(0, 0), (w, 0), (0, h), (w, h)])
    rs = np.linspace(0, 2.5 * r_corner, 2000)
    r2 = rs ** 2
    rd = rs * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    # đơn điệu cho tới khi bán kính méo vượt quá góc ảnh
    reach = np.nonzero(rd >= r_corner)[0]
    end = reach[0] if len(reach) else len(rs)
    if len(reach) == 0:
        return False
    return bool(np.all(np.diff(rd[:end + 1]) > 0))


def calibrate(results: list[ImageResult], cfg: BoardConfig, opt: CalibOptions,
              progress: Optional[Callable[[str, float], None]] = None) -> CalibResult:
    progress = progress or (lambda msg, f: None)
    board = cfg.make_board(unit="m")
    fisheye = opt.model == "fisheye"

    for r in results:
        r.excluded_auto = False
        r.reproj_err = None
    sig = board_signature(cfg)
    if any(r.board_sig and r.board_sig != sig for r in results if r.used):
        raise ValueError("Ảnh được nhận diện với thông số bảng cũ (số ô, từ điển hoặc tỉ lệ marker đã đổi). "
                         "Cần phân tích lại ảnh trước khi hiệu chỉnh.")
    cand = [r for r in results if r.used and r.detected and r.n_corners >= opt.min_corners and not r.error]
    if not cand:
        raise ValueError("Không có ảnh nào đủ góc để hiệu chỉnh.")
    sizes = {}
    for r in cand:
        sizes[(r.width, r.height)] = sizes.get((r.width, r.height), 0) + 1
    size = max(sizes, key=sizes.get)
    skipped_size = [r for r in cand if (r.width, r.height) != size]
    cand = [r for r in cand if (r.width, r.height) == size]
    if len(cand) < 4:
        raise ValueError(f"Chỉ có {len(cand)} ảnh hợp lệ — cần ít nhất 4 (khuyến nghị 20–30).")

    data = []
    for r in cand:
        obj, img = board.matchImagePoints(r.corners, r.ids)
        if obj is not None and len(obj) >= opt.min_corners:
            data.append((r, obj.astype(np.float32), img.astype(np.float32)))

    # ---- vòng hiệu chỉnh + tự loại ảnh xấu ----
    progress("Đang hiệu chỉnh…", 0.05)
    active = list(data)
    rejected = []
    while True:
        objs = [d[1] for d in active]
        imgs = [d[2] for d in active]
        rms, K, D, rvecs, tvecs, std_in = _calib_once(objs, imgs, size, opt, extended=not fisheye)
        errs, resid = _view_errors(objs, imgs, rvecs, tvecs, K, D, fisheye)
        if not opt.auto_reject or len(active) <= opt.min_keep:
            break
        med = float(np.median(errs))
        thr = max(opt.reject_factor * med, med + 0.3)
        worst = int(np.argmax(errs))
        if errs[worst] <= thr:
            break
        rejected.append(active[worst][0])
        active.pop(worst)
    for r in rejected:
        r.excluded_auto = True
    per_view = {}
    for (r, _, _), e in zip(active, errs):
        r.reproj_err = e
        per_view[r.name] = e
    # sai số các ảnh bị loại (tính với K cuối, solvePnP riêng)
    for r in rejected:
        obj, img = board.matchImagePoints(r.corners, r.ids)
        e = _holdout_error(obj, img, K, D, fisheye)
        r.reproj_err = e

    res = CalibResult(image_size=size, model=opt.model, K=K, dist=D.ravel(), rms=float(rms),
                      used_names=[d[0].name for d in active], per_view=per_view,
                      std_intrinsics=std_in, rejected=[r.name for r in rejected],
                      board=cfg.to_dict(), options=asdict(opt),
                      residuals=np.vstack(resid))

    # ---- độ ổn định: hiệu chỉnh trên tập con, đo sai số ảnh để riêng ----
    runs = opt.stability_runs if len(active) >= 8 else 0
    fxs, fys, cxs, cys, k1s, hold = [], [], [], [], [], []
    rng = random.Random(0)
    for k in range(runs):
        progress(f"Kiểm tra độ ổn định {k + 1}/{runs}…", 0.1 + 0.85 * k / max(1, runs))
        idx = list(range(len(active)))
        rng.shuffle(idx)
        n_tr = max(6, int(round(0.7 * len(idx))))
        tr, te = idx[:n_tr], idx[n_tr:]
        try:
            _, Ks, Ds, _, _, _ = _calib_once([active[i][1] for i in tr], [active[i][2] for i in tr], size, opt)
        except cv2.error:
            continue
        fxs.append(Ks[0, 0]); fys.append(Ks[1, 1]); cxs.append(Ks[0, 2]); cys.append(Ks[1, 2])
        k1s.append(float(Ds.ravel()[0]))
        for i in te:
            e = _holdout_error(active[i][1], active[i][2], Ks, Ds, fisheye)
            if e is not None:
                hold.append(e)
    if fxs:
        res.stability = {"runs": len(fxs),
                         "fx_std": float(np.std(fxs)), "fy_std": float(np.std(fys)),
                         "cx_std": float(np.std(cxs)), "cy_std": float(np.std(cys)),
                         "k1_std": float(np.std(k1s))}
        res.holdout_err = float(np.sqrt(np.mean(np.square(hold)))) if hold else None

    progress("Đánh giá kết quả…", 0.97)
    update_poses(results, board, K, D, fisheye)
    _evaluate(res, results, skipped_size)
    progress("Xong", 1.0)
    return res


def _holdout_error(obj, img, K, D, fisheye) -> Optional[float]:
    if obj is None or len(obj) < 4:
        return None
    try:
        if fisheye:
            und = cv2.fisheye.undistortPoints(img.reshape(1, -1, 2).astype(np.float64), K, D, P=K)
            ok, rv, tv = cv2.solvePnP(obj, und.reshape(-1, 1, 2), K, None)
            if not ok:
                return None
            proj, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3).astype(np.float64), rv, tv, K, D)
        else:
            ok, rv, tv = cv2.solvePnP(obj, img, K, D)
            if not ok:
                return None
            proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
    except cv2.error:
        return None
    d = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


def _evaluate(res: CalibResult, results: list[ImageResult], skipped_size):
    C = res.checks
    w, h = res.image_size
    n = len(res.used_names)
    data_problem = False

    # 1. RMS
    if res.rms <= 0.5:
        C.append(Check("ok", f"Sai số chiếu lại RMS = {res.rms:.3f} px (tốt, < 0.5 px)."))
    elif res.rms <= 1.0:
        C.append(Check("warn", f"Sai số chiếu lại RMS = {res.rms:.3f} px (chấp nhận được, nên < 0.5 px)."))
    else:
        C.append(Check("fail", f"Sai số chiếu lại RMS = {res.rms:.3f} px (quá lớn, > 1 px). "
                               "Thường do ảnh mờ, bảng không phẳng hoặc nhập sai kích thước ô."))

    # 2. Số ảnh
    if n >= 15:
        C.append(Check("ok", f"Dùng {n} ảnh."))
    else:
        C.append(Check("warn", f"Chỉ dùng {n} ảnh — nên có 20–30 ảnh."))
        data_problem = True
    if res.rejected:
        frac = len(res.rejected) / (n + len(res.rejected))
        C.append(Check("ok" if frac <= 0.15 else "warn",
                       f"Đã tự loại {len(res.rejected)} ảnh sai số lớn: " + ", ".join(res.rejected)))
    if skipped_size:
        C.append(Check("warn", f"Bỏ qua {len(skipped_size)} ảnh khác độ phân giải {w}×{h}."))

    used_set = set(res.used_names)
    blurry = [r.name for r in results if r.name in used_set and r.sharpness < SHARP_BAD]
    if blurry:
        C.append(Check("fail" if len(blurry) > 0.3 * n else "warn",
                       f"{len(blurry)} ảnh mờ vẫn đang được dùng ({', '.join(blurry[:6])}{'…' if len(blurry) > 6 else ''}) — "
                       "bấm “Bỏ chọn ảnh xấu” ở tab Ảnh rồi chạy lại."))
        data_problem = True

    # 2b. Kích thước bảng trong ảnh
    widths = []
    for r in results:
        if r.name in used_set and r.detected:
            p = r.corners.reshape(-1, 2)
            widths.append(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) / r.width * 100)
    if widths:
        mw = float(np.median(widths))
        if mw >= 15:
            C.append(Check("ok", f"Bảng chiếm trung bình {mw:.0f}% chiều rộng ảnh."))
        else:
            C.append(Check("fail" if mw < 8 else "warn",
                           f"Bảng quá nhỏ trong ảnh: chỉ chiếm khoảng {mw:.0f}% chiều rộng ảnh (nên ≥ 15–20%). "
                           "Khi bảng nhỏ, hiệu ứng phối cảnh yếu nên tiêu cự không xác định chính xác được. "
                           "In bảng khổ lớn hơn, hoặc lấy nét gần hơn nếu camera sẽ dùng ở khoảng cách gần."))
            data_problem = True

    # 3. Độ phủ
    grid = coverage_grid(results)
    total = max(1, grid.sum())
    weak = [CELL_NAMES[i][j] for i in range(3) for j in range(3) if grid[i, j] < 0.03 * total]
    corner_empty = [CELL_NAMES[i][j] for i, j in [(0, 0), (0, 2), (2, 0), (2, 2)] if grid[i, j] < 0.02 * total]
    if not weak:
        C.append(Check("ok", "Độ phủ khung hình đều ở cả 9 vùng."))
    elif corner_empty:
        C.append(Check("fail" if len(corner_empty) >= 3 else "warn",
                       "Thiếu dữ liệu ở góc ảnh (" + ", ".join(corner_empty) +
                       ") — hệ số méo không đáng tin ở vùng này."))
        data_problem = True
    else:
        C.append(Check("warn", "Độ phủ còn yếu ở: " + ", ".join(weak) + "."))
        data_problem = True

    # 4. Góc ngả
    used_r = [r for r in results if r.name in set(res.used_names)]
    tilted = [r for r in used_r if r.tilt_deg is not None and r.tilt_deg >= TILT_FRONTAL]
    if len(tilted) >= 0.5 * max(1, len(used_r)):
        C.append(Check("ok", f"{len(tilted)}/{len(used_r)} ảnh có ngả bảng ≥ {TILT_FRONTAL:.0f}°."))
    else:
        C.append(Check("warn", f"Chỉ {len(tilted)}/{len(used_r)} ảnh có ngả bảng — tiêu cự khó ước lượng chính xác."))
        data_problem = True

    # 5. fx / fy
    if not res.options.get("fix_aspect"):
        ratio_dev = abs(res.fx / res.fy - 1) * 100
        if ratio_dev <= 1.0:
            C.append(Check("ok", f"fx và fy gần bằng nhau (lệch {ratio_dev:.2f}%)."))
        else:
            C.append(Check("warn" if ratio_dev <= 3 else "fail",
                           f"fx và fy lệch nhau {ratio_dev:.1f}% — với camera thông thường hai giá trị này "
                           "gần như bằng nhau. Dấu hiệu dữ liệu chưa đủ (độ phủ, góc ngả)."))
            data_problem = True

    # 6. Tâm ảnh
    dx = abs(res.cx - w / 2) / w * 100
    dy = abs(res.cy - h / 2) / h * 100
    if max(dx, dy) <= 5:
        C.append(Check("ok", f"Tâm ảnh (cx, cy) gần giữa ảnh (lệch {dx:.1f}% / {dy:.1f}%)."))
    else:
        C.append(Check("warn", f"Tâm ảnh lệch xa khỏi giữa ảnh ({dx:.1f}% / {dy:.1f}%)."))

    # 7. Hàm méo hợp lệ
    if not _radial_monotonic(res.K, res.dist, res.image_size, res.model):
        C.append(Check("fail", "Mô hình méo không hợp lệ ở rìa ảnh (bị 'gập'), hệ số méo đang khớp nhiễu. "
                               "Cần thêm ảnh ở góc/cạnh ảnh, hoặc dùng mô hình ít hệ số hơn (k3 = 0)."))
        data_problem = True
    else:
        C.append(Check("ok", "Mô hình méo hợp lệ trên toàn khung hình."))

    # 8. Độ bất định & ổn định
    if res.std_intrinsics is not None and len(res.std_intrinsics) >= 2:
        sf = max(res.std_intrinsics[0] / res.fx, res.std_intrinsics[1] / res.fy) * 100
        C.append(Check("ok" if sf <= 0.5 else ("warn" if sf <= 1.5 else "fail"),
                       f"Độ bất định của tiêu cự ±{sf:.2f}% (σ của fx/fy)."))
    if res.stability:
        sf = max(res.stability["fx_std"] / res.fx, res.stability["fy_std"] / res.fy) * 100
        C.append(Check("ok" if sf <= 1.0 else ("warn" if sf <= 2.5 else "fail"),
                       f"Kiểm tra chéo {res.stability['runs']} lần trên tập con: tiêu cự dao động ±{sf:.2f}%."))
        if sf > 1.0:
            data_problem = True
    if res.holdout_err is not None:
        ratio = res.holdout_err / max(res.rms, 1e-6)
        lvl = "ok" if (res.holdout_err <= 1.0 and ratio <= 2.0) else "warn"
        C.append(Check(lvl, f"Sai số trên ảnh để riêng (không dùng để hiệu chỉnh) = {res.holdout_err:.3f} px "
                            f"(gấp {ratio:.1f} lần RMS)." + (" Mô hình có dấu hiệu khớp quá mức." if ratio > 2 else "")))

    # ---- kết luận ----
    levels = [c.level for c in C]
    if "fail" in levels:
        res.verdict_level = "fail"
        res.verdict = "KHÔNG ĐẠT — kết quả chưa dùng được."
    elif "warn" in levels and data_problem:
        res.verdict_level = "warn"
        res.verdict = "CẦN CHỤP THÊM — dùng tạm được nhưng chưa đáng tin."
    elif "warn" in levels:
        res.verdict_level = "warn"
        res.verdict = "ĐẠT, có lưu ý."
    else:
        res.verdict_level = "ok"
        res.verdict = "ĐẠT — kết quả đáng tin cậy."


# --------------------------------------------------------------------------
# Xuất / nhập kết quả
# --------------------------------------------------------------------------
def _fs_string(res: CalibResult) -> str:
    fs = cv2.FileStorage(".yaml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    fs.write("image_width", int(res.image_size[0]))
    fs.write("image_height", int(res.image_size[1]))
    fs.write("model", res.model)
    fs.write("camera_matrix", res.K)
    fs.write("dist_coeffs", res.dist.reshape(1, -1))
    fs.write("rms", float(res.rms))
    fs.write("n_images", len(res.used_names))
    fs.write("board", json.dumps(res.board, ensure_ascii=True))
    return fs.releaseAndGetString()


def save_yaml(res: CalibResult, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(_fs_string(res))


def result_dict(res: CalibResult) -> dict:
    d = {
        "image_width": res.image_size[0], "image_height": res.image_size[1],
        "model": res.model,
        "camera_matrix": res.K.tolist(),
        "dist_coeffs": res.dist.ravel().tolist(),
        "dist_names": res.dist_names(),
        "fx": res.fx, "fy": res.fy, "cx": res.cx, "cy": res.cy,
        "rms": res.rms, "holdout_err": res.holdout_err,
        "std_intrinsics": None if res.std_intrinsics is None else res.std_intrinsics.ravel().tolist(),
        "stability": res.stability,
        "verdict": res.verdict, "verdict_level": res.verdict_level,
        "checks": [asdict(c) for c in res.checks],
        "used_images": res.used_names, "rejected_images": res.rejected,
        "per_view_error": res.per_view,
        "board": res.board, "options": res.options,
    }
    return d


def save_json(res: CalibResult, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_dict(res), f, ensure_ascii=False, indent=2)


def save_ros(res: CalibResult, path: str, camera_name: str = "camera"):
    w, h = res.image_size
    K = res.K
    dist = res.dist.ravel()
    model = "equidistant" if res.model == "fisheye" else ("rational_polynomial" if len(dist) > 5 else "plumb_bob")
    if model == "plumb_bob":
        dist = np.concatenate([dist, np.zeros(5 - len(dist))])[:5]
    elif model == "rational_polynomial":
        dist = dist[:8]
    P = np.hstack([K, np.zeros((3, 1))])

    def mat(name, rows, cols, data):
        vals = ", ".join(f"{v:.10g}" for v in np.asarray(data).ravel())
        return f"{name}:\n  rows: {rows}\n  cols: {cols}\n  data: [{vals}]\n"

    txt = (f"image_width: {w}\nimage_height: {h}\ncamera_name: {camera_name}\n"
           + mat("camera_matrix", 3, 3, K)
           + f"distortion_model: {model}\n"
           + mat("distortion_coefficients", 1, len(dist), dist)
           + mat("rectification_matrix", 3, 3, np.eye(3))
           + mat("projection_matrix", 3, 4, P))
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def load_calibration(path: str) -> dict:
    """Đọc lại YAML (OpenCV) hoặc JSON do tool này xuất. Trả về dict có K, dist, size, model."""
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {"K": np.array(d["camera_matrix"], np.float64),
                "dist": np.array(d["dist_coeffs"], np.float64),
                "size": (int(d["image_width"]), int(d["image_height"])),
                "model": d.get("model", "pinhole5")}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ext = ".xml" if text.lstrip().startswith("<") else ".yaml"
    fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)

    def first(*names):
        for n in names:
            node = fs.getNode(n)
            if not node.empty():
                m = node.mat()
                if m is not None:
                    return m
        return None

    K = first("camera_matrix", "K", "cameraMatrix", "intrinsic_matrix")
    dist = first("dist_coeffs", "distortion_coefficients", "D", "dist", "distCoeffs")
    wn, hn = fs.getNode("image_width"), fs.getNode("image_height")
    mn = fs.getNode("model")
    model = mn.string() if not mn.empty() else "pinhole5"
    size = (int(wn.real()), int(hn.real())) if not wn.empty() else None
    fs.release()
    if K is None or dist is None:
        raise ValueError("File không có ma trận camera (camera_matrix) hoặc hệ số méo "
                         "(dist_coeffs / distortion_coefficients).")
    return {"K": K.astype(np.float64), "dist": dist.ravel().astype(np.float64), "size": size, "model": model or "pinhole5"}


# --------------------------------------------------------------------------
# Khử méo & kiểm tra trên ảnh mới
# --------------------------------------------------------------------------
def undistort(rgb: np.ndarray, K, dist, model: str, alpha: float = 0.0) -> np.ndarray:
    h, w = rgb.shape[:2]
    if model == "fisheye":
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, dist, (w, h), np.eye(3), balance=alpha)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), newK, (w, h), cv2.CV_16SC2)
        return cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    return cv2.undistort(rgb, K, dist, None, newK)


def verify_image(path: str, cfg: BoardConfig, K, dist, model: str) -> dict:
    """Kiểm tra độc lập: nhận diện bảng trên ảnh mới, đo sai số chiếu lại và khoảng cách bảng."""
    board = cfg.make_board(unit="m")
    r = detect_image(path, board, sig=board_signature(cfg))
    out = {"result": r, "err": None, "distance_m": None, "square_mm": None}
    if not r.detected or r.n_corners < 6:
        return out
    obj, img = board.matchImagePoints(r.corners, r.ids)
    fisheye = model == "fisheye"
    out["err"] = _holdout_error(obj, img, K, dist, fisheye)
    try:
        if fisheye:
            und = cv2.fisheye.undistortPoints(img.reshape(1, -1, 2).astype(np.float64), K, dist, P=K)
            ok, rv, tv = cv2.solvePnP(obj, und.reshape(-1, 1, 2), K, None)
        else:
            ok, rv, tv = cv2.solvePnP(obj, img, K, dist)
        if ok:
            out["distance_m"] = float(np.linalg.norm(tv))
            out["rvec"], out["tvec"] = rv, tv
    except cv2.error:
        pass
    update_poses([r], board, K, dist, fisheye)
    return out


def draw_detection(rgb: np.ndarray, r: ImageResult, board=None, K=None, dist=None, axes=None) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    s = max(1, int(round(max(w, h) / 1000)))
    if r.detected:
        pts = r.corners.reshape(-1, 2)
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.polylines(out, [hull], True, (42, 120, 214), 2 * s, cv2.LINE_AA)
        for (x, y), i in zip(pts, r.ids.ravel()):
            cv2.circle(out, (int(round(x)), int(round(y))), 4 * s, (27, 175, 122), -1, cv2.LINE_AA)
            cv2.circle(out, (int(round(x)), int(round(y))), 4 * s, (255, 255, 255), max(1, s // 2), cv2.LINE_AA)
            if s >= 2 or len(pts) <= 60:
                cv2.putText(out, str(int(i)), (int(x) + 5 * s, int(y) - 5 * s), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 * s, (235, 104, 52), max(1, s // 2), cv2.LINE_AA)
    if axes is not None and K is not None:
        rv, tv, length = axes
        try:
            cv2.drawFrameAxes(out, K, dist, rv, tv, length, 2 * s)
        except cv2.error:
            pass
    return out
