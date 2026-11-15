"""
Giao diện đồ họa ChArUco Calibrator (PySide6).
"""
from __future__ import annotations

import os
import traceback

import numpy as np
from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, QTimer, QSettings, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QIcon, QAction, QBrush, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QProgressBar, QGroupBox, QListWidget,
    QListWidgetItem, QSlider, QTextBrowser, QSizePolicy, QAbstractItemView, QFrame, QScrollArea,
    QDialog, QDialogButtonBox, QLineEdit, QStackedWidget, QButtonGroup,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

import core
import video as vid
from core import BoardConfig, CalibOptions, ImageResult, MODELS, DICT_NAMES, PAPERS
import report

APP_NAME = "ChArUco Calibrator"
STATUS_COLORS = {"ok": QColor("#0ca30c"), "warn": QColor("#b77b00"), "fail": QColor("#d03b3b")}
STATUS_BG = {"ok": QColor("#eef8ee"), "warn": QColor("#fdf5e2"), "fail": QColor("#fbecec")}


# ==========================================================================
# Tiện ích
# ==========================================================================
def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageView(QLabel):
    """Ảnh tự co giãn theo khung, giữ tỉ lệ."""

    def __init__(self, placeholder: str = ""):
        super().__init__()
        self._pix: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("QLabel { background:#f3f3f1; border:1px solid #e4e3df; border-radius:6px; color:#52514e; }")
        self.setText(placeholder)

    def set_array(self, arr: np.ndarray | None, placeholder: str = ""):
        if arr is None:
            self._pix = None
            self.setPixmap(QPixmap())
            self.setText(placeholder)
            return
        self._pix = QPixmap.fromImage(to_qimage(arr))
        self._refresh()

    def _refresh(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size() - QSize(4, 4), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()


class ChartBox(QWidget):
    """Khung chứa một biểu đồ matplotlib, thay được figure."""

    def __init__(self, min_h: int = 260, natural_height: bool = False):
        super().__init__()
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        self._natural = natural_height
        self.setMinimumHeight(min_h)

    def set_figure(self, fig):
        if self._canvas is not None:
            self._lay.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = FigureCanvasQTAgg(fig)
        if self._natural:
            h = int(fig.get_figheight() * fig.dpi)
            self._canvas.setMinimumHeight(h)
            self.setMinimumHeight(h)
        self._lay.addWidget(self._canvas)


class WorkerSignals(QObject):
    progress = Signal(int, int, object)
    message = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            out = self.fn(self.signals, *self.args, **self.kwargs)
        except Exception as e:
            msg = str(e) if isinstance(e, ValueError) else traceback.format_exc()
            self.signals.failed.emit(msg)
            return
        self.signals.done.emit(out)


def make_icon() -> QIcon:
    img = np.full((64, 64), 255, np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                img[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = 20
    img[20:28, 4:12] = 255
    return QIcon(QPixmap.fromImage(to_qimage(img)))


def hint(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet("color:#52514e;")
    return lb


def banner(level: str, text: str) -> str:
    c = {"ok": ("#e8f6e8", "#0b7d0b"), "warn": ("#fdf3dc", "#8a5d00"), "fail": ("#fbe9e9", "#b42f2f"),
         "info": ("#eef4fc", "#1c5cab")}[level]
    return (f"QLabel {{ background:{c[0]}; color:{c[1]}; border-radius:8px; padding:10px 14px; "
            f"font-weight:600; }}")


# ==========================================================================
# Tab 1 — Bảng hiệu chỉnh
# ==========================================================================
class BoardTab(QWidget):
    changed = Signal(bool)   # True nếu thay đổi cấu trúc (cần nhận diện lại)

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self._block = False
        self._last_struct = None

        self.sx = QSpinBox(); self.sx.setRange(3, 40)
        self.sy = QSpinBox(); self.sy.setRange(3, 40)
        self.sq = QDoubleSpinBox(); self.sq.setRange(1, 1000); self.sq.setDecimals(2); self.sq.setSuffix(" mm"); self.sq.setSingleStep(0.5)
        self.mk = QDoubleSpinBox(); self.mk.setRange(0.5, 1000); self.mk.setDecimals(2); self.mk.setSuffix(" mm"); self.mk.setSingleStep(0.5)
        self.dic = QComboBox(); self.dic.addItems(DICT_NAMES)
        self.legacy = QCheckBox("Legacy pattern (OpenCV < 4.6)")
        self.legacy.setToolTip("Chỉ bật khi dùng bảng tạo bằng OpenCV cũ (< 4.6) có số hàng chẵn.")
        self.paper = QComboBox(); self.paper.addItems(list(PAPERS.keys()))
        self.orient = QComboBox(); self.orient.addItems(["Ngang", "Dọc"])

        form = QFormLayout()
        form.addRow("Số ô ngang", self.sx)
        form.addRow("Số ô dọc", self.sy)
        form.addRow("Cạnh ô vuông", self.sq)
        form.addRow("Cạnh marker", self.mk)
        form.addRow("Từ điển ArUco", self.dic)
        form.addRow("", self.legacy)
        form.addRow("Khổ giấy", self.paper)
        form.addRow("Hướng giấy", self.orient)

        self.info = QLabel(); self.info.setWordWrap(True); self.info.setTextFormat(Qt.RichText)
        b_max = QPushButton("Tối đa hóa cạnh ô theo khổ giấy")
        b_pdf = QPushButton("Xuất PDF để in…"); b_pdf.setDefault(True)
        b_save = QPushButton("Lưu cấu hình bảng…")
        b_open = QPushButton("Mở cấu hình bảng…")
        b_max.clicked.connect(self.maximize)
        b_pdf.clicked.connect(self.export_pdf)
        b_save.clicked.connect(self.save_cfg)
        b_open.clicked.connect(self.open_cfg)

        left = QVBoxLayout()
        g = QGroupBox("Thông số bảng"); g.setLayout(form)
        left.addWidget(g)
        left.addWidget(self.info)
        left.addWidget(b_max)
        left.addWidget(b_pdf)
        row = QHBoxLayout(); row.addWidget(b_save); row.addWidget(b_open)
        left.addLayout(row)
        left.addWidget(hint("<b>Sau khi in:</b> đo cạnh ô và cạnh marker bằng thước rồi nhập <b>số đo thực tế</b> "
                            "vào các ô trên trước khi hiệu chỉnh. Dán tờ in phẳng lên tấm cứng (kính, mica, gỗ) — "
                            "giấy cong là nguồn sai số lớn nhất."))
        left.addStretch()
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(420)

        self.preview = ImageView("Xem trước")
        lay = QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(self.preview, 1)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(200)
        self._timer.timeout.connect(self._render)
        for w in (self.sx, self.sy, self.sq, self.mk):
            w.valueChanged.connect(self._on_change)
        for w in (self.dic, self.paper, self.orient):
            w.currentIndexChanged.connect(self._on_change)
        self.legacy.toggled.connect(self._on_change)
        self.set_config(main.cfg)

    # ---- đồng bộ ----
    def set_config(self, cfg: BoardConfig):
        self._block = True
        self.sx.setValue(cfg.squares_x); self.sy.setValue(cfg.squares_y)
        self.sq.setValue(cfg.square_mm); self.mk.setValue(cfg.marker_mm)
        self.dic.setCurrentText(cfg.dictionary); self.legacy.setChecked(cfg.legacy)
        self.paper.setCurrentText(cfg.paper); self.orient.setCurrentIndex(0 if cfg.landscape else 1)
        self._block = False
        self._last_struct = self._struct(cfg)
        self._render()

    def read_config(self) -> BoardConfig:
        return BoardConfig(self.sx.value(), self.sy.value(), self.sq.value(), self.mk.value(),
                           self.dic.currentText(), self.legacy.isChecked(), self.paper.currentText(),
                           self.orient.currentIndex() == 0)

    @staticmethod
    def _struct(c: BoardConfig):
        return (c.squares_x, c.squares_y, c.dictionary, c.legacy, c.square_mm / c.marker_mm)

    def _on_change(self, *_):
        if self._block:
            return
        cfg = self.read_config()
        self.main.cfg = cfg
        s = self._struct(cfg)
        structural = s[:4] != self._last_struct[:4] or abs(s[4] - self._last_struct[4]) > 1e-6
        self._last_struct = s
        self.changed.emit(structural)
        self._timer.start()

    def _render(self):
        cfg = self.main.cfg
        errs = cfg.validate()
        ok, msg = cfg.fits_paper()
        bw, bh = cfg.board_size_mm
        lines = [f"<b>{cfg.n_corners}</b> góc trong · <b>{cfg.n_markers}</b> marker · bảng <b>{bw:g} × {bh:g} mm</b>"]
        if cfg.squares_y % 2 == 0:
            lines.append("<span style='color:#8a5d00'>Số hàng chẵn: phần mềm khác cần cùng quy ước "
                         "legacy pattern mới nhận đúng.</span>")
        for e in errs:
            lines.append(f"<span style='color:#b42f2f'>✕ {e}</span>")
        if not ok:
            lines.append(f"<span style='color:#b42f2f'>✕ {msg}</span>")
        elif not errs:
            lines.append(f"<span style='color:#0b7d0b'>✓ Vừa khổ {cfg.paper}. Cạnh ô tối đa cho khổ này: "
                         f"{cfg.max_square_for_paper():g} mm.</span>")
        self.info.setText("<br>".join(lines))
        if errs:
            self.preview.set_array(None, "Thông số chưa hợp lệ")
            return
        try:
            pw, _ = cfg.paper_size_mm()
            ppm = max(1.0, min(5.0, 1400 / pw))
            self.preview.set_array(core.render_board_image(cfg, px_per_mm=ppm, with_paper=True))
        except Exception as e:
            self.preview.set_array(None, f"Không vẽ được: {e}")

    # ---- hành động ----
    def maximize(self):
        cfg = self.main.cfg
        s = cfg.max_square_for_paper()
        ratio = cfg.marker_mm / cfg.square_mm if cfg.square_mm else 0.75
        self._block = True
        self.sq.setValue(s)
        self.mk.setValue(round(s * ratio * 2) / 2)
        self._block = False
        self._on_change()

    def export_pdf(self):
        cfg = self.main.cfg
        default = os.path.join(self.main.last_dir(),
                               f"charuco_{cfg.squares_x}x{cfg.squares_y}_{cfg.square_mm:g}mm_{cfg.paper}.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "Xuất PDF", default, "PDF (*.pdf)")
        if not path:
            return
        try:
            core.generate_pdf(cfg, path)
            cfg.save(os.path.splitext(path)[0] + ".board.json")
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không xuất được PDF:\n{e}")
            return
        self.main.remember_dir(path)
        QMessageBox.information(self, APP_NAME,
                                f"Đã lưu:\n{path}\n\nKèm file cấu hình bảng (.board.json) cùng thư mục.\n\n"
                                "Khi in: chọn 100% / Actual size, KHÔNG chọn Fit to page. "
                                "Sau khi in, đo vạch kiểm tra phải đúng 100 mm.")

    def save_cfg(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu cấu hình bảng",
                                              os.path.join(self.main.last_dir(), "board.board.json"),
                                              "Cấu hình bảng (*.json)")
        if path:
            self.main.cfg.save(path)
            self.main.remember_dir(path)

    def open_cfg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mở cấu hình bảng", self.main.last_dir(), "Cấu hình bảng (*.json)")
        if not path:
            return
        try:
            cfg = BoardConfig.load(path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không đọc được file:\n{e}")
            return
        old = self._struct(self.main.cfg)
        self.main.cfg = cfg
        self.set_config(cfg)
        self.changed.emit(self._struct(cfg)[:4] != old[:4])
        self.main.remember_dir(path)


# ==========================================================================
# Hộp thoại video
# ==========================================================================
VIDEO_FILTER = "Video (" + " ".join("*" + e for e in vid.VIDEO_EXTS) + ")"


class VideoOptionsDialog(QDialog):
    def __init__(self, parent, path: str, info: dict, existing_size=None):
        super().__init__(parent)
        self.setWindowTitle("Trích khung hình từ video")
        self.setMinimumWidth(560)
        self.path = path
        lay = QVBoxLayout(self)
        dur = info.get("duration", 0)
        head = QLabel(f"<b>{os.path.basename(path)}</b><br>"
                      f"{info.get('width', 0)}×{info.get('height', 0)} · {info.get('fps', 0):.2f} khung/giây · "
                      f"{dur:.1f} giây ({info.get('n_frames', 0)} khung)")
        head.setWordWrap(True)
        lay.addWidget(head)
        if existing_size and (info.get("width"), info.get("height")) != tuple(existing_size) and info.get("width"):
            w = QLabel(f"Độ phân giải video khác với ảnh đã có trong danh sách ({existing_size[0]}×{existing_size[1]}). "
                       "Chỉ hiệu chỉnh chung khi cùng chế độ quay.")
            w.setWordWrap(True); w.setStyleSheet(banner("warn", ""))
            lay.addWidget(w)

        self.n = QSpinBox(); self.n.setRange(10, 300); self.n.setValue(40)
        self.fps = QDoubleSpinBox(); self.fps.setRange(1, 30); self.fps.setValue(6); self.fps.setSuffix(" khung/giây")
        self.speed = QDoubleSpinBox(); self.speed.setRange(0.1, 10); self.speed.setDecimals(1); self.speed.setValue(1.0)
        self.speed.setSuffix(" px/khung")
        self.gap = QDoubleSpinBox(); self.gap.setRange(0.0, 10); self.gap.setDecimals(1); self.gap.setValue(0.5)
        self.gap.setSuffix(" giây")
        self.out = QLineEdit(vid.default_out_dir(path))
        b_out = QPushButton("…"); b_out.setFixedWidth(32); b_out.clicked.connect(self._pick)
        row = QHBoxLayout(); row.addWidget(self.out, 1); row.addWidget(b_out)
        form = QFormLayout()
        form.addRow("Số khung cần lấy", self.n)
        form.addRow("Tần suất phân tích", self.fps)
        form.addRow("Ngưỡng đứng yên", self.speed)
        form.addRow("Cách nhau tối thiểu", self.gap)
        form.addRow("Lưu khung vào", row)
        lay.addLayout(form)
        est = dur * self.fps.value() * 0.07 if dur else 0
        lay.addWidget(hint("Tool phân tích video, bỏ khung mờ và khung bảng đang di chuyển (rolling shutter làm méo bảng), "
                           "rồi chọn các khung khác nhau nhiều nhất về vị trí, kích thước và góc ngả. "
                           "Ngưỡng đứng yên: tăng lên nếu có quá ít khung được chọn; giảm xuống nếu cần chặt hơn."
                           + (f"<br>Thời gian ước tính: khoảng {max(5, est):.0f} giây." if est else "")))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Bắt đầu")
        bb.button(QDialogButtonBox.Cancel).setText("Hủy")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "Thư mục lưu khung hình", os.path.dirname(self.out.text()))
        if d:
            self.out.setText(d)

    def options(self) -> "vid.VideoOptions":
        return vid.VideoOptions(target_frames=self.n.value(), analyze_fps=self.fps.value(),
                                max_speed=self.speed.value(), min_gap_s=self.gap.value(),
                                out_dir=self.out.text().strip())


class VideoResultDialog(QDialog):
    def __init__(self, parent, scan, opt):
        super().__init__(parent)
        self.setWindowTitle("Kết quả quét video")
        self.resize(980, 560)
        c = scan.counts()
        lay = QVBoxLayout(self)
        lvl = "ok" if c["selected"] >= min(opt.target_frames, 20) else ("warn" if c["selected"] >= 10 else "fail")
        top = QLabel(f"Đã chọn {c['selected']} khung từ {c['analyzed']} khung phân tích "
                     f"({c['detected']} thấy bảng · {c['still']} đủ nét và đứng yên).")
        top.setWordWrap(True); top.setStyleSheet(banner(lvl, ""))
        lay.addWidget(top)
        chart = ChartBox(300)
        chart.set_figure(report.fig_video_timeline(scan, opt.max_speed))
        lay.addWidget(chart, 1)
        tips = QTextBrowser(); tips.setMaximumHeight(120)
        tips.setHtml("<ul>" + "".join(f"<li>{t}</li>" for t in vid.scan_advice(scan, opt)) + "</ul>"
                     + (f"<p>Khung đã lưu tại: {scan.out_dir}</p>" if scan.out_dir else ""))
        lay.addWidget(tips)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.button(QDialogButtonBox.Ok).setText("Thêm vào danh sách ảnh" if c["selected"] else "Đóng")
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)


# ==========================================================================
# Tab 2 — Ảnh
# ==========================================================================
COLS = ["Dùng", "Ảnh", "Góc", "Độ nét", "Diện tích", "Ngả", "Hướng ngả", "Sai số (px)", "Trạng thái"]


class ImagesTab(QWidget):
    results_changed = Signal()

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self._updating = False
        self.setAcceptDrops(True)

        b_add = QPushButton("Thêm ảnh…")
        b_dir = QPushButton("Thêm thư mục…")
        b_vid = QPushButton("Thêm video…")
        b_vid.setStyleSheet("font-weight:600;")
        b_vid.clicked.connect(lambda: self.add_video())
        self.b_bad = QPushButton("Bỏ chọn ảnh xấu")
        b_del = QPushButton("Xóa ảnh đang chọn")
        b_clr = QPushButton("Xóa tất cả")
        self.b_re = QPushButton("Phân tích lại")
        for b in (b_add, b_dir):
            b.setStyleSheet("font-weight:600;")
        b_add.clicked.connect(self.add_files)
        b_dir.clicked.connect(self.add_dir)
        self.b_bad.clicked.connect(self.uncheck_bad)
        b_del.clicked.connect(self.remove_selected)
        b_clr.clicked.connect(self.clear_all)
        self.b_re.clicked.connect(self.reanalyze)
        bar = QHBoxLayout()
        for b in (b_add, b_dir, b_vid, self.b_bad, b_del, b_clr, self.b_re):
            bar.addWidget(b)
        bar.addStretch()

        self.warn = QLabel(); self.warn.setStyleSheet(banner("warn", "")); self.warn.hide()
        self.summary = QLabel("Chưa có ảnh. Bấm “Thêm ảnh…” / “Thêm video…” hoặc kéo-thả ảnh, video vào cửa sổ này.")
        self.progress = QProgressBar(); self.progress.hide()
        self.b_cancel = QPushButton("Hủy"); self.b_cancel.hide()
        self.b_cancel.clicked.connect(self._cancel_video)
        self._cancel = False
        self.pmsg = hint("")
        self.pmsg.hide()

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for i in range(len(COLS)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(COLS) - 1, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._show_selected)

        self.preview = ImageView("Chọn một ảnh để xem điểm nhận diện")
        self.caption = hint("")
        right = QVBoxLayout(); right.addWidget(self.preview, 1); right.addWidget(self.caption)
        rw = QWidget(); rw.setLayout(right)

        split = QSplitter()
        split.addWidget(self.table); split.addWidget(rw)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)

        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(self.warn)
        lay.addWidget(self.summary)
        prow = QHBoxLayout(); prow.addWidget(self.progress, 1); prow.addWidget(self.b_cancel)
        lay.addLayout(prow)
        lay.addWidget(self.pmsg)
        lay.addWidget(split, 1)
        lay.addWidget(hint(f"Độ nét đo trên vùng bảng: dưới {core.SHARP_BAD:.0f} là mờ (nên bỏ). "
                           "Góc ngả trước khi hiệu chỉnh là ước lượng; sau khi hiệu chỉnh sẽ tính lại chính xác."))

    # ---- kéo-thả ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths, videos = [], []
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.isdir(p):
                paths += self._list_dir(p)
            elif p.lower().endswith(core.IMAGE_EXTS):
                paths.append(p)
            elif p.lower().endswith(vid.VIDEO_EXTS):
                videos.append(p)
        if paths:
            self.add_paths(paths)
        elif videos:
            self.add_video(videos[0])

    # ---- thêm / xóa ----
    @staticmethod
    def _list_dir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(core.IMAGE_EXTS))

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Chọn ảnh", self.main.last_dir(),
                                                "Ảnh (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if paths:
            self.main.remember_dir(paths[0])
            self.add_paths(paths)

    def add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh", self.main.last_dir())
        if d:
            self.main.remember_dir(os.path.join(d, "x"))
            self.add_paths(self._list_dir(d))

    # ---- video ----
    def add_video(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Chọn video hiệu chỉnh", self.main.last_dir(), VIDEO_FILTER)
            if not path:
                return
        self.main.remember_dir(path)
        cfg = self.main.cfg
        if cfg.validate():
            QMessageBox.warning(self, APP_NAME, "Thông số bảng chưa hợp lệ (xem tab Bảng hiệu chỉnh).")
            return
        try:
            info = vid.video_info(path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, str(e))
            return
        sizes = [(r.width, r.height) for r in self.main.results if r.detected]
        existing = max(set(sizes), key=sizes.count) if sizes else None
        dlg = VideoOptionsDialog(self, path, info, existing)
        if dlg.exec() != QDialog.Accepted:
            return
        opt = dlg.options()
        self._video_opt = opt
        self._cancel = False
        self.main.set_busy(True)
        self.b_cancel.setEnabled(True); self.b_cancel.show()
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.show()
        self.pmsg.setText("Đang mở video…"); self.pmsg.show()

        def job(sig, path, cfg, opt):
            return vid.process_video(path, cfg, opt, progress=lambda m, f: sig.message.emit(m, f),
                                     cancel=lambda: self._cancel)

        w = Worker(job, path, cfg, opt)
        w.signals.message.connect(self._on_video_msg)
        w.signals.done.connect(self._on_video_done)
        w.signals.failed.connect(self._on_video_failed)
        self.main.start(w)

    def _cancel_video(self):
        self._cancel = True
        self.b_cancel.setEnabled(False)
        self.pmsg.setText("Đang hủy…")

    def _on_video_msg(self, m, f):
        self.progress.setValue(int(f * 100))
        self.pmsg.setText(m)
        self.main.statusBar().showMessage(m)

    def _video_ui_reset(self):
        self.progress.hide(); self.b_cancel.hide(); self.pmsg.hide()
        self.main.set_busy(False)

    def _on_video_failed(self, msg):
        self._video_ui_reset()
        if msg.strip() != "Đã hủy.":
            QMessageBox.warning(self, APP_NAME, f"Không xử lý được video:\n{msg}")

    def _on_video_done(self, scan):
        self._video_ui_reset()
        dlg = VideoResultDialog(self, scan, self._video_opt)
        dlg.exec()
        if scan.saved_paths:
            self.add_paths(scan.saved_paths)

    def add_paths(self, paths):
        have = {os.path.normcase(os.path.abspath(r.path)) for r in self.main.results}
        new = []
        for p in paths:
            k = os.path.normcase(os.path.abspath(p))
            if k not in have:
                have.add(k)
                new.append(p)
        if not new:
            return
        self.run_detection(new)

    def run_detection(self, paths, replace=False):
        cfg = self.main.cfg
        keep = {}
        if cfg.validate():
            QMessageBox.warning(self, APP_NAME, "Thông số bảng chưa hợp lệ (xem tab Bảng hiệu chỉnh).")
            return
        if replace:
            keep = {os.path.normcase(os.path.abspath(r.path)): r.used for r in self.main.results}
            self.main.results = []
            self.table.setRowCount(0)
        self.main.set_busy(True)
        self.progress.setRange(0, len(paths)); self.progress.setValue(0); self.progress.show()

        self._keep_used = keep if replace else {}

        def job(sig, paths, cfg):
            board = cfg.make_board()
            det = core.make_detector(board)
            bsig = core.board_signature(cfg)
            out = []
            for i, p in enumerate(paths):
                r = core.detect_image(p, board, det, sig=bsig)
                out.append(r)
                sig.progress.emit(i + 1, len(paths), r)
            core.update_poses(out, board)
            return out

        w = Worker(job, paths, cfg)
        w.signals.progress.connect(self._on_detect_progress)
        w.signals.done.connect(self._on_detect_done)
        w.signals.failed.connect(self._on_failed)
        self.main.start(w)

    def _on_detect_progress(self, i, n, r):
        self.progress.setValue(i)
        k = os.path.normcase(os.path.abspath(r.path))
        if k in self._keep_used:
            r.used = self._keep_used[k]
        self.main.results.append(r)
        self._add_row(len(self.main.results) - 1)
        self.main.statusBar().showMessage(f"Đang phân tích {i}/{n}: {r.name}")

    def _on_detect_done(self, _):
        self.progress.hide()
        self.main.set_busy(False)
        self.warn.hide()
        then, self._after_detect = getattr(self, "_after_detect", None), None
        if then is not None:
            QTimer.singleShot(0, then)
        self.main.calib_stale()
        self.refresh_all()
        self.main.statusBar().showMessage("Đã phân tích xong.", 5000)
        if self.table.currentRow() < 0 and self.table.rowCount():
            self.table.selectRow(0)
        self.results_changed.emit()

    def _on_failed(self, msg):
        self._after_detect = None
        self.progress.hide()
        self.main.set_busy(False)
        QMessageBox.critical(self, APP_NAME, msg)

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.main.results[r]
            self.table.removeRow(r)
        if rows:
            self.main.calib_stale()
            self.refresh_all()
            self.results_changed.emit()

    def clear_all(self):
        if not self.main.results:
            return
        if QMessageBox.question(self, APP_NAME, "Xóa toàn bộ ảnh khỏi danh sách?") != QMessageBox.Yes:
            return
        self.main.results = []
        self.table.setRowCount(0)
        self.preview.set_array(None, "Chọn một ảnh để xem điểm nhận diện")
        self.caption.setText("")
        self.main.calib_stale()
        self.refresh_all()
        self.results_changed.emit()

    def uncheck_bad(self):
        n = 0
        for r in self.main.results:
            if r.used and r.status_level() == "fail":
                r.used = False
                n += 1
        self.refresh_all()
        self.results_changed.emit()
        self.main.statusBar().showMessage(f"Đã bỏ chọn {n} ảnh xấu.", 5000)

    def reanalyze(self, then=None):
        paths = [r.path for r in self.main.results]
        if paths:
            self._after_detect = then
            self.run_detection(paths, replace=True)

    def needs_redetect(self) -> bool:
        sig = core.board_signature(self.main.cfg)
        return any(r.board_sig != sig for r in self.main.results)

    def structure_changed(self):
        if self.main.results:
            self.warn.setText("Thông số bảng đã thay đổi — ảnh sẽ được tự động phân tích lại khi chạy hiệu chỉnh "
                              "(hoặc bấm “Phân tích lại” ngay bây giờ).")
            self.warn.show()

    # ---- bảng ----
    def _add_row(self, i):
        self._updating = True
        self.table.insertRow(i)
        for c in range(len(COLS)):
            it = QTableWidgetItem()
            if c == 0:
                it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            elif c in (2, 3, 4, 5, 7):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, c, it)
        self._updating = False
        self._fill_row(i)

    def _fill_row(self, i):
        r: ImageResult = self.main.results[i]
        self._updating = True
        lvl = r.status_level()
        vals = ["", r.name, str(r.n_corners), f"{r.sharpness:.0f}", f"{r.area_pct:.1f}%",
                "" if r.tilt_deg is None else f"{r.tilt_deg:.0f}°", r.tilt_dir,
                "" if r.reproj_err is None else f"{r.reproj_err:.3f}", r.status_text()]
        for c, v in enumerate(vals):
            it = self.table.item(i, c)
            it.setText(v)
            if c == 0:
                it.setCheckState(Qt.Checked if r.used else Qt.Unchecked)
            it.setBackground(QBrush(STATUS_BG[lvl]) if lvl != "ok" else QBrush())
            if c == len(COLS) - 1:
                it.setForeground(QBrush(STATUS_COLORS[lvl]))
            if c == 1:
                it.setToolTip(r.path)
        self._updating = False

    def refresh_all(self):
        for i in range(len(self.main.results)):
            self._fill_row(i)
        rs = self.main.results
        n_det = sum(1 for r in rs if r.detected)
        n_ok = sum(1 for r in rs if r.status_level() == "ok")
        n_used = sum(1 for r in rs if r.used and r.detected)
        if rs:
            self.summary.setText(f"<b>{len(rs)}</b> ảnh · <b>{n_det}</b> nhận được bảng · "
                                 f"<b>{n_ok}</b> tốt · <b>{n_used}</b> đang chọn dùng")
        else:
            self.summary.setText("Chưa có ảnh. Bấm “Thêm ảnh…” / “Thêm video…” hoặc kéo-thả ảnh, video vào cửa sổ này.")

    def _item_changed(self, it):
        if self._updating or it.column() != 0:
            return
        r = self.main.results[it.row()]
        r.used = it.checkState() == Qt.Checked
        self.main.calib_stale()
        self.refresh_all()
        self.results_changed.emit()

    def _show_selected(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.main.results):
            return
        r = self.main.results[row]
        try:
            rgb = core.load_image(r.path)
        except Exception as e:
            self.preview.set_array(None, f"Không mở được ảnh: {e}")
            return
        self.preview.set_array(core.draw_detection(rgb, r))
        parts = [f"<b>{r.name}</b> · {r.width}×{r.height}", f"{r.n_markers} marker, {r.n_corners} góc"]
        if r.detected:
            parts.append(f"độ nét {r.sharpness:.0f}, bảng chiếm {r.area_pct:.1f}% ảnh")
        if r.tilt_deg is not None:
            parts.append(f"ngả {r.tilt_deg:.0f}° ({r.tilt_dir})")
        iss = r.issues()
        if iss:
            parts.append("<span style='color:#b42f2f'>" + "; ".join(t for _, t in iss) + "</span>")
        self.caption.setText(" · ".join(parts))


# ==========================================================================
# Tab 3 — Độ phủ
# ==========================================================================
class CoverageTab(QWidget):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.cov = ChartBox(320)
        self.tilt = ChartBox(320)
        self.tips = QTextBrowser()
        self.tips.setMinimumHeight(120)
        self.note = hint("")
        top = QHBoxLayout(); top.addWidget(self.cov, 3); top.addWidget(self.tilt, 2)
        lay = QVBoxLayout(self)
        lay.addLayout(top, 1)
        lay.addWidget(QLabel("<b>Gợi ý chụp thêm</b>"))
        lay.addWidget(self.tips)
        lay.addWidget(self.note)
        self.refresh()

    def refresh(self):
        rs = self.main.results
        size = self.main.calib.image_size if self.main.calib else None
        self.cov.set_figure(report.fig_coverage(rs, size))
        self.tilt.set_figure(report.fig_tilt(rs))
        tips = core.dataset_suggestions(rs) if rs else ["Chưa có ảnh."]
        self.tips.setHtml("<ul>" + "".join(f"<li style='margin-bottom:4px'>{t}</li>" for t in tips) + "</ul>")
        self.note.setText("Góc ngả đang tính bằng K của kết quả hiệu chỉnh." if self.main.calib
                          else "Góc ngả đang là ước lượng (chưa hiệu chỉnh). Mục tiêu: chấm phân tán ra mọi "
                               "hướng, phần lớn nằm giữa vòng 12° và vòng 40°.")


# ==========================================================================
# Tab 4 — Hiệu chỉnh
# ==========================================================================
class CalibTab(QWidget):
    calibrated = Signal()

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main

        self.model = QComboBox()
        for k, v in MODELS.items():
            self.model.addItem(v, k)
            self.model.setItemData(self.model.count() - 1, v, Qt.ToolTipRole)
        self.model.view().setMinimumWidth(420)
        self.fix_aspect = QCheckBox("Cố định fx = fy (điểm ảnh vuông)")
        self.fix_pp = QCheckBox("Cố định tâm ảnh ở giữa ảnh")
        self.zero_tan = QCheckBox("Bỏ méo tiếp tuyến (p1 = p2 = 0)")
        self.auto_rej = QCheckBox("Tự loại ảnh sai số lớn"); self.auto_rej.setChecked(True)
        self.stab = QSpinBox(); self.stab.setRange(0, 100); self.stab.setValue(20)
        self.model.currentIndexChanged.connect(self._model_changed)

        form = QFormLayout()
        form.addRow("Mô hình ống kính", self.model)
        form.addRow("", self.fix_aspect)
        form.addRow("", self.fix_pp)
        form.addRow("", self.zero_tan)
        form.addRow("", self.auto_rej)
        form.addRow("Số lần kiểm tra chéo", self.stab)
        g = QGroupBox("Tùy chọn"); g.setLayout(form)

        self.b_run = QPushButton("▶  Chạy hiệu chỉnh")
        self.b_run.setMinimumHeight(36)
        self.b_run.setStyleSheet("QPushButton { font-weight:600; font-size:14px; }")
        self.b_run.clicked.connect(self.run)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.hide()
        self.pmsg = hint("")

        self.b_yaml = QPushButton("Lưu kết quả (.yaml OpenCV)…")
        self.b_json = QPushButton("Lưu JSON…")
        self.b_ros = QPushButton("Lưu camera_info (ROS)…")
        self.b_html = QPushButton("Xuất báo cáo HTML…")
        self.b_yaml.clicked.connect(lambda: self.export("yaml"))
        self.b_json.clicked.connect(lambda: self.export("json"))
        self.b_ros.clicked.connect(lambda: self.export("ros"))
        self.b_html.clicked.connect(lambda: self.export("html"))
        exp = QGroupBox("Xuất kết quả")
        el = QVBoxLayout(exp)
        for b in (self.b_yaml, self.b_json, self.b_ros, self.b_html):
            el.addWidget(b)
        self._set_export(False)

        left = QVBoxLayout()
        left.addWidget(g)
        left.addWidget(hint("Khuyên dùng: Pinhole k3 = 0 cho camera điện thoại / webcam. Chỉ dùng 5 hệ số hoặc "
                            "Rational khi ống kính méo mạnh và bộ ảnh phủ kín 4 góc. "
                            "Bật “fx = fy” nếu dữ liệu ít mà camera là loại thông thường."))
        left.addWidget(self.b_run)
        left.addWidget(self.progress)
        left.addWidget(self.pmsg)
        left.addWidget(exp)
        left.addStretch()
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(380)

        # ---- kết quả ----
        self.verdict = QLabel("Chưa hiệu chỉnh."); self.verdict.setStyleSheet(banner("info", ""))
        self.verdict.setWordWrap(True)
        self.params = QTableWidget(0, 3)
        self.params.setHorizontalHeaderLabels(["Thông số", "Giá trị", "± σ"])
        self.params.verticalHeader().hide()
        self.params.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.params.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.params.setMinimumHeight(220)
        self.checks = QListWidget()
        self.checks.setWordWrap(True)
        self.checks.setMinimumHeight(220)
        top = QHBoxLayout(); top.addWidget(self.params, 2); top.addWidget(self.checks, 3)

        self.charts = QTabWidget()
        self.c_err = ChartBox(240, natural_height=True); self.c_res = ChartBox(260); self.c_dist = ChartBox(260)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self.c_err); sc.setFrameShape(QFrame.NoFrame)
        self.charts.addTab(sc, "Sai số từng ảnh")
        self.charts.addTab(self.c_res, "Sai số chi tiết")
        self.charts.addTab(self.c_dist, "Méo ống kính")

        right = QVBoxLayout()
        right.addWidget(self.verdict)
        right.addLayout(top)
        right.addWidget(self.charts, 1)
        rw = QWidget(); rw.setLayout(right)

        lay = QHBoxLayout(self)
        lay.addWidget(lw)
        lay.addWidget(rw, 1)

    def _model_changed(self):
        fe = self.model.currentData() == "fisheye"
        for w in (self.fix_aspect, self.zero_tan):
            w.setEnabled(not fe)

    def _set_export(self, on):
        for b in (self.b_yaml, self.b_json, self.b_ros, self.b_html):
            b.setEnabled(on)

    def options(self) -> CalibOptions:
        return CalibOptions(model=self.model.currentData(), fix_aspect=self.fix_aspect.isChecked(),
                            fix_principal=self.fix_pp.isChecked(), zero_tangent=self.zero_tan.isChecked(),
                            auto_reject=self.auto_rej.isChecked(), stability_runs=self.stab.value())

    def run(self):
        cfg = self.main.cfg
        if cfg.validate():
            QMessageBox.warning(self, APP_NAME, "Thông số bảng chưa hợp lệ.")
            return
        if self.main.results and self.main.t_images.needs_redetect():
            self.main.statusBar().showMessage("Thông số bảng đã đổi — đang phân tích lại ảnh trước khi hiệu chỉnh…")
            self.pmsg.setText("Đang phân tích lại ảnh theo thông số bảng mới…")
            self.main.t_images.reanalyze(then=self.run)
            return
        if not any(r.used and r.detected for r in self.main.results):
            if self.main.results:
                QMessageBox.warning(self, APP_NAME, "Không ảnh nào nhận được bảng với thông số hiện tại. Kiểm tra số ô, "
                                    "từ điển ArUco và tùy chọn legacy ở tab 1 có khớp với bảng đã in không.")
            else:
                QMessageBox.information(self, APP_NAME, "Chưa có ảnh nào. Thêm ảnh hoặc video ở tab “Ảnh”.")
            return
        self.main.set_busy(True)
        self.progress.setValue(0); self.progress.show()

        def job(sig, results, cfg, opt):
            return core.calibrate(results, cfg, opt, progress=lambda m, f: sig.message.emit(m, f))

        w = Worker(job, self.main.results, cfg, self.options())
        w.signals.message.connect(self._on_msg)
        w.signals.done.connect(self._on_done)
        w.signals.failed.connect(self._on_failed)
        self.main.start(w)

    def _on_msg(self, m, f):
        self.progress.setValue(int(f * 100))
        self.pmsg.setText(m)

    def _on_failed(self, msg):
        self.progress.hide(); self.pmsg.setText("")
        self.main.set_busy(False)
        QMessageBox.warning(self, APP_NAME, f"Không hiệu chỉnh được:\n{msg}")

    def _on_done(self, res):
        self.progress.hide(); self.pmsg.setText("")
        self.main.set_busy(False)
        self.main.calib = res
        self.show_result(res)
        self.calibrated.emit()

    def show_result(self, res):
        self.verdict.setText(res.verdict)
        self.verdict.setStyleSheet(banner(res.verdict_level, ""))
        std = res.std_intrinsics
        rows = [("fx  (tiêu cự)", res.fx, 0), ("fy  (tiêu cự)", res.fy, 1),
                ("cx  (tâm ảnh)", res.cx, 2), ("cy  (tâm ảnh)", res.cy, 3)]
        kind = {"k": "méo xuyên tâm", "p": "méo tiếp tuyến", "s": "méo lăng kính", "τ": "nghiêng cảm biến"}
        rows += [(f"{n}  ({kind.get(n[0], 'hệ số méo')})", v, 4 + k)
                 for k, (n, v) in enumerate(zip(res.dist_names(), res.dist.ravel()))]
        rows_out = []
        for name, v, si in rows:
            s = ""
            if std is not None and res.model != "fisheye" and len(std) > si and std[si] > 0:
                s = f"{std[si]:.2f}" if si < 4 else f"{std[si]:.4f}"
            rows_out.append((name, f"{v:.2f}" if si < 4 else f"{v:.6f}", s))
        rows_out.append(("RMS (px)", f"{res.rms:.4f}", ""))
        if res.holdout_err is not None:
            rows_out.append(("Sai số ảnh để riêng (px)", f"{res.holdout_err:.4f}", ""))
        rows_out.append(("Số ảnh dùng", str(len(res.used_names)), ""))
        self.params.setRowCount(len(rows_out))
        for i, row in enumerate(rows_out):
            for j, v in enumerate(row):
                it = QTableWidgetItem(v)
                if j:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.params.setItem(i, j, it)
        self.checks.clear()
        sym = {"ok": "✓", "warn": "!", "fail": "✕"}
        for c in res.checks:
            it = QListWidgetItem(f"{sym[c.level]}  {c.text}")
            it.setForeground(QBrush(STATUS_COLORS[c.level]))
            self.checks.addItem(it)
        self.c_err.set_figure(report.fig_errors(self.main.results, res.rms))
        self.c_res.set_figure(report.fig_residuals(res))
        self.c_dist.set_figure(report.fig_distortion(res))
        self._set_export(True)

    def mark_stale(self):
        if self.main.calib is not None:
            self.verdict.setText(self.main.calib.verdict + "   (Danh sách ảnh đã thay đổi — chạy lại để cập nhật.)")

    def export(self, kind):
        res = self.main.calib
        if res is None:
            return
        ext = {"yaml": ("calib.yaml", "YAML (*.yaml *.yml)"), "json": ("calib.json", "JSON (*.json)"),
               "ros": ("camera_info.yaml", "YAML (*.yaml)"), "html": ("bao_cao_hieu_chinh.html", "HTML (*.html)")}[kind]
        path, _ = QFileDialog.getSaveFileName(self, "Lưu", os.path.join(self.main.last_dir(), ext[0]), ext[1])
        if not path:
            return
        try:
            if kind == "yaml":
                core.save_yaml(res, path)
            elif kind == "json":
                core.save_json(res, path)
            elif kind == "ros":
                core.save_ros(res, path)
            else:
                report.build_report(res, self.main.results, self.main.cfg, path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không lưu được:\n{e}")
            return
        self.main.remember_dir(path)
        self.main.statusBar().showMessage(f"Đã lưu {path}", 8000)
        if kind == "html":
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))


# ==========================================================================
# Tab 5 — Khử méo & kiểm tra
# ==========================================================================
class UndistortTab(QWidget):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.loaded = None      # dict từ file
        self.rgb = None
        self.out = None
        self.img_path = None

        self.src = QLabel(); self.src.setWordWrap(True)
        b_load = QPushButton("Mở file kết quả (.yaml / .json)…")
        b_use = QPushButton("Dùng kết quả vừa hiệu chỉnh")
        b_img = QPushButton("Mở ảnh…")
        self.b_save = QPushButton("Lưu ảnh đã khử méo…"); self.b_save.setEnabled(False)
        self.b_verify = QPushButton("Kiểm tra trên ảnh mới…")
        b_load.clicked.connect(self.load_calib)
        b_use.clicked.connect(self.use_current)
        b_img.clicked.connect(self.open_image)
        self.b_save.clicked.connect(self.save_out)
        self.b_verify.clicked.connect(self.verify)

        self.alpha = QSlider(Qt.Horizontal); self.alpha.setRange(0, 100); self.alpha.setValue(0)
        self.alpha.valueChanged.connect(self._apply)
        self.alpha_lb = QLabel()

        row1 = QHBoxLayout()
        for b in (b_use, b_load, b_img, self.b_save, self.b_verify):
            row1.addWidget(b)
        row1.addStretch()
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Cắt viền đen"))
        row2.addWidget(self.alpha, 1)
        row2.addWidget(QLabel("Giữ toàn bộ điểm ảnh"))
        row2.addWidget(self.alpha_lb)

        self.v_in = ImageView("Ảnh gốc")
        self.v_out = ImageView("Ảnh đã khử méo")
        views = QHBoxLayout(); views.addWidget(self.v_in); views.addWidget(self.v_out)
        self.msg = QLabel(); self.msg.setWordWrap(True); self.msg.hide()

        lay = QVBoxLayout(self)
        lay.addWidget(self.src)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.msg)
        lay.addLayout(views, 1)
        lay.addWidget(hint("<b>Kiểm tra trên ảnh mới</b>: chụp thêm 1 ảnh bảng (không đưa vào hiệu chỉnh) ở một "
                           "góc khung hình. Sai số trên ảnh này gần bằng RMS là dấu hiệu kết quả tốt. "
                           "Khi khử méo, những đường thẳng ngoài thực tế (khung cửa, cạnh bàn) phải thẳng."))
        self._update_src()

    def calib(self):
        if self.loaded is not None:
            return self.loaded
        r = self.main.calib
        if r is None:
            return None
        return {"K": r.K, "dist": r.dist, "size": r.image_size, "model": r.model, "rms": r.rms}

    def _update_src(self):
        c = self.calib()
        if c is None:
            self.src.setText("Chưa có kết quả hiệu chỉnh. Chạy hiệu chỉnh ở tab 4, hoặc mở file kết quả đã lưu.")
        else:
            where = "file đã mở" if self.loaded is not None else "kết quả vừa hiệu chỉnh"
            size = f"{c['size'][0]}×{c['size'][1]}" if c.get("size") else "?"
            self.src.setText(f"Đang dùng <b>{where}</b>: fx = {c['K'][0, 0]:.1f}, fy = {c['K'][1, 1]:.1f}, "
                             f"cx = {c['K'][0, 2]:.1f}, cy = {c['K'][1, 2]:.1f} · ảnh {size} · {c['model']}")
        self.alpha_lb.setText(f"α = {self.alpha.value() / 100:.2f}")

    def on_calibrated(self):
        self.loaded = None
        self._update_src()
        self._apply()

    def use_current(self):
        if self.main.calib is None:
            QMessageBox.information(self, APP_NAME, "Chưa có kết quả hiệu chỉnh trong phiên này.")
            return
        self.on_calibrated()

    def load_calib(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mở kết quả", self.main.last_dir(), "Kết quả (*.yaml *.yml *.json)")
        if not path:
            return
        try:
            self.loaded = core.load_calibration(path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không đọc được file:\n{e}")
            return
        self.main.remember_dir(path)
        self._update_src()
        self._apply()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mở ảnh", self.main.last_dir(),
                                              "Ảnh (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        try:
            self.rgb = core.load_image(path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không mở được ảnh:\n{e}")
            return
        self.img_path = path
        self.main.remember_dir(path)
        self.msg.hide()
        self.v_in.set_array(self.rgb)
        self._apply()

    def _size_warning(self, c, w, h):
        if c.get("size") and tuple(c["size"]) != (w, h):
            return (f"Ảnh {w}×{h} khác độ phân giải lúc hiệu chỉnh {c['size'][0]}×{c['size'][1]} — "
                    "kết quả không áp dụng đúng cho ảnh này.")
        return ""

    def _apply(self):
        self._update_src()
        c = self.calib()
        if self.rgb is None or c is None:
            return
        h, w = self.rgb.shape[:2]
        sw = self._size_warning(c, w, h)
        if sw:
            self.msg.setText(sw); self.msg.setStyleSheet(banner("warn", "")); self.msg.show()
        try:
            self.out = core.undistort(self.rgb, c["K"], c["dist"], c["model"], self.alpha.value() / 100)
            self.v_out.set_array(self.out)
            self.b_save.setEnabled(True)
        except Exception as e:
            self.v_out.set_array(None, f"Lỗi khử méo: {e}")

    def save_out(self):
        if self.out is None:
            return
        base = os.path.splitext(os.path.basename(self.img_path or "anh"))[0]
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", os.path.join(self.main.last_dir(), base + "_undistorted.jpg"),
                                              "JPEG (*.jpg);;PNG (*.png)")
        if path:
            core.save_image(path, self.out)
            self.main.remember_dir(path)

    def verify(self):
        c = self.calib()
        if c is None:
            QMessageBox.information(self, APP_NAME, "Cần có kết quả hiệu chỉnh trước.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh bảng để kiểm tra", self.main.last_dir(),
                                              "Ảnh (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if path:
            self.main.remember_dir(path)
            self.verify_path(path)

    def verify_path(self, path):
        c = self.calib()
        if c is None:
            return
        try:
            v = core.verify_image(path, self.main.cfg, c["K"], c["dist"], c["model"])
            rgb = core.load_image(path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Lỗi:\n{e}")
            return
        r = v["result"]
        self.img_path, self.rgb = path, rgb
        if v.get("rvec") is not None:
            axes = (v["rvec"], v["tvec"], self.main.cfg.square_mm * 0.003)
            K = c["K"]; d = c["dist"] if c["model"] != "fisheye" else np.zeros(5)
            self.v_in.set_array(core.draw_detection(rgb, r, K=K, dist=d, axes=axes))
        else:
            self.v_in.set_array(core.draw_detection(rgb, r))
        self._apply()
        if v["err"] is None:
            self.msg.setText(f"Không nhận đủ góc bàn cờ trên ảnh này ({r.n_corners} góc).")
            self.msg.setStyleSheet(banner("fail", ""))
        else:
            rms = c.get("rms")
            ratio = v["err"] / rms if rms else None
            lvl = "ok" if v["err"] <= 1.0 and (ratio is None or ratio <= 2.0) else ("warn" if v["err"] <= 2.0 else "fail")
            txt = f"Sai số chiếu lại trên ảnh mới: {v['err']:.3f} px"
            if ratio is not None:
                txt += f" (gấp {ratio:.1f} lần RMS lúc hiệu chỉnh)"
            if v["distance_m"]:
                txt += f" · khoảng cách tới bảng ≈ {v['distance_m'] * 100:.1f} cm"
            if r.tilt_deg is not None:
                txt += f" · ngả {r.tilt_deg:.0f}°"
            sw = self._size_warning(c, r.width, r.height)
            if sw:
                txt += " · " + sw
                lvl = "warn"
            txt += {"ok": "  → Tốt.", "warn": "  → Tạm được.", "fail": "  → Kém, nên hiệu chỉnh lại."}[lvl]
            self.msg.setText(txt)
            self.msg.setStyleSheet(banner(lvl, ""))
        self.msg.show()


# ==========================================================================
# Cửa sổ chính
# ==========================================================================
HELP_HTML = """
<h3>Quy trình</h3>
<ol>
<li><b>Bảng hiệu chỉnh:</b> chọn thông số, xuất PDF, in 100% (Actual size), đo vạch kiểm tra = 100 mm,
dán phẳng lên tấm cứng, đo lại cạnh ô và nhập số đo thực tế.</li>
<li><b>Chụp ảnh hoặc quay video:</b> 20–30 ảnh, hoặc một video 1–3 phút, cùng chế độ và độ phân giải
sẽ dùng thật. Với video: di chuyển chậm, <b>dừng khoảng 1 giây</b> ở mỗi tư thế.</li>
<li><b>Ảnh:</b> thêm ảnh, xem độ nét / số góc, bỏ chọn ảnh xấu.</li>
<li><b>Độ phủ:</b> xem vùng nào của khung hình còn thiếu, hướng ngả nào còn thiếu.</li>
<li><b>Hiệu chỉnh:</b> chạy, đọc kết luận và các mục kiểm tra, xuất kết quả.</li>
<li><b>Khử méo & kiểm tra:</b> thử trên ảnh mới không dùng để hiệu chỉnh.</li>
</ol>
<h3>Cách chụp</h3>
<ul>
<li><b>Khóa lấy nét</b> (chế độ Pro/Manual) ở khoảng cách sẽ dùng thật và giữ nguyên suốt quá trình.
Không zoom, không đổi ống kính. Tắt chống rung điện tử (EIS) nếu có.</li>
<li>Giữ bảng trong <b>vùng nét</b>: không đưa bảng lại gần hơn khoảng cách đã lấy nét.</li>
<li>Đưa bảng đến <b>khắp 9 vùng</b> của khung hình, đặc biệt <b>4 góc</b> — nơi méo mạnh nhất.</li>
<li>Ở mỗi vị trí, <b>ngả bảng ra khỏi mặt phẳng</b> 20–40°: mép trên ra sau, mép dưới ra sau, mép trái ra xa,
mép phải ra xa. Xoay trong mặt phẳng (như kim đồng hồ) không thay thế được việc này.</li>
<li>Đủ sáng để ảnh không nhòe; giữ máy chắc tay hoặc đặt lên giá và di chuyển bảng.</li>
<li>Bảng nên chiếm ≥ 5–10% diện tích ảnh; nếu quá nhỏ, in khổ lớn hơn thay vì đến gần.</li>
</ul>
<h3>Đọc kết quả</h3>
<ul>
<li>RMS thấp <b>chưa đủ</b>: kiểm tra thêm độ phủ, fx ≈ fy, độ ổn định khi kiểm tra chéo và sai số trên ảnh để riêng.</li>
<li>Kết quả chỉ đúng cho đúng độ phân giải, chế độ chụp và trạng thái lấy nét lúc chụp.</li>
</ul>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon())
        self.resize(1360, 860)
        self.settings = QSettings("ChArUcoCalibrator", "app")
        self.pool = QThreadPool.globalInstance()
        self._workers = []
        self.results: list[ImageResult] = []
        self.calib = None
        self.cfg = BoardConfig()
        last_cfg = self.settings.value("board_cfg", "")
        if last_cfg:
            try:
                import json
                self.cfg = BoardConfig.from_dict(json.loads(last_cfg))
                if self.cfg.validate():
                    self.cfg = BoardConfig()
            except Exception:
                self.cfg = BoardConfig()

        self.tabs = QTabWidget()
        self.t_board = BoardTab(self)
        self.t_images = ImagesTab(self)
        self.t_cov = CoverageTab(self)
        self.t_calib = CalibTab(self)
        self.t_und = UndistortTab(self)
        self.tabs.addTab(self.t_board, "1. Bảng hiệu chỉnh")
        self.tabs.addTab(self.t_images, "2. Ảnh")
        self.tabs.addTab(self.t_cov, "3. Độ phủ")
        self.tabs.addTab(self.t_calib, "4. Hiệu chỉnh")
        self.tabs.addTab(self.t_und, "5. Khử méo && kiểm tra")

        from extrinsic_gui import ExtrinsicWidget
        self.ext = ExtrinsicWidget(self)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.tabs)
        self.stack.addWidget(self.ext)
        self.b_intr = QPushButton("Intrinsic — thông số trong của từng camera")
        self.b_extr = QPushButton("Extrinsic — vị trí các camera trên mặt sàn")
        grp = QButtonGroup(self)
        for k, b in enumerate((self.b_intr, self.b_extr)):
            b.setCheckable(True); b.setMinimumHeight(32)
            grp.addButton(b, k)
        self.b_intr.setChecked(True)
        grp.idClicked.connect(self.set_mode)
        mode_css = ("QPushButton { border:1px solid #c9c8c2; padding:4px 14px; background:#f3f3f1; color:#52514e; }"
                    "QPushButton:checked { background:#2a78d6; color:white; border-color:#2a78d6; font-weight:600; }")
        self.b_intr.setStyleSheet(mode_css + "QPushButton { border-top-left-radius:6px; border-bottom-left-radius:6px; }")
        self.b_extr.setStyleSheet(mode_css + "QPushButton { border-top-right-radius:6px; border-bottom-right-radius:6px; }")
        bar = QHBoxLayout(); bar.setSpacing(0)
        bar.addWidget(self.b_intr); bar.addWidget(self.b_extr); bar.addStretch()
        central = QWidget()
        cl = QVBoxLayout(central); cl.setContentsMargins(6, 6, 6, 0)
        cl.addLayout(bar); cl.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.t_board.changed.connect(self._board_changed)
        self.t_images.results_changed.connect(self.t_cov.refresh)
        self.t_calib.calibrated.connect(self._after_calib)
        self.tabs.currentChanged.connect(self._tab_changed)

        m = self.menuBar().addMenu("Tệp")
        a = QAction("Thêm ảnh…", self); a.triggered.connect(self.t_images.add_files); m.addAction(a)
        a = QAction("Thêm video…", self); a.triggered.connect(lambda: self.t_images.add_video()); m.addAction(a)
        a = QAction("Mở cấu hình bảng…", self); a.triggered.connect(self.t_board.open_cfg); m.addAction(a)
        a = QAction("Lưu cấu hình bảng…", self); a.triggered.connect(self.t_board.save_cfg); m.addAction(a)
        m.addSeparator()
        a = QAction("Thoát", self); a.triggered.connect(self.close); m.addAction(a)
        h = self.menuBar().addMenu("Trợ giúp")
        a = QAction("Hướng dẫn chụp ảnh và quy trình", self); a.triggered.connect(self.show_help); h.addAction(a)
        a = QAction("Giới thiệu", self); a.triggered.connect(self.show_about); h.addAction(a)
        self.statusBar().showMessage("Sẵn sàng. Bắt đầu ở tab 1 (in bảng) hoặc tab 2 (thêm ảnh).")

    def set_mode(self, k: int):
        self.stack.setCurrentIndex(k)
        (self.b_intr if k == 0 else self.b_extr).setChecked(True)
        if k == 1:
            self.ext.refresh_all()
            self.statusBar().showMessage("Chế độ Extrinsic: thêm camera ở tab 1, khai báo gạch và marker ở tab 2.", 8000)

    # ---- tiện ích ----
    def last_dir(self) -> str:
        d = self.settings.value("last_dir", "")
        return d if d and os.path.isdir(d) else os.path.expanduser("~")

    def remember_dir(self, path: str):
        self.settings.setValue("last_dir", os.path.dirname(os.path.abspath(path)))

    def start(self, worker: Worker):
        self._workers.append(worker)
        worker.signals.done.connect(lambda *_: self._workers.remove(worker) if worker in self._workers else None)
        worker.signals.failed.connect(lambda *_: self._workers.remove(worker) if worker in self._workers else None)
        self.pool.start(worker)

    def set_busy(self, busy: bool):
        for t in (self.t_board, self.t_images, self.t_calib):
            for b in t.findChildren(QPushButton):
                b.setEnabled(not busy)
        if not busy:
            self.t_calib._set_export(self.calib is not None)
        self.t_images.table.setEnabled(not busy)
        if busy and self.t_images.b_cancel.isVisible():
            self.t_images.b_cancel.setEnabled(True)
        QApplication.setOverrideCursor(Qt.BusyCursor) if busy else QApplication.restoreOverrideCursor()

    def calib_stale(self):
        self.t_calib.mark_stale()

    def _board_changed(self, structural: bool):
        import json
        self.settings.setValue("board_cfg", json.dumps(self.cfg.to_dict()))
        if structural:
            self.t_images.structure_changed()
        self.calib_stale()

    def _after_calib(self):
        self.t_images.refresh_all()
        self.t_cov.refresh()
        self.t_und.on_calibrated()
        self.statusBar().showMessage(f"Hiệu chỉnh xong — {self.calib.verdict}", 10000)

    def _tab_changed(self, i):
        if self.tabs.widget(i) is self.t_cov:
            self.t_cov.refresh()

    def show_help(self):
        box = QMessageBox(self)
        box.setWindowTitle("Hướng dẫn")
        box.setTextFormat(Qt.RichText)
        box.setText(HELP_HTML)
        box.exec()

    def show_about(self):
        import cv2
        QMessageBox.about(self, APP_NAME,
                          f"<b>{APP_NAME}</b><br>Hiệu chỉnh nội tại camera bằng bảng ChArUco.<br><br>"
                          f"OpenCV {cv2.__version__}")

    def closeEvent(self, e):
        if self._workers:
            if QMessageBox.question(self, APP_NAME, "Đang xử lý. Vẫn thoát?") != QMessageBox.Yes:
                e.ignore()
                return
        if not self.ext.confirm_discard():
            e.ignore()
            return
        e.accept()
