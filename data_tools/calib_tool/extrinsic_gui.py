"""
Giao diện hiệu chỉnh extrinsic nhiều camera (PySide6).
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPainter, QColor, QBrush, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QPushButton, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QDialog, QDialogButtonBox, QRadioButton, QSlider, QInputDialog, QApplication, QFrame,
    QScrollArea, QButtonGroup,
)

import core
import extrinsic_core as ex
import extrinsic_viz as xv
import video as vid
from gui import (ChartBox, Worker, banner, hint, to_qimage, APP_NAME, STATUS_COLORS, ImageView)

LEVEL_TXT = {"ok": "Đạt", "warn": "Cần xem lại", "fail": "Không đạt"}


# ==========================================================================
# Khung xem ảnh có zoom / kéo
# ==========================================================================
class ZoomView(QGraphicsView):
    clicked = Signal(float, float)

    def __init__(self, placeholder=""):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self._text = self._scene.addText(placeholder)
        self._text.setDefaultTextColor(QColor("#52514e"))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#ecebe7")))
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumSize(320, 220)
        self._auto_fit = True
        self._size = None
        self._press = None
        self._panning = False
        self.setToolTip("Lăn chuột để phóng to/thu nhỏ · kéo để di chuyển · nháy đúp để vừa khung")

    def set_image(self, rgb):
        if rgb is None:
            self._item.setPixmap(QPixmap())
            self._text.show()
            return
        self._text.hide()
        pm = QPixmap.fromImage(to_qimage(rgb))
        self._item.setPixmap(pm)
        size = (pm.width(), pm.height())
        self._scene.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))
        if self._auto_fit or size != self._size:
            self._size = size
            self.fit()

    def fit(self):
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self._auto_fit = True

    def wheelEvent(self, e):
        if self._item.pixmap().isNull():
            return
        f = 1.25 ** (e.angleDelta().y() / 120)
        cur = self.transform().m11()
        if (cur < 0.05 and f < 1) or (cur > 40 and f > 1):
            return
        self.scale(f, f)
        self._auto_fit = False

    def mousePressEvent(self, e):
        self._press = e.position()
        self._panning = False
        self._last = e.position()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None and e.buttons():
            d = e.position() - self._press
            if self._panning or abs(d.x()) + abs(d.y()) > 5:
                self._panning = True
                dd = e.position() - self._last
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(dd.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(dd.y()))
                self._auto_fit = False
        self._last = e.position()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._press is not None and not self._panning and e.button() == Qt.LeftButton \
                and not self._item.pixmap().isNull():
            p = self.mapToScene(e.position().toPoint())
            if 0 <= p.x() < self._item.pixmap().width() and 0 <= p.y() < self._item.pixmap().height():
                self.clicked.emit(p.x(), p.y())
        self._press = None
        self._panning = False
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._auto_fit:
            self.fit()


# ==========================================================================
# Hộp thoại
# ==========================================================================
class ManualPointDialog(QDialog):
    def __init__(self, parent, suggest=None, has_image=True):
        super().__init__(parent)
        self.setWindowTitle("Thêm điểm mốc")
        lay = QVBoxLayout(self)
        self.r_tile = QRadioButton("Góc gạch (giao điểm ron) — nhập vị trí theo số viên gạch")
        self.r_xyz = QRadioButton("Điểm có toạ độ đã đo (m)")
        self.r_tile.setChecked(True)
        self.gi = QDoubleSpinBox(); self.gj = QDoubleSpinBox()
        for s in (self.gi, self.gj):
            s.setRange(-500, 500); s.setDecimals(1); s.setSingleStep(1.0)
        if suggest is not None:
            self.gi.setValue(suggest[0]); self.gj.setValue(suggest[1])
        self.X = QDoubleSpinBox(); self.Y = QDoubleSpinBox(); self.Z = QDoubleSpinBox()
        for s in (self.X, self.Y, self.Z):
            s.setRange(-1000, 1000); s.setDecimals(3); s.setSingleStep(0.1); s.setSuffix(" m")
        g1 = QHBoxLayout(); g1.addWidget(QLabel("i")); g1.addWidget(self.gi); g1.addWidget(QLabel("j")); g1.addWidget(self.gj)
        g2 = QHBoxLayout()
        for n, w in (("X", self.X), ("Y", self.Y), ("Z", self.Z)):
            g2.addWidget(QLabel(n)); g2.addWidget(w)
        self.snap = QCheckBox("Bắt điểm vào góc / giao điểm ron gần nhất (dưới 1 pixel)")
        self.snap.setChecked(True)
        lay.addWidget(self.r_tile); lay.addLayout(g1)
        if suggest is not None:
            lay.addWidget(hint(f"Gợi ý từ tư thế hiện tại: góc gạch ({suggest[0]:g}, {suggest[1]:g})."))
        lay.addWidget(self.r_xyz); lay.addLayout(g2)
        lay.addWidget(hint("Toạ độ thế giới: X = i × kích thước gạch theo X, Y = j × kích thước gạch theo Y, Z = 0 là mặt sàn. "
                           "Điểm có độ cao (Z ≠ 0) như góc khung cửa giúp kết quả ổn định hơn."))
        lay.addWidget(self.snap)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Thêm"); bb.button(QDialogButtonBox.Cancel).setText("Hủy")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.r_tile.toggled.connect(self._mode)
        self._mode()

    def _mode(self):
        t = self.r_tile.isChecked()
        self.gi.setEnabled(t); self.gj.setEnabled(t)
        for w in (self.X, self.Y, self.Z):
            w.setEnabled(not t)


class FrameDialog(QDialog):
    """Chọn khung tham chiếu từ video (có thể lấy trung vị nhiều khung để xóa người đi qua)."""

    def __init__(self, parent, path):
        super().__init__(parent)
        self.setWindowTitle("Lấy khung tham chiếu từ video")
        self.resize(980, 680)
        self.path = path
        self.info = vid.video_info(path)
        self.n = max(1, self.info["n_frames"])
        self.fps = self.info["fps"]
        self.result_rgb = None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{os.path.basename(path)}</b> · {self.info['width']}×{self.info['height']} · "
                             f"{self.info['duration']:.1f} s"))
        self.view = ImageView("Đang tải…")
        lay.addWidget(self.view, 1)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, self.n - 1)
        self.tlabel = QLabel()
        row = QHBoxLayout(); row.addWidget(self.slider, 1); row.addWidget(self.tlabel)
        lay.addLayout(row)
        self.median = QCheckBox("Lấy trung vị nhiều khung quanh thời điểm này (xóa người / vật di chuyển)")
        self.span = QDoubleSpinBox(); self.span.setRange(1, 60); self.span.setValue(5); self.span.setSuffix(" giây")
        r2 = QHBoxLayout(); r2.addWidget(self.median); r2.addWidget(QLabel("trong khoảng ±")); r2.addWidget(self.span); r2.addStretch()
        lay.addLayout(r2)
        lay.addWidget(hint("Chọn lúc cảnh trống và các marker sàn đã đặt xong. Camera phải ở đúng vị trí sẽ quay dataset."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Dùng khung này"); bb.button(QDialogButtonBox.Cancel).setText("Hủy")
        bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(120)
        self._timer.timeout.connect(self._show)
        self.slider.valueChanged.connect(lambda *_: self._timer.start())
        self._show()

    def _read(self, idx):
        with vid.OpenedVideo(self.path) as cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
        if not ok or fr is None:
            return None
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    def _show(self):
        i = self.slider.value()
        self.tlabel.setText(f"{i / self.fps:6.1f} s")
        fr = self._read(i)
        self.view.set_array(fr, "Không đọc được khung này")

    def _accept(self):
        i = self.slider.value()
        if self.median.isChecked():
            span = int(self.span.value() * self.fps)
            idxs = np.linspace(max(0, i - span), min(self.n - 1, i + span), 15).astype(int)
            QApplication.setOverrideCursor(Qt.BusyCursor)
            try:
                frames = [f for f in (self._read(k) for k in idxs) if f is not None]
            finally:
                QApplication.restoreOverrideCursor()
            if not frames:
                QMessageBox.warning(self, APP_NAME, "Không đọc được khung hình."); return
            self.result_rgb = np.median(np.stack(frames), axis=0).astype(np.uint8)
        else:
            self.result_rgb = self._read(i)
        if self.result_rgb is None:
            QMessageBox.warning(self, APP_NAME, "Không đọc được khung hình."); return
        self.frame_index = i
        self.accept()


# ==========================================================================
# Tab 1 — Camera & dự án
# ==========================================================================
class CamerasTab(QWidget):
    def __init__(self, ext: "ExtrinsicWidget"):
        super().__init__()
        self.ext = ext
        # dự án
        b_new = QPushButton("Dự án mới"); b_open = QPushButton("Mở dự án…")
        b_save = QPushButton("Lưu"); b_saveas = QPushButton("Lưu thành…")
        b_new.clicked.connect(ext.new_project); b_open.clicked.connect(ext.open_project)
        b_save.clicked.connect(lambda: ext.save_project(False)); b_saveas.clicked.connect(lambda: ext.save_project(True))
        self.proj_label = hint("")
        gp = QGroupBox("Dự án"); gl = QGridLayout(gp)
        gl.addWidget(b_new, 0, 0); gl.addWidget(b_open, 0, 1); gl.addWidget(b_save, 1, 0); gl.addWidget(b_saveas, 1, 1)
        gl.addWidget(self.proj_label, 2, 0, 1, 2)
        # danh sách camera
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)
        b_add = QPushButton("Thêm camera"); b_add.setStyleSheet("font-weight:600;")
        b_del = QPushButton("Xóa"); b_ren = QPushButton("Đổi tên")
        b_add.clicked.connect(self.add_camera); b_del.clicked.connect(self.del_camera); b_ren.clicked.connect(self.rename)
        gc = QGroupBox("Camera"); cl = QVBoxLayout(gc)
        cl.addWidget(self.list, 1)
        r = QHBoxLayout(); r.addWidget(b_add); r.addWidget(b_del); r.addWidget(b_ren); cl.addLayout(r)
        left = QVBoxLayout(); left.addWidget(gp); left.addWidget(gc, 1)
        left.addWidget(hint("Mỗi camera cần: <b>file intrinsic</b> (.yaml/.json/.xml từ chế độ Intrinsic) và một "
                            "<b>ảnh tham chiếu</b> chụp từ đúng vị trí camera, lúc các marker sàn đã đặt xong — hoặc chọn thẳng "
                            "<b>video</b> để khi gán điểm có thể duyệt từng khung bằng phím ← →."))
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(360)
        # camera đang chọn
        self.cam_title = QLabel(); self.cam_title.setStyleSheet("font-weight:600; font-size:14px;")
        self.intr_info = QLabel(); self.intr_info.setWordWrap(True)
        self.ref_info = QLabel(); self.ref_info.setWordWrap(True)
        b_intr = QPushButton("Chọn file intrinsic…"); b_cur = QPushButton("Dùng kết quả intrinsic vừa hiệu chỉnh…")
        b_img = QPushButton("Chọn ảnh tham chiếu…"); b_vid = QPushButton("Lấy khung từ video…")
        b_vsrc = QPushButton("Chọn video (duyệt khung khi gán điểm)…")
        b_vsrc.setToolTip("Dùng cả video làm nguồn: ở tab Gán điểm dùng ← → để chuyển khung, "
                          "nhận diện marker / tìm lưới gạch chạy trên khung đang xem.")
        b_intr.clicked.connect(self.pick_intr); b_cur.clicked.connect(self.use_current_intr)
        b_img.clicked.connect(self.pick_image); b_vid.clicked.connect(self.pick_video)
        b_vsrc.clicked.connect(self.pick_video_source)
        self.warn = QLabel(); self.warn.setWordWrap(True); self.warn.hide()
        self.view = ZoomView("Chưa có ảnh tham chiếu")
        gi = QGridLayout()
        gi.addWidget(QLabel("Intrinsic:"), 0, 0); gi.addWidget(self.intr_info, 0, 1, 1, 3)
        gi.addWidget(b_intr, 1, 1); gi.addWidget(b_cur, 1, 2)
        gi.addWidget(QLabel("Ảnh tham chiếu:"), 2, 0); gi.addWidget(self.ref_info, 2, 1, 1, 3)
        gi.addWidget(b_img, 3, 1); gi.addWidget(b_vid, 3, 2); gi.addWidget(b_vsrc, 3, 3, Qt.AlignLeft)
        gi.setColumnStretch(3, 1)
        right = QVBoxLayout(); right.addWidget(self.cam_title); right.addLayout(gi); right.addWidget(self.warn)
        right.addWidget(self.view, 1)
        self.right = QWidget(); self.right.setLayout(right)
        lay = QHBoxLayout(self); lay.addWidget(lw); lay.addWidget(self.right, 1)
        self.refresh()

    def cam(self):
        r = self.list.currentRow()
        return self.ext.proj.cameras[r] if 0 <= r < len(self.ext.proj.cameras) else None

    def refresh(self):
        p = self.ext.proj
        self.proj_label.setText(f"File: {p.path}" if p.path else "Dự án chưa lưu.")
        cur = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for c in p.cameras:
            st = []
            st.append("✓ intrinsic" if c.K is not None else "✕ intrinsic")
            st.append(("✓ video" if c.is_video else "✓ ảnh") if c.image is not None else "✕ ảnh")
            if c.has_pose:
                st.append(f"đã giải ({LEVEL_TXT[c.result.level]})")
            it = QListWidgetItem(f"{c.name}   ·   {' · '.join(st)}")
            if c.has_pose:
                it.setForeground(QBrush(STATUS_COLORS[c.result.level]))
            self.list.addItem(it)
        self.list.blockSignals(False)
        if p.cameras:
            self.list.setCurrentRow(min(max(cur, 0), len(p.cameras) - 1))
        self._select()

    def _select(self, *_):
        c = self.cam()
        self.right.setEnabled(c is not None)
        if c is None:
            self.cam_title.setText("Chưa có camera — bấm “Thêm camera”.")
            self.intr_info.setText(""); self.ref_info.setText(""); self.warn.hide(); self.view.set_image(None)
            return
        self.cam_title.setText(c.name)
        if c.K is not None:
            sz = f" · {c.intr_size[0]}×{c.intr_size[1]}" if c.intr_size else ""
            self.intr_info.setText(f"{os.path.basename(c.intr_path)} — fx {c.K[0, 0]:.1f}, fy {c.K[1, 1]:.1f}, "
                                   f"cx {c.K[0, 2]:.1f}, cy {c.K[1, 2]:.1f}{sz} · {c.model}")
        else:
            self.intr_info.setText("<span style='color:#b42f2f'>Chưa có</span>")
        if c.is_video:
            self.ref_info.setText(f"Video {os.path.basename(c.video_path)} — {c.size[0]}×{c.size[1]} · {c.n_frames} khung · "
                                  f"đang dùng khung {c.frame_idx} ({c.frame_idx / c.fps:.1f} s). "
                                  f"Đổi khung ở tab “Gán điểm” bằng phím ← →.")
        elif c.video_path:
            self.ref_info.setText(f"<span style='color:#b42f2f'>Không mở được video {os.path.basename(c.video_path)}</span>")
        else:
            self.ref_info.setText(f"{os.path.basename(c.ref_path)} — {c.size[0]}×{c.size[1]}" if c.image is not None
                                  else "<span style='color:#b42f2f'>Chưa có</span>")
        w = c.size_warning()
        self.warn.setVisible(bool(w)); self.warn.setText(w); self.warn.setStyleSheet(banner("warn", ""))
        self.view.set_image(c.image)

    def add_camera(self):
        name, ok = QInputDialog.getText(self, "Thêm camera", "Tên camera:", text=self.ext.proj.unique_name())
        if not ok or not name.strip():
            return
        name = name.strip()
        if self.ext.proj.camera(name):
            QMessageBox.warning(self, APP_NAME, "Tên camera đã tồn tại."); return
        self.ext.proj.cameras.append(ex.CameraEx(name))
        self.ext.changed()
        self.refresh()
        self.list.setCurrentRow(len(self.ext.proj.cameras) - 1)

    def del_camera(self):
        c = self.cam()
        if c and QMessageBox.question(self, APP_NAME, f"Xóa camera {c.name}?") == QMessageBox.Yes:
            c.close_video()
            self.ext.proj.cameras.remove(c)
            for tp in self.ext.proj.test_points:
                tp.clicks.pop(c.name, None)
            self.ext.changed(); self.refresh()

    def rename(self):
        c = self.cam()
        if not c:
            return
        name, ok = QInputDialog.getText(self, "Đổi tên", "Tên mới:", text=c.name)
        name = name.strip()
        if ok and name and name != c.name:
            if self.ext.proj.camera(name):
                QMessageBox.warning(self, APP_NAME, "Tên camera đã tồn tại."); return
            for tp in self.ext.proj.test_points:
                if c.name in tp.clicks:
                    tp.clicks[name] = tp.clicks.pop(c.name)
            c.name = name
            self.ext.changed(); self.refresh()

    def pick_intr(self):
        c = self.cam()
        if not c:
            return
        p, _ = QFileDialog.getOpenFileName(self, "File intrinsic", self.ext.main.last_dir(),
                                           "Intrinsic (*.yaml *.yml *.json *.xml)")
        if not p:
            return
        try:
            c.load_intrinsics(p)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không đọc được:\n{e}"); return
        self.ext.main.remember_dir(p)
        c.result = None
        self.ext.changed(); self.refresh()

    def use_current_intr(self):
        c = self.cam()
        res = self.ext.main.calib
        if not c:
            return
        if res is None:
            QMessageBox.information(self, APP_NAME, "Chưa có kết quả hiệu chỉnh intrinsic trong phiên này. "
                                                    "Chạy hiệu chỉnh ở chế độ Intrinsic, hoặc chọn file đã lưu."); return
        p, _ = QFileDialog.getSaveFileName(self, "Lưu kết quả intrinsic cho camera này",
                                           os.path.join(self.ext.main.last_dir(), f"intr_{c.name}.yaml"), "YAML (*.yaml)")
        if not p:
            return
        core.save_yaml(res, p)
        c.load_intrinsics(p); c.result = None
        self.ext.main.remember_dir(p)
        self.ext.changed(); self.refresh()

    def pick_image(self):
        c = self.cam()
        if not c:
            return
        p, _ = QFileDialog.getOpenFileName(self, "Ảnh tham chiếu", self.ext.main.last_dir(),
                                           "Ảnh (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if not p:
            return
        try:
            c.load_image(p)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không mở được ảnh:\n{e}"); return
        self.ext.main.remember_dir(p)
        self.ext.changed(); self.refresh()

    def pick_video_source(self):
        c = self.cam()
        if not c:
            return
        p, _ = QFileDialog.getOpenFileName(self, "Video từ camera này", self.ext.main.last_dir(),
                                           "Video (" + " ".join("*" + e for e in vid.VIDEO_EXTS) + ")")
        if not p:
            return
        self.ext.main.remember_dir(p)
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            c.load_video(p)
            err = None
        except Exception as e:
            err = e
        QApplication.restoreOverrideCursor()
        if err is not None:
            QMessageBox.warning(self, APP_NAME, f"Không mở được video:\n{err}"); return
        self.ext.changed(); self.refresh()
        self.ext.main.statusBar().showMessage(
            f"{c.name}: dùng video làm nguồn ({c.n_frames} khung). Sang tab “Gán điểm” và dùng ← → để chọn khung.", 10000)

    def pick_video(self):
        c = self.cam()
        if not c:
            return
        p, _ = QFileDialog.getOpenFileName(self, "Video từ camera này", self.ext.main.last_dir(),
                                           "Video (" + " ".join("*" + e for e in vid.VIDEO_EXTS) + ")")
        if not p:
            return
        self.ext.main.remember_dir(p)
        try:
            dlg = FrameDialog(self, p)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, str(e)); return
        if dlg.exec() != QDialog.Accepted:
            return
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(os.path.dirname(p), f"{stem}_ref_f{dlg.frame_index:06d}.png")
        ok, buf = cv2.imencode(".png", cv2.cvtColor(dlg.result_rgb, cv2.COLOR_RGB2BGR))
        buf.tofile(out)
        c.load_image(out)
        self.ext.changed(); self.refresh()
        self.ext.main.statusBar().showMessage(f"Đã lưu khung tham chiếu: {out}", 8000)


# ==========================================================================
# Tab 2 — Hệ toạ độ
# ==========================================================================
class WorldTab(QWidget):
    def __init__(self, ext: "ExtrinsicWidget"):
        super().__init__()
        self.ext = ext
        self._block = False
        self.px = QDoubleSpinBox(); self.py = QDoubleSpinBox()
        for s in (self.px, self.py):
            s.setRange(20, 3000); s.setDecimals(1); s.setSuffix(" mm"); s.setSingleStep(0.5)
        self.dic = QComboBox(); self.dic.addItems(ex.FLOOR_DICTS)
        self.msize = QDoubleSpinBox(); self.msize.setRange(30, 800); self.msize.setDecimals(1); self.msize.setSuffix(" mm")
        f = QFormLayout()
        f.addRow("Kích thước 1 viên theo X", self.px)
        f.addRow("Kích thước 1 viên theo Y", self.py)
        g1 = QGroupBox("Gạch lát sàn"); v1 = QVBoxLayout(g1); v1.addLayout(f)
        v1.addWidget(hint("Tính cả ron. Đo chiều dài <b>10 viên liên tiếp</b> rồi chia 10. Kích thước gạch quyết định thang mét "
                          "của cả hệ thống: sai 0.5% thì mọi vị trí co giãn đều 0.5% (các camera vẫn khớp nhau)."))
        f2 = QFormLayout(); f2.addRow("Từ điển marker", self.dic); f2.addRow("Cạnh ô đen của marker", self.msize)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "i (viên)", "j (viên)", "Mũi tên chỉ"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self._table_changed)
        b_addm = QPushButton("Thêm marker"); b_delm = QPushButton("Xóa marker")
        b_addm.clicked.connect(self.add_marker); b_delm.clicked.connect(self.del_marker)
        self.paper = QComboBox(); self.paper.addItems(["A4", "A3"])
        b_pdf = QPushButton("Xuất PDF marker để in…"); b_pdf.clicked.connect(self.export_pdf)
        g2 = QGroupBox("Marker ArUco đặt trên sàn"); v2 = QVBoxLayout(g2); v2.addLayout(f2); v2.addWidget(self.table, 1)
        r = QHBoxLayout(); r.addWidget(b_addm); r.addWidget(b_delm); v2.addLayout(r)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Khổ giấy")); r2.addWidget(self.paper); r2.addWidget(b_pdf, 1); v2.addLayout(r2)
        v2.addWidget(hint("Đặt <b>tâm</b> marker trùng một góc gạch, hai đường nét đứt trên giấy trùng đường ron. "
                          "Nhập vị trí (i, j) = số viên gạch tính từ gốc, và hướng của mũi tên in trên giấy. "
                          "Mỗi camera nên thấy ít nhất 2 marker."))
        ax = QGroupBox("Quy ước trục"); va = QVBoxLayout(ax)
        va.addWidget(hint("Gốc (0, 0) tại một góc gạch — nên dán băng keo màu để đánh dấu. X và Y chạy dọc theo đường ron, "
                          "Z hướng lên, mặt sàn Z = 0. <b>Đứng ở gốc, nhìn theo hướng +X thì +Y ở bên TRÁI.</b>"))
        left = QVBoxLayout(); left.addWidget(ax); left.addWidget(g1); left.addWidget(g2, 1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(470)
        self.chart = ChartBox(380)
        lay = QHBoxLayout(self); lay.addWidget(lw); lay.addWidget(self.chart, 1)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(350)
        self._timer.timeout.connect(self._apply)
        for w in (self.px, self.py, self.msize):
            w.valueChanged.connect(self._changed)
        self.dic.currentIndexChanged.connect(self._changed)
        self.refresh()

    def refresh(self):
        w = self.ext.proj.world
        self._block = True
        self.px.setValue(w.pitch_x_mm); self.py.setValue(w.pitch_y_mm)
        self.dic.setCurrentText(w.marker_dict); self.msize.setValue(w.marker_mm)
        self.table.setRowCount(0)
        for m in w.markers:
            self._add_row(m)
        self._block = False
        self.chart.set_figure(xv.fig_world_setup(w, self.ext.proj.roi))

    def _add_row(self, m):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate([m.id, m.i, m.j]):
            self.table.setItem(r, c, QTableWidgetItem(f"{v:g}"))
        cb = QComboBox(); cb.addItems(list(ex.ARROWS.keys())); cb.setCurrentText(m.arrow)
        cb.currentIndexChanged.connect(self._changed)
        self.table.setCellWidget(r, 3, cb)

    def _read(self):
        w = self.ext.proj.world
        w.pitch_x_mm, w.pitch_y_mm = self.px.value(), self.py.value()
        w.marker_dict, w.marker_mm = self.dic.currentText(), self.msize.value()
        ms, bad = [], False
        for r in range(self.table.rowCount()):
            try:
                mid = int(float(self.table.item(r, 0).text()))
                i = float(self.table.item(r, 1).text().replace(",", "."))
                j = float(self.table.item(r, 2).text().replace(",", "."))
            except (ValueError, AttributeError):
                bad = True; continue
            ms.append(ex.MarkerPlacement(mid, i, j, self.table.cellWidget(r, 3).currentText()))
        ids = [m.id for m in ms]
        w.markers = ms
        return bad, len(ids) != len(set(ids))

    def _changed(self, *_):
        if not self._block:
            self._timer.start()

    def _table_changed(self, *_):
        self._changed()

    def _apply(self):
        bad, dup = self._read()
        if dup:
            self.ext.main.statusBar().showMessage("Có ID marker bị trùng — mỗi marker phải có ID riêng.", 8000)
        elif bad:
            self.ext.main.statusBar().showMessage("Có ô nhập không phải số trong bảng marker.", 8000)
        self.ext.world_changed()
        self.chart.set_figure(xv.fig_world_setup(self.ext.proj.world, self.ext.proj.roi))

    def add_marker(self):
        used = {m.id for m in self.ext.proj.world.markers}
        mid = next(k for k in range(1000) if k not in used)
        self._block = True
        self._add_row(ex.MarkerPlacement(mid, 0, 0, "+X"))
        self._block = False
        self._apply()

    def del_marker(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.currentRow() >= 0:
            rows = [self.table.currentRow()]
        self._block = True
        for r in rows:
            self.table.removeRow(r)
        self._block = False
        self._apply()

    def export_pdf(self):
        self._read()
        w = self.ext.proj.world
        if not w.markers:
            QMessageBox.information(self, APP_NAME, "Thêm marker vào bảng trước (mỗi dòng một marker)."); return
        p, _ = QFileDialog.getSaveFileName(self, "Xuất PDF marker sàn",
                                           os.path.join(self.ext.main.last_dir(), f"marker_san_{w.marker_dict}_{w.marker_mm:g}mm.pdf"),
                                           "PDF (*.pdf)")
        if not p:
            return
        try:
            ex.generate_marker_pdf(w, [m.id for m in w.markers], p, self.paper.currentText())
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không xuất được:\n{e}"); return
        self.ext.main.remember_dir(p)
        QMessageBox.information(self, APP_NAME, f"Đã lưu {p}\n\nIn 100% / Actual size, đo lại cạnh ô đen và nhập số đo "
                                               "vào ô “Cạnh ô đen của marker”.")


# ==========================================================================
# Tab 3 — Gán điểm & giải
# ==========================================================================
PCOLS = ["Dùng", "Nguồn", "Nhãn", "X (m)", "Y (m)", "Z (m)", "u (px)", "v (px)", "Sai số px", "Sai số cm"]


class PointsTab(QWidget):
    def __init__(self, ext: "ExtrinsicWidget"):
        super().__init__()
        self.ext = ext
        self._sel = None
        self._upd = False
        self.combo = QComboBox(); self.combo.setMinimumWidth(140)
        self.combo.currentIndexChanged.connect(self._cam_changed)
        self.b_mk = QPushButton("① Tự nhận diện marker"); self.b_tile = QPushButton("② Tìm lưới gạch")
        self.b_solve = QPushButton("Giải"); self.b_all = QPushButton("Giải tất cả camera")
        self.b_add = QPushButton("＋ Thêm điểm (click)"); self.b_add.setCheckable(True)
        for b in (self.b_mk, self.b_tile):
            b.setStyleSheet("font-weight:600;")
        self.b_mk.clicked.connect(self.detect_markers); self.b_tile.clicked.connect(self.find_tiles)
        self.b_solve.clicked.connect(self.solve); self.b_all.clicked.connect(self.solve_all)
        self.c_grid = QCheckBox("Lưới gạch"); self.c_err = QCheckBox("Vector sai số ×10"); self.c_roi = QCheckBox("Vùng quay")
        for c in (self.c_grid, self.c_roi):
            c.setChecked(True)
        for c in (self.c_grid, self.c_err, self.c_roi):
            c.toggled.connect(self.redraw)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Camera")); bar.addWidget(self.combo)
        for w in (self.b_mk, self.b_tile, self.b_add, self.b_solve, self.b_all):
            bar.addWidget(w)
        bar.addStretch()
        for c in (self.c_grid, self.c_err, self.c_roi):
            bar.addWidget(c)
        self.view = ZoomView("Chọn camera có ảnh tham chiếu")
        self.view.clicked.connect(self._clicked)
        legend = QLabel("<span style='color:#2a78d6'>●</span> marker &nbsp; <span style='color:#1baf7a'>●</span> góc gạch &nbsp; "
                        "<span style='color:#e87ba4'>●</span> thủ công &nbsp; <span style='color:#d03b3b'>✕</span> bị loại &nbsp; "
                        "<span style='color:#969696'>✕</span> đang tắt &nbsp; <span style='color:#eb6834'>━</span> vùng quay "
                        "&nbsp;·&nbsp; lưới trắng = lưới gạch theo tư thế đã giải (phải trùng ron thật)")
        legend.setStyleSheet("color:#52514e;")
        # thanh duyệt khung (khi nguồn là video)
        self.b_prev = QPushButton("◀"); self.b_next = QPushButton("▶")
        for b in (self.b_prev, self.b_next):
            b.setFixedWidth(36); b.setAutoRepeat(True)
        self.b_prev.clicked.connect(lambda: self.step_frame(-1)); self.b_next.clicked.connect(lambda: self.step_frame(1))
        self.fslider = QSlider(Qt.Horizontal); self.fslider.setFocusPolicy(Qt.NoFocus)
        self.flabel = QLabel(); self.flabel.setMinimumWidth(190)
        fb = QHBoxLayout(); fb.setContentsMargins(0, 0, 0, 0)
        fb.addWidget(self.b_prev); fb.addWidget(self.fslider, 1); fb.addWidget(self.b_next); fb.addWidget(self.flabel)
        self.fbar = QWidget(); self.fbar.setLayout(fb)
        self.fbar.setToolTip("← / → : 1 khung · Shift + ← / → : 10 khung · Ctrl + ← / → : 1 giây")
        self._ftimer = QTimer(self); self._ftimer.setSingleShot(True); self._ftimer.setInterval(30)
        self._ftimer.timeout.connect(self._load_frame)
        self.fslider.valueChanged.connect(self._slider_moved)
        for key, step in (("Left", -1), ("Right", 1), ("Shift+Left", -10), ("Shift+Right", 10),
                          ("Ctrl+Left", "-s"), ("Ctrl+Right", "+s")):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda st=step: self.step_frame(st))
        lv = QVBoxLayout(); lv.setContentsMargins(0, 0, 0, 0); lv.addWidget(self.view, 1); lv.addWidget(self.fbar)
        lv.addWidget(legend)
        vw = QWidget(); vw.setLayout(lv)
        # bên phải
        self.status = QLabel("Chưa giải."); self.status.setWordWrap(True); self.status.setStyleSheet(banner("info", ""))
        self.summary = QLabel(); self.summary.setWordWrap(True); self.summary.setTextFormat(Qt.RichText)
        self.warns = QListWidget(); self.warns.setWordWrap(True); self.warns.setMaximumHeight(130)
        self.table = QTableWidget(0, len(PCOLS))
        self.table.setHorizontalHeaderLabels(PCOLS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for i in range(len(PCOLS)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._row_selected)
        b_tog = QPushButton("Bật / tắt điểm chọn"); b_del = QPushButton("Xóa điểm chọn"); b_clr = QPushButton("Xóa mọi góc gạch")
        b_tog.clicked.connect(self.toggle_sel); b_del.clicked.connect(self.delete_sel); b_clr.clicked.connect(self.clear_tiles)
        rb = QHBoxLayout(); rb.addWidget(b_tog); rb.addWidget(b_del); rb.addWidget(b_clr)
        rv = QVBoxLayout(); rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.status); rv.addWidget(self.summary); rv.addWidget(self.warns)
        rv.addWidget(QLabel("<b>Điểm mốc</b>")); rv.addWidget(self.table, 1); rv.addLayout(rb)
        rw = QWidget(); rw.setLayout(rv)
        split = QSplitter(); split.addWidget(vw); split.addWidget(rw)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self); lay.addLayout(bar); lay.addWidget(split, 1)
        lay.addWidget(hint("Quy trình: ① nhận diện marker → tự giải tư thế ban đầu → ② tìm lưới gạch (tự dự đoán và tinh chỉnh "
                           "hàng trăm giao điểm ron) → kiểm tra lưới trắng trùng ron thật ở cả vùng gần và xa. Không có marker: "
                           "bật “Thêm điểm”, click ≥ 4 góc gạch trải rộng rồi bấm Giải. Lăn chuột để phóng to khi click. "
                           "Nguồn là video: ← → để chuyển khung (Shift: 10 khung, Ctrl: 1 giây) — chọn khung thấy rõ mốc rồi "
                           "click; nhận diện marker / tìm lưới gạch chạy trên khung đang xem."))

    # ---- tiện ích ----
    def cam(self):
        name = self.combo.currentText()
        return self.ext.proj.camera(name)

    def refresh(self):
        cur = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        for c in self.ext.proj.cameras:
            self.combo.addItem(c.name)
        if cur and self.combo.findText(cur) >= 0:
            self.combo.setCurrentText(cur)
        self.combo.blockSignals(False)
        self._cam_changed()

    def _cam_changed(self, *_):
        self._sel = None
        c = self.cam()
        ok = c is not None and c.ready()
        for b in (self.b_mk, self.b_tile, self.b_solve, self.b_add):
            b.setEnabled(ok)
        self.fbar.setVisible(c is not None and c.is_video)
        if c is not None and c.is_video:
            self.fslider.blockSignals(True)
            self.fslider.setRange(0, c.n_frames - 1); self.fslider.setValue(c.frame_idx)
            self.fslider.blockSignals(False)
            self._frame_label(c.frame_idx)
        self.fill_table()
        self.show_result()
        self.redraw(fit=True)

    def redraw(self, *_, fit=False):
        c = self.cam()
        if c is None or c.image is None:
            self.view.set_image(None)
            return
        img = xv.draw_overlay(c, self.ext.proj.world, self.ext.proj.roi, show_grid=self.c_grid.isChecked(),
                              show_err=self.c_err.isChecked(), show_roi=self.c_roi.isChecked(), selected=self._sel)
        self.view.set_image(img)

    def fill_table(self):
        c = self.cam()
        self._upd = True
        self.table.setRowCount(0)
        if c is None:
            self._upd = False; return
        self.table.setRowCount(len(c.corrs))
        for r, p in enumerate(c.corrs):
            vals = ["", ex.SOURCES.get(p.source, p.source), p.label, f"{p.X:.3f}", f"{p.Y:.3f}", f"{p.Z:.3f}",
                    f"{p.u:.1f}", f"{p.v:.1f}", "" if p.err_px is None else f"{p.err_px:.2f}",
                    "" if p.err_cm is None else f"{p.err_cm:.1f}"]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k == 0:
                    it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    it.setCheckState(Qt.Checked if p.enabled else Qt.Unchecked)
                elif k >= 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if not p.enabled:
                    it.setForeground(QBrush(QColor("#969696")))
                elif c.has_pose and not p.inlier:
                    it.setForeground(QBrush(STATUS_COLORS["fail"]))
                self.table.setItem(r, k, it)
        self._upd = False

    def show_result(self):
        c = self.cam()
        self.warns.clear()
        if c is None:
            self.status.setText("Chưa có camera."); self.status.setStyleSheet(banner("info", "")); self.summary.setText("")
            return
        if not c.ready():
            miss = [n for n, ok in (("intrinsic", c.K is not None), ("ảnh tham chiếu", c.image is not None)) if not ok]
            self.status.setText(f"{c.name}: thiếu {', '.join(miss)} (xem tab Camera)."); self.status.setStyleSheet(banner("warn", ""))
            self.summary.setText(""); return
        r = c.result
        n = {s: sum(1 for p in c.corrs if p.source == s) for s in ex.SOURCES}
        cnt = f"{n['marker'] // 4 if n['marker'] else 0} marker ({n['marker']} góc) · {n['tile']} góc gạch · {n['manual']} điểm thủ công"
        if r is None:
            self.status.setText(f"{c.name}: chưa giải."); self.status.setStyleSheet(banner("info", ""))
            self.summary.setText(cnt); return
        self.status.setText(f"{c.name}: {LEVEL_TXT[r.level]}")
        self.status.setStyleSheet(banner(r.level, ""))
        C = r.C
        cs = r.C_std_cm
        pm = f" ± {max(cs):.1f} cm" if cs else ""
        R = c.R()
        zc = R.T @ np.array([0, 0, 1.0])
        tilt = math.degrees(math.asin(max(-1, min(1, -zc[2]))))
        self.summary.setText(
            f"{cnt}<br>Dùng <b>{r.n_inliers}/{r.n_total}</b> điểm · RMS <b>{r.rms:.2f} px</b> · sai số trên sàn trung vị "
            f"<b>{r.med_cm:.1f} cm</b> (lớn nhất {r.max_cm:.1f})<br>Vị trí camera: X = {C[0]:.3f}, Y = {C[1]:.3f}, "
            f"<b>cao {C[2]:.3f} m</b>{pm} · nhìn chúc xuống {tilt:.0f}°")
        for lv, t in r.warnings:
            it = QListWidgetItem(("✕  " if lv == "fail" else "!  " if lv == "warn" else "·  ") + t)
            it.setForeground(QBrush(STATUS_COLORS.get(lv, QColor("#52514e"))))
            self.warns.addItem(it)
        if not r.warnings:
            it = QListWidgetItem("✓  Không có cảnh báo."); it.setForeground(QBrush(STATUS_COLORS["ok"]))
            self.warns.addItem(it)

    def after_change(self, resolve=True):
        c = self.cam()
        if resolve and c is not None and c.has_pose:
            try:
                ex.solve(c, self.ext.proj.world, self.ext.proj.roi)
            except Exception:
                pass
        self.fill_table(); self.show_result(); self.redraw()
        self.ext.changed()

    # ---- duyệt khung video ----
    def showEvent(self, e):
        super().showEvent(e)
        self.view.setFocus()           # để phím ← → dùng được ngay

    def _frame_label(self, i):
        c = self.cam()
        if c is not None and c.is_video:
            self.flabel.setText(f"khung {i} / {c.n_frames - 1} · {i / c.fps:.2f} s")

    def step_frame(self, step):
        c = self.cam()
        if c is None or not c.is_video:
            return
        if isinstance(step, str):
            step = (1 if step[0] == "+" else -1) * max(1, round(c.fps))
        self.fslider.setValue(self.fslider.value() + step)

    def _slider_moved(self, i):
        self._frame_label(i)
        if not self._ftimer.isActive():
            self._ftimer.start()

    def _load_frame(self):
        c = self.cam()
        if c is None or not c.is_video:
            return
        i = self.fslider.value()
        if i == c.frame_idx:
            return
        if not c.set_frame(i):
            self.ext.main.statusBar().showMessage(f"Không đọc được khung {i}.", 4000)
            return
        self.redraw()
        self.ext.changed()
        if self.fslider.value() != i:      # thanh trượt đã chạy tiếp trong lúc đọc
            self._ftimer.start()

    def _frame_note(self, c):
        return f" (khung {c.frame_idx})" if c.is_video else ""

    # ---- hành động ----
    def detect_markers(self):
        c = self.cam()
        if not c or not c.ready():
            return
        w = self.ext.proj.world
        if not w.markers:
            QMessageBox.information(self, APP_NAME, "Chưa khai báo marker nào ở tab “Hệ toạ độ”."); return
        corrs, unknown, side = ex.detect_floor_markers(c, w)
        old = {p.label: p.enabled for p in c.corrs if p.source == "marker"}
        for p in corrs:
            p.enabled = old.get(p.label, True)
        c.corrs = [p for p in c.corrs if p.source != "marker"] + corrs
        ids = sorted({int(p.label[1:].split("-")[0]) for p in corrs})
        msg = [f"Tìm thấy {len(ids)} marker{self._frame_note(c)}: {', '.join(map(str, ids)) if ids else '—'}."]
        if unknown:
            msg.append(f"Marker chưa khai báo (bỏ qua): {', '.join(map(str, sorted(set(unknown))))}.")
        if side is not None and side < 20:
            msg.append(f"Marker nhỏ nhất chỉ {side:.0f} px — nên in marker to hơn để góc chính xác hơn.")
        if len([p for p in c.corrs if p.enabled]) >= 4:
            try:
                ex.solve(c, w, self.ext.proj.roi)
                msg.append("Đã giải tư thế ban đầu. Bấm “② Tìm lưới gạch” để thêm điểm.")
            except Exception as e:
                msg.append(f"Chưa giải được: {e}")
        else:
            msg.append("Cần ít nhất 1–2 marker (hoặc thêm điểm thủ công) để giải.")
        self.after_change(resolve=False)
        self.ext.main.statusBar().showMessage(" ".join(msg), 12000)
        if not ids:
            QMessageBox.information(self, APP_NAME, "\n".join(msg + ["", "Kiểm tra từ điển marker ở tab “Hệ toạ độ” và "
                                                                       "marker có nằm trong ảnh, đủ to, đủ sáng."
                                                                       + (" Với video: thử chuyển sang khung khác (← →)."
                                                                          if c.is_video else "")]))

    def find_tiles(self):
        c = self.cam()
        if not c or not c.ready():
            return
        if not c.has_pose:
            QMessageBox.information(self, APP_NAME, "Cần tư thế ban đầu: nhận diện marker hoặc thêm ≥ 4 điểm thủ công rồi bấm Giải.")
            return
        proj = self.ext.proj

        def job(sig, cam):
            return ex.auto_tiles(cam, proj.world, proj.roi, progress=lambda m, f: sig.message.emit(m, f))

        def done(n):
            self.after_change(resolve=False)
            self.ext.main.statusBar().showMessage(f"{c.name}: tìm được {n} góc gạch{self._frame_note(c)}.", 10000)
        self.ext.run_job(job, done, f"Đang tìm lưới gạch cho {c.name}…", c)

    def solve(self):
        c = self.cam()
        if not c:
            return
        try:
            ex.solve(c, self.ext.proj.world, self.ext.proj.roi)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, str(e)); return
        self.after_change(resolve=False)

    def solve_all(self):
        errs = []
        for c in self.ext.proj.cameras:
            if c.ready() and sum(1 for p in c.corrs if p.enabled) >= 4:
                try:
                    ex.solve(c, self.ext.proj.world, self.ext.proj.roi)
                except Exception as e:
                    errs.append(f"{c.name}: {e}")
            elif c.ready():
                errs.append(f"{c.name}: chưa đủ điểm mốc.")
        self.after_change(resolve=False)
        if errs:
            QMessageBox.information(self, APP_NAME, "\n".join(errs))

    def _clicked(self, x, y):
        c = self.cam()
        if not c or not c.ready():
            return
        if self.b_add.isChecked():
            suggest = None
            if c.has_pose:
                p = c.backproject(np.array([[x, y]]), 0.0)[0]
                if not np.isnan(p).any():
                    w = self.ext.proj.world
                    suggest = (round(p[0] / w.px), round(p[1] / w.py))
            dlg = ManualPointDialog(self, suggest)
            if dlg.exec() != QDialog.Accepted:
                return
            u, v = x, y
            if dlg.snap.isChecked():
                gray = cv2.cvtColor(c.image, cv2.COLOR_RGB2GRAY)
                u, v = ex.snap_point(gray, x, y, win=max(4, int(c.size[0] / 400)))
            if dlg.r_tile.isChecked():
                gi, gj = dlg.gi.value(), dlg.gj.value()
                X, Y = self.ext.proj.world.grid_to_world(gi, gj)
                p = ex.Corr(X, Y, 0.0, u, v, "manual", f"({gi:g},{gj:g})", gi=gi, gj=gj)
            else:
                p = ex.Corr(dlg.X.value(), dlg.Y.value(), dlg.Z.value(), u, v, "manual",
                            f"({dlg.X.value():g},{dlg.Y.value():g},{dlg.Z.value():g})")
            c.corrs.append(p)
            k = len(c.corrs) - 1
            self.after_change(resolve=True)
            self.table.selectRow(k)
            self.table.scrollToItem(self.table.item(k, 0))
            return
        # chọn điểm gần nhất
        if not c.corrs:
            return
        d = [math.hypot(p.u - x, p.v - y) for p in c.corrs]
        k = int(np.argmin(d))
        tol = 15 * max(1.0, 1.0 / max(self.view.transform().m11(), 1e-3))
        if d[k] <= tol:
            self.table.selectRow(k)
            self.table.scrollToItem(self.table.item(k, 0))

    def _row_selected(self):
        r = self.table.currentRow()
        self._sel = r if r >= 0 else None
        self.redraw()

    def _item_changed(self, it):
        if self._upd or it.column() != 0:
            return
        c = self.cam()
        c.corrs[it.row()].enabled = it.checkState() == Qt.Checked
        self.after_change(resolve=True)

    def _selected_rows(self):
        return sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)

    def toggle_sel(self):
        c = self.cam()
        if not c:
            return
        for r in self._selected_rows():
            c.corrs[r].enabled = not c.corrs[r].enabled
        self.after_change(resolve=True)

    def delete_sel(self):
        c = self.cam()
        if not c:
            return
        for r in self._selected_rows():
            del c.corrs[r]
        self._sel = None
        self.after_change(resolve=True)

    def clear_tiles(self):
        c = self.cam()
        if c and any(p.source == "tile" for p in c.corrs):
            c.corrs = [p for p in c.corrs if p.source != "tile"]
            self._sel = None
            self.after_change(resolve=True)


# ==========================================================================
# Tab 4 — Kiểm tra đa camera
# ==========================================================================
class CheckTab(QWidget):
    def __init__(self, ext: "ExtrinsicWidget"):
        super().__init__()
        self.ext = ext
        self.inner = QTabWidget()
        # nhìn từ trên xuống
        self.top = ChartBox(420)
        b_top = QPushButton("Cập nhật"); b_top.clicked.connect(self.update_top)
        w1 = QWidget(); l1 = QVBoxLayout(w1); r1 = QHBoxLayout(); r1.addWidget(b_top); r1.addStretch()
        r1.addWidget(hint("Màu xanh: số camera thấy trọn một người (chân và đầu 1.7 m) tại mỗi vị trí. Nên ≥ 2 trên toàn vùng quay."))
        l1.addLayout(r1); l1.addWidget(self.top, 1)
        # ảnh ghép
        self.mosaic = ZoomView("Bấm “Tạo ảnh ghép”")
        b_mos = QPushButton("Tạo ảnh ghép"); b_mos.clicked.connect(self.update_mosaic)
        w2 = QWidget(); l2 = QVBoxLayout(w2); r2 = QHBoxLayout(); r2.addWidget(b_mos); r2.addStretch()
        r2.addWidget(hint("Ảnh mọi camera chiếu xuống mặt sàn rồi chồng lên nhau. Ron gạch và marker phải trùng khít — "
                          "nếu thấy bóng mờ / đường ron đôi là extrinsic chưa khớp. (Vật có chiều cao sẽ bị kéo dài — bình thường.)"))
        l2.addLayout(r2); l2.addWidget(self.mosaic, 1)
        # độ bất định
        self.ucombo = QComboBox(); self.ucombo.currentIndexChanged.connect(self.update_unc)
        self.unc = ChartBox(420)
        w3 = QWidget(); l3 = QVBoxLayout(w3); r3 = QHBoxLayout(); r3.addWidget(QLabel("Camera")); r3.addWidget(self.ucombo)
        r3.addStretch(); r3.addWidget(hint("Sai số vị trí trên sàn chỉ do sai số tư thế camera (chưa tính sai số khi xác định chân người)."))
        l3.addLayout(r3); l3.addWidget(self.unc, 1)
        # điểm kiểm tra
        w4 = self._build_tests()
        self.inner.addTab(w1, "Nhìn từ trên xuống")
        self.inner.addTab(w2, "Ảnh ghép mặt sàn")
        self.inner.addTab(w3, "Độ bất định trên sàn")
        self.inner.addTab(w4, "Điểm kiểm tra chéo")
        self.inner.currentChanged.connect(lambda *_: self.refresh())
        lay = QVBoxLayout(self); lay.addWidget(self.inner)

    def _build_tests(self):
        self.tp_list = QListWidget(); self.tp_list.currentRowChanged.connect(self._tp_selected)
        b_add = QPushButton("Thêm điểm"); b_del = QPushButton("Xóa"); b_ren = QPushButton("Đổi tên")
        b_add.clicked.connect(self.add_tp); b_del.clicked.connect(self.del_tp); b_ren.clicked.connect(self.ren_tp)
        self.tp_z = QDoubleSpinBox(); self.tp_z.setRange(-5, 10); self.tp_z.setDecimals(3); self.tp_z.setSuffix(" m")
        self.tp_z.valueChanged.connect(self._tp_z)
        self.tcombo = QComboBox(); self.tcombo.currentIndexChanged.connect(self.redraw_tp)
        self.b_all = QPushButton("Xem trên mọi camera"); self.b_all.setCheckable(True); self.b_all.toggled.connect(self.redraw_tp)
        b_clear = QPushButton("Xóa điểm click ở camera này"); b_clear.clicked.connect(self.clear_click)
        self.tp_table = QTableWidget(0, 4)
        self.tp_table.setHorizontalHeaderLabels(["Điểm", "Số camera", "X, Y (m)", "Lệch max (cm)"])
        self.tp_table.verticalHeader().hide(); self.tp_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tp_table.horizontalHeader().setStretchLastSection(True)
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Điểm kiểm tra</b>")); left.addWidget(self.tp_list, 1)
        r = QHBoxLayout(); r.addWidget(b_add); r.addWidget(b_del); r.addWidget(b_ren); left.addLayout(r)
        f = QFormLayout(); f.addRow("Độ cao điểm", self.tp_z); f.addRow("Camera đang click", self.tcombo); left.addLayout(f)
        left.addWidget(b_clear); left.addWidget(self.b_all)
        left.addWidget(self.tp_table, 1)
        left.addWidget(hint("Chọn một điểm vật lý nhìn thấy ở nhiều camera (chân một người đứng yên, chân ghế, vết băng keo). "
                            "Click đúng điểm đó ở từng camera. Tool chiếu từng click xuống sàn và báo độ lệch giữa các camera "
                            "(mục tiêu < 5–10 cm). Hình trụ cao 1.7 m được vẽ tại vị trí trung bình trên mọi camera."))
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(380)
        self.tview = ZoomView("Chọn camera")
        self.tview.clicked.connect(self._tp_click)
        w = QWidget(); l = QHBoxLayout(w); l.addWidget(lw); l.addWidget(self.tview, 1)
        return w

    def refresh(self):
        names = [c.name for c in self.ext.proj.cameras if c.has_pose]
        for cb in (self.ucombo, self.tcombo):
            cur = cb.currentText()
            cb.blockSignals(True); cb.clear(); cb.addItems(names)
            if cur in names:
                cb.setCurrentText(cur)
            cb.blockSignals(False)
        i = self.inner.currentIndex()
        if i == 0:
            self.update_top()
        elif i == 2:
            self.update_unc()
        elif i == 3:
            self.fill_tps(); self.redraw_tp()

    def update_top(self):
        self.top.set_figure(xv.fig_topview(self.ext.proj))

    def update_mosaic(self):
        proj = self.ext.proj
        if not proj.roi.valid():
            QMessageBox.information(self, APP_NAME, "Đặt vùng quay ở tab “Vùng lưới & xuất” trước."); return
        cams = [c for c in proj.cameras if c.has_pose and c.image is not None]
        if not cams:
            return

        def job(sig, _):
            m, ext_, n = ex.ground_mosaic(cams, proj.roi)
            return m

        self.ext.run_job(job, lambda m: self.mosaic.set_image(m), "Đang tạo ảnh ghép mặt sàn…", None)

    def update_unc(self, *_):
        c = self.ext.proj.camera(self.ucombo.currentText())
        if c is None:
            self.unc.set_figure(xv.fig_uncertainty(self.ext.proj, ex.CameraEx("—")))
            return
        self.unc.set_figure(xv.fig_uncertainty(self.ext.proj, c))

    # ---- điểm kiểm tra ----
    def tp(self):
        r = self.tp_list.currentRow()
        tps = self.ext.proj.test_points
        return tps[r] if 0 <= r < len(tps) else None

    def fill_tps(self):
        tps = self.ext.proj.test_points
        cur = self.tp_list.currentRow()
        self.tp_list.blockSignals(True); self.tp_list.clear()
        for t in tps:
            self.tp_list.addItem(f"{t.name}  (z = {t.z:g} m, {len(t.clicks)} click)")
        self.tp_list.blockSignals(False)
        if tps:
            self.tp_list.setCurrentRow(min(max(cur, 0), len(tps) - 1))
        self.tp_table.setRowCount(len(tps))
        for r, t in enumerate(tps):
            sm = t.summary(self.ext.proj.cameras)
            vals = [t.name, str(sm["n"]) if sm else "0",
                    f"{sm['mean'][0]:.3f}, {sm['mean'][1]:.3f}" if sm else "",
                    f"{sm['max_pair_cm']:.1f}" if sm and sm["max_pair_cm"] is not None else ""]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k == 3 and sm and sm["max_pair_cm"] is not None:
                    lv = "ok" if sm["max_pair_cm"] <= 5 else ("warn" if sm["max_pair_cm"] <= 10 else "fail")
                    it.setForeground(QBrush(STATUS_COLORS[lv]))
                self.tp_table.setItem(r, k, it)
        t = self.tp()
        self.tp_z.blockSignals(True); self.tp_z.setValue(t.z if t else 0.0); self.tp_z.blockSignals(False)

    def _tp_selected(self, *_):
        t = self.tp()
        self.tp_z.blockSignals(True); self.tp_z.setValue(t.z if t else 0.0); self.tp_z.blockSignals(False)

    def add_tp(self):
        k = len(self.ext.proj.test_points) + 1
        self.ext.proj.test_points.append(ex.TestPoint(f"P{k}"))
        self.ext.changed(); self.fill_tps(); self.tp_list.setCurrentRow(len(self.ext.proj.test_points) - 1)
        self.redraw_tp()

    def del_tp(self):
        t = self.tp()
        if t:
            self.ext.proj.test_points.remove(t); self.ext.changed(); self.fill_tps(); self.redraw_tp()

    def ren_tp(self):
        t = self.tp()
        if not t:
            return
        n, ok = QInputDialog.getText(self, "Đổi tên", "Tên điểm:", text=t.name)
        if ok and n.strip():
            t.name = n.strip(); self.ext.changed(); self.fill_tps(); self.redraw_tp()

    def _tp_z(self, v):
        t = self.tp()
        if t:
            t.z = v; self.ext.changed(); self.fill_tps(); self.redraw_tp()

    def clear_click(self):
        t = self.tp()
        if t:
            t.clicks.pop(self.tcombo.currentText(), None); self.ext.changed(); self.fill_tps(); self.redraw_tp()

    def _tp_click(self, x, y):
        if self.b_all.isChecked():
            return
        t = self.tp()
        c = self.ext.proj.camera(self.tcombo.currentText())
        if t is None:
            self.add_tp(); t = self.tp()
        if c is None:
            return
        t.clicks[c.name] = [float(x), float(y)]
        self.ext.changed(); self.fill_tps(); self.redraw_tp()

    def redraw_tp(self, *_):
        proj = self.ext.proj
        if self.b_all.isChecked():
            cams = [c for c in proj.cameras if c.has_pose and c.image is not None]
            imgs = []
            for c in cams:
                im = c.image.copy()
                xv.draw_test_points(im, c, proj.test_points, proj)
                imgs.append(im)
            self.tview.set_image(xv.composite(imgs, [c.name for c in cams], cols=2 if len(cams) <= 4 else 3))
            return
        c = proj.camera(self.tcombo.currentText())
        if c is None or c.image is None:
            self.tview.set_image(None); return
        im = c.image.copy()
        xv.draw_test_points(im, c, proj.test_points, proj)
        self.tview.set_image(im)


# ==========================================================================
# Tab 5 — Vùng lưới & xuất
# ==========================================================================
class ExportTab(QWidget):
    def __init__(self, ext: "ExtrinsicWidget"):
        super().__init__()
        self.ext = ext
        self._block = False
        self.x0, self.y0, self.x1, self.y1 = (QDoubleSpinBox() for _ in range(4))
        for s in (self.x0, self.y0, self.x1, self.y1):
            s.setRange(-200, 200); s.setDecimals(2); s.setSuffix(" m"); s.setSingleStep(0.1)
            s.valueChanged.connect(self._changed)
        self.cell = QDoubleSpinBox(); self.cell.setRange(0.5, 100); self.cell.setDecimals(1); self.cell.setSuffix(" cm")
        self.cell.valueChanged.connect(self._changed)
        self.dims = QLabel()
        b_auto = QPushButton("Lấy theo phạm vi điểm mốc"); b_auto.clicked.connect(self.auto_roi)
        f = QGridLayout()
        f.addWidget(QLabel("X từ"), 0, 0); f.addWidget(self.x0, 0, 1); f.addWidget(QLabel("đến"), 0, 2); f.addWidget(self.x1, 0, 3)
        f.addWidget(QLabel("Y từ"), 1, 0); f.addWidget(self.y0, 1, 1); f.addWidget(QLabel("đến"), 1, 2); f.addWidget(self.y1, 1, 3)
        f.addWidget(QLabel("Cỡ ô lưới"), 2, 0); f.addWidget(self.cell, 2, 1); f.addWidget(self.dims, 2, 2, 1, 2)
        g1 = QGroupBox("Vùng quay && lưới mặt đất cho mô hình"); v1 = QVBoxLayout(g1); v1.addLayout(f); v1.addWidget(b_auto)
        v1.addWidget(hint("Vùng người đi lại sẽ được gán nhãn. Ví dụ: Wildtrack dùng ô 2.5 cm. Vùng này cũng dùng cho bản đồ "
                          "độ phủ, ảnh ghép và giới hạn phạm vi tìm lưới gạch."))
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Camera", "Trạng thái", "Điểm", "RMS (px)", "Sàn (cm)", "Vị trí X, Y, Z (m)", "Thấy % vùng"])
        self.table.verticalHeader().hide(); self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.unit = QComboBox(); self.unit.addItems(["m", "cm"])
        b_exp = QPushButton("Xuất kết quả ra thư mục…"); b_exp.setStyleSheet("font-weight:600;")
        b_rep = QPushButton("Xuất báo cáo HTML…")
        b_exp.clicked.connect(self.export); b_rep.clicked.connect(self.report)
        g2 = QGroupBox("Xuất"); v2 = QVBoxLayout(g2)
        r = QHBoxLayout(); r.addWidget(QLabel("Đơn vị tvec trong extr_*.xml")); r.addWidget(self.unit); r.addStretch(); v2.addLayout(r)
        v2.addWidget(b_exp); v2.addWidget(b_rep)
        v2.addWidget(hint("Mỗi camera: <code>intr_&lt;tên&gt;.xml</code> (camera_matrix, distortion_coefficients) và "
                          "<code>extr_&lt;tên&gt;.xml</code> (rvec, tvec). Kèm <code>calibration.json</code>: K, hệ số méo, R, t, "
                          "vị trí camera, ma trận chiếu P, homography mặt sàn → ảnh, vùng quay và kích thước lưới."))
        left = QVBoxLayout(); left.addWidget(g1); left.addWidget(g2); left.addStretch()
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(470)
        right = QVBoxLayout(); right.addWidget(QLabel("<b>Tổng hợp camera</b>")); right.addWidget(self.table, 1)
        rw = QWidget(); rw.setLayout(right)
        lay = QHBoxLayout(self); lay.addWidget(lw); lay.addWidget(rw, 1)

    def refresh(self):
        r = self.ext.proj.roi
        self._block = True
        for s, v in ((self.x0, r.x0), (self.y0, r.y0), (self.x1, r.x1), (self.y1, r.y1), (self.cell, r.cell_cm)):
            s.setValue(v)
        self._block = False
        self._dims()
        self.fill_table()

    def _dims(self):
        r = self.ext.proj.roi
        if r.valid():
            h, w = r.grid_shape()
            self.dims.setText(f"→ lưới {h} × {w} ô ({r.x1 - r.x0:.2f} × {r.y1 - r.y0:.2f} m)")
        else:
            self.dims.setText("<span style='color:#b42f2f'>vùng chưa hợp lệ</span>")

    def _changed(self, *_):
        if self._block:
            return
        r = self.ext.proj.roi
        r.x0, r.y0, r.x1, r.y1, r.cell_cm = self.x0.value(), self.y0.value(), self.x1.value(), self.y1.value(), self.cell.value()
        self._dims()
        self.ext.changed()
        QTimer.singleShot(400, self.fill_table)

    def auto_roi(self):
        bb = self.ext.proj.ground_corr_bbox()
        if bb is None:
            QMessageBox.information(self, APP_NAME, "Chưa có điểm mốc trên sàn."); return
        (a, b), (c, d) = bb
        w = self.ext.proj.world
        r = self.ext.proj.roi
        r.x0, r.y0 = math.floor(a / w.px) * w.px, math.floor(b / w.py) * w.py
        r.x1, r.y1 = math.ceil(c / w.px) * w.px, math.ceil(d / w.py) * w.py
        self.ext.changed(); self.refresh()

    def fill_table(self):
        proj = self.ext.proj
        cams = proj.cameras
        self.table.setRowCount(len(cams))
        g = ex.ground_points(proj.roi, 0.2) if proj.roi.valid() else None
        for k, c in enumerate(cams):
            r = c.result
            vis = ""
            if r is not None and g is not None and len(g):
                _, m = c.project(g)
                vis = f"{m.mean() * 100:.0f}%"
            vals = [c.name, LEVEL_TXT[r.level] if r else "chưa giải", f"{r.n_inliers}" if r else "",
                    f"{r.rms:.2f}" if r else "", f"{r.med_cm:.1f}" if r else "",
                    f"{r.C[0]:.2f}, {r.C[1]:.2f}, {r.C[2]:.2f}" if r else "", vis]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 1 and r:
                    it.setForeground(QBrush(STATUS_COLORS[r.level]))
                self.table.setItem(k, j, it)

    def export(self):
        proj = self.ext.proj
        if not any(c.has_pose for c in proj.cameras):
            QMessageBox.information(self, APP_NAME, "Chưa có camera nào được giải."); return
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất", self.ext.main.last_dir())
        if not d:
            return
        try:
            files = ex.export_project(proj, d, self.unit.currentText())
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không xuất được:\n{e}"); return
        self.ext.main.remember_dir(os.path.join(d, "x"))
        bad = [c.name for c in proj.cameras if c.has_pose and c.result.level == "fail"]
        msg = f"Đã xuất {len(files)} file vào:\n{d}"
        if bad:
            msg += f"\n\nLưu ý: {', '.join(bad)} đang ở trạng thái “Không đạt”."
        skipped = [c.name for c in proj.cameras if not c.has_pose]
        if skipped:
            msg += f"\nBỏ qua (chưa giải): {', '.join(skipped)}."
        QMessageBox.information(self, APP_NAME, msg)

    def report(self):
        proj = self.ext.proj
        if not any(c.has_pose for c in proj.cameras):
            QMessageBox.information(self, APP_NAME, "Chưa có camera nào được giải."); return
        p, _ = QFileDialog.getSaveFileName(self, "Báo cáo extrinsic", os.path.join(self.ext.main.last_dir(), "bao_cao_extrinsic.html"),
                                           "HTML (*.html)")
        if not p:
            return

        def job(sig, _):
            xv.build_report(proj, p)
            return p

        def done(path):
            self.ext.main.remember_dir(path)
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        self.ext.run_job(job, done, "Đang tạo báo cáo…", None)


# ==========================================================================
# Widget chính
# ==========================================================================
class ExtrinsicWidget(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.proj = ex.Project()
        self.dirty = False
        self._job = None
        self.tabs = QTabWidget()
        self.t_cams = CamerasTab(self)
        self.t_world = WorldTab(self)
        self.t_pts = PointsTab(self)
        self.t_check = CheckTab(self)
        self.t_exp = ExportTab(self)
        self.tabs.addTab(self.t_cams, "1. Camera && dự án")
        self.tabs.addTab(self.t_world, "2. Hệ toạ độ")
        self.tabs.addTab(self.t_pts, "3. Gán điểm && giải")
        self.tabs.addTab(self.t_check, "4. Kiểm tra đa camera")
        self.tabs.addTab(self.t_exp, "5. Vùng lưới && xuất")
        self.tabs.currentChanged.connect(self._tab_changed)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.tabs)

    def _tab_changed(self, i):
        w = self.tabs.widget(i)
        if hasattr(w, "refresh"):
            w.refresh()

    def refresh_all(self):
        for t in (self.t_cams, self.t_world, self.t_pts, self.t_exp):
            t.refresh()
        if self.tabs.currentWidget() is self.t_check:
            self.t_check.refresh()

    def changed(self):
        self.dirty = True

    def world_changed(self):
        """Cập nhật toạ độ thế giới của điểm mốc theo hệ toạ độ mới rồi giải lại các camera đã giải."""
        w = self.proj.world
        for c in self.proj.cameras:
            keep = []
            for p in c.corrs:
                if p.source == "tile" and p.gi is not None:
                    p.X, p.Y = w.grid_to_world(p.gi, p.gj)
                elif p.source == "manual" and p.gi is not None:
                    p.X, p.Y = w.grid_to_world(p.gi, p.gj)
                elif p.source == "marker":
                    try:
                        mid, k = p.label[1:].split("-"); mid, k = int(mid), int(k)
                    except ValueError:
                        keep.append(p); continue
                    m = w.placement(mid)
                    if m is None:
                        continue
                    W = w.marker_world_corners(m)
                    p.X, p.Y, p.Z = map(float, W[k])
                keep.append(p)
            c.corrs = keep
            if c.has_pose:
                try:
                    ex.solve(c, w, self.proj.roi)
                except Exception:
                    c.result = None
        self.changed()

    def run_job(self, fn, done, label, arg):
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self.main.statusBar().showMessage(label)
        w = Worker(fn, arg)

        def finish(res):
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._job = None
            self.main.statusBar().clearMessage()
            done(res)

        def fail(msg):
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._job = None
            QMessageBox.warning(self, APP_NAME, msg)
        w.signals.message.connect(lambda m, f: self.main.statusBar().showMessage(m))
        w.signals.done.connect(finish)
        w.signals.failed.connect(fail)
        self._job = w
        self.main.start(w)

    # ---- dự án ----
    def _close_videos(self):
        for c in self.proj.cameras:
            c.close_video()

    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, "Dự án extrinsic có thay đổi chưa lưu. Lưu lại?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save_project(False)
        return r == QMessageBox.Discard

    def new_project(self):
        if not self.confirm_discard():
            return
        self._close_videos()
        self.proj = ex.Project(); self.dirty = False
        self.refresh_all()

    def open_project(self):
        if not self.confirm_discard():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Mở dự án extrinsic", self.main.last_dir(), "Dự án extrinsic (*.json)")
        if not p:
            return
        try:
            proj, errs = ex.Project.load(p)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không mở được dự án:\n{e}"); return
        self._close_videos()
        self.proj = proj; self.dirty = False
        self.main.remember_dir(p)
        self.refresh_all()
        if errs:
            QMessageBox.warning(self, APP_NAME, "Một số file không mở được:\n" + "\n".join(errs))

    def save_project(self, as_new=False) -> bool:
        p = self.proj.path
        if as_new or not p:
            p, _ = QFileDialog.getSaveFileName(self, "Lưu dự án extrinsic",
                                               os.path.join(self.main.last_dir(), "du_an_extrinsic.json"), "Dự án extrinsic (*.json)")
            if not p:
                return False
        try:
            self.proj.save(p)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Không lưu được:\n{e}"); return False
        self.dirty = False
        self.main.remember_dir(p)
        self.t_cams.refresh()
        self.main.statusBar().showMessage(f"Đã lưu dự án: {p}", 6000)
        return True
