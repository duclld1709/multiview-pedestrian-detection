"""
Hiển thị cho phần extrinsic: vẽ lớp chồng lên ảnh camera, biểu đồ nhìn từ trên xuống, báo cáo HTML.
Không phụ thuộc Qt.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import math

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

import extrinsic_core as ex
from report import _style, _seq_cmap, INK, INK2, GRID, SURFACE, BLUE, ORANGE

# màu (RGB) — cùng bảng màu với phần intrinsic
C_MARKER = (42, 120, 214)       # xanh dương
C_TILE = (27, 175, 122)         # xanh ngọc
C_MANUAL = (232, 123, 164)      # hồng
C_OUT = (208, 59, 59)
C_OFF = (150, 150, 150)
C_ROI = (235, 104, 52)
C_SEL = (250, 178, 25)
SRC_COLOR = {"marker": C_MARKER, "tile": C_TILE, "manual": C_MANUAL}
TP_COLORS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (232, 123, 164), (98, 80, 214), (237, 161, 0)]


def _polyline(img, cam, P, color, th, closed=False):
    uv, ok = cam.project(P, margin=200)
    pts = uv
    n = len(P)
    rng = range(n) if closed else range(n - 1)
    for k in rng:
        a, b = k, (k + 1) % n
        if ok[a] and ok[b]:
            cv2.line(img, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), color, th, cv2.LINE_AA)


def draw_overlay(cam: ex.CameraEx, world: ex.WorldSetup, roi: ex.ROI = None, show_grid=True, show_err=True,
                 show_roi=True, selected=None, test_points=None, proj=None, show_points=True):
    img = cam.image.copy()
    h, w = img.shape[:2]
    s = max(1, int(round(w / 1000)))
    if cam.has_pose:
        rv, tv = cam.pose()
        if show_grid:
            layer = img.copy()
            fp = ex.floor_footprint(cam, 25.0)
            x0, y0 = fp.min(axis=0); x1, y1 = fp.max(axis=0)
            if roi is not None and roi.valid():
                x0, y0 = max(x0, roi.x0 - 3), max(y0, roi.y0 - 3)
                x1, y1 = min(x1, roi.x1 + 3), min(y1, roi.y1 + 3)
            I = np.arange(math.floor(x0 / world.px), math.ceil(x1 / world.px) + 1)
            J = np.arange(math.floor(y0 / world.py), math.ceil(y1 / world.py) + 1)
            if len(I) * len(J) < 20000:
                ys = np.linspace(J[0] * world.py, J[-1] * world.py, max(2, len(J) * 4))
                xs = np.linspace(I[0] * world.px, I[-1] * world.px, max(2, len(I) * 4))
                for i in I:
                    _polyline(layer, cam, np.c_[np.full_like(ys, i * world.px), ys, np.zeros_like(ys)], (255, 255, 255), s)
                for j in J:
                    _polyline(layer, cam, np.c_[xs, np.full_like(xs, j * world.py), np.zeros_like(xs)], (255, 255, 255), s)
                img = cv2.addWeighted(layer, 0.55, img, 0.45, 0)
            # trục toạ độ tại gốc
            L = 2 * min(world.px, world.py)
            o, ok = cam.project(np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]]), margin=500)
            if ok[0]:
                for k, col in [(1, (220, 40, 40)), (2, (30, 170, 60)), (3, (40, 90, 230))]:
                    if ok[k]:
                        cv2.arrowedLine(img, tuple(np.round(o[0]).astype(int)), tuple(np.round(o[k]).astype(int)),
                                        col, 3 * s, cv2.LINE_AA, tipLength=0.15)
                        cv2.putText(img, "XYZ"[k - 1], tuple(np.round(o[k]).astype(int) + 6 * s),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, col, 2 * s, cv2.LINE_AA)
        if show_roi and roi is not None and roi.valid():
            n = 60
            t = np.linspace(0, 1, n)
            P = np.vstack([np.c_[roi.x0 + t * (roi.x1 - roi.x0), np.full(n, roi.y0)],
                           np.c_[np.full(n, roi.x1), roi.y0 + t * (roi.y1 - roi.y0)],
                           np.c_[roi.x1 - t * (roi.x1 - roi.x0), np.full(n, roi.y1)],
                           np.c_[np.full(n, roi.x0), roi.y1 - t * (roi.y1 - roi.y0)]])
            _polyline(img, cam, np.c_[P, np.zeros(len(P))], C_ROI, 3 * s, closed=True)
    if show_points:
        for k, c in enumerate(cam.corrs):
            p = (int(round(c.u)), int(round(c.v)))
            if not c.enabled:
                col = C_OFF
            elif not c.inlier and cam.has_pose:
                col = C_OUT
            else:
                col = SRC_COLOR.get(c.source, C_MANUAL)
            r = (4 if c.source == "tile" else 5) * s
            if c.enabled and (c.inlier or not cam.has_pose):
                cv2.circle(img, p, r, col, -1, cv2.LINE_AA)
                cv2.circle(img, p, r, (255, 255, 255), max(1, s // 2), cv2.LINE_AA)
            else:
                d = r + s
                cv2.line(img, (p[0] - d, p[1] - d), (p[0] + d, p[1] + d), col, 2 * s, cv2.LINE_AA)
                cv2.line(img, (p[0] - d, p[1] + d), (p[0] + d, p[1] - d), col, 2 * s, cv2.LINE_AA)
            if show_err and cam.has_pose and c.err_px is not None:
                q = cam.project_raw(c.world, *cam.pose())[0]
                dv = (q - [c.u, c.v]) * 10
                cv2.line(img, p, (int(round(c.u + dv[0])), int(round(c.v + dv[1]))), (255, 220, 0), max(1, s), cv2.LINE_AA)
            if c.source in ("manual",) or (c.source == "marker" and c.label.endswith("-0")):
                lab = c.label.split("-")[0] if c.source == "marker" else c.label
                cv2.putText(img, lab, (p[0] + 7 * s, p[1] - 7 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s,
                            (20, 20, 20), 3 * s, cv2.LINE_AA)
                cv2.putText(img, lab, (p[0] + 7 * s, p[1] - 7 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s,
                            (255, 255, 255), s, cv2.LINE_AA)
        if selected is not None and 0 <= selected < len(cam.corrs):
            c = cam.corrs[selected]
            cv2.circle(img, (int(round(c.u)), int(round(c.v))), 14 * s, C_SEL, 3 * s, cv2.LINE_AA)
    if test_points and cam.has_pose:
        draw_test_points(img, cam, test_points, proj, s)
    return img


def draw_test_points(img, cam, test_points, proj, s=None):
    if s is None:
        s = max(1, int(round(img.shape[1] / 1000)))
    for k, tp in enumerate(test_points):
        col = TP_COLORS[k % len(TP_COLORS)]
        sm = tp.summary(proj.cameras) if proj else None
        if sm is not None:
            bot, top = ex.cylinder_points(sm["mean"][:2])
            _polyline(img, cam, bot, col, 2 * s, closed=True)
            _polyline(img, cam, top, col, 2 * s, closed=True)
            for a, b in [(0, 0), (6, 6), (12, 12), (18, 18)]:
                _polyline(img, cam, np.vstack([bot[a], top[b]]), col, 2 * s)
            uv, ok = cam.project(np.array([[*sm["mean"][:2], 0.0]]), margin=50)
            if ok[0]:
                cv2.putText(img, tp.name, tuple(np.round(uv[0]).astype(int) + [8 * s, 18 * s]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, col, 2 * s, cv2.LINE_AA)
        if cam.name in tp.clicks:
            u, v = tp.clicks[cam.name]
            p = (int(round(u)), int(round(v)))
            cv2.drawMarker(img, p, col, cv2.MARKER_CROSS, 22 * s, 3 * s, cv2.LINE_AA)


def composite(images, names, cols=2, width=1920):
    if not images:
        return None
    rows = math.ceil(len(images) / cols)
    tw = width // cols
    th = int(tw * images[0].shape[0] / images[0].shape[1])
    out = np.full((rows * th, cols * tw, 3), 245, np.uint8)
    for k, (im, n) in enumerate(zip(images, names)):
        r, c = divmod(k, cols)
        t = cv2.resize(im, (tw, int(tw * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)[:th]
        out[r * th:r * th + t.shape[0], c * tw:c * tw + tw] = t
        cv2.rectangle(out, (c * tw, r * th), (c * tw + 12 + 16 * len(n), r * th + 36), (255, 255, 255), -1)
        cv2.putText(out, n, (c * tw + 8, r * th + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Biểu đồ nhìn từ trên xuống
# --------------------------------------------------------------------------
CAM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _cam_glyph(ax, cam, color, size=0.35):
    C = cam.center()
    R = cam.R()
    zc = R.T @ np.array([0, 0, 1.0])
    d = zc[:2] / (np.linalg.norm(zc[:2]) + 1e-9)
    ax.plot([C[0]], [C[1]], marker="o", ms=9, color=color, mec=SURFACE, mew=1.5, zorder=6)
    ax.annotate("", xy=(C[0] + d[0] * size * 2, C[1] + d[1] * size * 2), xytext=(C[0], C[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2), zorder=6)
    ax.text(C[0], C[1] - size * 0.9, f"{cam.name}\nh={C[2]:.2f} m", ha="center", va="top", fontsize=8, color=INK, zorder=7)


def fig_topview(proj: ex.Project, show_coverage=True, step=0.1):
    fig = Figure(figsize=(7.2, 6.0), dpi=100)
    ax = fig.add_subplot(111)
    cams = [c for c in proj.cameras if c.has_pose]
    roi = proj.roi
    pts = [c.center()[:2] for c in cams]
    if roi.valid():
        pts += [[roi.x0, roi.y0], [roi.x1, roi.y1]]
    if not pts:
        ax.text(0.5, 0.5, "Chưa có camera nào được giải.", ha="center", va="center", transform=ax.transAxes, color=INK2)
        _style(fig); return fig
    P = np.array(pts)
    lo, hi = P.min(axis=0) - 1.0, P.max(axis=0) + 1.0
    if show_coverage and roi.valid() and cams:
        xs, ys, cnt = ex.coverage_map(cams, roi, step)
        n = len(cams)
        cols = ["#f3f3f1"] + [c for c in ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b", "#0a2a52", "#071d3a"][:n]]
        cmap = ListedColormap(cols)
        norm = BoundaryNorm(np.arange(-0.5, n + 1.5), cmap.N)
        im = ax.imshow(cnt, origin="lower", extent=(roi.x0, roi.x1, roi.y0, roi.y1), cmap=cmap, norm=norm,
                       interpolation="nearest", zorder=1)
        cb = fig.colorbar(im, ax=ax, ticks=range(n + 1), fraction=0.04, pad=0.02)
        cb.set_label("số camera thấy cả người (chân + đầu 1.7 m)", fontsize=8.5, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2); cb.outline.set_visible(False)
    # lưới gạch mờ
    w = proj.world
    for i in np.arange(math.floor(lo[0] / w.px), math.ceil(hi[0] / w.px) + 1):
        ax.axvline(i * w.px, color=GRID, lw=0.6, zorder=0)
    for j in np.arange(math.floor(lo[1] / w.py), math.ceil(hi[1] / w.py) + 1):
        ax.axhline(j * w.py, color=GRID, lw=0.6, zorder=0)
    if roi.valid():
        ax.add_patch(Rectangle((roi.x0, roi.y0), roi.x1 - roi.x0, roi.y1 - roi.y0, fill=False, ec=ORANGE, lw=2, zorder=4))
    for k, c in enumerate(cams):
        col = CAM_COLORS[k % len(CAM_COLORS)]
        fp = ex.footprint_polygon(c, max_dist=max(8.0, float(np.max(hi - lo))))
        ax.add_patch(Polygon(fp, closed=True, fill=False, ec=col, lw=1.2, ls="--", zorder=3))
        _cam_glyph(ax, c, col)
    for m in w.markers:
        x, y = w.grid_to_world(m.i, m.j)
        ax.plot([x], [y], marker="s", ms=6, color=INK, zorder=5)
        ax.text(x + 0.08, y + 0.08, f"M{m.id}", fontsize=7.5, color=INK2, zorder=5)
    for k, tp in enumerate(proj.test_points):
        sm = tp.summary(proj.cameras)
        if sm:
            ax.plot([sm["mean"][0]], [sm["mean"][1]], marker="X", ms=10, color=ORANGE, mec=SURFACE, zorder=8)
            ax.text(sm["mean"][0] + 0.1, sm["mean"][1], tp.name, fontsize=8, color=INK, zorder=8)
    ax.annotate("", xy=(0.6, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#dc2828", lw=2), zorder=9)
    ax.annotate("", xy=(0, 0.6), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#1eaa3c", lw=2), zorder=9)
    ax.text(0.65, 0, "X", color="#dc2828", fontsize=9, va="center"); ax.text(0, 0.68, "Y", color="#1eaa3c", fontsize=9, ha="center")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=8.5); ax.set_ylabel("Y (m)", fontsize=8.5)
    ax.set_title("Nhìn từ trên xuống", fontsize=10, loc="left")
    ax.text(0.0, -0.09, "Nét đứt: vùng sàn mỗi camera nhìn thấy · khung cam: vùng quay · ô vuông đen: marker · X cam: điểm kiểm tra",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    _style(fig)
    fig.tight_layout()
    return fig


def fig_uncertainty(proj: ex.Project, cam: ex.CameraEx):
    fig = Figure(figsize=(7.2, 6.0), dpi=100)
    ax = fig.add_subplot(111)
    roi = proj.roi
    res = ex.ground_uncertainty(cam, roi) if (roi.valid() and cam.has_pose) else None
    if res is None:
        ax.text(0.5, 0.5, "Cần vùng quay hợp lệ và camera đã giải (≥ 6 điểm).", ha="center", va="center",
                transform=ax.transAxes, color=INK2)
        _style(fig); return fig
    xs, ys, sd = res
    im = ax.imshow(sd, origin="lower", extent=(roi.x0, roi.x1, roi.y0, roi.y1), cmap=_seq_cmap(), zorder=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("độ bất định vị trí trên sàn (cm)", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2); cb.outline.set_visible(False)
    pts = np.array([[c.X, c.Y] for c in cam.corrs if c.enabled and c.inlier and abs(c.Z) < 1e-6])
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=INK, alpha=0.5, zorder=3, label="điểm mốc")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    _cam_glyph(ax, cam, ORANGE)
    C = cam.center()
    ax.set_xlim(min(roi.x0, C[0]) - 0.8, max(roi.x1, C[0]) + 0.8)
    ax.set_ylim(min(roi.y0, C[1]) - 0.8, max(roi.y1, C[1]) + 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=8.5); ax.set_ylabel("Y (m)", fontsize=8.5)
    ax.set_title(f"{cam.name}: sai số vị trí trên sàn do sai số tư thế camera", fontsize=10, loc="left")
    _style(fig)
    fig.tight_layout()
    return fig


def fig_world_setup(world: ex.WorldSetup, roi: ex.ROI = None):
    """Sơ đồ hệ toạ độ & vị trí marker để người dùng đối chiếu với thực tế."""
    fig = Figure(figsize=(6.0, 5.4), dpi=100)
    ax = fig.add_subplot(111)
    xs = [0.0]; ys = [0.0]
    for m in world.markers:
        x, y = world.grid_to_world(m.i, m.j); xs.append(x); ys.append(y)
    if roi is not None and roi.valid():
        xs += [roi.x0, roi.x1]; ys += [roi.y0, roi.y1]
    pad = 2 * max(world.px, world.py)
    lo = (min(xs) - pad, min(ys) - pad); hi = (max(xs) + pad, max(ys) + pad)
    if hi[0] - lo[0] < 4 * world.px:
        hi = (lo[0] + 6 * world.px, hi[1])
    if hi[1] - lo[1] < 4 * world.py:
        hi = (hi[0], lo[1] + 6 * world.py)
    for i in np.arange(math.floor(lo[0] / world.px), math.ceil(hi[0] / world.px) + 1):
        ax.axvline(i * world.px, color="#c9c8c2", lw=0.8, zorder=0)
    for j in np.arange(math.floor(lo[1] / world.py), math.ceil(hi[1] / world.py) + 1):
        ax.axhline(j * world.py, color="#c9c8c2", lw=0.8, zorder=0)
    if roi is not None and roi.valid():
        ax.add_patch(Rectangle((roi.x0, roi.y0), roi.x1 - roi.x0, roi.y1 - roi.y0, fill=False, ec=ORANGE, lw=2))
    L = 1.6 * min(world.px, world.py)
    ax.annotate("", xy=(L, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#dc2828", lw=2.5))
    ax.annotate("", xy=(0, L), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#1eaa3c", lw=2.5))
    ax.text(L * 1.05, 0, "+X", color="#dc2828", fontsize=10, va="center", fontweight="bold")
    ax.text(0, L * 1.08, "+Y", color="#1eaa3c", fontsize=10, ha="center", fontweight="bold")
    ax.plot([0], [0], "o", color=INK, ms=6)
    ax.text(-0.05, -0.05, "gốc (0, 0)", ha="right", va="top", fontsize=8.5, color=INK)
    for m in world.markers:
        W = world.marker_world_corners(m)
        ax.add_patch(Polygon(W[:, :2], closed=True, fc="#0b0b0b", ec="#0b0b0b", zorder=3))
        cx, cy = world.grid_to_world(m.i, m.j)
        a = math.radians(ex.ARROWS.get(m.arrow, 0))
        r = max(world.marker_mm / 1000 * 0.9, 0.6 * min(world.px, world.py))
        ax.annotate("", xy=(cx + r * math.cos(a), cy + r * math.sin(a)), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8), zorder=4)
        ax.text(cx, cy - world.marker_mm / 1000 * 0.75, f"M{m.id} ({m.i:g},{m.j:g})", ha="center", va="top",
                fontsize=8, color=INK, zorder=4)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=8.5); ax.set_ylabel("Y (m)", fontsize=8.5)
    ax.set_title("Sơ đồ nhìn từ trên xuống — đối chiếu với sàn thật", fontsize=10, loc="left")
    _style(fig)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Báo cáo HTML
# --------------------------------------------------------------------------
def _png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode()


def _jpg(rgb, width=1280):
    h, w = rgb.shape[:2]
    if w > width:
        rgb = cv2.resize(rgb, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


LV_TXT = {"ok": "Đạt", "warn": "Cần xem lại", "fail": "Không đạt"}


def build_report(proj: ex.Project, path: str):
    e = html.escape
    cams = [c for c in proj.cameras if c.has_pose and c.image is not None]
    rows, sections = [], []
    for c in cams:
        r = c.result
        C = r.C
        cs = r.C_std_cm or [float("nan")] * 3
        rows.append(f"<tr><td>{e(c.name)}</td><td><span class='dot {r.level}'></span>{LV_TXT[r.level]}</td>"
                    f"<td class=n>{r.n_inliers}/{r.n_total}</td><td class=n>{r.rms:.3f}</td><td class=n>{r.med_cm:.2f}</td>"
                    f"<td class=n>{C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}</td><td class=n>{max(cs):.2f}</td></tr>")
        warns = "".join(f"<li class='{lv}'>{e(t)}</li>" for lv, t in r.warnings)
        ov = draw_overlay(c, proj.world, proj.roi, show_err=False, test_points=proj.test_points, proj=proj)
        sections.append(f"<h2>{e(c.name)}</h2><figure><img src='data:image/jpeg;base64,{_jpg(ov)}'></figure>"
                        f"<p class=muted>rvec = {np.round(r.rvec, 6).tolist()} · tvec = {np.round(r.tvec, 4).tolist()} m</p>"
                        + (f"<ul class=w>{warns}</ul>" if warns else ""))
    tps = []
    for tp in proj.test_points:
        sm = tp.summary(proj.cameras)
        if sm:
            pair = "" if sm["max_pair_cm"] is None else "%.1f" % sm["max_pair_cm"]
            tps.append(f"<tr><td>{e(tp.name)}</td><td class=n>{sm['n']}</td><td class=n>{sm['mean'][0]:.3f}, {sm['mean'][1]:.3f}</td>"
                       f"<td class=n>{pair}</td></tr>")
    charts = [_png(fig_topview(proj))]
    mos = ""
    if proj.roi.valid() and cams:
        m, _, _ = ex.ground_mosaic(cams, proj.roi)
        mos = f"<h2>Ảnh ghép mặt sàn</h2><figure><img src='data:image/jpeg;base64,{_jpg(m, 1100)}'></figure>"
    roi = proj.roi
    roi_txt = (f"X {roi.x0:g} → {roi.x1:g} m, Y {roi.y0:g} → {roi.y1:g} m, ô {roi.cell_cm:g} cm, lưới "
               f"{roi.grid_shape()[0]} × {roi.grid_shape()[1]}") if roi.valid() else "chưa đặt"
    w = proj.world
    doc = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Báo cáo extrinsic</title>
<style>
:root{{--bg:#f9f9f7;--s:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI",system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px 60px}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:30px 0 10px}}
.muted{{color:var(--ink2)}} table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);font-size:14px}}
th{{color:var(--ink2);font-weight:500}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.card{{background:var(--s);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow-x:auto}}
figure{{margin:0 0 12px;background:var(--s);border:1px solid var(--line);border-radius:10px;padding:6px}} figure img{{width:100%;display:block}} figure.narrow img{{max-width:760px;margin:0 auto}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}} .dot.ok{{background:#0ca30c}} .dot.warn{{background:#fab219}} .dot.fail{{background:#d03b3b}}
ul.w li.fail{{color:#b42f2f}} ul.w li.warn{{color:#8a5d00}} ul.w li.ok{{color:var(--ink2)}}
</style></head><body><main>
<h1>Báo cáo hiệu chỉnh extrinsic</h1>
<div class=muted>{e(_dt.datetime.now().strftime('%d/%m/%Y %H:%M'))} · {len(cams)} camera · gạch {w.pitch_x_mm:g} × {w.pitch_y_mm:g} mm ·
marker {e(w.marker_dict)} {w.marker_mm:g} mm · vùng quay: {e(roi_txt)}</div>
<h2>Tổng hợp</h2><div class=card><table><tr><th>Camera</th><th>Trạng thái</th><th>Điểm dùng</th><th>RMS (px)</th>
<th>Sai số sàn trung vị (cm)</th><th>Vị trí camera X, Y, Z (m)</th><th>± vị trí (cm)</th></tr>{''.join(rows)}</table></div>
{'<h2>Điểm kiểm tra đa camera</h2><div class=card><table><tr><th>Điểm</th><th>Số camera</th><th>Vị trí X, Y (m)</th><th>Lệch lớn nhất giữa camera (cm)</th></tr>' + ''.join(tps) + '</table></div>' if tps else ''}
<h2>Nhìn từ trên xuống</h2><figure class=narrow><img src='data:image/png;base64,{charts[0]}'></figure>
{mos}
{''.join(sections)}
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
