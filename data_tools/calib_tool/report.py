"""
Biểu đồ (matplotlib, dùng chung cho giao diện và báo cáo) và báo cáo HTML tự chứa.
Dùng matplotlib.figure.Figure trực tiếp (không pyplot) để an toàn khi chạy trong luồng nền.
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
from matplotlib.patches import Circle

from core import (CalibResult, ImageResult, BoardConfig, coverage_grid, tilt_summary,
                  dataset_suggestions, CELL_NAMES, TILT_FRONTAL, MODELS)

# ---- bảng màu (palette tham chiếu, đã kiểm chứng) ----
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
STATUS = {"ok": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b"}


def _style(fig: Figure):
    fig.patch.set_facecolor(SURFACE)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.title.set_color(INK)
        ax.xaxis.label.set_color(INK2)
        ax.yaxis.label.set_color(INK2)


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seqblue", SEQ)


# --------------------------------------------------------------------------
def fig_coverage(results: list[ImageResult], size=None) -> Figure:
    fig = Figure(figsize=(6.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    used = [r for r in results if r.used and r.detected]
    if size is None:
        size = (used[0].width, used[0].height) if used else (1920, 1080)
    W, H = size
    grid = coverage_grid(results)
    total = max(1, grid.sum())
    vmax = max(1, grid.max())
    ax.imshow(grid, cmap=_seq_cmap(), vmin=0, vmax=vmax, extent=(0, W, H, 0), interpolation="nearest", alpha=0.85)
    for r in used:
        p = r.corners.reshape(-1, 2) * [W / r.width, H / r.height]
        ax.scatter(p[:, 0], p[:, 1], s=4, color=INK, alpha=0.35, linewidths=0)
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            low = v < 0.03 * total
            ax.text((j + 0.5) * W / 3, (i + 0.5) * H / 3, f"{v}" + ("\nthiếu" if low else ""),
                    ha="center", va="center", fontsize=11 if not low else 10,
                    color=("#ffffff" if v > 0.6 * vmax else INK), fontweight="bold" if low else "normal")
    for k in (1, 2):
        ax.axvline(k * W / 3, color=SURFACE, lw=2)
        ax.axhline(k * H / 3, color=SURFACE, lw=2)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Độ phủ khung hình — số góc bàn cờ trong mỗi vùng", fontsize=10, loc="left")
    _style(fig)
    fig.tight_layout()
    return fig


def fig_tilt(results: list[ImageResult]) -> Figure:
    fig = Figure(figsize=(4.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    pts = [(r.tilt_x, r.tilt_y) for r in results if r.used and r.tilt_x is not None]
    lim = 50
    ax.add_patch(Circle((0, 0), TILT_FRONTAL, facecolor="#f0efec", edgecolor=GRID, lw=1))
    ax.add_patch(Circle((0, 0), 40, facecolor="none", edgecolor=GRID, lw=1, ls="--"))
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    if pts:
        p = np.array(pts)
        ax.scatter(p[:, 0], p[:, 1], s=46, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.text(0, 0, "thẳng\nmặt", ha="center", va="center", fontsize=8, color=INK2)
    ax.text(lim - 2, 2, "mép phải\nxa hơn", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.text(-lim + 2, 2, "mép trái\nxa hơn", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.text(2, lim - 2, "mép dưới xa hơn", ha="left", va="top", fontsize=8, color=INK2)
    ax.text(2, -lim + 2, "mép trên xa hơn", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(-lim, lim); ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    ax.set_xlabel("ngả trái ↔ phải (°)", fontsize=8.5)
    ax.set_ylabel("ngả trên ↔ dưới (°)", fontsize=8.5)
    ax.set_title("Góc ngả của bảng (mỗi chấm một ảnh)", fontsize=10, loc="left")
    _style(fig)
    fig.tight_layout()
    return fig


def fig_errors(results: list[ImageResult], rms: float | None = None) -> Figure:
    rows = [r for r in results if r.reproj_err is not None]
    h = max(2.4, 0.22 * len(rows) + 1.0)
    fig = Figure(figsize=(6.4, h), dpi=100)
    ax = fig.add_subplot(111)
    if rows:
        names = [r.name for r in rows]
        vals = [r.reproj_err for r in rows]
        colors = [ORANGE if r.excluded_auto else BLUE for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(y, names, fontsize=8)
        ax.invert_yaxis()
        if rms is not None:
            ax.axvline(rms, color=INK2, lw=1, ls="--")
            ax.text(rms, -0.9, f" RMS {rms:.3f}", color=INK2, fontsize=8, va="bottom")
        ax.set_xlabel("sai số chiếu lại (px)", fontsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        from matplotlib.patches import Patch
        if any(r.excluded_auto for r in rows):
            handles = [Patch(color=BLUE, label="dùng để hiệu chỉnh"), Patch(color=ORANGE, label="đã tự loại")]
            ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right",
                      bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
    ax.set_title("Sai số từng ảnh", fontsize=10, loc="left", pad=14)
    _style(fig)
    fig.tight_layout()
    return fig


def fig_residuals(res: CalibResult) -> Figure:
    fig = Figure(figsize=(6.4, 3.2), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    R = res.residuals
    W, H = res.image_size
    if R is not None and len(R):
        lim = max(0.5, float(np.percentile(np.abs(R[:, 2:]), 99)) * 1.2)
        ax1.scatter(R[:, 2], R[:, 3], s=5, color=BLUE, alpha=0.45, linewidths=0)
        ax1.add_patch(Circle((0, 0), res.rms, facecolor="none", edgecolor=INK2, lw=1, ls="--"))
        ax1.set_xlim(-lim, lim); ax1.set_ylim(lim, -lim); ax1.set_aspect("equal")
        ax1.axhline(0, color=GRID, lw=1); ax1.axvline(0, color=GRID, lw=1)
        ax1.set_xlabel("dx (px)", fontsize=8.5); ax1.set_ylabel("dy (px)", fontsize=8.5)
        mag = np.hypot(R[:, 2], R[:, 3])
        scale = 0.06 * W / max(1e-6, float(np.percentile(mag, 95)))
        ax2.quiver(R[:, 0], R[:, 1], R[:, 2] * scale, R[:, 3] * scale, angles="xy", scale_units="xy",
                   scale=1, color=BLUE, width=0.004, alpha=0.8)
        ax2.set_xlim(0, W); ax2.set_ylim(H, 0); ax2.set_aspect("equal")
        ax2.set_xticks([]); ax2.set_yticks([])
        for s in ax2.spines.values():
            s.set_visible(True); s.set_color(GRID)
    ax1.set_title("Phân bố sai số", fontsize=10, loc="left")
    ax2.set_title("Vector sai số trên ảnh (phóng to)", fontsize=10, loc="left")
    _style(fig)
    for s in ax2.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    return fig


def distortion_field(K, dist, size, model, n=40):
    W, H = size
    xs, ys = np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, H - 1, int(n * H / W) + 1))
    pts = np.stack([xs, ys], -1).reshape(-1, 1, 2).astype(np.float64)
    if model == "fisheye":
        und = cv2.fisheye.undistortPoints(pts.reshape(1, -1, 2), K, dist, P=K).reshape(-1, 2)
    else:
        und = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
    d = und - pts.reshape(-1, 2)
    return xs, ys, d[:, 0].reshape(xs.shape), d[:, 1].reshape(xs.shape)


def fig_distortion(res: CalibResult) -> Figure:
    fig = Figure(figsize=(6.4, 3.9), dpi=100)
    ax = fig.add_subplot(111)
    W, H = res.image_size
    try:
        xs, ys, dx, dy = distortion_field(res.K, res.dist, res.image_size, res.model)
        mag = np.hypot(dx, dy)
        cs = ax.contourf(xs, ys, mag, levels=8, cmap=_seq_cmap())
        cb = fig.colorbar(cs, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("độ dịch do méo (px)", fontsize=8.5, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_visible(False)
        st = max(1, xs.shape[1] // 14)
        ax.quiver(xs[::st, ::st], ys[::st, ::st], dx[::st, ::st], dy[::st, ::st], angles="xy",
                  scale_units="xy", scale=1, color=INK, width=0.0025, alpha=0.6)
        ax.plot([res.cx], [res.cy], marker="+", ms=12, color=ORANGE, mew=2)
    except cv2.error:
        ax.text(0.5, 0.5, "Không tính được trường méo", transform=ax.transAxes, ha="center")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Mức méo ống kính trên khung hình (+ = tâm ảnh)", fontsize=10, loc="left")
    _style(fig)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Báo cáo HTML
# --------------------------------------------------------------------------
def _png(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode()


ICON = {"ok": "✓", "warn": "!", "fail": "✕"}


def build_report(res: CalibResult, results: list[ImageResult], cfg: BoardConfig, path: str):
    e = html.escape
    std = res.std_intrinsics
    rows = []

    def pm(i):
        return f" ± {std[i]:.2f}" if std is not None and len(std) > i and std[i] > 0 else ""

    rows += [("fx", f"{res.fx:.2f}{pm(0)} px"), ("fy", f"{res.fy:.2f}{pm(1)} px"),
             ("cx", f"{res.cx:.2f}{pm(2)} px"), ("cy", f"{res.cy:.2f}{pm(3)} px")]
    for k, (name, v) in enumerate(zip(res.dist_names(), res.dist.ravel())):
        s = (f" ± {std[4 + k]:.4f}" if (std is not None and len(std) > 4 + k and std[4 + k] > 0
                                         and res.model != "fisheye") else " (cố định)" if res.model != "fisheye"
             and std is not None and len(std) > 4 + k else "")
        rows.append((name, f"{v:.6f}{s}"))
    param_html = "".join(f"<tr><th>{e(a)}</th><td>{e(b)}</td></tr>" for a, b in rows)

    K = res.K
    kmat = "\n".join("  ".join(f"{v:12.4f}" for v in row) for row in K)
    dvec = ", ".join(f"{v:.8f}" for v in res.dist.ravel())

    checks_html = "".join(
        f'<li class="{c.level}"><span class="ic">{ICON[c.level]}</span><span>{e(c.text)}</span></li>' for c in res.checks)

    stab = res.stability
    stab_html = ""
    if stab:
        stab_html = (f"<p class='muted'>Kiểm tra chéo {stab['runs']} lần (mỗi lần dùng ~70% ảnh): "
                     f"σ(fx) = {stab['fx_std']:.2f} px, σ(fy) = {stab['fy_std']:.2f} px, "
                     f"σ(cx) = {stab['cx_std']:.2f} px, σ(cy) = {stab['cy_std']:.2f} px. "
                     + (f"Sai số trên ảnh để riêng: {res.holdout_err:.3f} px." if res.holdout_err else "") + "</p>")

    img_rows = []
    for r in results:
        lvl = r.status_level()
        img_rows.append(
            "<tr>"
            f"<td>{e(r.name)}</td><td class='n'>{r.n_corners}</td><td class='n'>{r.sharpness:.0f}</td>"
            f"<td class='n'>{r.area_pct:.1f}%</td>"
            f"<td class='n'>{'' if r.tilt_deg is None else f'{r.tilt_deg:.0f}°'}</td><td>{e(r.tilt_dir)}</td>"
            f"<td class='n'>{'' if r.reproj_err is None else f'{r.reproj_err:.3f}'}</td>"
            f"<td>{'có' if (r.name in res.used_names) else 'không'}</td>"
            f"<td><span class='dot {lvl}'></span>{e(r.status_text())}</td>"
            "</tr>")

    tips = "".join(f"<li>{e(t)}</li>" for t in dataset_suggestions(results))
    charts = [
        ("Độ phủ khung hình", _png(fig_coverage(results, res.image_size))),
        ("Góc ngả", _png(fig_tilt(results))),
        ("Sai số từng ảnh", _png(fig_errors(results, res.rms))),
        ("Sai số chi tiết", _png(fig_residuals(res))),
        ("Méo ống kính", _png(fig_distortion(res))),
    ]
    charts_html = "".join(f'<figure><img alt="{e(t)}" src="data:image/png;base64,{b}"></figure>' for t, b in charts)

    now = _dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    doc = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo cáo hiệu chỉnh camera</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df;
  --ok:#0ca30c; --warn:#b77b00; --fail:#d03b3b; --okbg:#e8f6e8; --warnbg:#fdf3dc; --failbg:#fbe9e9; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 "Segoe UI", system-ui, sans-serif; }}
main {{ max-width:980px; margin:0 auto; padding:28px 16px 60px }}
h1 {{ font-size:24px; margin:0 0 4px }} h2 {{ font-size:17px; margin:32px 0 10px }}
.muted {{ color:var(--ink2) }}
.verdict {{ margin:18px 0; padding:14px 18px; border-radius:10px; font-weight:600; font-size:17px }}
.verdict.ok {{ background:var(--okbg); color:var(--ok) }} .verdict.warn {{ background:var(--warnbg); color:var(--warn) }}
.verdict.fail {{ background:var(--failbg); color:var(--fail) }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px }}
@media (max-width:720px) {{ .grid2 {{ grid-template-columns:1fr }} }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px }}
table {{ border-collapse:collapse; width:100% }}
th, td {{ text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); font-size:14px; vertical-align:top }}
th {{ color:var(--ink2); font-weight:500 }} td.n {{ text-align:right; font-variant-numeric:tabular-nums }}
pre {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:10px; overflow:auto; font-size:13px }}
ul.checks {{ list-style:none; padding:0; margin:0 }}
ul.checks li {{ display:flex; gap:10px; padding:7px 0; border-bottom:1px solid var(--line) }}
.ic {{ flex:0 0 22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
  color:#fff; font-size:12px; font-weight:700 }}
li.ok .ic {{ background:var(--ok) }} li.warn .ic {{ background:#fab219; color:#0b0b0b }} li.fail .ic {{ background:var(--fail) }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px }}
.dot.ok {{ background:var(--ok) }} .dot.warn {{ background:#fab219 }} .dot.fail {{ background:var(--fail) }}
figure {{ margin:0 0 16px; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:8px }}
figure img {{ width:100%; height:auto; display:block }}
.scroll {{ overflow-x:auto }}
</style></head><body><main>
<h1>Báo cáo hiệu chỉnh camera</h1>
<div class="muted">{e(now)} · {res.image_size[0]}×{res.image_size[1]} px · mô hình: {e(MODELS.get(res.model, res.model))}<br>
Bảng: {e(cfg.describe())}</div>
<div class="verdict {res.verdict_level}">{e(res.verdict)}</div>

<div class="grid2">
 <div class="card"><h2 style="margin-top:0">Thông số camera</h2><table>{param_html}
  <tr><th>RMS</th><td>{res.rms:.4f} px</td></tr><tr><th>Số ảnh dùng</th><td>{len(res.used_names)}</td></tr></table>
  {stab_html}</div>
 <div class="card"><h2 style="margin-top:0">Kiểm tra độ tin cậy</h2><ul class="checks">{checks_html}</ul></div>
</div>

<h2>Ma trận camera K</h2><pre>{e(kmat)}</pre>
<h2>Hệ số méo ({e(", ".join(res.dist_names()))})</h2><pre>{e(dvec)}</pre>

<h2>Gợi ý cho bộ ảnh</h2><div class="card"><ul>{tips}</ul></div>

<h2>Biểu đồ</h2>{charts_html}

<h2>Chi tiết từng ảnh</h2>
<div class="card scroll"><table>
<tr><th>Ảnh</th><th>Góc</th><th>Độ nét</th><th>Diện tích</th><th>Ngả</th><th>Hướng ngả</th><th>Sai số (px)</th><th>Dùng</th><th>Trạng thái</th></tr>
{''.join(img_rows)}
</table></div>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# --------------------------------------------------------------------------
# Dòng thời gian quét video
# --------------------------------------------------------------------------
def fig_video_timeline(scan, max_speed: float) -> Figure:
    fig = Figure(figsize=(7.6, 3.0), dpi=100)
    ax = fig.add_subplot(111)
    fr = scan.frames
    cap = max(4 * max_speed, 1.0)
    groups = [
        ("Không thấy bảng / thiếu góc", lambda f: f.status in ("không thấy bảng", "thiếu góc"), "#b5b4ae", 10),
        ("Đang di chuyển", lambda f: f.status == "đang di chuyển", ORANGE, 12),
        ("Mờ", lambda f: f.status == "mờ", "#eda100", 12),
        ("Tốt, không chọn", lambda f: f.status in ("ứng viên", "trùng thời gian", "ít đa dạng", "không đo được chuyển động"), "#9ec5f4", 14),
        ("Được chọn", lambda f: f.status == "chọn", BLUE, 34),
    ]
    for label, pred, color, size in groups:
        pts = [(f.t, 0.0 if f.n_corners == 0 or f.speed is None else min(f.speed, cap)) for f in fr if pred(f)]
        if pts:
            p = np.array(pts)
            ax.scatter(p[:, 0], p[:, 1], s=size, color=color, label=f"{label} ({len(pts)})",
                       edgecolors=SURFACE, linewidths=0.8 if size > 20 else 0, zorder=3 if size > 20 else 2)
    ax.axhline(max_speed, color=INK2, lw=1, ls="--")
    ax.text(0, max_speed, f" ngưỡng đứng yên {max_speed:g} px/khung", color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(-0.05 * cap, cap * 1.05)
    ax.set_xlabel("thời gian (s)", fontsize=8.5)
    ax.set_ylabel("tốc độ bảng (px/khung)", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Dòng thời gian video — mỗi chấm là một khung đã phân tích", fontsize=10, loc="left")
    _style(fig)
    fig.tight_layout()
    return fig
