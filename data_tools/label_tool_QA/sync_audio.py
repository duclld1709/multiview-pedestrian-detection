r"""Estimate camera time alignment from shared audio, then verify it in independent windows.

python sync_audio.py --root F:\thesis_dataset\WaveTrack --out projects/wavetrack_sync
Audio caches and the report are written under --out; source videos are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal, ndimage

RATE = 8000
HOP = 80


def audio_samples(path: Path, cache: Path) -> np.ndarray:
    import av
    fingerprint = f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}:audio-v1"
    target = cache / (hashlib.sha256(fingerprint.encode()).hexdigest()[:20] + ".npy")
    if target.exists():
        return np.load(target, mmap_mode="r")
    cache.mkdir(parents=True, exist_ok=True)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio: {path}")
        stream = container.streams.audio[0]
        video = container.streams.video[0]
        origin = float((video.start_time or 0) * video.time_base)
        duration = float(container.duration / av.time_base)
        samples = np.zeros(math.ceil((duration + 1) * RATE), np.float32)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
        end = 0
        def append(frame):
            nonlocal end
            values = frame.to_ndarray().reshape(-1)
            start = round((float(frame.pts * frame.time_base) - origin) * RATE)
            if start < 0:
                values, start = values[-start:], 0
            stop = min(start + len(values), len(samples))
            if stop > start:
                samples[start:stop] = values[:stop-start]
                end = max(end, stop)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                append(output)
        for output in resampler.resample(None):
            append(output)
    np.save(target, samples[:end])
    return np.load(target, mmap_mode="r")


def envelope(samples):
    filtered = signal.sosfilt(signal.butter(3, [250, 3000], btype="bandpass", fs=RATE, output="sos"), samples)
    n = len(filtered) // HOP
    values = np.log(np.sqrt(np.mean(filtered[:n*HOP].reshape(n, HOP)**2, axis=1)) + 1e-6)
    values -= ndimage.uniform_filter1d(values, size=101)
    return values.astype(np.float32)


def normalized_match(template, target):
    """Pearson correlation for every position where the complete template fits."""
    template = np.asarray(template, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(target) < len(template) or len(template) < 2:
        raise ValueError("Audio window is too short")
    template -= template.mean()
    n = len(template)
    sums = np.r_[0., np.cumsum(target)]
    squares = np.r_[0., np.cumsum(target**2)]
    energy = squares[n:] - squares[:-n] - (sums[n:] - sums[:-n])**2 / n
    cross = signal.correlate(target, template, mode="valid", method="fft")
    return cross / np.sqrt(np.maximum(energy * np.sum(template**2), 1e-20))


def align(reference, target):
    """target_time = reference_time + offset; robust line also estimates recorder clock drift."""
    ref_env, target_env = envelope(reference), envelope(target)
    # Multiple independent 30-second snippets locate overlap even for a shorter or unrelated take.
    coarse = []
    window = 3000
    for start in np.linspace(1000, max(1000, len(ref_env)-window-1000), 7).astype(int):
        template = ref_env[start:start+window]
        if len(template) < window or len(target_env) < window:
            continue
        scores = normalized_match(template, target_env)
        best = int(np.argmax(scores))
        coarse.append({"reference_s": start/100, "offset_s": (best-start)/100, "score": float(scores[best])})
    if not coarse:
        return {"accepted": False, "reason": "insufficient audio", "windows": []}
    leader = max(coarse, key=lambda row: sum(r["score"] for r in coarse if abs(r["offset_s"]-row["offset_s"]) < 0.5))
    agreeing = [r for r in coarse if abs(r["offset_s"]-leader["offset_s"]) < 0.5 and r["score"] > 0.15]
    if len(agreeing) < 3:
        return {"accepted": False, "reason": "no consistent shared audio", "coarse": coarse, "windows": []}
    measurements = []
    for row in agreeing:
        for delta in range(2, 29, 2):
            start = round((row["reference_s"] + delta) * RATE)
            n = RATE
            left = max(0, start + round((row["offset_s"] - 0.12)*RATE))
            right = min(len(target), start + n + round((row["offset_s"] + 0.12)*RATE))
            if right-left < n:
                continue
            scores = normalized_match(reference[start:start+n].copy(), target[left:right])
            best = int(np.argmax(np.abs(scores)))
            measurements.append({"reference_s": start/RATE + 0.5, "offset_s": (left+best-start)/RATE,
                                 "score": float(abs(scores[best]))})
    reliable = [r for r in measurements if r["score"] >= 0.12]
    if len(reliable) < 10:
        return {"accepted": False, "reason": "waveform verification failed", "coarse": coarse, "windows": measurements}
    x = np.array([r["reference_s"] for r in reliable])
    y = np.array([r["offset_s"] for r in reliable])
    from scipy.stats import theilslopes
    slope, intercept, _, _ = theilslopes(y, x)
    residual = y - (intercept + slope*x)
    keep = abs(residual) <= 0.025
    if keep.sum() < 10 or np.ptp(x[keep]) <= 100:
        return {"accepted": False, "reason": "insufficient consistent waveform windows", "coarse": coarse, "windows": measurements}
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    error = float(np.max(np.abs(y[keep] - (intercept + slope*x[keep]))))
    return {"accepted": bool(keep.sum() >= 10 and np.ptp(x[keep]) > 100 and error < 0.035 and abs(slope) < 0.002),
            "offset_s": float(intercept), "rate": float(1+slope), "residual_ms": error*1000,
            "coarse": coarse, "windows": measurements}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-files", nargs="+", help="Known consecutive clips of the first camera")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.root.glob("Cam*/*") if p.suffix.lower() in {".mp4", ".mov"})
    samples = {}
    for path in paths:
        print(f"Audio: {path.parent.name}/{path.name}", flush=True)
        samples[str(path)] = audio_samples(path, args.out / "audio_cache")
    if args.reference_files:
        build_profile(args.root, args.out, paths, samples, args.reference_files)
        return
    reference = paths[0]
    results = {}
    for path in paths[1:]:
        print(f"Align: {path.parent.name}/{path.name}", flush=True)
        result = align(samples[str(reference)], samples[str(path)])
        results[str(path)] = result
        print(json.dumps(result), flush=True)
        (args.out / "audio_alignment.json").write_text(json.dumps({"reference": str(reference), "results": results}, indent=2))


def load_profile(root: Path, directory: Path) -> dict | None:
    """Offer a verified profile only while the source files still have the measured fingerprints."""
    file = directory / "sync_profile.json"
    if not file.is_file():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        if Path(data["root"]).resolve() != root.resolve():
            return None
        for item in data["files"]:
            path = root / item["path"]
            if not path.is_file() or path.stat().st_size != item["size"] or path.stat().st_mtime_ns != item["mtime_ns"]:
                return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return data


def validate_profile(profile, samples, out):
    """Check held-out 10-second energy envelopes without adjusting the waveform-fitted clocks."""
    import av
    root = Path(profile["root"])
    videos = profile["settings"]["videos"]
    reference_name = next(iter(videos))
    reference_parts = []
    for name in videos[reference_name].get("files", [videos[reference_name]["file"]]):
        path = root / reference_name / name
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)
        reference_parts.append(samples[str(path)][:round(duration*RATE)])
    reference = envelope(np.concatenate(reference_parts))
    env_rate = RATE // HOP
    report, stats = {}, {}
    for name, config in videos.items():
        if name == reference_name:
            continue
        target = envelope(samples[str(root / name / config["file"])])
        clock = profile["clocks"][name]
        rows = []
        for start in range(60, int(len(reference)/env_rate)-10, 120):
            # Resample only this small search window into the reference clock.
            margin = round(.4*env_rate)
            times = start + np.arange(-margin, 10*env_rate+margin)/env_rate
            positions = (clock["offset_s"] + times*clock["rate"])*env_rate
            left, right = max(0, int(positions[0])-1), min(len(target), math.ceil(positions[-1])+2)
            if positions[0] < 0 or positions[-1] >= len(target):
                continue
            corrected = np.interp(positions, np.arange(left, right), target[left:right])
            scores = normalized_match(reference[start*env_rate:(start+10)*env_rate].copy(), corrected)
            best = int(np.argmax(scores))
            rows.append({"reference_s": start, "remaining_lag_ms": round((best-margin)/env_rate*1000, 3),
                         "correlation": round(float(scores[best]), 4)})
        report[name] = rows
        reliable = [abs(r["remaining_lag_ms"]) for r in rows if r["correlation"] >= .3]
        stats[name] = {"reliable_windows": len(reliable),
                       "median_absolute_lag_ms": float(np.median(reliable)) if reliable else None,
                       "max_absolute_lag_ms": max(reliable) if reliable else None}
    profile["independent_validation"] = stats
    profile["validation_method"] = "Held-out 10-second band-limited RMS energy envelopes, 10 ms steps; correlation >= 0.3."
    profile["limitations"] = ("Audio propagation, microphone processing and frame quantization remain. "
                              "Held-out audio lag is not a measurement of visual frame-lock accuracy.")
    (out / "holdout_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_report(profile, out):
    fps = profile['settings']['timeline_fps']
    count = math.ceil(profile['duration_s'] * fps - 1e-9)
    lines = ["# Đồng bộ thời gian WaveTrack", "",
             f"Đoạn chung: {profile['duration_s']:.3f} giây, đủ {len(profile['clocks'])} camera. Lấy {fps:g} frame/giây: {count} frame.", "",
             "| Camera | Video | Mốc đầu (giây) | Hệ số đồng hồ |", "|---|---|---:|---:|"]
    for name, config in profile["settings"]["videos"].items():
        files = " → ".join(config.get("files", [config["file"]]))
        lines.append(f"| {name} | {files} | {config['offset_s']:.6f} | {config['rate']:.9f} |")
    lines += ["", "Trong Setup, đọc dataset rồi bấm **Áp dụng đoạn đồng bộ 5 camera**. "
              "Tạo project mới để chạy label trên các frame đã đồng bộ; nhãn cũ không tự chuyển sang mốc mới.", "",
              "Camera 4 dùng IMG_2221.MOV. IMG_2223.MOV không có âm thanh trùng với phiên quay này.", "",
              "Tool ghép hai clip GoPro theo thứ tự, bù mốc bắt đầu và trôi đồng hồ, đọc timestamp thực của video "
              "và xử lý metadata xoay ảnh. Mọi camera được giới hạn vào cùng đoạn thời gian.", "",
              "Kiểm tra độc lập đường bao năng lượng âm thanh trong các cửa sổ 10 giây, bước đo 10 ms "
              "(chỉ thống kê tương quan ≥ 0.3; các cửa sổ còn lại không đủ tin cậy):", "",
              "| Camera | Cửa sổ tin cậy | Trung vị sai lệch (ms) | Sai lệch lớn nhất (ms) |", "|---|---:|---:|---:|"]
    for name, stats in profile.get("independent_validation", {}).items():
        lines.append(f"| {name} | {stats['reliable_windows']} | {stats['median_absolute_lag_ms']} | {stats['max_absolute_lag_ms']} |")
    lines += ["", "Đây là căn thời gian bằng phần mềm, còn ảnh hưởng của truyền âm, xử lý microphone và lượng tử hóa frame. "
              "Sai lệch âm thanh không phải phép đo chính xác sai lệch hình ảnh; chưa thể khẳng định đồng bộ từng frame như hệ thống khóa phần cứng.", "",
              "Số đo: `sync_measurements.json`, `holdout_validation.json`. Cấu hình áp dụng: `sync_profile.json`.", "",
              "Ảnh đối chiếu 5 camera: [120 giây](verify_120s.jpg), [700 giây](verify_700s.jpg), [1200 giây](verify_1200s.jpg)."]
    (out / "SYNC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_profile(root, out, paths, samples, reference_files):
    import av
    from scipy.stats import theilslopes
    from sources import Dataset
    reference_folder = paths[0].parent
    reference_paths = [reference_folder / name for name in reference_files]
    def duration(path):
        with av.open(str(path)) as container:
            s = container.streams.video[0]
            return float(s.duration * s.time_base)
    ref_duration = sum(duration(p) for p in reference_paths)
    clocks = {reference_folder.name: {"offset_s": 0., "rate": 1., "residual_ms": 0., "duration_s": ref_duration}}
    videos = {reference_folder.name: {"file": reference_files[0], "files": reference_files}}
    report, used = {}, list(reference_paths)
    for folder in sorted({p.parent for p in paths} - {reference_folder}):
        candidates = []
        for path in [p for p in paths if p.parent == folder]:
            first = align(samples[str(reference_paths[0])], samples[str(path)])
            report[str(path)] = {"first_clip": first}
            print(json.dumps({"file": path.name, **{k:v for k,v in first.items() if k not in ('coarse','windows')}}), flush=True)
            if first["accepted"]:
                candidates.append((path, first))
        if not candidates:
            (out / "sync_measurements.json").write_text(json.dumps(report, indent=1))
            raise ValueError(f"No verified overlapping take for {folder.name}; inspect sync_measurements.json")
        path, first = max(candidates, key=lambda item: np.median([r['score'] for r in item[1]['coarse']]))
        rows = list(first["windows"])
        elapsed = duration(reference_paths[0])
        for ref in reference_paths[1:]:
            later = align(samples[str(ref)], samples[str(path)])
            report[str(path)][ref.name] = later
            if not later["accepted"]:
                (out / "sync_measurements.json").write_text(json.dumps(report, indent=1))
                raise ValueError(f"Cannot verify later reference clip against {path}")
            rows += [{**r, "reference_s": r["reference_s"]+elapsed, "offset_s": r["offset_s"]-elapsed}
                     for r in later["windows"]]
            elapsed += duration(ref)
        rows = [r for r in rows if r["score"] >= 0.12]
        x, y = np.array([r["reference_s"] for r in rows]), np.array([r["offset_s"] for r in rows])
        slope, intercept, _, _ = theilslopes(y, x)
        keep = abs(y-(intercept+slope*x)) <= .025
        if keep.sum() < 10 or np.ptp(x[keep]) <= 100:
            raise ValueError(f"Insufficient consistent recorder clock measurements for {path}")
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = abs(y[keep]-(intercept+slope*x[keep]))
        if keep.sum() < 10 or abs(slope) > .002 or max(residual) > .035:
            raise ValueError(f"Inconsistent recorder clock for {path}")
        clocks[folder.name] = {"offset_s": float(intercept), "rate": float(1+slope),
                               "residual_ms": float(np.quantile(residual, .95)*1000),
                               "verified_windows": int(keep.sum()), "duration_s": duration(path)}
        videos[folder.name] = {"file": path.name}
        used.append(path)
    start = max(0., *[-c["offset_s"]/c["rate"] for c in clocks.values()])
    end = min((c["duration_s"]-c["offset_s"])/c["rate"] for c in clocks.values())
    # Avoid the capture startup and end boundary; still retain the complete useful shared interval.
    start, end = start + 1., end - 1.
    if end <= start:
        raise ValueError("No common time interval for every camera")
    for name, config in videos.items():
        c = clocks[name]
        config.update(offset_s=c["offset_s"]+start*c["rate"], rate=c["rate"],
                      end_s=c["offset_s"]+end*c["rate"], read_mode="timestamp")
    settings = {"videos": videos, "timeline_fps": 2.}
    dataset = Dataset(root, **settings)
    assert all(dataset.active(f, dataset.frames[-1]) for f in dataset.folders)
    profile = {"root": str(root.resolve()), "method": "shared audio: coarse envelope + independent waveform windows + clock drift fit",
               "duration_s": end-start, "reference_start_s": start, "clocks": clocks, "settings": settings,
               "summary": f"Đã đối chiếu âm thanh ở cả hai clip GoPro: đoạn chung {end-start:.1f} giây, đủ 5 camera. Chọn đúng take, bù độ lệch và trôi đồng hồ; đọc theo timestamp video.",
               "files": [{"path": str(p.relative_to(root)), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in used]}
    (out / "sync_measurements.json").write_text(json.dumps(report, indent=1))
    validate_profile(profile, samples, out)
    (out / "sync_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(profile, out)
    print(json.dumps({"duration_s": end-start, "clocks": clocks}), flush=True)


if __name__ == "__main__":
    main()
