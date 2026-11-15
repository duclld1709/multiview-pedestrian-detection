"""Shared model catalogue with project-specific execution requirements."""
import os


def weight_exists(filename):
    if not filename:
        return False
    path = os.path.abspath(filename)
    # Training run names frequently exceed Windows' legacy 260-character limit.
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\UNC\\" + path[2:] if path.startswith("\\\\") else "\\\\?\\" + path
    return os.path.isfile(path)


def describe_model(spec, kind, cameras, layout=None):
    reasons = []
    required = spec.get("num_cam")
    padded = kind == "wavetrack" and cameras == 5 and spec.get("backend") == "mvdetr" and required in (6, 7)
    if required != cameras and not padded:
        reasons.append(f"Weight yêu cầu {required} camera; project có {cameras} camera.")
    if kind is None:
        reasons.append(f"Chưa có adapter BEV cho {layout or 'dataset này'} (calibration và lưới mặt sàn).")
    elif spec.get("dataset") != kind and not padded:
        reasons.append(f"Weight dành cho {spec.get('dataset')}; project thuộc {kind}.")
    weights = spec.get("weights") or ""
    exists = weight_exists(weights)
    if not exists:
        reasons.append("Không tìm thấy file weight đã đăng ký.")
    return {"id": spec["id"], "method": spec.get("method", spec.get("backend", "Model")),
            "name": spec.get("name", spec["id"]), "backend": spec.get("backend"),
            "dataset": spec.get("dataset"), "num_cam": required, "weights": weights,
            "zip_member": spec.get("zip_member"), "weights_exists": exists,
            "moda": spec.get("test_moda"), "compatible": not reasons, "reasons": reasons,
            "experimental": padded, "warning": (f"Thử nghiệm: thêm {required-5} camera đen cho weight {spec.get('dataset')}. "
                                                "Chưa hiệu chỉnh cho WaveTrack; số MODA chỉ thuộc dataset gốc.") if padded else None,
            "status": "ready" if not reasons else "blocked", "best": False}


def model_catalog(models, kind, cameras, layout=None):
    rows = [describe_model(spec, kind, cameras, layout) for spec in models.values()]
    rows.sort(key=lambda row: (not row["compatible"], -(row["moda"] or 0), row["id"]))
    best = next((row for row in rows if row["compatible"] and not row["experimental"]), None)
    if best:
        best["best"] = True
    return rows
