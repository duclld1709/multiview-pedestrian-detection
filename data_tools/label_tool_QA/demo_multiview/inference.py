"""Checkpoint-backed inference for the thesis detectors (MVDet, MVDeTr).

Both repositories ship a package called ``multiview_detector``; each backend imports its own copy once and
then removes it from ``sys.modules`` so the two can be loaded side by side in one server process.
"""

from __future__ import annotations

import ast
import base64
import bisect
import hashlib
import io
import json
import os
import re
import sys
import threading
import types
import zipfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")   # MVDeTr imports pyplot at module level

import kinds

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPOS = {"mvdet": PROJECT / "MVDet", "mvdetr": PROJECT / "MVDeTr"}
BACKEND_LABELS = {"mvdet": "MVDet", "mvdetr": "MVDeTr"}
DEFAULT_THRESHOLDS = {"mvdet": 0.4, "mvdetr": 0.6}


def default_frame(kind: str) -> int:
    """First test frame (last 10 %) of the kind's dataset."""
    return kinds.get(kind).default_frame

PACKAGE = "multiview_detector"
WEIGHT_NAME = "MultiviewDetector.pth"
NMS_RADIUS = 20       # grid cells, as in the trainers' test loops
LONG_PREFIX = "\\\\?\\"
GPU_LOCK = threading.Lock()


class PomIndex:
    """Sparse on-disk index for the large rectangles.pom file."""

    def __init__(self, path: Path, stride: int = 512):
        self.path = path
        self.checkpoints = {}
        self.stride = stride
        if not path.is_file():
            return
        with path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.startswith(b"RECTANGLE "):
                    continue
                parts = line.split(maxsplit=3)
                if len(parts) < 3:
                    continue
                try:
                    camera, position = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                checkpoints = self.checkpoints.setdefault(camera, [])
                if not checkpoints or position - checkpoints[-1][0] >= stride:
                    checkpoints.append((position, offset))

    def boxes(self, position: int) -> dict[int, list[int]]:
        boxes = {}
        if not self.checkpoints:
            return boxes
        with self.path.open("rb") as source:
            for camera, checkpoints in self.checkpoints.items():
                index = bisect.bisect_right(checkpoints, (position, float("inf"))) - 1
                if index < 0:
                    continue
                source.seek(checkpoints[index][1])
                for line in source:
                    if not line.startswith(b"RECTANGLE "):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    try:
                        line_camera, line_position = int(parts[1]), int(parts[2])
                    except ValueError:
                        continue
                    if line_camera != camera or line_position > position:
                        break
                    if line_position == position and len(parts) >= 7:
                        try:
                            left, top, right, bottom = map(int, parts[3:7])
                        except ValueError:
                            break
                        if right > left and bottom > top:
                            boxes[camera] = [max(0, left), max(0, top), min(1919, right), min(1079, bottom)]
                        break
        return boxes


# ---------------------------------------------------------------------------------------------------- imports

_REPO_LOCK = threading.Lock()
_REPO_SYMBOLS = {}


def _package_modules() -> list[str]:
    return [name for name in sys.modules if name == PACKAGE or name.startswith(PACKAGE + ".")]


def _mvdet_symbols():
    from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.models.persp_trans_detector import PerspTransDetector
    from multiview_detector.utils.nms import nms
    return SimpleNamespace(Wildtrack=Wildtrack, MultiviewX=MultiviewX, Model=PerspTransDetector, nms=nms)


class _PyTorchDeformAttn:
    """Stands in for the compiled MultiScaleDeformableAttention op (same maths, eval only)."""

    def __init__(self, core):
        self.core = core

    def apply(self, value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return self.core(value, shapes, sampling_locations, attention_weights)


class _CompiledDeformAttn:
    """The CUDA op only dispatches float/double, so run it in fp32 when autocast hands it fp16 tensors."""

    def __init__(self, function):
        self.function = function

    def apply(self, value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        output = self.function.apply(value.float(), shapes, level_start_index, sampling_locations.float(),
                                     attention_weights.float(), im2col_step)
        return output.to(value.dtype)


def _mvdetr_symbols():
    import torch  # noqa: F401  (the compiled op links against torch's DLLs)
    native_ops = str(HERE / "native")   # built by build_deform_attn.ps1
    sys.path.insert(0, native_ops)
    try:
        import MultiScaleDeformableAttention  # noqa: F401
        compiled = True
    except ImportError:
        sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")
        compiled = False
    finally:
        sys.path.remove(native_ops)
    from multiview_detector.models.ops.functions import ms_deform_attn_func
    from multiview_detector.models.ops.modules import ms_deform_attn
    if compiled and os.environ.get("THESIS_DEMO_DEFORM") != "pytorch":
        ms_deform_attn.MSDeformAttnFunction = _CompiledDeformAttn(ms_deform_attn_func.MSDeformAttnFunction)
    else:
        compiled = False
        ms_deform_attn.MSDeformAttnFunction = _PyTorchDeformAttn(ms_deform_attn_func.ms_deform_attn_core_pytorch)
    from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.models import mvdetr
    from multiview_detector.utils.decode import mvdet_decode
    from multiview_detector.utils.nms import nms
    # The checkpoint overwrites the backbone, so skip the ImageNet download MVDeTr.__init__ asks for.
    backbone = mvdetr.resnet18
    mvdetr.resnet18 = lambda pretrained=False, **kwargs: backbone(pretrained=False, **kwargs)
    return SimpleNamespace(Wildtrack=Wildtrack, MultiviewX=MultiviewX, Model=mvdetr.MVDeTr,
                           decode=mvdet_decode, nms=nms, compiled_deform=compiled)


SYMBOL_LOADERS = {"mvdet": _mvdet_symbols, "mvdetr": _mvdetr_symbols}


def repo_symbols(backend: str) -> SimpleNamespace:
    with _REPO_LOCK:
        if backend not in _REPO_SYMBOLS:
            for name in _package_modules():
                del sys.modules[name]
            root = str(REPOS[backend])
            sys.path.insert(0, root)
            try:
                _REPO_SYMBOLS[backend] = SYMBOL_LOADERS[backend]()
            finally:
                sys.path.remove(root)
                for name in _package_modules():
                    del sys.modules[name]
        return _REPO_SYMBOLS[backend]


# ---------------------------------------------------------------------------------------------------- runners

def native(path) -> str:
    """Path string that also opens beyond Windows' 260-character limit (the MVDeTr run folders are longer)."""
    text = os.path.abspath(str(path))
    if os.name == "nt" and not text.startswith(LONG_PREFIX):
        return LONG_PREFIX + text
    return text


def read_weights(spec: dict):
    import torch
    if spec.get("zip_member"):
        with zipfile.ZipFile(native(spec["weights"])) as archive, archive.open(spec["zip_member"]) as weight_file:
            return torch.load(io.BytesIO(weight_file.read()), map_location="cpu", weights_only=True)
    with open(native(spec["weights"]), "rb") as weight_file:
        return torch.load(io.BytesIO(weight_file.read()), map_location="cpu", weights_only=True)


class GenericBase:
    """Dataset object for a kind of kinds.json, with the attributes the MVDet and MVDeTr constructors read, in the
    conventions of the trainers' Wildtrack / MultiviewX classes: calibration units; MVDet maps heatmap (row, col) to
    the world with worldgrid2worldcoord_mat, MVDeTr maps grid (x, y) with worldcoord_from_worldgrid_mat."""

    def __init__(self, kind: "kinds.Kind", root: Path):
        import numpy as np
        self.__name__ = kind.name
        self.root, self.kind = str(root), kind
        self.img_shape, self.worldgrid_shape = list(kinds.IMAGE_SHAPE), kind.worldgrid_shape()
        self.num_cam, self.indexing = kind.cameras, kind.indexing
        swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        self.world_indexing_from_xy_mat = np.eye(3) if kind.indexing == "xy" else swap
        self.world_indexing_from_ij_mat = swap if kind.indexing == "xy" else np.eye(3)
        self.img_xy_from_xy_mat, self.img_xy_from_ij_mat = np.eye(3), swap
        self.worldcoord_unit = 1.0 / kind.unit_per_m
        ox, oy = kind.origin
        self.worldcoord_from_worldgrid_mat = np.array([[kind.cell, 0, ox], [0, kind.cell, oy], [0, 0, 1]])
        # MVDet indexes its heatmap (row, col); rows of an xy grid run along world Y (MultiviewX: [[0, .025], [.025, 0]])
        self.worldgrid2worldcoord_mat = self.worldcoord_from_worldgrid_mat @ (swap if kind.indexing == "xy" else np.eye(3))
        self.intrinsic_matrices, self.extrinsic_matrices = kind.read_calibration(Path(root))


class Runner:
    """One checkpoint bound to one calibration folder, loaded lazily onto the GPU."""

    def __init__(self, spec: dict, kind: str, calib_root: Path, fp16: bool = False, gpu_gap_s: float | None = None):
        self.spec, self.kind, self.calib_root, self.fp16 = spec, kind, Path(calib_root), fp16
        self.gpu_gap_s = gpu_gap_s   # None: run at full speed; else yield the GPU to other programs (Unity)
        self.threshold = float(spec.get("threshold", DEFAULT_THRESHOLDS[spec["backend"]]))
        self.model = None
        self.lock = threading.Lock()

    def base(self, symbols):
        """The dataset object the model constructors read the cameras and the grid from: the trainers' own
        Wildtrack / MultiviewX classes, or GenericBase for the kinds of kinds.json."""
        if self.kind == "wildtrack":
            return symbols.Wildtrack(str(self.calib_root))
        if self.kind == "multiviewx":
            return symbols.MultiviewX(str(self.calib_root))
        return GenericBase(kinds.get(self.kind), self.calib_root)

    def predict(self, images: list) -> tuple:
        """Images of every camera (PIL or RGB uint8 arrays) -> (heatmap [H, W], [(x, y, score)] in world-grid cells)."""
        import torch
        with self.lock:
            if self.model is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("These checkpoints need CUDA, but no CUDA GPU is available.")
                self.load()
                if self.gpu_gap_s is not None:
                    self.share_gpu(self.gpu_gap_s)
            if len(images) != self.num_cam:
                raise ValueError(f"{self.spec['method']} expects {self.num_cam} cameras, got {len(images)}")
            try:
                with GPU_LOCK, torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=self.fp16):
                    return self.forward(self.batch(images))
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory: other programs hold the VRAM (check nvidia-smi / Task "
                                   "Manager), or start with -MaxLoaded 1 -Fp16.") from error

    @staticmethod
    def batch(images: list):
        """[1, N, 3, 720, 1280] ImageNet-normalised on the GPU (both trainers feed 1280x720 crops of 1920x1080)."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        arrays = [image if isinstance(image, np.ndarray) else np.asarray(image.convert("RGB")) for image in images]
        pixels = torch.from_numpy(np.stack(arrays)).cuda()[..., :3]   # RGBA frames from Unity: drop alpha on the GPU
        pixels = pixels.permute(0, 3, 1, 2).float() / 255
        if tuple(pixels.shape[-2:]) != (720, 1280):
            pixels = F.interpolate(pixels, size=(720, 1280), mode="bilinear", align_corners=False, antialias=True)
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((pixels - mean) / std).unsqueeze(0)

    def unload(self):
        with self.lock:
            self.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def _yielding(module, gap_s: float, chunked: bool):
    """Wrap a sub-network so it runs in short GPU bursts with a pause after each.

    On Windows the GPU is time-sliced between processes at the boundaries of submitted work: one 200 ms burst of
    CUDA kernels leaves the Unity editor (same GPU) without a single frame. Running the backbone one camera at a
    time and synchronising in between lets Unity render between bursts.
    """
    import time
    import torch

    class Yielding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = module

        def forward(self, x, *args, **kwargs):
            if not chunked:
                out = self.inner(x, *args, **kwargs)
                torch.cuda.synchronize()
                time.sleep(gap_s)
                return out
            outs = []
            for part in x.split(1):
                outs.append(self.inner(part))
                torch.cuda.synchronize()
                time.sleep(gap_s)
            return torch.cat(outs)

    return Yielding()


class MVDetRunner(Runner):
    def share_gpu(self, gap_s: float):
        self.model.base_pt2 = _yielding(self.model.base_pt2, gap_s, chunked=False)   # already one camera per call

    def load(self):
        symbols = repo_symbols("mvdet")
        base = self.base(symbols)
        adapter = SimpleNamespace(base=base, num_cam=base.num_cam, img_shape=base.img_shape,
                                  reducedgrid_shape=[dimension // 4 for dimension in base.worldgrid_shape],
                                  grid_reduce=4, img_reduce=4)
        model = symbols.Model(adapter, self.spec.get("arch", "resnet18"))
        model.load_state_dict(read_weights(self.spec), strict=True)
        self.nms, self.num_cam, self.model = symbols.nms, base.num_cam, model.eval()
        self.indexing = base.indexing

    def forward(self, batch):
        heatmap, _ = self.model(batch)
        heatmap = heatmap.detach().float().cpu().squeeze()
        rows = (heatmap > self.threshold).nonzero()
        scores = heatmap[heatmap > self.threshold]
        coordinates = (rows[:, [1, 0]] if self.indexing == "xy" else rows).float() * 4
        return heatmap.numpy(), self.suppress(coordinates, scores)

    def suppress(self, coordinates, scores):
        if not len(scores):
            return []
        keep, count = self.nms(coordinates, scores, NMS_RADIUS, float("inf"))
        return [(*map(float, coordinates[index].tolist()), float(scores[index])) for index in keep[:count]]


class MVDeTrRunner(MVDetRunner):
    def load(self):
        import numpy as np
        symbols = repo_symbols("mvdetr")
        base = self.base(symbols)
        adapter = SimpleNamespace(base=base, num_cam=base.num_cam, img_shape=base.img_shape,
                                  worldgrid_shape=base.worldgrid_shape, world_reduce=4, img_reduce=12,
                                  Rworld_shape=[dimension // 4 for dimension in base.worldgrid_shape],
                                  Rimg_shape=np.ceil(np.array(base.img_shape) / 12).astype(int).tolist())
        model = symbols.Model(adapter, self.spec.get("arch", "resnet18"),
                              world_feat_arch=self.spec.get("world_feat", "deform_trans"),
                              bottleneck_dim=int(self.spec.get("bottleneck_dim", 128)),
                              outfeat_dim=int(self.spec.get("outfeat_dim", 0)), droupout=0.0)
        model.load_state_dict(read_weights(self.spec), strict=True)
        self.decode, self.nms, self.num_cam = symbols.decode, symbols.nms, base.num_cam
        self.indexing = base.indexing
        self.model = model.cuda().eval()

    def share_gpu(self, gap_s: float):
        self.model.base = _yielding(self.model.base, gap_s, chunked=True)
        self.model.world_feat = _yielding(self.model.world_feat, gap_s, chunked=False)

    def forward(self, batch):
        import torch
        affine = torch.eye(3).repeat(1, batch.shape[1], 1, 1)
        (world_heatmap, world_offset), _ = self.model(batch, affine)
        heatmap = torch.sigmoid(world_heatmap.detach().float().cpu())
        xys = self.decode(heatmap, world_offset.detach().float().cpu(), reduce=4)[0]
        positions = xys[:, :2] if self.indexing == "xy" else xys[:, [1, 0]]
        keep = xys[:, 2] > self.threshold
        return heatmap.squeeze().numpy(), self.suppress(positions[keep], xys[keep, 2])


RUNNERS = {"mvdet": MVDetRunner, "mvdetr": MVDeTrRunner}


class RunnerPool:
    """Keeps at most ``max_loaded`` networks on the GPU (6 GB cards hold one or two comfortably)."""

    def __init__(self, max_loaded: int = 2, fp16: bool = False):
        self.max_loaded, self.fp16 = max(1, max_loaded), fp16
        self.lock = threading.Lock()
        self.runners = OrderedDict()

    def get(self, spec: dict, kind: str, calib_root: Path, fp16: bool | None = None,
            gpu_gap_s: float | None = None) -> Runner:
        fp16 = self.fp16 if fp16 is None else fp16
        key = (spec["id"], str(Path(calib_root).resolve()), fp16, gpu_gap_s)
        with self.lock:
            runner = self.runners.pop(key, None) or RUNNERS[spec["backend"]](spec, kind, calib_root, fp16, gpu_gap_s)
            self.runners[key] = runner
            evicted = []
            while len(self.runners) > self.max_loaded:
                evicted.append(self.runners.popitem(last=False)[1])
        for old in evicted:
            old.unload()
        return runner

    def clear(self):
        """Free the GPU (the calibration-free estimation needs all of it)."""
        with self.lock:
            runners = list(self.runners.values())
            self.runners.clear()
        for runner in runners:
            runner.unload()


def heatmap_png(heatmap, threshold: float) -> str:
    import numpy as np
    from PIL import Image
    peak = max(float(np.max(heatmap)), threshold)
    intensity = np.clip((heatmap - threshold * 0.25) / max(peak - threshold * 0.25, 1e-6), 0, 1)
    rgba = np.empty((*intensity.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 20 + intensity * 225
    rgba[..., 1] = 185 + intensity * 55
    rgba[..., 2] = 165 - intensity * 105
    rgba[..., 3] = intensity * 225
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------------------------------- registry

SETTINGS = re.compile(r"Settings:\s*\n(\{.*?\})\s*\n", re.S)
MODA = re.compile(r"moda: ([0-9.]+)%")
STAMP = re.compile(r"\d{4}-?\d{2}-?\d{2}_\d{2}-?\d{2}-?\d{2}")


def parse_log(text: str) -> tuple[dict, float | None]:
    settings = {}
    match = SETTINGS.search(text)
    if match:
        try:
            settings = ast.literal_eval(match[1])
        except (ValueError, SyntaxError):
            settings = {}
    modas = MODA.findall(text)
    return settings, float(modas[-1]) if modas else None


def _argument(command: list, flag: str):
    if flag in command and command.index(flag) + 1 < len(command):
        return command[command.index(flag) + 1]
    return None


def _drop(*values) -> int:
    for value in values:
        match = re.search(r"drop[_ ]?(\d+)", str(value)) if isinstance(value, str) else None
        if match:
            return int(match[1])
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def kind_of(text: str, declared: str = "", cameras: int | None = None) -> str:
    """Kind of a checkpoint: the dataset its log declares when that is a registered kind, else the longest kind name
    or alias in the declared name, path and manifest ("wildtrack_warehouse" is the warehouse), else (weights loaded
    by hand) the first kind with that many cameras."""
    declared = (declared or "").lower()
    if declared in kinds.KINDS and (cameras is None or kinds.KINDS[declared].cameras == cameras):
        return declared
    named = kinds.match_text(f"{declared} {text}", cameras)
    if named:
        return named
    candidates = kinds.by_cameras(cameras) if cameras is not None else []
    return candidates[0].key if candidates else ""


def local_path(text: str) -> Path:
    """A path written by the training machine, moved by THESIS_DEMO_PATH_MAP ("from=to;from=to"; the Docker image
    maps D:\\mapanything\\datasets to the calibrations copied into it). Separators and case do not matter."""
    plain = text.replace("\\", "/")
    for pair in filter(None, os.environ.get("THESIS_DEMO_PATH_MAP", "").split(";")):
        source, _, target = pair.partition("=")
        source = source.replace("\\", "/").rstrip("/")
        if plain.lower() == source.lower() or plain.lower().startswith(source.lower() + "/"):
            return Path(target) / plain[len(source):].lstrip("/")
    return Path(text)


def describe_checkpoint(backend: str, weights: Path, member: str | None, log_text: str,
                        manifest: dict, cameras: int | None = None) -> dict | None:
    settings, moda = parse_log(log_text)
    try:
        relative = weights.relative_to(PROJECT).as_posix()
    except ValueError:   # loaded from the browser, outside the repo
        relative = weights.as_posix()
    relative += f"!{member}" if member else ""
    config = manifest.get("config", {})
    command = manifest.get("train_command", [])
    data_text = str(settings.get("data_path") or settings.get("data_root") or "")
    kind = kind_of(f"{relative} {json.dumps(manifest)} {data_text}",
                   settings.get("dataset") or manifest.get("dataset") or "", cameras)
    if not kind:
        return None
    drop = _drop(settings.get("drop_ratio"), manifest.get("drop_ratio"), config.get("DROP_RATIO"), relative)
    threshold = settings.get("cls_thres") or _argument(command, "--cls_thres") or config.get("CLS_THRES")
    tags = [part for part in Path(relative.split("!")[0]).parts if part.startswith(("brl", "no_brl", "mse", "bce"))]
    tag = manifest.get("method") if isinstance(manifest.get("method"), str) else None
    tag = tag or (tags[0].split("_pseudo")[0] if tags else settings.get("loss") or "default")
    stamp = STAMP.findall(relative)
    data_root = local_path(settings.get("data_path") or settings.get("data_root") or "/nonexistent")
    spec = {"id": f"{backend}-{hashlib.sha1(relative.encode()).hexdigest()[:8]}", "num_cam": kinds.get(kind).cameras,
            "method": f"{BACKEND_LABELS[backend]} · {'drop ' + str(drop) + '%' if drop else 'full labels'}",
            "name": " · ".join(filter(None, [tag, stamp[-1] if stamp else weights.stem if member else weights.parent.name,
                                             f"MODA {moda:.1f}" if moda is not None else None])),
            "dataset": kind, "backend": backend, "arch": settings.get("arch", "resnet18"),
            "threshold": float(threshold) if threshold is not None else DEFAULT_THRESHOLDS[backend],
            "default_frame": default_frame(kind), "weights": str(weights), "source": relative,
            "test_moda": moda}
    if member:
        spec["zip_member"] = member
    if backend == "mvdetr":
        spec.update(world_feat=settings.get("world_feat", "deform_trans"),
                    bottleneck_dim=settings.get("bottleneck_dim", 128), outfeat_dim=settings.get("outfeat_dim", 0))
    # Trained on its own calibration (e.g. MapAnything estimated cameras): evaluate with the same one.
    if data_root.is_dir() and (data_root / "calibrations").is_dir():
        spec["calib_root"] = str(data_root)
    return spec


SKIP_DIRS = {"scripts", "__pycache__", ".git", "pseudo_cache", "pseudo_check", "node_modules"}


def _walk(folder: Path):
    """Files under folder (long paths included), skipping copied sources and caches."""
    prefix = native(folder)
    for directory, subdirectories, files in os.walk(prefix):
        subdirectories[:] = sorted(name for name in subdirectories if name not in SKIP_DIRS)
        base = Path(directory[len(LONG_PREFIX):] if directory.startswith(LONG_PREFIX) else directory)
        for name in sorted(files):
            yield base / name


def _read_text(path: Path) -> str:
    try:
        with open(native(path), encoding="utf-8", errors="replace") as source:
            return source.read()
    except OSError:
        return ""


def discover_models(project: Path = PROJECT) -> list[dict]:
    specs = []
    for backend, repo in REPOS.items():
        for folder in (repo, project / "experiment_logs" / repo.name):
            if not folder.is_dir():
                continue
            files = list(_walk(folder))
            for weights in (path for path in files if path.name == WEIGHT_NAME):
                manifest = _read_text(weights.parent.parent / "manifest.json") or                     _read_text(weights.parent.parent / "run_manifest.json")
                spec = describe_checkpoint(backend, weights, None, _read_text(weights.with_name("log.txt")),
                                           json.loads(manifest) if manifest else {})
                if spec:
                    specs.append(spec)
            for archive in (path for path in files if path.suffix == ".zip"):
                if archive.with_suffix("").is_dir():
                    continue   # already extracted next to the zip
                try:
                    with zipfile.ZipFile(native(archive)) as bundle:
                        names = set(bundle.namelist())
                        member = next((name for name in sorted(names) if name.endswith(WEIGHT_NAME)), None)
                        if member is None:
                            continue
                        prefix = member[:-len(WEIGHT_NAME)]
                        read = lambda name: bundle.read(name).decode("utf-8", "replace") if name in names else ""
                        log_text = read(prefix + "log.txt")
                        manifest_text = read("manifest.json") or read("run_manifest.json")
                except (zipfile.BadZipFile, OSError):
                    continue
                spec = describe_checkpoint(backend, archive, member, log_text,
                                           json.loads(manifest_text) if manifest_text else {})
                if spec:
                    specs.append(spec)
    return specs


def architecture_from_weights(state: dict) -> dict:
    """Backend and the constructor settings a state dict fixes. The fusion layers are sized by the number of
    cameras (MVDet: map_classifier input = 512 x cameras + 2; MVDeTr: one level embedding per camera), so the
    weights only load for that many views, in the training order."""
    def has(prefix):
        return any(key.startswith(prefix) for key in state)

    if has("depth_classifier.") or has("feat_before_merge."):
        raise ValueError("This is a SHOT checkpoint; the demo has MVDet and MVDeTr runners so far.")
    kernel = lambda key: int(state[key].shape[-1]) if key in state else 7
    if "map_classifier.0.weight" in state and has("base_pt1."):
        return {"backend": "mvdet", "num_cam": (int(state["map_classifier.0.weight"].shape[1]) - 2) // 512,
                "arch": "resnet18" if kernel("base_pt1.0.weight") == 7 else "vgg11"}
    if has("world_feat.") and "world_heatmap.0.weight" in state:
        base_dim = int(state["bottleneck.0.weight"].shape[0]) if "bottleneck.0.weight" in state else 0
        dim = base_dim or 512
        head = state["world_heatmap.0.weight"]
        spec = {"backend": "mvdetr", "arch": "resnet18" if kernel("base.0.weight") == 7 else "vgg11",
                "bottleneck_dim": base_dim, "outfeat_dim": 0 if head.shape[0] == 1 else int(head.shape[0])}
        if "world_feat.lvl_embedding" in state:
            return {**spec, "world_feat": "deform_trans", "num_cam": int(state["world_feat.lvl_embedding"].shape[0])}
        if "world_feat.merge.0.weight" in state:
            return {**spec, "world_feat": "aio", "num_cam": int(state["world_feat.merge.0.weight"].shape[1]) // dim}
        if has("world_feat.deform_conv."):
            return {**spec, "world_feat": "deform_conv",
                    "num_cam": int(state["world_feat.merge_linear.0.weight"].shape[1]) // dim}
        if has("world_feat.encoder."):
            return {**spec, "world_feat": "trans", "num_cam": int(state["world_feat.downsample.0.weight"].shape[1]) // dim}
        if "world_feat.world_feat.0.weight" in state:
            return {**spec, "world_feat": "conv",
                    "num_cam": (int(state["world_feat.world_feat.0.weight"].shape[1]) - 2) // dim}
    if has("model.") or has("state_dict"):
        raise ValueError("This looks like a Lightning (TrackTacular / EarlyBird) checkpoint; no runner for it yet.")
    raise ValueError("Unknown checkpoint layout: expected MVDet (map_classifier) or MVDeTr (world_feat) weights.")


def inspect_checkpoint(path: str | Path) -> dict:
    """Spec for a checkpoint picked in the browser: architecture from the weights, the rest (threshold, test MODA,
    estimated calibration) from a log.txt next to it when there is one."""
    import torch
    weights = Path(path)
    member = None
    if weights.suffix.lower() == ".zip":
        with zipfile.ZipFile(native(weights)) as bundle:
            member = next((name for name in sorted(bundle.namelist()) if name.endswith(".pth")), None)
            if member is None:
                raise ValueError("The zip holds no .pth file.")
            log_text = next((bundle.read(name).decode("utf-8", "replace") for name in bundle.namelist()
                             if name.endswith("log.txt") and Path(name).parent == Path(member).parent), "")
    else:
        if not os.path.isfile(native(weights)):
            raise ValueError(f"Not a file: {weights}")
        log_text = _read_text(weights.with_name("log.txt"))
    spec = {"weights": str(weights), **({"zip_member": member} if member else {})}
    state = read_weights(spec)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    architecture = architecture_from_weights(state)
    settings, _ = parse_log(log_text)
    kind = kind_of(f"{weights} {settings.get('data_path') or ''}", settings.get("dataset") or "",
                   architecture["num_cam"])
    if not kind:
        grids = ", ".join(f"{item.name} ({item.cameras})" for item in kinds.all_kinds())
        raise ValueError(f"{architecture['num_cam']}-camera weights: no scene kind has that many cameras "
                         f"(known: {grids}). Add one to kinds.json.")
    described = describe_checkpoint(architecture["backend"], weights, member, log_text,
                                    {"dataset": kind}, architecture["num_cam"]) or {}
    described.update(architecture)
    described.update(dataset=kind, default_frame=default_frame(kind), source=str(weights), user=True,
                     id=f"user-{hashlib.sha1(os.path.abspath(str(weights)).encode()).hexdigest()[:8]}")
    described.setdefault("threshold", DEFAULT_THRESHOLDS[architecture["backend"]])
    described["method"] = f"{BACKEND_LABELS[architecture['backend']]} · loaded"
    label = weights.parent.name if weights.stem == WEIGHT_NAME[:-4] else weights.stem
    stamps = STAMP.findall(label)   # MVDeTr run folders: 120 characters of settings, then the date
    label = stamps[-1] if stamps else label[:48]
    described["name"] = " · ".join(filter(None, [label,
                                                 f"MODA {described['test_moda']:.1f}" if described.get("test_moda")
                                                 else None]))
    return described


def load_model_specs(config_path: Path | None, discover: bool = True) -> list[dict]:
    specs = discover_models() if discover else []
    if config_path and config_path.is_file():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Model config must be a JSON list")
        for item in raw:
            if item.get("backend") not in RUNNERS or item.get("dataset") not in kinds.KINDS:
                raise ValueError("Unsupported model backend or dataset")
            if not Path(item["weights"]).is_file():
                raise FileNotFoundError(item["weights"])
            item.setdefault("threshold", DEFAULT_THRESHOLDS[item["backend"]])
            item.setdefault("default_frame", default_frame(item["dataset"]))
            item.setdefault("num_cam", kinds.get(item["dataset"]).cameras)
            specs.append(item)
    if len({item["id"] for item in specs}) != len(specs):
        raise ValueError("Duplicate model id")
    return sorted(specs, key=lambda item: (item["dataset"], item["method"], item["name"]))
