"""Kiểm tra venv của một model: torch thấy GPU, có kernel cho đúng kiến trúc GPU, entrypoint import đủ thư viện."""
import importlib.util
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    'mvdet': ('models/MVDet', 'main.py'),
    'mvdetr': ('models/MVDeTr', 'main.py'),
    'shot': ('models/shot', 'main.py'),
    'earlybird': ('models/EarlyBird/EarlyBird', 'main.py'),
    'tracktacular': ('models/TrackTacular/WorldTrack', 'world_track.py'),
    'tools': ('data_tools', 'generate_pseudo_cache.py'),
}


def arch_supported(cap, arch_list):
    # SASS chạy được trên GPU cùng major, minor >= bản build; PTX (compute_xx) JIT được lên GPU mới hơn
    major, minor = cap
    for arch in arch_list:
        kind, num = arch.split('_')
        a_major, a_minor = int(num[:-1]), int(num[-1])
        if kind == 'sm' and a_major == major and a_minor <= minor:
            return True
        if kind == 'compute' and (a_major, a_minor) <= (major, minor):
            return True
    return False


def check_msda():
    # chạy MSDeformAttn trên GPU; có CUDA op thì so với bản PyTorch (forward + backward)
    import torch
    from multiview_detector.models.ops.functions.ms_deform_attn_func import (
        MSDA, MSDeformAttnFunction, ms_deform_attn_core_pytorch)
    torch.manual_seed(0)
    shapes = torch.as_tensor([[6, 4], [3, 2]], dtype=torch.long, device='cuda')
    level_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    N, M, D, Lq, L, P = 1, 2, 8, 5, 2, 2
    S = int(shapes.prod(1).sum())
    value = torch.rand(N, S, M, D, device='cuda', dtype=torch.float64, requires_grad=True)
    loc = torch.rand(N, Lq, M, L, P, 2, device='cuda', dtype=torch.float64)
    attn = torch.rand(N, Lq, M, L, P, device='cuda', dtype=torch.float64) + 1e-5
    attn = attn / attn.sum((-1, -2), keepdim=True)
    ref = ms_deform_attn_core_pytorch(value, shapes, loc, attn)
    if MSDA is None:
        print('[check] MSDeformAttn: PyTorch (fallback, CUDA op chưa build)')
        return
    out = MSDeformAttnFunction.apply(value, shapes, level_start, loc, attn, 2)
    g_ref, = torch.autograd.grad(ref.sum(), value)
    g_out, = torch.autograd.grad(out.sum(), value)
    if not (torch.allclose(out, ref, atol=1e-8) and torch.allclose(g_out, g_ref, atol=1e-8)):
        sys.exit('[check] MSDeformAttn: CUDA op cho kết quả khác bản PyTorch')
    print('[check] MSDeformAttn: CUDA op (khớp bản PyTorch)')


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # console Windows mặc định cp1252, không in được tiếng Việt
    model = sys.argv[1]
    model_dir, entry = MODELS[model]

    import torch
    print(f'[check] python {platform.python_version()} | torch {torch.__version__} | CUDA {torch.version.cuda}')
    if not torch.cuda.is_available():
        sys.exit('[check] torch không thấy GPU (driver / CUDA_VISIBLE_DEVICES?)')
    arch_list = torch.cuda.get_arch_list()
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        if not arch_supported(cap, arch_list):
            sys.exit(f'[check] torch này không có kernel cho {name} (sm_{cap[0]}{cap[1]}), có: {arch_list}; '
                     f'thử TORCH_CUDA khác')
        print(f'[check] GPU {i}: {name} sm_{cap[0]}{cap[1]}')
    x = torch.randn(64, 64, device='cuda')
    assert torch.isfinite(x @ x).all()

    # import entrypoint (đều có guard __main__) -> lộ ra mọi thư viện thiếu / không tương thích
    sys.dont_write_bytecode = True  # không để lại __pycache__ trong thư mục model
    path = os.path.join(ROOT, model_dir)
    os.chdir(path)
    sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location('_mvpd_entry', os.path.join(path, entry))
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    print(f'[check] import {model_dir}/{entry}: OK')

    if model == 'mvdetr':
        check_msda()
    if model == 'tools':
        import ultralytics
        print(f'[check] ultralytics {ultralytics.__version__}')
    print(f'[check] {model}: OK')


if __name__ == '__main__':
    main()
