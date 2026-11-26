"""Check the installed backend can execute the demo's core operators."""

import cv2
import PIL
import torch
import torch.nn.functional as F
import torchvision

from realtime_demo.device import inference_device


def main():
    device = inference_device()
    cuda, hip, xpu = (getattr(torch.version, name, None) for name in ('cuda', 'hip', 'xpu'))
    accelerated = device.type != 'cpu'
    if not accelerated and any((cuda, hip, xpu)):
        print('PyTorch could not initialize a GPU with the installed backend.', flush=True)
        return 2
    try:
        dtype = torch.float16 if accelerated else torch.float32
        image = torch.ones((1, 3, 8, 8), device=device, dtype=dtype)
        kernel = torch.ones((4, 3, 3, 3), device=device, dtype=dtype)
        grid = torch.zeros((1, 4, 4, 2), device=device, dtype=dtype)
        with torch.inference_mode(), torch.autocast(device.type, enabled=accelerated):
            features = F.conv2d(image, kernel, padding=1)
            value = F.grid_sample(features, grid, align_corners=True).float().mean().item()
        if value != 27:
            raise ValueError(f'Unexpected inference result {value}')
        hardware = getattr(torch, device.type).get_device_name() if accelerated else 'CPU'
    except (RuntimeError, ValueError) as exc:
        print(f'Inference check failed on {device}: {exc}', flush=True)
        return 2 if accelerated else 1
    backend = f'ROCm {hip}' if hip else f'CUDA {cuda}' if cuda else 'XPU' if accelerated else 'CPU'
    print(f'{hardware} | {backend} | torch {torch.__version__} | torchvision {torchvision.__version__} | OpenCV {cv2.__version__} | Pillow {PIL.__version__}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
