"""Choose a supported PyTorch inference device."""

import torch


def inference_device():
    for name in ('cuda', 'xpu'):
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_available():
            return torch.device(name)
    return torch.device('cpu')
