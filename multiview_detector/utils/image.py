"""ImageNet input transforms."""
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_tensor(bgr, size):
    """BGR uint8 (H, W, 3) -> normalised (3, h, w) float tensor at `size` = (w, h)."""
    import cv2
    img = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD
