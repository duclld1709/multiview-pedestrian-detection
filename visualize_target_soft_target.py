import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from multiview_detector.datasets import frameDataset, MultiviewX, Wildtrack


def build_dataset(dataset_name, data_root):
    if dataset_name == "wildtrack":
        base = Wildtrack(os.path.join(data_root, "Wildtrack"))
    elif dataset_name == "multiviewx":
        base = MultiviewX(os.path.join(data_root, "MultiviewX"))
    else:
        raise ValueError("dataset must be one of: wildtrack, multiviewx")

    transform = T.Compose([T.Resize([720, 1280]), T.ToTensor()])
    return frameDataset(base, train=False, transform=transform, grid_reduce=4, force_download=False)


def make_soft_target(target, kernel):
    target_4d = target.unsqueeze(0)
    kernel_size = kernel.shape[-1]
    padding = int((kernel_size - 1) / 2)
    with torch.no_grad():
        soft_target = F.conv2d(target_4d, kernel.float(), padding=padding)
    return soft_target.squeeze(0)


def visualize(dataset_name, data_root, index, output, cmap):
    test_set = build_dataset(dataset_name, data_root)
    if index < 0 or index >= len(test_set):
        raise IndexError(f"index {index} is out of range for test set with {len(test_set)} frames")

    _, target, _, frame = test_set[index]
    soft_target = make_soft_target(target, test_set.map_kernel)

    target_np = target.squeeze(0).cpu().numpy()
    soft_target_np = soft_target.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    target_im = axes[0].imshow(target_np, cmap=cmap)
    axes[0].set_title(f"Target frame={frame}")
    axes[0].axis("off")
    fig.colorbar(target_im, ax=axes[0], fraction=0.046, pad=0.04)

    soft_im = axes[1].imshow(soft_target_np, cmap=cmap)
    axes[1].set_title("Soft target (Gaussian)")
    axes[1].axis("off")
    fig.colorbar(soft_im, ax=axes[1], fraction=0.046, pad=0.04)

    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"dataset={dataset_name}")
    print(f"index={index}, frame={frame}")
    print(f"target shape={tuple(target.shape)}")
    print(f"soft target min={soft_target.min().item():.6f}, max={soft_target.max().item():.6f}")
    print(f"saved visualization to {output}")


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.join(os.path.dirname(current_dir), "Dataset")

    parser = argparse.ArgumentParser(description="Visualize target and Gaussian soft target for one test frame.")
    parser.add_argument("--dataset", type=str, default="wildtrack", choices=["wildtrack", "multiviewx"])
    parser.add_argument("--data_root", type=str, default=default_data_root)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=str, default="target_soft_target.png")
    parser.add_argument("--cmap", type=str, default="jet")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args.dataset, args.data_root, args.index, args.output, args.cmap)
